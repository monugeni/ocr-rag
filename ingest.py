#!/usr/bin/env python3
"""
Document RAG Ingestion (Marker JSON + pdfplumber tables)
=========================================================
Pipeline:
  - Marker JSON: per-page content with OCR, headings, section hierarchy
  - pdfplumber: accurate table extraction (replaces Marker tables where possible)

Workflow:
  1. Run Marker:  marker_single input.pdf --output_dir out/ --output_format json
  2. Ingest:      python ingest.py --pdf input.pdf --db docs.db
  3. LLM extracts metadata automatically (needs ANTHROPIC_API_KEY)

Usage:
    python ingest.py --project /documents/EPL-251/ --db docs.db
    python ingest.py --pdf source.pdf --db docs.db --project-name EPL-251
    python ingest.py --pdf source.pdf --marker output.json --db docs.db
"""

import argparse
from collections import Counter
import json
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    title       TEXT NOT NULL,
    filename    TEXT,
    pdf_path    TEXT,
    total_pages INTEGER,
    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading     TEXT NOT NULL,
    level       INTEGER NOT NULL,
    page_start  INTEGER,
    page_end    INTEGER,
    parent_id   INTEGER REFERENCES sections(id),
    breadcrumb  TEXT,
    seq         INTEGER
);

CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_num    INTEGER NOT NULL,
    section_id  INTEGER REFERENCES sections(id),
    content     TEXT NOT NULL,
    breadcrumb  TEXT,
    page_type   TEXT DEFAULT 'text',
    char_count  INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id     INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_start  INTEGER,
    page_end    INTEGER,
    breadcrumb  TEXT,
    text        TEXT NOT NULL,
    chunk_type  TEXT DEFAULT 'text',
    heading_id  INTEGER,
    confidence  REAL DEFAULT 0,
    metadata    TEXT
);

-- FTS5 index on page content and breadcrumbs
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    content,
    breadcrumb,
    content='pages',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, content, breadcrumb)
    VALUES (new.id, new.content, new.breadcrumb);
END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, content, breadcrumb)
    VALUES ('delete', old.id, old.content, old.breadcrumb);
END;
CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, content, breadcrumb)
    VALUES ('delete', old.id, old.content, old.breadcrumb);
    INSERT INTO pages_fts(rowid, content, breadcrumb)
    VALUES (new.id, new.content, new.breadcrumb);
END;

-- FTS5 index on paragraph/clause chunks. This is the primary retrieval index
-- for tender Q&A because it preserves the precise breadcrumb at hit level.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    breadcrumb,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, breadcrumb)
    VALUES (new.id, new.text, new.breadcrumb);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, breadcrumb)
    VALUES ('delete', old.id, old.text, old.breadcrumb);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, breadcrumb)
    VALUES ('delete', old.id, old.text, old.breadcrumb);
    INSERT INTO chunks_fts(rowid, text, breadcrumb)
    VALUES (new.id, new.text, new.breadcrumb);
END;

CREATE INDEX IF NOT EXISTS idx_pages_doc_page ON pages(doc_id, page_num);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_start);

CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    category    TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_by  TEXT DEFAULT 'llm'
);

CREATE TABLE IF NOT EXISTS cross_references (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_doc_id   INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    relationship    TEXT NOT NULL,
    page_num        INTEGER,
    context         TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quality_flags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_num    INTEGER,
    flag_type   TEXT NOT NULL,
    reason      TEXT,
    related_doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    resolved    BOOLEAN DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS page_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id     INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text  TEXT NOT NULL,
    breadcrumb  TEXT,
    chunk_type  TEXT DEFAULT 'text',
    embedding   BLOB NOT NULL,
    model       TEXT NOT NULL,
    metadata    TEXT
);

CREATE INDEX IF NOT EXISTS idx_corrections_doc ON corrections(doc_id);
CREATE INDEX IF NOT EXISTS idx_cross_references_doc ON cross_references(doc_id);
CREATE INDEX IF NOT EXISTS idx_quality_flags_doc ON quality_flags(doc_id);
CREATE INDEX IF NOT EXISTS idx_page_embeddings_page ON page_embeddings(page_id);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    project     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT DEFAULT '',
    error       TEXT,
    doc_id      INTEGER,
    started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS upload_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project      TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    filename     TEXT NOT NULL,
    action       TEXT NOT NULL,
    reason       TEXT,
    job_id       TEXT,
    details      TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_upload_events_project_created
    ON upload_events(project, created_at DESC);

CREATE TABLE IF NOT EXISTS chat_threads (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_project ON chat_threads(project, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    sources     TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, id);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _ensure_table_columns(conn, "ingestion_jobs", {
        "source_path": "TEXT",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
        "delete_after_cancel": "INTEGER NOT NULL DEFAULT 0",
        "cancel_reason": "TEXT",
    })
    _ensure_table_columns(conn, "page_embeddings", {
        "breadcrumb": "TEXT",
        "chunk_type": "TEXT DEFAULT 'text'",
        "chunk_id": "INTEGER",
        "metadata": "TEXT",
    })
    return conn


def _ensure_table_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]):
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


# ---------------------------------------------------------------------------
# HTML to Markdown conversion
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, stripping tags."""
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    return text


def _html_table_to_markdown(html: str) -> str:
    """Convert HTML table to markdown table."""
    # Extract rows
    rows = []
    for tr_match in re.finditer(r'<tr>(.*?)</tr>', html, re.DOTALL):
        cells = []
        for cell_match in re.finditer(r'<t[hd][^>]*>(.*?)</t[hd]>', tr_match.group(1), re.DOTALL):
            cell_text = _html_to_text(cell_match.group(1))
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return _html_to_text(html)

    # Normalize column count
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append('')

    lines = []
    lines.append('| ' + ' | '.join(rows[0]) + ' |')
    lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
    for r in rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def _block_to_markdown(block: dict) -> str:
    """Convert a Marker JSON block to markdown text."""
    bt = block.get('block_type', '')
    html = block.get('html', '')

    if bt == 'SectionHeader':
        m = re.match(r'<h(\d)>(.*?)</h\d>', html, re.DOTALL)
        if m:
            level = int(m.group(1))
            text = _html_to_text(m.group(2))
            return '#' * level + ' ' + text
        return _html_to_text(html)

    if bt in ('Table', 'Form'):
        return _html_table_to_markdown(html)

    if bt == 'ListGroup':
        items = re.findall(r'<li[^>]*>(.*?)</li>', html, re.DOTALL)
        if items:
            return '\n'.join(f'- {_html_to_text(item)}' for item in items)
        return _html_to_text(html)

    # Text, other
    return _html_to_text(html)


# ---------------------------------------------------------------------------
# pdfplumber table enhancement
# ---------------------------------------------------------------------------

def _is_real_table(table_rows: list[list]) -> bool:
    """Filter out pdfplumber false positives."""
    if not table_rows or len(table_rows) < 2:
        return False
    ncols = max(len(r) for r in table_rows)
    if ncols < 2:
        return False
    first_cell = str(table_rows[0][0] or '')
    if ncols == 1 and len(first_cell) > 200:
        return False
    return True


def _plumber_table_to_markdown(rows: list[list]) -> str:
    """Convert pdfplumber table rows to markdown."""
    if not rows or len(rows) < 2:
        return ''
    clean = []
    for row in rows:
        clean.append([str(c).replace('\n', ' ').strip() if c else '' for c in row])

    ncols = max(len(r) for r in clean)
    for r in clean:
        while len(r) < ncols:
            r.append('')

    lines = []
    lines.append('| ' + ' | '.join(clean[0]) + ' |')
    lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
    for r in clean[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def get_pdfplumber_tables(pdf_path: str) -> dict:
    """Extract tables per page from PDF using pdfplumber.

    Returns {page_num: [markdown_table_str, ...]}
    """
    try:
        import pdfplumber
    except ImportError:
        return {}

    result = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for plumber_page in pdf.pages:
                tables = plumber_page.extract_tables()
                real = []
                for t in tables:
                    if _is_real_table(t):
                        md = _plumber_table_to_markdown(t)
                        if md:
                            real.append(md)
                if real:
                    result[plumber_page.page_number] = real
    except Exception as e:
        print(f"  pdfplumber error: {e}")

    return result


# ---------------------------------------------------------------------------
# Marker JSON parsing
# ---------------------------------------------------------------------------

def parse_marker_json(marker_path: str, pdf_path: str = None) -> tuple[list[dict], list[dict]]:
    """Parse Marker JSON output into per-page content with breadcrumbs.

    Returns (pages, sections) where:
      pages = [{'page_num': int, 'content': str, 'breadcrumb': str}, ...]
      sections = [{'heading': str, 'level': int, 'page_num': int, 'breadcrumb': str}, ...]
    """
    with open(marker_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'children' not in data:
        return [], []

    # Get pdfplumber tables for enhancement
    plumber_tables = {}
    if pdf_path:
        plumber_tables = get_pdfplumber_tables(pdf_path)
        if plumber_tables:
            print(f"  pdfplumber: {sum(len(v) for v in plumber_tables.values())} tables from {len(plumber_tables)} pages")

    # Build header index: block_id -> (level, text)
    header_index = {}
    for page_data in data['children']:
        for block in page_data.get('children', []):
            if block.get('block_type') == 'SectionHeader':
                bid = block.get('id', '')
                html = block.get('html', '')
                m = re.match(r'<h(\d)>(.*?)</h\d>', html, re.DOTALL)
                if m:
                    level = int(m.group(1))
                    text = _html_to_text(m.group(2))
                    header_index[bid] = (level, text)

    pages = []
    sections = []
    seen_sections = set()

    for page_idx, page_data in enumerate(data['children']):
        page_num = page_idx + 1
        blocks = page_data.get('children', [])
        section_hierarchy = page_data.get('section_hierarchy', {})

        # Resolve breadcrumb from section_hierarchy
        resolved = {}
        for depth, path in section_hierarchy.items():
            if path in header_index:
                resolved[depth] = header_index[path]
        breadcrumb = ' > '.join(text for _, (_, text) in sorted(resolved.items()))

        # Track sections (deduplicated)
        for depth, path in sorted(section_hierarchy.items()):
            if path in header_index and path not in seen_sections:
                seen_sections.add(path)
                level, text = header_index[path]
                sections.append({
                    'heading': text,
                    'level': level,
                    'page_num': page_num,
                    'breadcrumb': breadcrumb,
                })

        # Convert blocks to markdown
        page_parts = []
        table_idx = 0  # track tables on this page for pdfplumber replacement
        plumber_page_tables = plumber_tables.get(page_num, [])

        for block in blocks:
            bt = block.get('block_type', '')
            if bt in ('Table', 'Form') and table_idx < len(plumber_page_tables):
                # Use pdfplumber table
                page_parts.append(plumber_page_tables[table_idx])
                table_idx += 1
            else:
                md = _block_to_markdown(block)
                if md:
                    page_parts.append(md)

        content = '\n\n'.join(page_parts)
        if content.strip():
            pages.append({
                'page_num': page_num,
                'content': content.strip(),
                'breadcrumb': breadcrumb,
            })

    return pages, sections


def detect_document_boundaries(pages: list[dict], sections: list[dict]) -> list[int]:
    """Detect likely document boundaries within a single PDF's parsed output.

    Looks for H1 headers that appear after the first page — these indicate
    a new logical document started mid-PDF (e.g., the pdf splitter merged
    a piping spec and a civil works spec into one file).

    Returns list of page numbers where new documents begin.
    """
    if not pages or not sections:
        return []

    first_page = pages[0]['page_num']
    min_level = min(s['level'] for s in sections) if sections else 1
    boundaries = []

    for s in sections:
        if s['level'] == min_level and s['page_num'] > first_page:
            boundaries.append(s['page_num'])

    return sorted(set(boundaries))


def split_into_subdocuments(
    pages: list[dict], sections: list[dict], boundaries: list[int]
) -> list[tuple[list[dict], list[dict], str]]:
    """Split pages and sections into sub-documents at boundary pages.

    Returns list of (sub_pages, sub_sections, title_hint) tuples.
    title_hint is the H1 heading text at the start of each sub-document.
    """
    if not boundaries:
        first_heading = sections[0]['heading'] if sections else ''
        return [(pages, sections, first_heading)]

    # Break points: start of first doc, each boundary, past the last page
    breaks = [pages[0]['page_num']] + boundaries + [pages[-1]['page_num'] + 1]

    subdocs = []
    for i in range(len(breaks) - 1):
        start, end = breaks[i], breaks[i + 1]

        sub_pages = [p for p in pages if start <= p['page_num'] < end]
        sub_sections = [s for s in sections if start <= s['page_num'] < end]

        if not sub_pages:
            continue

        # Rebuild breadcrumbs relative to this sub-document's own hierarchy
        # (strip parent context from the previous sub-document)
        if sub_sections:
            top_heading = sub_sections[0]['heading']
            for p in sub_pages:
                bc = p.get('breadcrumb', '')
                # If breadcrumb contains the top heading, trim everything before it
                if top_heading in bc:
                    idx = bc.index(top_heading)
                    p['breadcrumb'] = bc[idx:]
        else:
            top_heading = ''

        subdocs.append((sub_pages, sub_sections, top_heading))

    return subdocs


def parse_marker_md(marker_path: str) -> tuple[list[dict], list[dict]]:
    """Fallback: parse Marker markdown output (no per-page info)."""
    with open(marker_path, 'r', encoding='utf-8') as f:
        text = f.read()

    MD_HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    headings = []
    for m in MD_HEADING.finditer(text):
        h_text = m.group(2).strip().strip('*').strip()
        if h_text:
            headings.append({
                'level': len(m.group(1)),
                'text': h_text,
                'pos': m.start()
            })

    # Build sections with breadcrumbs
    sections = []
    stack = []
    for h in headings:
        while stack and stack[-1][0] >= h['level']:
            stack.pop()
        stack.append((h['level'], h['text']))
        breadcrumb = ' > '.join(t for _, t in stack)
        sections.append({
            'heading': h['text'],
            'level': h['level'],
            'page_num': 1,
            'breadcrumb': breadcrumb,
        })

    pages = [{'page_num': 1, 'content': text.strip(), 'breadcrumb': sections[-1]['breadcrumb'] if sections else ''}] if text.strip() else []
    return pages, sections


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def classify_page(text: str, page_num: int) -> str:
    n = len(text.strip())
    if n < 60:
        return 'drawing'
    if page_num <= 2 and n < 500:
        return 'cover'
    lower = text.lower()
    if any(p in lower for p in ['table of contents', 'list of contents']):
        if re.search(r'\.{3,}\s*\d+', text):
            return 'toc'
    return 'text'


# ---------------------------------------------------------------------------
# Chunk storage for retrieval
# ---------------------------------------------------------------------------

def _json_default(value):
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _insert_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: list[dict],
    page_id_by_num: dict[int, int],
) -> int:
    """Store paragraph/clause chunks for chunk-level FTS retrieval."""
    if not chunks:
        return 0

    inserted = 0
    for idx, chunk in enumerate(chunks):
        page_start = int(chunk.get("page_start") or chunk.get("page_num") or 1)
        page_end = int(chunk.get("page_end") or page_start)
        page_id = page_id_by_num.get(page_start)
        if page_id is None:
            continue

        text = (chunk.get("text") or chunk.get("chunk_text") or "").strip()
        if not text:
            continue

        metadata = chunk.get("metadata")
        if metadata is not None and not isinstance(metadata, str):
            metadata = json.dumps(metadata, ensure_ascii=False, default=_json_default)

        conn.execute(
            """INSERT INTO chunks
               (doc_id, page_id, chunk_index, page_start, page_end, breadcrumb,
                text, chunk_type, heading_id, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                page_id,
                int(chunk.get("chunk_index") or chunk.get("chunk_id") or idx + 1),
                page_start,
                page_end,
                chunk.get("breadcrumb") or "",
                text,
                chunk.get("chunk_type") or chunk.get("type") or "text",
                chunk.get("heading_id"),
                float(chunk.get("confidence") or 0.0),
                metadata,
            ),
        )
        inserted += 1
    return inserted


def _fallback_chunks_for_pages(pages: list[dict]) -> list[dict]:
    """Create retrieval chunks for non-fast ingestion paths."""
    chunks = []
    for page in pages:
        page_num = int(page["page_num"])
        for idx, chunk in enumerate(structured_chunks_for_embedding(
            page.get("content") or "",
            page.get("breadcrumb") or "",
        )):
            chunks.append({
                "page_start": page_num,
                "page_end": page_num,
                "chunk_index": idx + 1,
                "breadcrumb": chunk.get("breadcrumb") or page.get("breadcrumb") or "",
                "text": chunk.get("text") or "",
                "chunk_type": chunk.get("type") or "text",
                "confidence": 0.0,
            })
    return chunks


# ---------------------------------------------------------------------------
# Paragraph / clause aware embedding chunks
# ---------------------------------------------------------------------------

MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
DECIMAL_CLAUSE_RE = re.compile(
    r'^(\d+(?:\.\d+){0,8})(?:[.)])?\s+(.{2,160})$'
)
LETTER_CLAUSE_RE = re.compile(r'^([A-Z])(?:[.)])\s+(.{2,140})$')
ROMAN_CLAUSE_RE = re.compile(
    r'^((?:[IVX]{1,8}|iv|ix|v?i{1,3}|x{1,3}))(?:[.)])\s+(.{2,140})$',
    re.IGNORECASE,
)
STRUCTURE_LABEL_RE = re.compile(
    r'^(section|part|chapter|appendix|annex(?:ure)?|attachment|schedule|'
    r'exhibit|volume)\b',
    re.IGNORECASE,
)


def _normalize_heading_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip(':- ')


def _decimal_depth_from_heading(text: str) -> int:
    m = re.match(r'^(\d+(?:\.\d+)*)\b', _normalize_heading_text(text))
    if not m:
        return 0
    return len(m.group(1).split('.'))


def _looks_like_structural_heading(line: str) -> tuple[int, str, str] | None:
    """Return (level, heading_text, type) when a line should affect breadcrumbs."""
    text = _normalize_heading_text(line)
    if not text or len(text) < 3:
        return None

    md = MD_HEADING_RE.match(text)
    if md:
        return min(len(md.group(1)), 6), _normalize_heading_text(md.group(2)), 'heading'

    # Avoid treating table rows or prose sentences as clause headings.
    words = text.split()
    if len(words) > 18 or len(text) > 180:
        return None
    if text.count('|') >= 2:
        return None
    if re.search(r'[a-z].*,\s+[a-z]', text) and not text.endswith(':'):
        return None

    m = DECIMAL_CLAUSE_RE.match(text)
    if m:
        number, title = m.groups()
        if '.' not in number and len(number) > 2:
            return None
        if title and len(title.split()) <= 14:
            return min(len(number.split('.')), 6), f"{number} {title.strip()}", 'clause'

    m = LETTER_CLAUSE_RE.match(text)
    if m and len(m.group(2).split()) <= 12:
        return 4, f"{m.group(1)}. {m.group(2).strip()}", 'clause'

    m = ROMAN_CLAUSE_RE.match(text)
    if m and len(m.group(2).split()) <= 12:
        return 4, f"{m.group(1)}. {m.group(2).strip()}", 'clause'

    alpha = [c for c in text if c.isalpha()]
    is_caps = bool(alpha) and text == text.upper()
    if (STRUCTURE_LABEL_RE.match(text) or text.endswith(':') or is_caps) and 2 <= len(words) <= 12:
        level = 1 if STRUCTURE_LABEL_RE.match(text) else 3
        return level, text.rstrip(':'), 'heading'

    return None


def _append_chunk(chunks: list[dict], breadcrumb: str, parts: list[str],
                  chunk_type: str = 'text') -> None:
    body = '\n'.join(part.strip() for part in parts if part.strip()).strip()
    if not body:
        return
    context = breadcrumb.strip()
    chunk_text = f"Breadcrumb: {context}\n\n{body}" if context else body
    chunks.append({
        'text': chunk_text,
        'breadcrumb': context,
        'type': chunk_type,
    })


def structured_chunks_for_embedding(
    content: str,
    page_breadcrumb: str = '',
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[dict]:
    """Split page content into paragraph/clause chunks with local breadcrumbs.

    Page-level breadcrumbs are still the fallback. When the page text contains
    markdown headings, numbered clauses, or short heading-shaped lines, the
    current breadcrumb stack is updated before subsequent paragraphs are
    chunked. This gives embeddings a more precise context than arbitrary
    page-level word windows.
    """
    base_parts = [
        _normalize_heading_text(part)
        for part in (page_breadcrumb or '').split(' > ')
        if _normalize_heading_text(part)
    ]
    first_numbered_idx = next(
        (idx for idx, part in enumerate(base_parts) if _decimal_depth_from_heading(part)),
        len(base_parts),
    )
    decimal_level_offset = first_numbered_idx
    base_stack = []
    for idx, part in enumerate(base_parts):
        decimal_depth = _decimal_depth_from_heading(part)
        level = decimal_level_offset + decimal_depth if decimal_depth else idx + 1
        if base_stack and level <= base_stack[-1][0]:
            level = base_stack[-1][0] + 1
        base_stack.append((level, part))
    stack = list(base_stack)
    chunks: list[dict] = []
    paragraph: list[str] = []

    def current_breadcrumb() -> str:
        return ' > '.join(text for _level, text in stack if text)

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = '\n'.join(paragraph).strip()
        paragraph = []
        if not text:
            return
        words = text.split()
        if len(words) <= chunk_size:
            _append_chunk(chunks, current_breadcrumb(), [text])
            return
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            _append_chunk(chunks, current_breadcrumb(), [' '.join(words[start:end])])
            if end == len(words):
                break
            start = max(end - overlap, start + 1)

    for raw_line in (content or '').splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        structural = _looks_like_structural_heading(line)
        if structural:
            flush_paragraph()
            level, heading_text, chunk_type = structural
            decimal_depth = _decimal_depth_from_heading(heading_text)
            if decimal_depth:
                level = decimal_level_offset + decimal_depth
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            _append_chunk(chunks, current_breadcrumb(), [line], chunk_type)
            continue

        paragraph.append(raw_line.rstrip())

    flush_paragraph()
    if chunks:
        return chunks

    return [
        {'text': chunk, 'breadcrumb': page_breadcrumb or '', 'type': 'text'}
        for chunk in chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    ]


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks for embedding."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def compute_embeddings(conn, doc_id, model_name='all-MiniLM-L6-v2'):
    """Compute optional semantic embeddings.

    Chunk-level FTS is the primary retrieval path. Embeddings are stored against
    chunks when available, and only fall back to legacy page-derived chunks for
    older ingestions.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        print(f"  sentence-transformers import failed: {e}")
        print("  Install: pip install sentence-transformers")
        return
    except Exception as e:
        print(f"  Failed to load sentence-transformers: {e}")
        print("  If numpy/scipy version mismatch, try: pip install --upgrade scipy numpy")
        return

    _ensure_table_columns(conn, "page_embeddings", {
        "breadcrumb": "TEXT",
        "chunk_type": "TEXT DEFAULT 'text'",
        "chunk_id": "INTEGER",
        "metadata": "TEXT",
    })

    print(f"  Computing embeddings ({model_name})...")
    model = SentenceTransformer(model_name)

    pages = conn.execute(
        "SELECT id, page_num, content, breadcrumb FROM pages "
        "WHERE doc_id = ? ORDER BY page_num",
        (doc_id,)
    ).fetchall()

    # Delete existing embeddings for these pages
    page_ids = [p['id'] for p in pages]
    if page_ids:
        ph = ','.join('?' * len(page_ids))
        conn.execute(f"DELETE FROM page_embeddings WHERE page_id IN ({ph})", page_ids)

    total_chunks = 0
    stored_chunks = conn.execute(
        """SELECT c.id, c.page_id, c.chunk_index, c.text, c.breadcrumb,
                  c.chunk_type, c.metadata
           FROM chunks c
           JOIN pages p ON p.id = c.page_id
           WHERE c.doc_id = ?
           ORDER BY c.page_start, c.chunk_index, c.id""",
        (doc_id,),
    ).fetchall()

    if stored_chunks:
        embedding_inputs = [
            f"Breadcrumb: {row['breadcrumb']}\n\n{row['text']}".strip()
            if row["breadcrumb"] else row["text"]
            for row in stored_chunks
        ]
        embeddings = model.encode(embedding_inputs)
        for row, emb in zip(stored_chunks, embeddings):
            conn.execute(
                "INSERT INTO page_embeddings "
                "(page_id, chunk_id, chunk_index, chunk_text, breadcrumb, chunk_type, "
                "embedding, model, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["page_id"],
                    row["id"],
                    row["chunk_index"],
                    row["text"],
                    row["breadcrumb"] or "",
                    row["chunk_type"] or "text",
                    emb.astype(np.float32).tobytes(),
                    model_name,
                    row["metadata"],
                )
            )
            total_chunks += 1
        conn.commit()
        print(f"  Embeddings: {total_chunks} stored retrieval chunks from {len(pages)} pages")
        return

    for page in pages:
        chunks = structured_chunks_for_embedding(
            page['content'],
            page['breadcrumb'] or '',
        )
        if not chunks:
            continue

        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = model.encode(chunk_texts)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                "INSERT INTO page_embeddings "
                "(page_id, chunk_index, chunk_text, breadcrumb, chunk_type, embedding, model, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    page['id'],
                    i,
                    chunk['text'],
                    chunk.get('breadcrumb', ''),
                    chunk.get('type', 'text'),
                    emb.astype(np.float32).tobytes(),
                    model_name,
                    json.dumps({"source": "legacy_page_chunk"}, ensure_ascii=False),
                )
            )
            total_chunks += 1

    conn.commit()
    print(f"  Embeddings: {total_chunks} chunks from {len(pages)} pages")


def compute_embeddings_for_project(conn, project, model_name='all-MiniLM-L6-v2'):
    """Compute embeddings for all documents in a project."""
    docs = conn.execute(
        "SELECT id, title FROM documents WHERE project = ? ORDER BY id",
        (project,)
    ).fetchall()

    for d in docs:
        # Check if embeddings already exist
        count = conn.execute("""
            SELECT COUNT(*) FROM page_embeddings pe
            JOIN pages p ON p.id = pe.page_id
            WHERE p.doc_id = ?
        """, (d['id'],)).fetchone()[0]

        if count > 0:
            print(f"\n  [{d['id']}] {d['title']} — {count} embeddings exist, skipping")
            continue

        print(f"\n  [{d['id']}] {d['title']}")
        compute_embeddings(conn, d['id'], model_name)


# ---------------------------------------------------------------------------
# Database ingestion
# ---------------------------------------------------------------------------

def ingest_document(
    conn: sqlite3.Connection,
    pages: list[dict],
    sections: list[dict],
    project: str,
    title: str,
    filename: str = None,
    pdf_path: str = None,
    metadata: dict = None,
    replace: bool = False,
    chunks: list[dict] | None = None,
) -> int:
    cur = conn.cursor()
    existing_doc_id = None

    if replace and filename:
        cur.execute("SELECT id FROM documents WHERE filename = ? AND project = ?", (filename, project))
        row = cur.fetchone()
        if row:
            existing_doc_id = row['id']
            cur.execute(
                "DELETE FROM page_embeddings WHERE page_id IN "
                "(SELECT id FROM pages WHERE doc_id = ?)",
                (existing_doc_id,),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
            cur.execute("DELETE FROM pages WHERE doc_id = ?", (existing_doc_id,))
            cur.execute("DELETE FROM sections WHERE doc_id = ?", (existing_doc_id,))
            print(f"  Replaced: {filename}")

    if existing_doc_id is not None:
        cur.execute(
            """UPDATE documents
               SET project = ?, title = ?, filename = ?, pdf_path = ?,
                   total_pages = ?, ingested_at = CURRENT_TIMESTAMP, metadata = ?
               WHERE id = ?""",
            (project, title, filename, pdf_path, len(pages), json.dumps(metadata or {}), existing_doc_id),
        )
        doc_id = existing_doc_id
    else:
        cur.execute(
            """INSERT INTO documents (project, title, filename, pdf_path, total_pages, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project, title, filename, pdf_path, len(pages), json.dumps(metadata or {}))
        )
        doc_id = cur.lastrowid

    # Insert sections
    stack = []
    section_ids = {}  # page_num -> section_id
    for seq, s in enumerate(sections):
        while stack and stack[-1][0] >= s['level']:
            stack.pop()
        parent_id = stack[-1][2] if stack else None
        breadcrumb = s.get('breadcrumb') or ' > '.join([x[1] for x in stack] + [s['heading']])

        cur.execute(
            """INSERT INTO sections (doc_id, heading, level, page_start, parent_id, breadcrumb, seq)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, s['heading'], s['level'], s['page_num'], parent_id, breadcrumb, seq)
        )
        sid = cur.lastrowid
        stack.append((s['level'], s['heading'], sid))
        section_ids[s['page_num']] = sid

    # Fill page_end
    cur.execute("SELECT id, page_start FROM sections WHERE doc_id = ? ORDER BY seq", (doc_id,))
    srows = cur.fetchall()
    for i, r in enumerate(srows):
        end = srows[i + 1]['page_start'] - 1 if i + 1 < len(srows) else (pages[-1]['page_num'] if pages else 1)
        cur.execute("UPDATE sections SET page_end = ? WHERE id = ?", (max(end, r['page_start']), r['id']))

    # Insert pages
    sec_id = None
    page_id_by_num = {}
    for p in pages:
        if p['page_num'] in section_ids:
            sec_id = section_ids[p['page_num']]

        ptype = p.get('page_type') or classify_page(p['content'], p['page_num'])
        cur.execute(
            """INSERT INTO pages (doc_id, page_num, section_id, content, breadcrumb, page_type, char_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, p['page_num'], sec_id, p['content'], p.get('breadcrumb', ''), ptype, len(p['content']))
        )
        page_id_by_num[int(p['page_num'])] = cur.lastrowid

    retrieval_chunks = chunks if chunks is not None else _fallback_chunks_for_pages(pages)
    chunk_count = _insert_chunks(conn, doc_id, retrieval_chunks, page_id_by_num)

    conn.commit()

    types = {}
    for p in pages:
        t = p.get('page_type') or classify_page(p['content'], p['page_num'])
        types[t] = types.get(t, 0) + 1

    print(f"  Title: {title}")
    print(f"  Pages: {len(pages)} ({', '.join(f'{v} {k}' for k, v in sorted(types.items()))})")
    print(f"  Sections: {len(sections)}")
    print(f"  Retrieval chunks: {chunk_count}")
    return doc_id


# ---------------------------------------------------------------------------
# LLM metadata extraction
# ---------------------------------------------------------------------------

METADATA_PROMPT = """Analyze these first pages of an engineering document. Return ONLY valid JSON:

{{
  "title": "full document title",
  "document_number": "doc/reference number or null",
  "revision": "revision number/letter or null",
  "date": "document date or null",
  "prepared_by": "authoring company or null",
  "prepared_for": "client/recipient or null",
  "project_name": "project name or null",
  "project_number": "project/job number or null",
  "document_type": "standard|specification|datasheet|vendor_document|drawing_list|calculation|procedure|report|manual|correspondence|other",
  "summary": "2-3 sentence summary",
  "equipment_tags": ["list", "of", "tags"],
  "applicable_codes": ["API 560", "ASME B31.3"],
  "keywords": ["5-10", "search", "keywords"]
}}

DOCUMENT TEXT:
---
{text}
---"""


def extract_metadata_llm(pages, api_key=None, model="claude-sonnet-4-20250514"):
    import urllib.request, urllib.error

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  No ANTHROPIC_API_KEY, skipping metadata extraction")
        return None

    text = "\n\n---\n\n".join(
        f"[Page {p['page_num']}]\n{p['content']}" for p in pages[:3]
    )
    if len(text) > 4000:
        text = text[:4000] + "\n...[truncated]"

    try:
        print(f"  Extracting metadata via LLM...")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": METADATA_PROMPT.format(text=text)}]
            }).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        raw = ''.join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        meta = json.loads(raw)
        print(f"  Metadata: {meta.get('title', '?')}")
        return meta

    except Exception as e:
        print(f"  LLM metadata failed: {e}")
        return None


def replay_corrections(conn, doc_id, pdf_path):
    """Replay sidecar corrections after re-ingestion.

    Applies corrections in order: skips → reclassifications → heading fixes →
    breadcrumb overrides → OCR fixes → metadata → cross-refs → quality flags.
    Each step is idempotent.
    """
    from corrections import (
        _load_sidecar, _sidecar_path, _rebuild_breadcrumbs,
        _recalculate_section_page_end, _renumber_sections,
        _update_doc_metadata, _update_doc_total_pages,
    )

    sidecar = _load_sidecar(pdf_path)
    if not sidecar:
        return

    applied = []

    # --- Skipped pages ---
    for page_num in sidecar.get('skipped_pages', []):
        conn.execute(
            "UPDATE pages SET page_type = 'skipped' "
            "WHERE doc_id = ? AND page_num = ?",
            (doc_id, page_num)
        )
    if sidecar.get('skipped_pages'):
        applied.append(f"skipped {len(sidecar['skipped_pages'])} pages")

    # --- Page reclassifications ---
    for page_str, new_type in sidecar.get('page_reclassifications', {}).items():
        conn.execute(
            "UPDATE pages SET page_type = ? WHERE doc_id = ? AND page_num = ?",
            (new_type, doc_id, int(page_str))
        )
    if sidecar.get('page_reclassifications'):
        applied.append(f"reclassified {len(sidecar['page_reclassifications'])} pages")

    # --- Heading removals ---
    for rem in sidecar.get('removed_headings', []):
        sec = conn.execute(
            "SELECT id FROM sections WHERE doc_id = ? AND page_start = ? "
            "AND heading LIKE ?",
            (doc_id, rem['page_num'], rem['text_prefix'] + '%')
        ).fetchone()
        if sec:
            conn.execute(
                "UPDATE pages SET section_id = NULL WHERE section_id = ?",
                (sec['id'],)
            )
            conn.execute("DELETE FROM sections WHERE id = ?", (sec['id'],))
    if sidecar.get('removed_headings'):
        applied.append(f"removed {len(sidecar['removed_headings'])} headings")

    # --- New headings ---
    for new_h in sidecar.get('new_headings', []):
        exists = conn.execute(
            "SELECT 1 FROM sections WHERE doc_id = ? AND page_start = ? "
            "AND heading = ?",
            (doc_id, new_h['page_num'], new_h['text'])
        ).fetchone()
        if not exists:
            max_seq = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM sections "
                "WHERE doc_id = ? AND page_start <= ?",
                (doc_id, new_h['page_num'])
            ).fetchone()[0]
            conn.execute(
                "UPDATE sections SET seq = seq + 1 "
                "WHERE doc_id = ? AND seq > ?",
                (doc_id, max_seq)
            )
            conn.execute(
                "INSERT INTO sections (doc_id, heading, level, page_start, seq) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, new_h['text'], new_h['level'],
                 new_h['page_num'], max_seq + 1)
            )
    if sidecar.get('new_headings'):
        applied.append(f"added {len(sidecar['new_headings'])} headings")

    # --- Heading overrides (level changes, renames) ---
    for key, override in sidecar.get('heading_overrides', {}).items():
        parts = key.split(':', 1)
        if len(parts) != 2:
            continue
        page_num, text_prefix = int(parts[0]), parts[1]
        section = conn.execute(
            "SELECT id FROM sections WHERE doc_id = ? AND page_start = ? "
            "AND heading LIKE ?",
            (doc_id, page_num, text_prefix + '%')
        ).fetchone()
        if section:
            if not override.get('is_heading', True):
                conn.execute(
                    "UPDATE pages SET section_id = NULL WHERE section_id = ?",
                    (section['id'],))
                conn.execute("DELETE FROM sections WHERE id = ?",
                             (section['id'],))
            else:
                updates = []
                params = []
                if 'level' in override:
                    updates.append("level = ?")
                    params.append(override['level'])
                if 'corrected_text' in override:
                    updates.append("heading = ?")
                    params.append(override['corrected_text'])
                if updates:
                    params.append(section['id'])
                    conn.execute(
                        f"UPDATE sections SET {', '.join(updates)} "
                        f"WHERE id = ?", params
                    )
    if sidecar.get('heading_overrides'):
        applied.append(f"applied {len(sidecar['heading_overrides'])} heading overrides")

    # Rebuild sections after heading changes
    if any(sidecar.get(k) for k in
           ('removed_headings', 'new_headings', 'heading_overrides')):
        _renumber_sections(conn, doc_id)
        _recalculate_section_page_end(conn, doc_id)
        _rebuild_breadcrumbs(conn, doc_id)

    # --- Breadcrumb overrides ---
    for page_str, bc in sidecar.get('breadcrumb_overrides', {}).items():
        conn.execute(
            "UPDATE pages SET breadcrumb = ? WHERE doc_id = ? AND page_num = ?",
            (bc, doc_id, int(page_str))
        )
    if sidecar.get('breadcrumb_overrides'):
        applied.append(f"overrode {len(sidecar['breadcrumb_overrides'])} breadcrumbs")

    # --- OCR fixes ---
    for fix in sidecar.get('ocr_fixes', []):
        page = conn.execute(
            "SELECT id, content FROM pages "
            "WHERE doc_id = ? AND page_num = ?",
            (doc_id, fix['page_num'])
        ).fetchone()
        if page and fix['old_text'] in page['content']:
            new_content = page['content'].replace(
                fix['old_text'], fix['new_text'])
            conn.execute(
                "UPDATE pages SET content = ?, char_count = ? WHERE id = ?",
                (new_content, len(new_content), page['id'])
            )
    if sidecar.get('ocr_fixes'):
        applied.append(f"applied {len(sidecar['ocr_fixes'])} OCR fixes")

    # --- Metadata overrides ---
    for page_range, overrides in sidecar.get('metadata_overrides', {}).items():
        for key, value in overrides.items():
            _update_doc_metadata(conn, doc_id, key, value)
    if sidecar.get('metadata_overrides'):
        applied.append("applied metadata overrides")

    # --- Document titles ---
    titles = sidecar.get('document_titles', {})
    if titles:
        # Use the first title override (there's typically one per doc)
        for pr, title in titles.items():
            conn.execute(
                "UPDATE documents SET title = ? WHERE id = ?",
                (title, doc_id)
            )
            break
        applied.append("applied title override")

    # --- Quality flags ---
    for qf in sidecar.get('quality_flags', []):
        conn.execute(
            "INSERT INTO quality_flags (doc_id, page_num, flag_type, reason) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, qf.get('page_num'), qf['type'],
             qf.get('reason', ''))
        )
    if sidecar.get('quality_flags'):
        applied.append(f"restored {len(sidecar['quality_flags'])} quality flags")

    conn.commit()

    if applied:
        print(f"  Sidecar corrections replayed: {'; '.join(applied)}")


def apply_metadata(conn, doc_id, meta):
    cur = conn.cursor()
    if meta.get("title"):
        cur.execute("UPDATE documents SET title = ? WHERE id = ?", (meta["title"], doc_id))

    existing = cur.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,)).fetchone()
    merged = json.loads(existing["metadata"]) if existing and existing["metadata"] else {}
    merged.update(meta)
    cur.execute("UPDATE documents SET metadata = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=False), doc_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Project ingestion
# ---------------------------------------------------------------------------

def find_marker_output(pdf_path: Path) -> Optional[Path]:
    """Find Marker JSON or MD output for a given PDF."""
    stem = pdf_path.stem
    parent = pdf_path.parent

    # Prefer JSON over MD
    candidates = [
        parent / stem / f"{stem}.json",
        parent / f"{stem}.json",
        parent / stem / f"{stem}.md",
        parent / f"{stem}.md",
        parent / stem / "output.json",
        parent / stem / "output.md",
    ]

    subfolder = parent / stem
    if subfolder.is_dir():
        json_files = list(subfolder.glob("*.json"))
        # Exclude _meta.json
        json_files = [f for f in json_files if not f.name.endswith('_meta.json')]
        if json_files:
            candidates = [json_files[0]] + candidates
        md_files = list(subfolder.glob("*.md"))
        if md_files:
            candidates.append(md_files[0])

    for c in candidates:
        if c.exists():
            return c
    return None


def _page_type_from_fast_chunks(chunks: list[dict], page_num: int, content: str) -> str:
    roles = [
        chunk.get("chunk_type") or "text"
        for chunk in chunks
        if int(chunk.get("page_start") or 0) <= page_num <= int(chunk.get("page_end") or 0)
    ]
    if roles:
        common = Counter(role for role in roles if role not in {"heading", "text"})
        if common:
            return common.most_common(1)[0][0]
    return classify_page(content, page_num)


def ingest_fast_pdf(
    conn: sqlite3.Connection,
    pdf_path: Path,
    project: str,
    replace=False,
    skip_llm=False,
    skip_embeddings=False,
    api_key=None,
    llm_model="claude-sonnet-4-20250514",
    ocr=False,
    ocr_jobs=4,
):
    """Ingest a PDF with the fast Poppler/PyMuPDF heading pipeline.

    This path stores chunk-level FTS records with document/chapter/section
    breadcrumbs. It does not require Marker output.
    """
    from fast_pipeline import run_fast_pipeline

    print(f"\nFast ingesting: {pdf_path.name}")
    with tempfile.TemporaryDirectory(prefix="ocr-rag-fast-ingest-") as tmp:
        artifacts = run_fast_pipeline(pdf_path, Path(tmp), ocr=ocr, ocr_jobs=ocr_jobs)

    chunks = []
    chunk_dicts = [asdict(chunk) for chunk in artifacts.chunks]
    for chunk in chunk_dicts:
        chunk["metadata"] = {
            "source": "fast_pipeline",
            "document_title": artifacts.document_title,
            "document_context": asdict(artifacts.document_context),
            "extractor": artifacts.extractor,
            "warnings": artifacts.warnings,
        }
        chunks.append(chunk)

    pages = []
    for page in artifacts.pages:
        content = page.get("content") or ""
        page_num = int(page["page_num"])
        pages.append({
            "page_num": page_num,
            "content": content,
            "breadcrumb": page.get("breadcrumb") or artifacts.document_context.breadcrumb_root,
            "page_type": _page_type_from_fast_chunks(chunk_dicts, page_num, content),
        })

    sections = []
    for section in artifacts.sections:
        sections.append({
            "heading": section["heading"],
            "level": section["level"],
            "page_num": section.get("page_num") or section.get("page_start") or 1,
            "breadcrumb": section.get("breadcrumb") or "",
        })

    title = artifacts.document_title or pdf_path.stem.replace('-', ' ').replace('_', ' ')
    metadata = {
        "document_type": "fast_pdf",
        "document_context": asdict(artifacts.document_context),
        "extractor": artifacts.extractor,
        "warnings": artifacts.warnings,
        "heading_count": len(artifacts.headings),
        "toc_entry_count": len(artifacts.toc_entries),
        "chunk_count": len(artifacts.chunks),
    }
    doc_id = ingest_document(
        conn, pages, sections, project, title,
        filename=pdf_path.name,
        pdf_path=str(pdf_path.resolve()),
        metadata=metadata,
        replace=replace,
        chunks=chunks,
    )

    if not skip_llm:
        meta = extract_metadata_llm(pages, api_key=api_key, model=llm_model)
        if meta:
            meta["project"] = project
            apply_metadata(conn, doc_id, meta)

    replay_corrections(conn, doc_id, str(pdf_path.resolve()))

    if not skip_embeddings:
        compute_embeddings(conn, doc_id)

    return doc_id


def ingest_pdf(conn, pdf_path: Path, marker_path: Path, project: str,
               replace=False, skip_llm=False, skip_embeddings=False,
               api_key=None, llm_model="claude-sonnet-4-20250514"):
    """Ingest a single PDF using Marker JSON + pdfplumber tables."""
    print(f"\nIngesting: {pdf_path.name}")
    print(f"  Marker: {marker_path}")

    if marker_path.suffix == '.json':
        pages, sections = parse_marker_json(str(marker_path), str(pdf_path))
    else:
        print(f"  WARNING: Markdown input — no per-page splitting. Re-run Marker with --output_format json")
        pages, sections = parse_marker_md(str(marker_path))

    if not pages:
        print(f"  No pages extracted, skipping")
        return None

    title = pdf_path.stem.replace('-', ' ').replace('_', ' ')
    doc_id = ingest_document(
        conn, pages, sections, project, title,
        filename=pdf_path.name,
        pdf_path=str(pdf_path.resolve()),
        replace=replace
    )

    if not skip_llm:
        meta = extract_metadata_llm(pages, api_key=api_key, model=llm_model)
        if meta:
            meta["project"] = project
            apply_metadata(conn, doc_id, meta)

    # Replay sidecar corrections from previous LLM feedback
    replay_corrections(conn, doc_id, str(pdf_path.resolve()))

    # Compute embeddings for semantic search
    if not skip_embeddings:
        compute_embeddings(conn, doc_id)

    return doc_id


def ingest_project(project_dir, db_path, replace=False, skip_llm=False,
                   skip_embeddings=False, api_key=None,
                   llm_model="claude-sonnet-4-20250514",
                   fast_pdf=False, ocr=False, ocr_jobs=4):
    project_dir = Path(project_dir)
    project = project_dir.name

    print(f"\n{'=' * 60}")
    print(f"Project: {project}")
    print(f"Folder:  {project_dir}")
    print(f"DB:      {db_path}")
    print(f"{'=' * 60}")

    conn = init_db(db_path)
    pdfs = sorted(project_dir.glob("*.pdf")) + sorted(project_dir.glob("*/*.pdf"))

    existing = set()
    if not replace:
        for row in conn.execute("SELECT filename FROM documents WHERE project = ?", (project,)):
            existing.add(row["filename"])

    ingested = 0
    for pdf in pdfs:
        if pdf.name in existing:
            print(f"\nSkipping (exists): {pdf.name}")
            continue

        if fast_pdf:
            doc_id = ingest_fast_pdf(
                conn, pdf, project,
                replace=replace, skip_llm=skip_llm,
                skip_embeddings=skip_embeddings,
                api_key=api_key, llm_model=llm_model,
                ocr=ocr, ocr_jobs=ocr_jobs,
            )
        else:
            marker_file = find_marker_output(pdf)
            if not marker_file:
                print(f"\nSkipping (no Marker output): {pdf.name}")
                continue

            doc_id = ingest_pdf(
                conn, pdf, marker_file, project,
                replace=replace, skip_llm=skip_llm,
                skip_embeddings=skip_embeddings,
                api_key=api_key, llm_model=llm_model
            )
        if doc_id:
            ingested += 1

    # Summary
    cur = conn.cursor()
    print(f"\n{'=' * 60}")
    print(f"PROJECT: {project}")
    print(f"{'=' * 60}")

    stats = cur.execute("""
        SELECT COUNT(DISTINCT d.id) as docs,
               SUM(d.total_pages) as pages,
               (SELECT COUNT(*) FROM sections s JOIN documents dd ON dd.id=s.doc_id WHERE dd.project=?) as sections
        FROM documents d WHERE d.project = ?
    """, (project, project)).fetchone()

    print(f"Documents: {stats['docs']} ({ingested} new)")
    print(f"Pages: {stats['pages']}")
    print(f"Sections: {stats['sections']}")

    print(f"\nDocument list:")
    for row in cur.execute(
        "SELECT id, title, filename, total_pages, metadata FROM documents WHERE project = ? ORDER BY id",
        (project,)
    ):
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        dtype = meta.get("document_type", "")
        print(f"  [{row['id']}] {row['title']}")
        print(f"      {row['filename']} | {row['total_pages']}pp | {dtype}")

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Ingest PDFs into SQLite FTS5 database (Marker JSON + pdfplumber)',
        epilog="""
Examples:
  python ingest.py --project /documents/EPL-251/ --db docs.db
  python ingest.py --pdf source.pdf --db docs.db --project-name EPL-251
  python ingest.py --pdf source.pdf --marker output.json --db docs.db
"""
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--project', help='Project folder (contains PDFs + Marker output)')
    mode.add_argument('--pdf', help='Single PDF file to ingest')
    mode.add_argument('--embed-only', metavar='PROJECT_DIR',
                      help='Only compute embeddings for an already-ingested project (no re-ingest)')

    p.add_argument('--marker', '-m', help='Marker output file (.json preferred, .md fallback)')
    p.add_argument('--db', '-d', default='docs.db', help='Central database path')
    p.add_argument('--project-name', help='Project name (default: folder name or "default")')
    p.add_argument('--replace', '-r', action='store_true', help='Replace existing documents')
    p.add_argument('--skip-llm', action='store_true', help='Skip LLM metadata extraction')
    p.add_argument('--skip-embeddings', action='store_true', help='Skip embedding computation')
    p.add_argument('--fast-pdf', action='store_true',
                   help='Use the fast Poppler/PyMuPDF PDF pipeline instead of Marker output')
    p.add_argument('--ocr', action='store_true',
                   help='With --fast-pdf, run ocrmypdf --skip-text before extraction')
    p.add_argument('--ocr-jobs', type=int, default=4,
                   help='ocrmypdf worker count for --ocr (default: 4)')
    p.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                   help='Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)')
    p.add_argument('--api-key', help='Anthropic API key (or ANTHROPIC_API_KEY env)')
    p.add_argument('--llm-model', default='claude-sonnet-4-20250514', help='Model for metadata')

    args = p.parse_args()

    if args.embed_only:
        # Only compute embeddings for existing project
        project_dir = Path(args.embed_only)
        project_name = args.project_name or project_dir.name
        conn = init_db(args.db)
        print(f"\nComputing embeddings for project: {project_name}")
        print(f"Model: {args.embedding_model}")
        compute_embeddings_for_project(conn, project_name, args.embedding_model)
        conn.close()
        print("\nDone.")
    elif args.project:
        ingest_project(
            args.project, args.db,
            replace=args.replace, skip_llm=args.skip_llm,
            skip_embeddings=args.skip_embeddings,
            api_key=args.api_key, llm_model=args.llm_model,
            fast_pdf=args.fast_pdf,
            ocr=args.ocr, ocr_jobs=args.ocr_jobs,
        )
    else:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            sys.exit(1)

        project = args.project_name or "default"
        conn = init_db(args.db)
        if args.fast_pdf:
            ingest_fast_pdf(
                conn, pdf_path, project,
                replace=args.replace, skip_llm=args.skip_llm,
                skip_embeddings=args.skip_embeddings,
                api_key=args.api_key, llm_model=args.llm_model,
                ocr=args.ocr, ocr_jobs=args.ocr_jobs,
            )
        else:
            if args.marker:
                marker_path = Path(args.marker)
            else:
                marker_path = find_marker_output(pdf_path)

            if not marker_path or not marker_path.exists():
                print(f"Marker output not found for {pdf_path.name}")
                print(f"  Run: marker_single {pdf_path} --output_dir {pdf_path.parent}/ --output_format json")
                sys.exit(1)

            ingest_pdf(
                conn, pdf_path, marker_path, project,
                replace=args.replace, skip_llm=args.skip_llm,
                skip_embeddings=args.skip_embeddings,
                api_key=args.api_key, llm_model=args.llm_model
            )
        conn.close()


if __name__ == '__main__':
    main()
