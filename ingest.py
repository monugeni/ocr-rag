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
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
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

CREATE INDEX IF NOT EXISTS idx_pages_doc_page ON pages(doc_id, page_num);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


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
    replace: bool = False
) -> int:
    cur = conn.cursor()

    if replace and filename:
        cur.execute("SELECT id FROM documents WHERE filename = ? AND project = ?", (filename, project))
        row = cur.fetchone()
        if row:
            cur.execute("DELETE FROM pages WHERE doc_id = ?", (row['id'],))
            cur.execute("DELETE FROM sections WHERE doc_id = ?", (row['id'],))
            cur.execute("DELETE FROM documents WHERE id = ?", (row['id'],))
            print(f"  Replaced: {filename}")

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
        breadcrumb = ' > '.join([x[1] for x in stack] + [s['heading']])

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
    for p in pages:
        if p['page_num'] in section_ids:
            sec_id = section_ids[p['page_num']]

        ptype = classify_page(p['content'], p['page_num'])
        cur.execute(
            """INSERT INTO pages (doc_id, page_num, section_id, content, breadcrumb, page_type, char_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, p['page_num'], sec_id, p['content'], p.get('breadcrumb', ''), ptype, len(p['content']))
        )

    conn.commit()

    types = {}
    for p in pages:
        t = classify_page(p['content'], p['page_num'])
        types[t] = types.get(t, 0) + 1

    print(f"  Title: {title}")
    print(f"  Pages: {len(pages)} ({', '.join(f'{v} {k}' for k, v in sorted(types.items()))})")
    print(f"  Sections: {len(sections)}")
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


def ingest_pdf(conn, pdf_path: Path, marker_path: Path, project: str,
               replace=False, skip_llm=False, api_key=None,
               llm_model="claude-sonnet-4-20250514"):
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

    return doc_id


def ingest_project(project_dir, db_path, replace=False, skip_llm=False,
                   api_key=None, llm_model="claude-sonnet-4-20250514"):
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

        marker_file = find_marker_output(pdf)
        if not marker_file:
            print(f"\nSkipping (no Marker output): {pdf.name}")
            continue

        doc_id = ingest_pdf(
            conn, pdf, marker_file, project,
            replace=replace, skip_llm=skip_llm,
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

    p.add_argument('--marker', '-m', help='Marker output file (.json preferred, .md fallback)')
    p.add_argument('--db', '-d', default='docs.db', help='Central database path')
    p.add_argument('--project-name', help='Project name (default: folder name or "default")')
    p.add_argument('--replace', '-r', action='store_true', help='Replace existing documents')
    p.add_argument('--skip-llm', action='store_true', help='Skip LLM metadata extraction')
    p.add_argument('--api-key', help='Anthropic API key (or ANTHROPIC_API_KEY env)')
    p.add_argument('--llm-model', default='claude-sonnet-4-20250514', help='Model for metadata')

    args = p.parse_args()

    if args.project:
        ingest_project(
            args.project, args.db,
            replace=args.replace, skip_llm=args.skip_llm,
            api_key=args.api_key, llm_model=args.llm_model
        )
    else:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            sys.exit(1)

        if args.marker:
            marker_path = Path(args.marker)
        else:
            marker_path = find_marker_output(pdf_path)

        if not marker_path or not marker_path.exists():
            print(f"Marker output not found for {pdf_path.name}")
            print(f"  Run: marker_single {pdf_path} --output_dir {pdf_path.parent}/ --output_format json")
            sys.exit(1)

        project = args.project_name or "default"
        conn = init_db(args.db)
        ingest_pdf(
            conn, pdf_path, marker_path, project,
            replace=args.replace, skip_llm=args.skip_llm,
            api_key=args.api_key, llm_model=args.llm_model
        )
        conn.close()


if __name__ == '__main__':
    main()
