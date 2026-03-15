#!/usr/bin/env python3
"""
Document RAG MCP Server
========================
Single server for all projects. Every search requires a project name.

Tools:
    Discovery:
        list_projects       - List all projects
        list_documents      - List documents in a project
        get_document_info   - Full metadata for a document
        get_toc             - Section tree for a document

    Search:
        search_pages        - Full-text keyword search (primary tool)
        search_sections     - Search section headings

    Navigation:
        get_page            - Get full page content with context
        get_adjacent        - Get next/previous page

    Fallback (LLM calls these when Marker output looks wrong):
        reextract_page      - Re-extract page text from original PDF via pdfplumber
        reextract_table     - Re-extract table from original PDF via pdfplumber

Usage:
    python mcp_server.py --db docs.db --port 8200
"""

import argparse
import json
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from corrections import register_correction_tools


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = "docs.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def sanitize_fts(query: str) -> str:
    """Sanitize query for FTS5. Splits hyphenated terms, quotes parts."""
    words = query.strip().split()
    out = []
    for w in words:
        if w.upper() in ('AND', 'OR', 'NOT', 'NEAR'):
            continue
        if re.search(r'[-/\\@#$%&*()+=]', w):
            parts = re.split(r'[-/\\]', w)
            out.extend(f'"{p.strip()}"' for p in parts if p.strip())
        else:
            w = w.strip('"\'')
            if w:
                out.append(w)
    return ' '.join(out)


def project_doc_ids(conn, project: str) -> list[int]:
    return [r['id'] for r in conn.execute(
        "SELECT id FROM documents WHERE project = ?", (project,)
    ).fetchall()]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

SSE_PORT = 8200

mcp = FastMCP(
    "Document RAG",
    instructions="""\
Search and navigate engineering project documents (specifications, procedures, QAPs, tender docs).
All searches are scoped by project. Use list_projects first.

=== SEARCH STRATEGY (CRITICAL) ===

search_pages uses FTS keyword matching — NOT semantic/vector search. It only finds exact word matches.
This means you MUST try multiple query variations to get good recall. Treat every search like this:

1. NEVER search just once. Always run 3-8 query variations for any question.
2. Strip question words: "What is the MOC of superheater coil tubes?" → search for keywords only.
3. Try DIFFERENT word combinations, not just the obvious one:
   - Full terms: "superheater coil tube material"
   - Subset pairs: "superheater tube", "superheater material", "tube material"
   - Reordered: "material specification superheater", "tube grade boiler"
4. Expand abbreviations — these documents use both forms:
   MOC = material of construction | OD = outside diameter | ID = inside diameter
   NB = nominal bore | CS = carbon steel | SS = stainless steel | AS/LAS = alloy/low alloy steel
   QAP = quality assurance plan | IBR = Indian boiler regulations | PWHT = post weld heat treatment
   NDE/NDT = non-destructive examination/testing | MTC = material test certificate
   PSV/PRV = pressure safety/relief valve | WPS = welding procedure specification
   LSG = lower steam generator | USG = upper steam generator | TA = turnaround
5. Try SYNONYMS — engineering docs use varied terminology:
   tube ↔ pipe ↔ coil ↔ tubing | material ↔ grade ↔ alloy ↔ specification
   repair ↔ replacement ↔ maintenance | inspection ↔ testing ↔ examination
   drawing ↔ diagram ↔ figure | specification ↔ standard ↔ requirement
   superheater ↔ super heater | economizer ↔ economiser | boiler ↔ steam generator
   refractory ↔ lining ↔ castable | schedule ↔ sch (wall thickness)
6. Try ASME/SA material spec numbers if relevant:
   SA 106 (CS pipe), SA 213 (alloy tube), SA 516 (CS plate), SA 240 (SS plate)
   Grade T9, T11, T22 (chrome-moly alloys)
7. Look for ANNEXURE references — specs often say "as per Annexure X" and the detail is elsewhere.
   Follow those references by searching for the annexure content.
8. Use search_sections to find which DOCUMENT likely has the answer, then search_pages within that doc.

Example — searching for "MOC of superheater coil tubes":
  search 1: "superheater coil tube material"
  search 2: "superheater tube material"
  search 3: "material of construction superheater"
  search 4: "SA 213 superheater"
  search 5: "tube specification grade superheater"
  search 6: "coil material supply boiler"
  search 7: "QAP superheater tube"
→ Combine all results, read the most promising pages with get_page.

After finding relevant pages, use get_page with include_adjacent=true to see surrounding context.
If Marker output looks garbled, use reextract_page or reextract_table for a fresh PDF extraction.

=== DATA CORRECTION (YOU CAN AND SHOULD FIX WHAT YOU SEE) ===

*** CRITICAL RULE: NEVER ALTER DOCUMENT CONTENT ***
The page content is extracted from official engineering documents — specifications,
contracts, procedures, QAPs. This content is SACRED. Even if you see a spelling mistake,
a grammatical error, or a wrong number in the document text, DO NOT CHANGE IT. That is
what the original document says, and it must remain exactly as-is. These are legal and
contractual documents — altering their content would be falsification.

What you CAN correct (extraction artifacts only):
  ✓ Heading structure — levels, missing/false headings (the EXTRACTOR got these wrong, not the document)
  ✓ OCR artifacts — garbled characters from scanning (e.g. "tbe" that is clearly "the" due to
    a scanning error, random symbols injected by OCR). Only fix text that is OBVIOUSLY a scanning
    artifact, never "correct" what might be the original document's actual text.
  ✓ Document structure — splits, merges, page classification, breadcrumbs
  ✓ Metadata — titles, types, revision numbers, keywords (these are YOUR labels, not document content)
  ✓ Running headers/footers — repeated extraction artifacts cluttering every page
  ✓ Quality flags — flagging problems for human review
  ✓ Cross-references and keywords — adding search aids

What you must NEVER do:
  ✗ Fix spelling or grammar in document content — that's what the document actually says
  ✗ "Correct" numbers, dates, or technical values — even if they look wrong
  ✗ Rewrite or rephrase any document text
  ✗ Remove content that looks redundant — it may be intentional
  When in doubt, leave the content alone and flag_low_quality instead.

Your corrections improve the system permanently: they write to both the live database
(immediate effect on search results) AND a sidecar JSON file next to the PDF. When the
document is re-ingested later, your corrections are automatically replayed. The system
gets better every time you use it.

WHEN TO CORRECT (do this as you go, not as a separate task):
- You see a heading that shouldn't be one (e.g. a table row detected as H2):
  → remove_heading(doc_id, page_num, text_prefix)
- A real heading was missed by the extractor:
  → add_heading(doc_id, page_num, text, level)
- Heading is at the wrong level (H3 should be H1):
  → change_heading_level(doc_id, page_num, text_prefix, new_level)
- Heading text is garbled from OCR (not a document correction — an extraction fix):
  → rename_heading(doc_id, page_num, old_text_prefix, new_text)
- Two documents should actually be one (were incorrectly split):
  → merge_documents(doc_id_a, doc_id_b)
- One document contains multiple logical documents:
  → split_document(doc_id, at_page)
- OCR artifact in content (garbled characters from scanning, NOT a document typo):
  → fix_ocr_text(doc_id, page_num, old_text, new_text)
- A page is classified wrong (text marked as drawing, or vice versa):
  → reclassify_page(doc_id, page_num, new_type)
- A page is junk (blank, duplicate, garbage OCR):
  → skip_page(doc_id, page_num)
- A page belongs in a different document:
  → move_page_to_document(page_num, from_doc_id, to_doc_id)
- The breadcrumb/context on a page is wrong:
  → set_page_breadcrumb(doc_id, page_num, breadcrumb)
- A repeated header/footer clutters the content (extraction artifact on every page):
  → add_running_header(doc_id, text) — strips it from all pages immediately
- Document title is wrong or unhelpful:
  → set_document_title(doc_id, title)
- You can identify the document type, number, or revision:
  → set_document_type, set_document_number, set_revision
- You spot cross-references between documents:
  → add_cross_reference(doc_id, page_num, target_doc_id, context)
  → link_documents(doc_id, related_doc_id, relationship)
- You want to tag keywords or equipment for better future search:
  → add_keywords(doc_id, keywords_csv)
  → add_equipment_tags(doc_id, tags_csv)
- A page has terrible OCR quality or suspicious content:
  → flag_low_quality(doc_id, page_num, reason) — flag for human review, don't alter
  → suggest_reocr(doc_id, reason)
- Two documents look like duplicates:
  → flag_duplicate(doc_id, duplicate_of_doc_id)

Be proactive with STRUCTURAL fixes. If the document title is "f 41  maintenance work
procedure   s" and you can see from the content it's actually "F-41: Maintenance Work
Procedure - Scaffolding", fix the title. If a page has 50 false headings from table rows,
remove them. But never touch the actual document text.
""",
    port=SSE_PORT,
)

register_correction_tools(mcp, get_db)


# ===== Discovery =====

@mcp.tool()
def list_projects() -> str:
    """List all projects with document and page counts."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT project, COUNT(*) as docs, SUM(total_pages) as pages
            FROM documents GROUP BY project ORDER BY project
        """).fetchall()
        return json.dumps([
            {"project": r["project"], "documents": r["docs"], "total_pages": r["pages"]}
            for r in rows
        ], indent=2)


@mcp.tool()
def list_documents(project: str) -> str:
    """List all documents in a project.

    Args:
        project: Project name (from list_projects).
    """
    with get_db() as conn:
        docs = conn.execute(
            "SELECT * FROM documents WHERE project = ? ORDER BY id", (project,)
        ).fetchall()

        results = []
        for d in docs:
            meta = json.loads(d["metadata"]) if d["metadata"] else {}
            sec_count = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE doc_id = ?", (d["id"],)
            ).fetchone()[0]

            results.append({
                "id": d["id"],
                "title": d["title"],
                "filename": d["filename"],
                "total_pages": d["total_pages"],
                "sections": sec_count,
                "document_type": meta.get("document_type"),
                "prepared_by": meta.get("prepared_by"),
                "summary": meta.get("summary"),
            })

        return json.dumps(results, indent=2)


@mcp.tool()
def get_document_info(doc_id: int) -> str:
    """Get full metadata for a document including LLM-extracted details.

    Args:
        doc_id: Document ID.
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})

        meta = json.loads(d["metadata"]) if d["metadata"] else {}

        types = {}
        for r in conn.execute(
            "SELECT page_type, COUNT(*) as n FROM pages WHERE doc_id = ? GROUP BY page_type",
            (doc_id,)
        ):
            types[r["page_type"]] = r["n"]

        return json.dumps({
            "id": d["id"], "project": d["project"], "title": d["title"],
            "filename": d["filename"], "pdf_path": d["pdf_path"],
            "total_pages": d["total_pages"], "page_types": types,
            "document_number": meta.get("document_number"),
            "revision": meta.get("revision"),
            "date": meta.get("date"),
            "prepared_by": meta.get("prepared_by"),
            "prepared_for": meta.get("prepared_for"),
            "project_name": meta.get("project_name"),
            "document_type": meta.get("document_type"),
            "summary": meta.get("summary"),
            "equipment_tags": meta.get("equipment_tags"),
            "applicable_codes": meta.get("applicable_codes"),
            "keywords": meta.get("keywords"),
        }, indent=2)


@mcp.tool()
def get_toc(doc_id: int, max_level: int = 4) -> str:
    """Get the section tree (table of contents) for a document.

    Args:
        doc_id: Document ID.
        max_level: Maximum heading depth (default 4).
    """
    with get_db() as conn:
        d = conn.execute("SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})

        sections = conn.execute(
            """SELECT heading, level, page_start, page_end, breadcrumb
               FROM sections WHERE doc_id = ? AND level <= ? ORDER BY seq""",
            (doc_id, max_level)
        ).fetchall()

        return json.dumps({
            "doc_title": d["title"],
            "sections": [
                {"heading": s["heading"], "level": s["level"],
                 "pages": f"{s['page_start']}-{s['page_end']}",
                 "breadcrumb": s["breadcrumb"]}
                for s in sections
            ]
        }, indent=2)


# ===== Search =====

@mcp.tool()
def search_pages(
    project: str,
    query: str,
    doc_id: Optional[int] = None,
    page_type: Optional[str] = None,
    max_results: int = 10
) -> str:
    """Full-text keyword search across a project's pages. This is FTS, not semantic search.

    IMPORTANT: Always call this multiple times with different query variations.
    A single query will miss results because FTS only matches exact words.
    Try 3-8 variations: rearrange words, use synonyms, expand abbreviations,
    try 2-word subsets. See server instructions for detailed search strategy.

    Args:
        project: Project name (required).
        query: 2-5 keyword terms (not a full sentence). Drop stop words.
               Examples: "superheater tube material", "SA 213 grade", "QAP coil tube"
        doc_id: Optional. Restrict to one document.
        page_type: Optional. Filter: 'text', 'drawing', 'table', 'toc', 'cover'.
        max_results: Default 10, max 25.

    Returns:
        Matching pages with doc_title, page_num, breadcrumb, snippet, and rank.
    """
    clean = sanitize_fts(query)
    if not clean:
        return json.dumps({"error": "Empty query"})

    with get_db() as conn:
        ids = project_doc_ids(conn, project)
        if not ids:
            return json.dumps({"error": f"No documents in project '{project}'"})

        conds = ["pages_fts MATCH ?"]
        params = [clean]

        ph = ','.join('?' * len(ids))
        conds.append(f"p.doc_id IN ({ph})")
        params.extend(ids)

        if doc_id is not None:
            conds.append("p.doc_id = ?")
            params.append(doc_id)
        if page_type:
            conds.append("p.page_type = ?")
            params.append(page_type)

        params.append(min(max_results, 25))

        try:
            rows = conn.execute(f"""
                SELECT p.doc_id, d.title as doc_title, d.total_pages,
                       p.page_num, p.breadcrumb, p.page_type, p.char_count,
                       snippet(pages_fts, 0, '>>>', '<<<', '...', 40) as snippet, rank
                FROM pages_fts
                JOIN pages p ON p.id = pages_fts.rowid
                JOIN documents d ON d.id = p.doc_id
                WHERE {' AND '.join(conds)}
                ORDER BY rank LIMIT ?
            """, params).fetchall()
        except sqlite3.OperationalError:
            # Retry with simplified query
            params[0] = ' '.join(w for w in clean.split() if not w.startswith('"'))
            try:
                rows = conn.execute(f"""
                    SELECT p.doc_id, d.title as doc_title, d.total_pages,
                           p.page_num, p.breadcrumb, p.page_type, p.char_count,
                           snippet(pages_fts, 0, '>>>', '<<<', '...', 40) as snippet, rank
                    FROM pages_fts
                    JOIN pages p ON p.id = pages_fts.rowid
                    JOIN documents d ON d.id = p.doc_id
                    WHERE {' AND '.join(conds)}
                    ORDER BY rank LIMIT ?
                """, params).fetchall()
            except sqlite3.OperationalError as e:
                return json.dumps({"error": str(e), "query": clean})

        return json.dumps({
            "query": query, "result_count": len(rows),
            "results": [
                {"doc_id": r["doc_id"], "doc_title": r["doc_title"],
                 "page_num": r["page_num"], "total_pages": r["total_pages"],
                 "breadcrumb": r["breadcrumb"], "page_type": r["page_type"],
                 "snippet": r["snippet"], "rank": round(r["rank"], 4)}
                for r in rows
            ]
        }, indent=2)


@mcp.tool()
def search_sections(project: str, query: str, doc_id: Optional[int] = None) -> str:
    """Search section headings within a project.

    Args:
        project: Project name.
        query: Keywords to match in section headings.
        doc_id: Optional. Restrict to one document.
    """
    with get_db() as conn:
        conds = ["(s.heading LIKE ? OR s.breadcrumb LIKE ?)", "d.project = ?"]
        params = [f"%{query}%", f"%{query}%", project]

        if doc_id is not None:
            conds.append("s.doc_id = ?")
            params.append(doc_id)

        rows = conn.execute(f"""
            SELECT s.doc_id, d.title as doc_title, s.heading, s.level,
                   s.breadcrumb, s.page_start, s.page_end
            FROM sections s JOIN documents d ON d.id = s.doc_id
            WHERE {' AND '.join(conds)}
            ORDER BY s.doc_id, s.seq
        """, params).fetchall()

        return json.dumps({
            "query": query, "result_count": len(rows),
            "results": [dict(r) for r in rows]
        }, indent=2)


# ===== Navigation =====

@mcp.tool()
def get_page(doc_id: int, page_num: int, include_adjacent: bool = False) -> str:
    """Get full content of a specific page with section context.

    Args:
        doc_id: Document ID (from search results).
        page_num: Page number.
        include_adjacent: If True, also returns prev/next page content.
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})

        p = conn.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_num = ?", (doc_id, page_num)
        ).fetchone()
        if not p:
            return json.dumps({"error": f"Page {page_num} not found", "total_pages": d["total_pages"]})

        result = {
            "doc_id": doc_id, "doc_title": d["title"], "project": d["project"],
            "page_num": p["page_num"], "total_pages": d["total_pages"],
            "breadcrumb": p["breadcrumb"], "page_type": p["page_type"],
            "content": p["content"]
        }

        if include_adjacent:
            for direction, offset in [("prev_page", -1), ("next_page", 1)]:
                adj = conn.execute(
                    "SELECT page_num, content, breadcrumb, page_type FROM pages WHERE doc_id = ? AND page_num = ?",
                    (doc_id, page_num + offset)
                ).fetchone()
                if adj:
                    result[direction] = {
                        "page_num": adj["page_num"], "breadcrumb": adj["breadcrumb"],
                        "page_type": adj["page_type"], "content": adj["content"]
                    }

        return json.dumps(result, indent=2)


@mcp.tool()
def get_pages(doc_id: int, page_start: int, page_end: int) -> str:
    """Get full content of a range of pages. Max 10 pages per call.

    Args:
        doc_id: Document ID.
        page_start: First page number (inclusive).
        page_end: Last page number (inclusive).
    """
    with get_db() as conn:
        d = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})

        page_end = min(page_end, page_start + 9)  # cap at 10 pages

        rows = conn.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_num BETWEEN ? AND ? "
            "ORDER BY page_num",
            (doc_id, page_start, page_end)
        ).fetchall()

        if not rows:
            return json.dumps({
                "error": f"No pages found in range {page_start}-{page_end}",
                "total_pages": d["total_pages"]
            })

        return json.dumps({
            "doc_id": doc_id, "doc_title": d["title"], "project": d["project"],
            "total_pages": d["total_pages"],
            "pages": [
                {"page_num": p["page_num"], "breadcrumb": p["breadcrumb"],
                 "page_type": p["page_type"], "content": p["content"]}
                for p in rows
            ]
        }, indent=2)


@mcp.tool()
def get_adjacent(doc_id: int, page_num: int, direction: str = "next") -> str:
    """Get the next or previous page.

    Args:
        doc_id: Document ID.
        page_num: Current page number.
        direction: "next" or "prev".
    """
    target = page_num + (1 if direction == "next" else -1)
    return get_page(doc_id, target)


# ===== Fallback extraction tools =====

@mcp.tool()
def reextract_page(doc_id: int, page_start: int, page_end: Optional[int] = None) -> str:
    """Re-extract text from original PDF using pdfplumber (layout-preserved).

    Use this when Marker's output for a page looks garbled, has missing text,
    or seems to have OCR errors. This goes back to the source PDF for a fresh extraction.

    Args:
        doc_id: Document ID.
        page_start: First page to re-extract (1-indexed).
        page_end: Last page (default: same as page_start).
    """
    if page_end is None:
        page_end = page_start

    with get_db() as conn:
        d = conn.execute("SELECT pdf_path, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})
        if not d["pdf_path"] or not Path(d["pdf_path"]).exists():
            return json.dumps({"error": f"Original PDF not found at {d['pdf_path']}"})

    try:
        import pdfplumber
    except ImportError:
        return json.dumps({"error": "pdfplumber not installed. Run: pip install pdfplumber"})

    try:
        results = []
        with pdfplumber.open(d["pdf_path"]) as pdf:
            for pg_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
                page = pdf.pages[pg_num - 1]
                text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or ""
                results.append({
                    "page_num": pg_num,
                    "content": text,
                    "char_count": len(text),
                    "width": page.width,
                    "height": page.height
                })

        return json.dumps({
            "doc_id": doc_id, "doc_title": d["title"],
            "source": "pdfplumber (layout mode)",
            "pages": results
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Extraction failed: {e}"})


@mcp.tool()
def reextract_table(doc_id: int, page_start: int, page_end: Optional[int] = None) -> str:
    """Re-extract tables from original PDF using pdfplumber with structure detection.

    Use this when a table in Marker's output has misaligned columns, merged cells,
    or garbled content. Returns structured table data with identified headers and rows.

    Args:
        doc_id: Document ID.
        page_start: First page containing the table (1-indexed).
        page_end: Last page of the table (for multi-page tables).
    """
    if page_end is None:
        page_end = page_start

    with get_db() as conn:
        d = conn.execute("SELECT pdf_path, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not d:
            return json.dumps({"error": f"Document {doc_id} not found"})
        if not d["pdf_path"] or not Path(d["pdf_path"]).exists():
            return json.dumps({"error": f"Original PDF not found at {d['pdf_path']}"})

    try:
        import pdfplumber
    except ImportError:
        return json.dumps({"error": "pdfplumber not installed. Run: pip install pdfplumber"})

    try:
        all_tables = []

        with pdfplumber.open(d["pdf_path"]) as pdf:
            for pg_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
                page = pdf.pages[pg_num - 1]

                # Try strict line-based extraction first
                tables = page.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                })

                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                    })

                for raw_table in (tables or []):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Clean cells
                    cleaned = []
                    for row in raw_table:
                        cleaned.append([
                            (str(c).strip().replace('\n', ' ') if c else '')
                            for c in row
                        ])

                    # Detect headers (first row usually)
                    headers = cleaned[0]
                    data = cleaned[1:]

                    # Build markdown
                    cols = max(len(r) for r in cleaned)
                    headers = headers + [''] * (cols - len(headers))
                    md_lines = [
                        '| ' + ' | '.join(headers) + ' |',
                        '| ' + ' | '.join(['---'] * cols) + ' |',
                    ]
                    for row in data:
                        padded = row + [''] * (cols - len(row))
                        md_lines.append('| ' + ' | '.join(padded) + ' |')

                    all_tables.append({
                        "page_num": pg_num,
                        "headers": headers,
                        "rows": data,
                        "row_count": len(data),
                        "col_count": cols,
                        "markdown": '\n'.join(md_lines)
                    })

        return json.dumps({
            "doc_id": doc_id, "doc_title": d["title"],
            "source": "pdfplumber (table extraction)",
            "page_range": f"{page_start}-{page_end}",
            "tables_found": len(all_tables),
            "tables": all_tables
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Table extraction failed: {e}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DB_PATH, SSE_PORT

    p = argparse.ArgumentParser(description='Document RAG MCP Server')
    p.add_argument('--db', '-d', required=True, help='SQLite database path')
    p.add_argument('--port', type=int, default=8200, help='SSE port (default: 8200)')

    args = p.parse_args()
    DB_PATH = args.db

    if not Path(DB_PATH).exists():
        print(f"Error: {DB_PATH} not found. Run ingest.py first.")
        sys.exit(1)

    # Update port on the mcp instance
    mcp.settings.port = args.port

    with get_db() as conn:
        projects = conn.execute(
            "SELECT project, COUNT(*) as n FROM documents GROUP BY project"
        ).fetchall()
        total_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

        print(f"Document RAG MCP Server")
        print(f"Database: {DB_PATH}")
        proj_list = ', '.join(f"{r['project']} ({r['n']} docs)" for r in projects)
        print(f"Projects: {proj_list}")
        print(f"Total pages indexed: {total_pages}")
        print(f"Starting on port {args.port}...")

    mcp.settings.host = "0.0.0.0"
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    mcp.run(transport="sse")


if __name__ == '__main__':
    main()
