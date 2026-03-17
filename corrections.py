#!/usr/bin/env python3
"""
LLM Correction Tools for PDF Extraction Pipeline
===================================================
25 MCP tools that let the LLM correct mistakes made by the heuristic
extractor: false-positive headings, wrong levels, missed document
boundaries, OCR garbage, metadata errors, etc.

Dual persistence:
  - SQLite: immediate effect (corrections, cross_references, quality_flags tables)
  - Sidecar JSON: survives re-ingestion ({stem}_corrections.json next to PDF)

Usage:
    from corrections import register_correction_tools
    register_correction_tools(mcp, get_db)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


CORRECTIONS_SUFFIX = '_corrections.json'
ToolResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_doc(conn, doc_id):
    """Get document row or None."""
    return conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()


def _get_doc_pdf_path(conn, doc_id):
    """Get (doc_row, pdf_path_str|None). Returns (None, None) if doc missing."""
    doc = _get_doc(conn, doc_id)
    if not doc:
        return None, None
    pdf_path = doc['pdf_path']
    if pdf_path and Path(pdf_path).exists():
        return doc, pdf_path
    return doc, None


def _sidecar_path(pdf_path):
    p = Path(pdf_path)
    return p.parent / (p.stem + CORRECTIONS_SUFFIX)


def _load_sidecar(pdf_path):
    """Load sidecar JSON. Returns empty dict if missing or invalid."""
    if not pdf_path:
        return {}
    sp = _sidecar_path(pdf_path)
    if sp.exists():
        try:
            with open(sp, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_sidecar(pdf_path, data):
    """Atomically write sidecar JSON."""
    if not pdf_path:
        return
    data['version'] = 2
    sp = _sidecar_path(pdf_path)
    fd, tmp = tempfile.mkstemp(dir=sp.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, sp)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _log_correction(conn, doc_id, category, action, payload):
    """Insert into corrections table."""
    conn.execute(
        "INSERT INTO corrections (doc_id, category, action, payload) VALUES (?, ?, ?, ?)",
        (doc_id, category, action, json.dumps(payload, ensure_ascii=False))
    )


def _page_range_key(conn, doc_id):
    """Page range string for sidecar keys, e.g. '1-14'."""
    row = conn.execute(
        "SELECT MIN(page_num) as mn, MAX(page_num) as mx FROM pages WHERE doc_id = ?",
        (doc_id,)
    ).fetchone()
    if row and row['mn'] is not None:
        return f"{row['mn']}-{row['mx']}"
    return "0-0"


def _update_doc_metadata(conn, doc_id, key, value):
    """Update a single key in the document's metadata JSON."""
    row = conn.execute(
        "SELECT metadata FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    meta = json.loads(row['metadata']) if row and row['metadata'] else {}
    meta[key] = value
    conn.execute(
        "UPDATE documents SET metadata = ? WHERE id = ?",
        (json.dumps(meta, ensure_ascii=False), doc_id)
    )


def _update_doc_total_pages(conn, doc_id):
    """Recalculate total_pages on a document."""
    count = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE documents SET total_pages = ? WHERE id = ?", (count, doc_id)
    )


def _renumber_sections(conn, doc_id):
    """Renumber seq for all sections by page_start order."""
    sections = conn.execute(
        "SELECT id FROM sections WHERE doc_id = ? ORDER BY page_start, id",
        (doc_id,)
    ).fetchall()
    for i, s in enumerate(sections):
        conn.execute("UPDATE sections SET seq = ? WHERE id = ?", (i, s['id']))


def _recalculate_section_page_end(conn, doc_id):
    """Fix page_end on all sections in a document."""
    sections = conn.execute(
        "SELECT id, page_start FROM sections WHERE doc_id = ? ORDER BY seq",
        (doc_id,)
    ).fetchall()
    if not sections:
        return
    max_page = conn.execute(
        "SELECT MAX(page_num) FROM pages WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0] or 1
    for i, s in enumerate(sections):
        if i + 1 < len(sections):
            page_end = max(sections[i + 1]['page_start'] - 1, s['page_start'])
        else:
            page_end = max(max_page, s['page_start'])
        conn.execute("UPDATE sections SET page_end = ? WHERE id = ?",
                     (page_end, s['id']))


def _rebuild_breadcrumbs(conn, doc_id):
    """Rebuild breadcrumbs for all sections and pages in a document.

    This re-derives parent_id + breadcrumb on every section from the
    level hierarchy, then propagates to pages (including section_id).
    Page UPDATE triggers keep FTS in sync automatically.
    """
    # --- sections ---
    sections = conn.execute(
        "SELECT id, heading, level, page_start FROM sections "
        "WHERE doc_id = ? ORDER BY seq",
        (doc_id,)
    ).fetchall()

    stack = []  # [(level, heading, section_id)]
    for s in sections:
        while stack and stack[-1][0] >= s['level']:
            stack.pop()
        stack.append((s['level'], s['heading'], s['id']))
        bc = ' > '.join(text for _, text, _ in stack)
        parent_id = stack[-2][2] if len(stack) >= 2 else None
        conn.execute(
            "UPDATE sections SET breadcrumb = ?, parent_id = ? WHERE id = ?",
            (bc, parent_id, s['id'])
        )

    # --- pages ---
    sections = conn.execute(
        "SELECT id, page_start, breadcrumb FROM sections "
        "WHERE doc_id = ? ORDER BY seq",
        (doc_id,)
    ).fetchall()
    pages = conn.execute(
        "SELECT id, page_num FROM pages WHERE doc_id = ? ORDER BY page_num",
        (doc_id,)
    ).fetchall()

    for p in pages:
        best = None
        for s in sections:
            if s['page_start'] <= p['page_num']:
                best = s
        bc = best['breadcrumb'] if best else ''
        sid = best['id'] if best else None
        conn.execute(
            "UPDATE pages SET breadcrumb = ?, section_id = ? WHERE id = ?",
            (bc, sid, p['id'])
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_correction_tools(mcp, get_db):
    """Register all 25 correction tools with the MCP server."""

    # ===== Document-level (5) =====

    @mcp.tool()
    def merge_documents(doc_id_a: int, doc_id_b: int) -> ToolResult:
        """Merge document B into document A. Moves all pages and sections, then deletes B.
        Refuses if page numbers overlap between the two documents.

        Args:
            doc_id_a: Target document (receives pages/sections).
            doc_id_b: Source document (deleted after merge).
        """
        with get_db() as conn:
            doc_a = _get_doc(conn, doc_id_a)
            doc_b = _get_doc(conn, doc_id_b)
            if not doc_a:
                return {"error": f"Document {doc_id_a} not found"}
            if not doc_b:
                return {"error": f"Document {doc_id_b} not found"}

            overlap = conn.execute(
                "SELECT a.page_num FROM pages a "
                "JOIN pages b ON a.page_num = b.page_num "
                "WHERE a.doc_id = ? AND b.doc_id = ? LIMIT 1",
                (doc_id_a, doc_id_b)
            ).fetchone()
            if overlap:
                return {
                    "error": f"Page numbers overlap (e.g. page {overlap['page_num']}). Cannot merge."
                }

            max_seq = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM sections WHERE doc_id = ?",
                (doc_id_a,)
            ).fetchone()[0]

            conn.execute(
                "UPDATE pages SET doc_id = ? WHERE doc_id = ?",
                (doc_id_a, doc_id_b)
            )
            conn.execute(
                "UPDATE sections SET doc_id = ?, seq = seq + ? WHERE doc_id = ?",
                (doc_id_a, max_seq + 1, doc_id_b)
            )
            # Move cross_references and quality_flags too
            conn.execute(
                "UPDATE cross_references SET doc_id = ? WHERE doc_id = ?",
                (doc_id_a, doc_id_b)
            )
            conn.execute(
                "UPDATE quality_flags SET doc_id = ? WHERE doc_id = ?",
                (doc_id_a, doc_id_b)
            )
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id_b,))

            _renumber_sections(conn, doc_id_a)
            _recalculate_section_page_end(conn, doc_id_a)
            _rebuild_breadcrumbs(conn, doc_id_a)
            _update_doc_total_pages(conn, doc_id_a)

            payload = {"doc_id_a": doc_id_a, "doc_id_b": doc_id_b,
                       "title_b": doc_b['title']}
            _log_correction(conn, doc_id_a, "document", "merge", payload)
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id_a)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('document_merges', []).append({
                    "doc_a_pages": _page_range_key(conn, doc_id_a),
                    "doc_b_title": doc_b['title'],
                })
                _save_sidecar(pdf_path, sidecar)

            total = conn.execute(
                "SELECT total_pages FROM documents WHERE id = ?", (doc_id_a,)
            ).fetchone()['total_pages']
            return {
                "status": "merged", "doc_id": doc_id_a,
                "deleted_doc_id": doc_id_b, "total_pages": total,
            }

    @mcp.tool()
    def split_document(doc_id: int, at_page: int) -> ToolResult:
        """Split a document at a page boundary. Pages >= at_page become a new document.
        Sections spanning the boundary are cloned. Breadcrumbs rebuilt for both.

        Args:
            doc_id: Document to split.
            at_page: First page number of the new document.
        """
        with get_db() as conn:
            doc = _get_doc(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            if not conn.execute(
                "SELECT 1 FROM pages WHERE doc_id = ? AND page_num = ?",
                (doc_id, at_page)
            ).fetchone():
                return {
                    "error": f"Page {at_page} not found in document {doc_id}"
                }

            before = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE doc_id = ? AND page_num < ?",
                (doc_id, at_page)
            ).fetchone()[0]
            after = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE doc_id = ? AND page_num >= ?",
                (doc_id, at_page)
            ).fetchone()[0]
            if before == 0 or after == 0:
                return {"error": "Split would create an empty document"}

            # Title for new doc
            heading_row = conn.execute(
                "SELECT heading FROM sections WHERE doc_id = ? AND page_start >= ? "
                "ORDER BY seq LIMIT 1", (doc_id, at_page)
            ).fetchone()
            new_title = heading_row['heading'] if heading_row else f"{doc['title']} (Part 2)"

            conn.execute(
                "INSERT INTO documents (project, title, filename, pdf_path, "
                "total_pages, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (doc['project'], new_title, doc['filename'],
                 doc['pdf_path'], after, doc['metadata'])
            )
            new_doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Move pages
            conn.execute(
                "UPDATE pages SET doc_id = ? WHERE doc_id = ? AND page_num >= ?",
                (new_doc_id, doc_id, at_page)
            )

            # Sections entirely in new doc → move
            conn.execute(
                "UPDATE sections SET doc_id = ? "
                "WHERE doc_id = ? AND page_start >= ?",
                (new_doc_id, doc_id, at_page)
            )

            # Sections spanning boundary → clone
            spanning = conn.execute(
                "SELECT * FROM sections WHERE doc_id = ? "
                "AND page_start < ? AND page_end >= ?",
                (doc_id, at_page, at_page)
            ).fetchall()
            for s in spanning:
                conn.execute(
                    "UPDATE sections SET page_end = ? WHERE id = ?",
                    (at_page - 1, s['id'])
                )
                conn.execute(
                    "INSERT INTO sections "
                    "(doc_id, heading, level, page_start, page_end, breadcrumb, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (new_doc_id, s['heading'], s['level'], at_page,
                     s['page_end'], s['breadcrumb'], s['seq'])
                )

            # Rebuild both docs
            for did in (doc_id, new_doc_id):
                _update_doc_total_pages(conn, did)
                _renumber_sections(conn, did)
                _recalculate_section_page_end(conn, did)
                _rebuild_breadcrumbs(conn, did)

            _log_correction(conn, doc_id, "document", "split", {
                "at_page": at_page, "new_doc_id": new_doc_id,
            })
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('document_splits', []).append({
                    "at_page": at_page, "title_hint": new_title,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "split", "original_doc_id": doc_id,
                "original_pages": before, "new_doc_id": new_doc_id,
                "new_title": new_title, "new_pages": after,
            }

    @mcp.tool()
    def set_document_title(doc_id: int, title: str) -> ToolResult:
        """Set or correct a document's title.

        Args:
            doc_id: Document ID.
            title: New title.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            conn.execute(
                "UPDATE documents SET title = ? WHERE id = ?", (title, doc_id)
            )
            _log_correction(conn, doc_id, "document", "set_title",
                            {"title": title})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('document_titles', {})[pr] = title
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id, "title": title
            }

    @mcp.tool()
    def set_document_type(doc_id: int, doc_type: str) -> ToolResult:
        """Set the document type classification.

        Args:
            doc_id: Document ID.
            doc_type: One of: standard, specification, datasheet, vendor_document,
                      drawing_list, calculation, procedure, report, manual,
                      correspondence, other.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            _update_doc_metadata(conn, doc_id, 'document_type', doc_type)
            _log_correction(conn, doc_id, "document", "set_type",
                            {"doc_type": doc_type})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('document_types', {})[pr] = doc_type
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id,
                "document_type": doc_type,
            }

    @mcp.tool()
    def link_documents(doc_id: int, related_doc_id: int,
                       relationship: str) -> ToolResult:
        """Create a cross-reference link between two documents.

        Args:
            doc_id: Source document.
            related_doc_id: Target document being referenced.
            relationship: E.g. 'references', 'supersedes', 'appendix_to', 'related'.
        """
        with get_db() as conn:
            if not _get_doc(conn, doc_id):
                return {"error": f"Document {doc_id} not found"}
            target = _get_doc(conn, related_doc_id)
            if not target:
                return {"error": f"Document {related_doc_id} not found"}

            conn.execute(
                "INSERT INTO cross_references "
                "(doc_id, target_doc_id, relationship) VALUES (?, ?, ?)",
                (doc_id, related_doc_id, relationship)
            )
            _log_correction(conn, doc_id, "document", "link", {
                "related_doc_id": related_doc_id,
                "relationship": relationship,
            })
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('cross_references', []).append({
                    "target_doc_title": target['title'],
                    "relationship": relationship,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "linked", "doc_id": doc_id,
                "related_doc_id": related_doc_id,
                "relationship": relationship,
            }

    # ===== Heading hierarchy (4) =====

    @mcp.tool()
    def add_heading(doc_id: int, page_num: int, text: str,
                    level: int) -> ToolResult:
        """Add a heading the extractor missed. Inserts a section and rebuilds breadcrumbs.

        Args:
            doc_id: Document ID.
            page_num: Page where the heading appears.
            text: Heading text.
            level: Heading level (1 = top-level, 2, 3, 4).
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            after_seq = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM sections "
                "WHERE doc_id = ? AND page_start <= ?",
                (doc_id, page_num)
            ).fetchone()[0]

            conn.execute(
                "UPDATE sections SET seq = seq + 1 "
                "WHERE doc_id = ? AND seq > ?",
                (doc_id, after_seq)
            )
            conn.execute(
                "INSERT INTO sections (doc_id, heading, level, page_start, seq) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, text, level, page_num, after_seq + 1)
            )

            _recalculate_section_page_end(conn, doc_id)
            _rebuild_breadcrumbs(conn, doc_id)
            _log_correction(conn, doc_id, "heading", "add", {
                "page_num": page_num, "text": text, "level": level,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('new_headings', []).append({
                    "page_num": page_num, "text": text, "level": level,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "added", "doc_id": doc_id,
                "heading": text, "level": level, "page_num": page_num,
            }

    @mcp.tool()
    def remove_heading(doc_id: int, page_num: int,
                       text_prefix: str) -> ToolResult:
        """Remove a false-positive heading. Re-parents its children and rebuilds breadcrumbs.

        Args:
            doc_id: Document ID.
            page_num: Page where the heading appears.
            text_prefix: Start of the heading text (enough to identify it).
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            section = conn.execute(
                "SELECT * FROM sections "
                "WHERE doc_id = ? AND page_start = ? AND heading LIKE ?",
                (doc_id, page_num, text_prefix + '%')
            ).fetchone()
            if not section:
                return {
                    "error": f"No heading starting with '{text_prefix}' "
                             f"on page {page_num}"
                }

            # Clear page references to this section
            conn.execute(
                "UPDATE pages SET section_id = NULL WHERE section_id = ?",
                (section['id'],)
            )
            conn.execute(
                "UPDATE sections SET parent_id = ? WHERE parent_id = ?",
                (section['parent_id'], section['id'])
            )
            conn.execute("DELETE FROM sections WHERE id = ?", (section['id'],))

            _renumber_sections(conn, doc_id)
            _recalculate_section_page_end(conn, doc_id)
            _rebuild_breadcrumbs(conn, doc_id)
            _log_correction(conn, doc_id, "heading", "remove", {
                "page_num": page_num, "text_prefix": text_prefix,
                "heading": section['heading'],
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('removed_headings', []).append({
                    "page_num": page_num, "text_prefix": text_prefix,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "removed", "doc_id": doc_id,
                "heading": section['heading'], "page_num": page_num,
            }

    @mcp.tool()
    def change_heading_level(doc_id: int, page_num: int,
                             text_prefix: str, new_level: int) -> ToolResult:
        """Change a heading's level (e.g. H3 to H2). Rebuilds breadcrumbs.

        Args:
            doc_id: Document ID.
            page_num: Page where the heading appears.
            text_prefix: Start of the heading text.
            new_level: New level (1-4).
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            section = conn.execute(
                "SELECT * FROM sections "
                "WHERE doc_id = ? AND page_start = ? AND heading LIKE ?",
                (doc_id, page_num, text_prefix + '%')
            ).fetchone()
            if not section:
                return {
                    "error": f"No heading starting with '{text_prefix}' "
                             f"on page {page_num}"
                }

            old_level = section['level']
            conn.execute(
                "UPDATE sections SET level = ? WHERE id = ?",
                (new_level, section['id'])
            )

            _rebuild_breadcrumbs(conn, doc_id)
            _log_correction(conn, doc_id, "heading", "change_level", {
                "page_num": page_num, "text_prefix": text_prefix,
                "old_level": old_level, "new_level": new_level,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                key = f"{page_num}:{section['heading'][:50]}"
                sidecar.setdefault('heading_overrides', {})[key] = {
                    "level": new_level, "is_heading": True,
                }
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id,
                "heading": section['heading'],
                "old_level": old_level, "new_level": new_level,
            }

    @mcp.tool()
    def rename_heading(doc_id: int, page_num: int,
                       old_text_prefix: str, new_text: str) -> ToolResult:
        """Rename a heading. Updates the section and rebuilds breadcrumbs.

        Args:
            doc_id: Document ID.
            page_num: Page where the heading appears.
            old_text_prefix: Start of the current heading text.
            new_text: New heading text.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            section = conn.execute(
                "SELECT * FROM sections "
                "WHERE doc_id = ? AND page_start = ? AND heading LIKE ?",
                (doc_id, page_num, old_text_prefix + '%')
            ).fetchone()
            if not section:
                return {
                    "error": f"No heading starting with '{old_text_prefix}' "
                             f"on page {page_num}"
                }

            old_text = section['heading']
            conn.execute(
                "UPDATE sections SET heading = ? WHERE id = ?",
                (new_text, section['id'])
            )

            _rebuild_breadcrumbs(conn, doc_id)
            _log_correction(conn, doc_id, "heading", "rename", {
                "page_num": page_num,
                "old_text": old_text, "new_text": new_text,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                key = f"{page_num}:{old_text[:50]}"
                sidecar.setdefault('heading_overrides', {})[key] = {
                    "corrected_text": new_text, "is_heading": True,
                }
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "renamed", "doc_id": doc_id,
                "old_text": old_text, "new_text": new_text,
            }

    # ===== Page-level (4) =====

    @mcp.tool()
    def reclassify_page(doc_id: int, page_num: int,
                        new_type: str) -> ToolResult:
        """Change a page's type classification.

        Args:
            doc_id: Document ID.
            page_num: Page number.
            new_type: New type: text, drawing, table, toc, cover, skipped.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            updated = conn.execute(
                "UPDATE pages SET page_type = ? "
                "WHERE doc_id = ? AND page_num = ?",
                (new_type, doc_id, page_num)
            ).rowcount
            if not updated:
                return {
                    "error": f"Page {page_num} not found in document {doc_id}"
                }

            _log_correction(conn, doc_id, "page", "reclassify", {
                "page_num": page_num, "new_type": new_type,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('page_reclassifications', {})[
                    str(page_num)] = new_type
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "reclassified", "doc_id": doc_id,
                "page_num": page_num, "new_type": new_type,
            }

    @mcp.tool()
    def skip_page(doc_id: int, page_num: int) -> ToolResult:
        """Mark a page as skipped (excluded from search results).

        Args:
            doc_id: Document ID.
            page_num: Page number to skip.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            updated = conn.execute(
                "UPDATE pages SET page_type = 'skipped' "
                "WHERE doc_id = ? AND page_num = ?",
                (doc_id, page_num)
            ).rowcount
            if not updated:
                return {
                    "error": f"Page {page_num} not found in document {doc_id}"
                }

            _log_correction(conn, doc_id, "page", "skip",
                            {"page_num": page_num})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                skipped = sidecar.setdefault('skipped_pages', [])
                if page_num not in skipped:
                    skipped.append(page_num)
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "skipped", "doc_id": doc_id,
                "page_num": page_num,
            }

    @mcp.tool()
    def move_page_to_document(page_num: int, from_doc_id: int,
                              to_doc_id: int) -> ToolResult:
        """Move a single page from one document to another. Rebuilds both documents.

        Args:
            page_num: Page number to move.
            from_doc_id: Source document.
            to_doc_id: Target document.
        """
        with get_db() as conn:
            if not _get_doc(conn, from_doc_id):
                return {"error": f"Document {from_doc_id} not found"}
            if not _get_doc(conn, to_doc_id):
                return {"error": f"Document {to_doc_id} not found"}

            updated = conn.execute(
                "UPDATE pages SET doc_id = ? "
                "WHERE doc_id = ? AND page_num = ?",
                (to_doc_id, from_doc_id, page_num)
            ).rowcount
            if not updated:
                return {
                    "error": f"Page {page_num} not found in "
                             f"document {from_doc_id}"
                }

            for did in (from_doc_id, to_doc_id):
                _update_doc_total_pages(conn, did)
                _recalculate_section_page_end(conn, did)
                _rebuild_breadcrumbs(conn, did)

            _log_correction(conn, from_doc_id, "page", "move", {
                "page_num": page_num, "to_doc_id": to_doc_id,
            })
            conn.commit()

            return {
                "status": "moved", "page_num": page_num,
                "from_doc_id": from_doc_id, "to_doc_id": to_doc_id,
            }

    @mcp.tool()
    def set_page_breadcrumb(doc_id: int, page_num: int,
                            breadcrumb: str) -> ToolResult:
        """Override the breadcrumb (heading context) for a specific page.

        Args:
            doc_id: Document ID.
            page_num: Page number.
            breadcrumb: New breadcrumb (e.g. 'SCOPE > Welding > Procedures').
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            updated = conn.execute(
                "UPDATE pages SET breadcrumb = ? "
                "WHERE doc_id = ? AND page_num = ?",
                (breadcrumb, doc_id, page_num)
            ).rowcount
            if not updated:
                return {
                    "error": f"Page {page_num} not found in document {doc_id}"
                }

            _log_correction(conn, doc_id, "page", "set_breadcrumb", {
                "page_num": page_num, "breadcrumb": breadcrumb,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('breadcrumb_overrides', {})[
                    str(page_num)] = breadcrumb
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id,
                "page_num": page_num, "breadcrumb": breadcrumb,
            }

    # ===== Content (3) =====

    @mcp.tool()
    def fix_ocr_text(doc_id: int, page_num: int,
                     old_text: str, new_text: str) -> ToolResult:
        """Fix OCR scanning artifacts ONLY — garbled characters produced by the scanner.

        IMPORTANT: Only use this for obvious OCR/scanning errors (e.g. random symbols,
        character substitution from poor scan quality). NEVER use this to fix spelling
        mistakes, grammar, or any text that might be what the original document actually
        says. These are legal/contractual engineering documents — their content is sacred.
        When in doubt, use flag_low_quality instead.

        Args:
            doc_id: Document ID.
            page_num: Page number.
            old_text: The garbled OCR text (exact match).
            new_text: What the scan clearly intended.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            page = conn.execute(
                "SELECT id, content FROM pages "
                "WHERE doc_id = ? AND page_num = ?",
                (doc_id, page_num)
            ).fetchone()
            if not page:
                return {
                    "error": f"Page {page_num} not found in document {doc_id}"
                }

            if old_text not in page['content']:
                return {
                    "error": f"Text not found on page {page_num}: "
                             f"'{old_text[:80]}'"
                }

            occurrences = page['content'].count(old_text)
            new_content = page['content'].replace(old_text, new_text)
            conn.execute(
                "UPDATE pages SET content = ?, char_count = ? WHERE id = ?",
                (new_content, len(new_content), page['id'])
            )

            _log_correction(conn, doc_id, "content", "fix_ocr", {
                "page_num": page_num,
                "old_text": old_text, "new_text": new_text,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('ocr_fixes', []).append({
                    "page_num": page_num,
                    "old_text": old_text, "new_text": new_text,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "fixed", "doc_id": doc_id,
                "page_num": page_num, "replacements": occurrences,
            }

    @mcp.tool()
    def add_running_header(doc_id: int, text: str) -> ToolResult:
        """Strip a repeated header/footer from all pages of a document.
        Lines containing this text are removed immediately.

        Args:
            doc_id: Document ID.
            text: The running header/footer text to strip.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            pages = conn.execute(
                "SELECT id, content FROM pages WHERE doc_id = ?", (doc_id,)
            ).fetchall()

            lines_removed = 0
            for page in pages:
                lines = page['content'].split('\n')
                new_lines = [l for l in lines if text not in l]
                removed = len(lines) - len(new_lines)
                if removed:
                    new_content = '\n'.join(new_lines)
                    conn.execute(
                        "UPDATE pages SET content = ?, char_count = ? "
                        "WHERE id = ?",
                        (new_content, len(new_content), page['id'])
                    )
                    lines_removed += removed

            _log_correction(conn, doc_id, "content", "add_running_header", {
                "text": text, "lines_removed": lines_removed,
            })
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                headers = sidecar.setdefault('running_headers_add', [])
                if text not in headers:
                    headers.append(text)
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "stripped", "doc_id": doc_id,
                "header_text": text, "lines_removed": lines_removed,
            }

    @mcp.tool()
    def remove_running_header(doc_id: int, text: str) -> ToolResult:
        """Mark a running header for removal on next re-extraction.
        Sidecar-only — no immediate content change. The extractor will
        stop stripping this text on the next run.

        Args:
            doc_id: Document ID.
            text: Running header text that should NOT be stripped.
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            _log_correction(conn, doc_id, "content",
                            "remove_running_header", {"text": text})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                headers = sidecar.setdefault('running_headers_remove', [])
                if text not in headers:
                    headers.append(text)
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "recorded", "doc_id": doc_id,
                "note": "Takes effect on next re-extraction of the PDF",
            }

    # ===== Metadata (5) =====

    @mcp.tool()
    def set_document_number(doc_id: int, number: str) -> ToolResult:
        """Set or correct the document reference number.

        Args:
            doc_id: Document ID.
            number: Document/reference number (e.g. 'ABC-123-REV2').
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            _update_doc_metadata(conn, doc_id, 'document_number', number)
            _log_correction(conn, doc_id, "metadata",
                            "set_document_number", {"number": number})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('metadata_overrides', {}).setdefault(
                    pr, {})['document_number'] = number
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id,
                "document_number": number,
            }

    @mcp.tool()
    def set_revision(doc_id: int, revision: str) -> ToolResult:
        """Set or correct the document revision.

        Args:
            doc_id: Document ID.
            revision: Revision identifier (e.g. 'Rev. 3', 'B', '2.0').
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            _update_doc_metadata(conn, doc_id, 'revision', revision)
            _log_correction(conn, doc_id, "metadata", "set_revision",
                            {"revision": revision})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('metadata_overrides', {}).setdefault(
                    pr, {})['revision'] = revision
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id, "revision": revision,
            }

    @mcp.tool()
    def add_cross_reference(doc_id: int, page_num: int,
                            target_doc_id: int, context: str) -> ToolResult:
        """Record a cross-reference from a page to another document.

        Args:
            doc_id: Source document.
            page_num: Page where the reference appears.
            target_doc_id: Document being referenced.
            context: Description (e.g. 'references piping spec section 3.2').
        """
        with get_db() as conn:
            if not _get_doc(conn, doc_id):
                return {"error": f"Document {doc_id} not found"}
            target = _get_doc(conn, target_doc_id)
            if not target:
                return {"error": f"Document {target_doc_id} not found"}

            conn.execute(
                "INSERT INTO cross_references "
                "(doc_id, target_doc_id, relationship, page_num, context) "
                "VALUES (?, ?, 'references', ?, ?)",
                (doc_id, target_doc_id, page_num, context)
            )
            _log_correction(conn, doc_id, "metadata",
                            "add_cross_reference", {
                                "page_num": page_num,
                                "target_doc_id": target_doc_id,
                                "context": context,
                            })
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('cross_references', []).append({
                    "page_num": page_num,
                    "target_doc_title": target['title'],
                    "context": context,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "added", "doc_id": doc_id,
                "page_num": page_num, "target_doc_id": target_doc_id,
            }

    @mcp.tool()
    def add_keywords(doc_id: int, keywords_csv: str) -> ToolResult:
        """Add search keywords to a document's metadata. Merges with existing keywords.

        Args:
            doc_id: Document ID.
            keywords_csv: Comma-separated keywords (e.g. 'piping, welding, ASME B31.3').
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            keywords = [k.strip() for k in keywords_csv.split(',')
                        if k.strip()]
            meta = json.loads(doc['metadata']) if doc['metadata'] else {}
            existing = meta.get('keywords', [])
            merged = list(dict.fromkeys(existing + keywords))

            _update_doc_metadata(conn, doc_id, 'keywords', merged)
            _log_correction(conn, doc_id, "metadata", "add_keywords",
                            {"keywords": keywords})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('metadata_overrides', {}).setdefault(
                    pr, {})['keywords'] = merged
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id, "keywords": merged,
            }

    @mcp.tool()
    def add_equipment_tags(doc_id: int, tags_csv: str) -> ToolResult:
        """Add equipment tag references to a document's metadata.

        Args:
            doc_id: Document ID.
            tags_csv: Comma-separated tags (e.g. 'P-101, V-201, E-301').
        """
        with get_db() as conn:
            doc, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if not doc:
                return {"error": f"Document {doc_id} not found"}

            tags = [t.strip() for t in tags_csv.split(',') if t.strip()]
            meta = json.loads(doc['metadata']) if doc['metadata'] else {}
            existing = meta.get('equipment_tags', [])
            merged = list(dict.fromkeys(existing + tags))

            _update_doc_metadata(conn, doc_id, 'equipment_tags', merged)
            _log_correction(conn, doc_id, "metadata",
                            "add_equipment_tags", {"tags": tags})
            conn.commit()

            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                pr = _page_range_key(conn, doc_id)
                sidecar.setdefault('metadata_overrides', {}).setdefault(
                    pr, {})['equipment_tags'] = merged
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "updated", "doc_id": doc_id,
                "equipment_tags": merged,
            }

    # ===== Quality (3) =====

    @mcp.tool()
    def flag_low_quality(doc_id: int, page_num: int,
                         reason: str) -> ToolResult:
        """Flag a page as low quality (OCR errors, garbled text, missing content).

        Args:
            doc_id: Document ID.
            page_num: Page with quality issues.
            reason: Description of the problem.
        """
        with get_db() as conn:
            if not _get_doc(conn, doc_id):
                return {"error": f"Document {doc_id} not found"}

            conn.execute(
                "INSERT INTO quality_flags "
                "(doc_id, page_num, flag_type, reason) "
                "VALUES (?, ?, 'low_quality', ?)",
                (doc_id, page_num, reason)
            )
            _log_correction(conn, doc_id, "quality", "flag_low_quality", {
                "page_num": page_num, "reason": reason,
            })
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('quality_flags', []).append({
                    "page_num": page_num,
                    "type": "low_quality", "reason": reason,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "flagged", "doc_id": doc_id,
                "page_num": page_num, "flag": "low_quality",
            }

    @mcp.tool()
    def flag_duplicate(doc_id: int, duplicate_of_doc_id: int) -> ToolResult:
        """Flag a document as a duplicate of another.

        Args:
            doc_id: The duplicate document.
            duplicate_of_doc_id: The original document.
        """
        with get_db() as conn:
            if not _get_doc(conn, doc_id):
                return {"error": f"Document {doc_id} not found"}
            if not _get_doc(conn, duplicate_of_doc_id):
                return {"error": f"Document {duplicate_of_doc_id} not found"}

            conn.execute(
                "INSERT INTO quality_flags "
                "(doc_id, flag_type, reason, related_doc_id) "
                "VALUES (?, 'duplicate', 'Duplicate document', ?)",
                (doc_id, duplicate_of_doc_id)
            )
            _log_correction(conn, doc_id, "quality", "flag_duplicate", {
                "duplicate_of_doc_id": duplicate_of_doc_id,
            })
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('quality_flags', []).append({
                    "type": "duplicate",
                    "duplicate_of_doc_id": duplicate_of_doc_id,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "flagged", "doc_id": doc_id,
                "flag": "duplicate",
                "duplicate_of": duplicate_of_doc_id,
            }

    @mcp.tool()
    def suggest_reocr(doc_id: int, reason: str) -> ToolResult:
        """Suggest that a document needs OCR re-processing.

        Args:
            doc_id: Document ID.
            reason: Why re-OCR is needed (e.g. 'garbled text on pages 5-8').
        """
        with get_db() as conn:
            if not _get_doc(conn, doc_id):
                return {"error": f"Document {doc_id} not found"}

            conn.execute(
                "INSERT INTO quality_flags "
                "(doc_id, flag_type, reason) VALUES (?, 'needs_reocr', ?)",
                (doc_id, reason)
            )
            _log_correction(conn, doc_id, "quality", "suggest_reocr",
                            {"reason": reason})
            conn.commit()

            _, pdf_path = _get_doc_pdf_path(conn, doc_id)
            if pdf_path:
                sidecar = _load_sidecar(pdf_path)
                sidecar.setdefault('quality_flags', []).append({
                    "type": "needs_reocr", "reason": reason,
                })
                _save_sidecar(pdf_path, sidecar)

            return {
                "status": "flagged", "doc_id": doc_id,
                "flag": "needs_reocr",
            }
