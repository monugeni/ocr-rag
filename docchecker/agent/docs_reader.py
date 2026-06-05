"""Read ingested document text from the ocr-rag docs DB (stdlib sqlite only).

Keeps the agent module self-contained — no dependency on the app package.
"""
from __future__ import annotations

import sqlite3


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def read_pages(db_path: str, doc_id: int) -> list[dict]:
    """Return [{page_num, content, breadcrumb, page_type}] ordered by page."""
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT page_num, content, breadcrumb, page_type FROM pages "
            "WHERE doc_id = ? ORDER BY page_num",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_sections(db_path: str, doc_id: int) -> list[dict]:
    """Return [{heading, level, page_start, page_end, breadcrumb}] for a doc."""
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT heading, level, page_start, page_end, breadcrumb FROM sections "
            "WHERE doc_id = ? ORDER BY seq",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def document_text(db_path: str, doc_id: int, *, max_chars: int = 120_000) -> str:
    """Concatenate a document's pages with explicit page markers, capped."""
    parts: list[str] = []
    total = 0
    for p in read_pages(db_path, doc_id):
        chunk = f"\n[Page {p['page_num']}]\n{p['content']}"
        if total + len(chunk) > max_chars:
            parts.append(f"\n[... truncated at {max_chars} chars ...]")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts).strip()


def page_text(db_path: str, doc_id: int, page_num: int) -> str:
    conn = _conn(db_path)
    try:
        row = conn.execute(
            "SELECT content FROM pages WHERE doc_id = ? AND page_num = ?",
            (doc_id, page_num),
        ).fetchone()
        return row["content"] if row else ""
    finally:
        conn.close()
