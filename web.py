#!/usr/bin/env python3
"""
OCR-RAG Web GUI
================
Folder & document management with PDF ingestion.

  python web.py --db docs.db --port 8201

Routes:
  /                                     Web GUI
  /api/folders                          List / create folders
  /api/folders/{name}/documents         List docs + pending uploads
  /api/folders/{name}/chats             List / create chat threads
  /api/folders/{name}/upload            Upload PDFs
  /api/folders/{name}/ingest            Ingest all pending PDFs
  /api/documents/{id}                   Document info
  /api/documents/{id}/toc               Section tree
  /api/documents/{id}/pages/{num}       Single page
  /api/documents/{id}/pages?start=&end= Page range
  /api/documents/{id}/search?q=         Search within doc
  /api/chats/{id}/messages              Get / append chat messages
  /api/ingestion/jobs                   Active ingestion jobs
  /api/folders/{name}/quality           Quality flags + corrections
"""

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chat_mcp_runner import run_folder_chat
from ingest import (
    init_db, ingest_document, replay_corrections,
    extract_metadata_llm, apply_metadata, compute_embeddings,
)
from extractor import extract_pdf
from file_extractors import (
    extract_file, is_archive, extract_archive,
    INGESTABLE_EXTENSIONS, IMAGE_EXTENSIONS,
)
from splitter import PDFSplitter


CHAT_REVIEW_MAX_PAGES = 10
CHAT_REVIEW_ALLOWED_EXTENSIONS = INGESTABLE_EXTENSIONS
CHAT_REVIEW_MAX_PAGE_CHARS = 3200
CHAT_REVIEW_MAX_EXCERPT_CHARS = 9000
CHAT_SOURCE_MAX_CHARS = 1600
CHAT_SOURCE_SNIPPET_CHARS = 280
CHAT_CONTEXT_MAX_SOURCES = 10
CHAT_CONTEXT_TARGET_SOURCES = 6
CHAT_CONTEXT_MAX_OPEN_PAGES = 8
CHAT_CONTEXT_MAX_SECTION_PAGES = 2
CHAT_CONTEXT_MAX_NEIGHBOR_PAGES = 2
CHAT_SEARCH_MAX_PAGE_CANDIDATES = 8
CHAT_SEARCH_MAX_SECTION_CANDIDATES = 6
CHAT_SEARCH_MAX_DOC_SHORTLIST = 3


# ---------------------------------------------------------------------------
# Split-document filename parser
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(
    r'^(.+?)_part(\d{3})_p(\d+)-(\d+)(?:_(.+))?\.pdf$'
)


def parse_split_info(filename: str) -> Optional[dict]:
    """Parse split-document filename -> parent, part number, page range."""
    m = _SPLIT_RE.match(filename or '')
    if not m:
        return None
    return {
        "parent": m.group(1) + ".pdf",
        "part": int(m.group(2)),
        "page_start": int(m.group(3)),
        "page_end": int(m.group(4)),
    }


# ---------------------------------------------------------------------------
# Config (set by main())
# ---------------------------------------------------------------------------

DB_PATH = "docs.db"
UPLOADS_DIR = "./uploads"
MCP_SERVER_URL = "http://127.0.0.1:8200/sse"
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Esteem Folder Knowledge", docs_url="/api/docs")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _normalize_folder_name(name: str) -> str:
    cleaned = re.sub(r"/+", "/", (name or "").replace("\\", "/").strip())
    return cleaned.strip("/")


def _validate_folder_name(name: str, field_name: str = "Folder name") -> str:
    folder = _normalize_folder_name(name)
    if not folder:
        raise HTTPException(400, f"{field_name} required")

    for segment in folder.split("/"):
        if not segment:
            raise HTTPException(400, "Folder path contains an empty segment")
        if segment in {".", ".."}:
            raise HTTPException(400, "Folder path cannot contain . or ..")
        if re.search(r'[<>:"|?*]', segment):
            raise HTTPException(400, f"Invalid characters in folder segment: {segment}")
    return folder


def _folder_disk_path(folder: str) -> Path:
    folder = _normalize_folder_name(folder)
    base = Path(UPLOADS_DIR)
    return base.joinpath(*folder.split("/")) if folder else base


def _prune_empty_upload_ancestors(folder_dir: Path) -> None:
    """Remove empty parent directories left behind after deleting a folder."""
    uploads_root = Path(UPLOADS_DIR).resolve()
    current = folder_dir.resolve().parent

    while current != uploads_root:
        try:
            current.relative_to(uploads_root)
        except ValueError:
            break

        try:
            current.rmdir()
        except OSError:
            break

        current = current.parent


def _folder_scope_sql(column: str = "project") -> str:
    return f"({column} = ? OR {column} LIKE ?)"


def _folder_scope_params(folder: str) -> list[str]:
    return [folder, f"{folder}/%"]


def _folder_relative_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return _normalize_folder_name(rel)


def _folder_root(folder: str) -> str:
    folder = _normalize_folder_name(folder)
    return folder.split("/", 1)[0] if folder else ""


def _sidecar_path_for_pdf(pdf_path: str) -> Path:
    path = Path(pdf_path)
    return path.parent / f"{path.stem}_corrections.json"


def _search_upload_tree_for_filename(scope_dir: Path, filename: str) -> Optional[Path]:
    if not filename or not scope_dir.exists():
        return None
    for current_root, dirnames, filenames in os.walk(scope_dir):
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".") and not name.startswith("_")
        ]
        if filename in filenames:
            return Path(current_root) / filename
    return None


def _resolve_document_source_path(conn, doc_row) -> Optional[str]:
    filename = doc_row["filename"] or ""
    project = _normalize_folder_name(doc_row["project"] or "")
    uploads_root = Path(UPLOADS_DIR).resolve()

    candidates: list[Path] = []
    if doc_row["pdf_path"]:
        candidates.append(Path(doc_row["pdf_path"]))
    if project and filename:
        candidates.append(_folder_disk_path(project) / filename)
    if filename and project:
        root_folder = _folder_root(project)
        if root_folder:
            root_match = _search_upload_tree_for_filename(_folder_disk_path(root_folder), filename)
            if root_match:
                candidates.append(root_match)
    if filename:
        global_match = _search_upload_tree_for_filename(uploads_root, filename)
        if global_match:
            candidates.append(global_match)

    seen = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not Path(resolved).exists():
            continue
        if doc_row["pdf_path"] != resolved:
            conn.execute("UPDATE documents SET pdf_path = ? WHERE id = ?", (resolved, doc_row["id"]))
        return resolved
    return None


def _iter_upload_folders() -> set[str]:
    uploads = Path(UPLOADS_DIR)
    if not uploads.exists():
        return set()

    folders = set()
    for current_root, dirnames, _filenames in os.walk(uploads):
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".") and not name.startswith("_")
        ]
        current_path = Path(current_root)
        if current_path == uploads:
            continue
        folder = _folder_relative_path(current_path, uploads)
        if folder:
            folders.add(folder)
    return folders


def _all_known_folders(conn) -> list[str]:
    folders = set()
    for table in ("documents", "ingestion_jobs", "chat_threads"):
        rows = conn.execute(f"SELECT DISTINCT project FROM {table}").fetchall()
        folders.update(
            _normalize_folder_name(row["project"])
            for row in rows
            if row["project"]
        )
    folders.update(_iter_upload_folders())

    expanded = set()
    for folder in folders:
        parts = folder.split("/")
        for i in range(1, len(parts) + 1):
            expanded.add("/".join(parts[:i]))
    return sorted(expanded)


def _folder_doc_ids(conn, folder: str) -> list[int]:
    return [r["id"] for r in conn.execute(
        f"SELECT id FROM documents WHERE {_folder_scope_sql('project')} ORDER BY id",
        _folder_scope_params(folder)
    ).fetchall()]


def _pending_uploads(conn, folder: str) -> list[dict]:
    folder_dir = _folder_disk_path(folder)
    if not folder_dir.exists():
        return []

    ingested = {
        (row["project"], row["filename"])
        for row in conn.execute(
            f"SELECT project, filename FROM documents WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        ).fetchall()
    }

    uploads_root = Path(UPLOADS_DIR)
    pending = []
    for current_root, dirnames, filenames in os.walk(folder_dir):
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".") and not name.startswith("_")
        ]
        current_path = Path(current_root)
        current_folder = _folder_relative_path(current_path, uploads_root)
        for name in sorted(filenames):
            file_path = current_path / name
            if name.startswith(".") or name.startswith("_"):
                continue
            if file_path.suffix.lower() not in INGESTABLE_EXTENSIONS:
                continue
            if (current_folder, name) in ingested:
                continue
            relative_path = file_path.relative_to(folder_dir).as_posix()
            pending.append({
                "folder": current_folder,
                "filename": name,
                "relative_path": relative_path,
                "size_mb": round(file_path.stat().st_size / 1048576, 1),
            })

    pending.sort(key=lambda item: (item["folder"], item["filename"]))
    return pending


def _rename_folder_value(value: str, old_folder: str, new_folder: str) -> str:
    if value == old_folder:
        return new_folder
    return new_folder + value[len(old_folder):]


def _rename_folder_references(conn, table: str, old_folder: str, new_folder: str):
    rows = conn.execute(
        f"SELECT DISTINCT project FROM {table} WHERE {_folder_scope_sql('project')}",
        _folder_scope_params(old_folder)
    ).fetchall()
    for row in rows:
        current = row["project"]
        updated = _rename_folder_value(current, old_folder, new_folder)
        conn.execute(
            f"UPDATE {table} SET project = ? WHERE project = ?",
            (updated, current)
        )


def _delete_document_data(conn, doc_id: int):
    conn.execute(
        "DELETE FROM page_embeddings WHERE page_id IN (SELECT id FROM pages WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM corrections WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM cross_references WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM quality_flags WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def _record_upload_event(
    conn,
    project: str,
    relative_path: str,
    filename: str,
    action: str,
    reason: str = "",
    job_id: str | None = None,
    details: str = "",
):
    conn.execute(
        """
        INSERT INTO upload_events (project, relative_path, filename, action, reason, job_id, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (project, relative_path, filename, action, reason or None, job_id, details or None),
    )


def _resolve_upload_path(upload_relative_path: str) -> Path:
    uploads_root = Path(UPLOADS_DIR).resolve()
    candidate = (uploads_root / _normalize_folder_name(upload_relative_path)).resolve()
    try:
        candidate.relative_to(uploads_root)
    except ValueError as exc:
        raise HTTPException(400, "Invalid upload path") from exc
    return candidate


def _prune_empty_upload_dirs(path: Path):
    uploads_root = Path(UPLOADS_DIR).resolve()
    current = path.parent
    while current != uploads_root:
        try:
            current.relative_to(uploads_root)
        except ValueError:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _discard_upload_path(
    upload_relative_path: str,
    *,
    reason: str = "",
    job_id: str | None = None,
    action: str = "discarded",
):
    upload_relative_path = _normalize_folder_name(upload_relative_path)
    upload_path = _resolve_upload_path(upload_relative_path)
    if not upload_path.exists():
        raise HTTPException(404, "Pending file not found")

    project = _folder_relative_path(upload_path.parent, Path(UPLOADS_DIR))
    conn = get_conn()
    try:
        _record_upload_event(
            conn,
            project=project,
            relative_path=upload_relative_path,
            filename=upload_path.name,
            action=action,
            reason=reason,
            job_id=job_id,
        )
        upload_path.unlink()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _prune_empty_upload_dirs(upload_path)
    return {
        "status": "deleted",
        "project": project,
        "relative_path": upload_relative_path,
        "filename": upload_path.name,
    }


def _raise_if_cancel_requested(jid: str):
    job = tracker.get(jid)
    if not job or not job.get("cancel_requested"):
        return
    reason = job.get("cancel_reason") or "Cancelled by user"
    raise IngestionCancelled(reason)


def _job_upload_relative_path(job: dict) -> str | None:
    candidates = [
        job.get("source_path"),
        f"{job.get('project', '')}/{job.get('filename', '')}",
        f"{_folder_root(job.get('project', ''))}/{job.get('filename', '')}",
    ]
    for candidate in candidates:
        candidate = _normalize_folder_name(candidate or "")
        if not candidate:
            continue
        try:
            resolved = _resolve_upload_path(candidate)
        except HTTPException:
            continue
        if resolved.exists():
            return _folder_relative_path(resolved, Path(UPLOADS_DIR))
    return None


def _rollback_split_ingestion(
    original_path: Path,
    archived_original_path: Path | None,
    part_paths: list[Path],
    created_doc_ids: list[int],
):
    """Undo partial split ingestion so the original upload can be retried cleanly."""
    if created_doc_ids:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            for doc_id in created_doc_ids:
                _delete_document_data(conn, doc_id)
            conn.commit()
        finally:
            conn.close()

    for part_path in part_paths:
        if part_path == original_path:
            continue
        try:
            if part_path.exists():
                part_path.unlink()
        except OSError:
            pass

    if archived_original_path and archived_original_path.exists() and not original_path.exists():
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived_original_path), str(original_path))


def _relocate_document_file(doc_row, target_folder: str):
    pdf_path = doc_row["pdf_path"]
    if not pdf_path:
        return None

    source_path = Path(pdf_path)
    uploads_root = Path(UPLOADS_DIR).resolve()
    try:
        source_resolved = source_path.resolve()
    except FileNotFoundError:
        return str(source_path)

    if not source_resolved.exists():
        return str(source_resolved)

    try:
        source_resolved.relative_to(uploads_root)
    except ValueError:
        return str(source_resolved)

    target_dir = _folder_disk_path(target_folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_resolved.name
    if target_path.exists() and target_path.resolve() != source_resolved:
        raise HTTPException(409, f"Target file already exists: {target_path.name}")

    if target_path.resolve() != source_resolved:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)
        sidecar_source = _sidecar_path_for_pdf(str(source_resolved))
        if sidecar_source.exists():
            sidecar_target = _sidecar_path_for_pdf(str(target_path))
            sidecar_source.rename(sidecar_target)
    return str(target_path.resolve())


def _sanitize_fts_query(query: str) -> str:
    words = query.strip().split()
    out = []
    for w in words:
        if w.upper() in ("AND", "OR", "NOT", "NEAR"):
            continue
        if re.search(r"\w\.\w", w):
            clean = re.sub(r"[^\w.]+", " ", w).strip()
            if clean:
                out.append(f'"{clean}"')
        elif re.search(r"[-/\\@#$%&*()+=,:;^~\[\]{}!?<>]", w):
            parts = re.split(r"[-/\\.,;:]+", w)
            out.extend(f'"{p.strip()}"' for p in parts if p.strip())
        else:
            clean = w.strip('"\'')
            if clean:
                out.append(clean)
    return " ".join(out)


def _query_focus_terms(text: str, limit: int = 10) -> list[str]:
    stop_words = {
        "the", "and", "for", "from", "with", "that", "this", "there",
        "what", "which", "where", "when", "into", "about", "their",
        "your", "have", "does", "dont", "need", "show", "list", "tell",
        "find", "page", "pages", "document", "documents", "folder",
        "same", "they", "them", "those", "these", "above",
    }
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", text or ""):
        clean = token.strip('"\'').lower()
        if len(clean) < 2 or clean in stop_words:
            continue
        tokens.append(clean)
    return _dedupe_strings(tokens, limit=limit)


def _query_expansion_terms(query: str) -> list[str]:
    text = (query or "").lower()
    extra = []
    if re.search(r"\bjb\b", text):
        extra.extend(["junction", "box"])
    if re.search(r"\bff\b", text) and ("junction" in text or re.search(r"\bjb\b", text)):
        extra.extend(["foundation", "fieldbus"])
    if "field bus" in text:
        extra.append("fieldbus")
    if re.search(r"\bvendor(s)?\b", text):
        extra.extend(["supplier", "bidders"])
    return _dedupe_strings(extra, limit=8)


def _domain_query_variants(question: str, retrieval_query: str, attachment_excerpt: str = "") -> list[str]:
    text = " ".join([
        question or "",
        retrieval_query or "",
        (attachment_excerpt or "")[:400],
    ]).lower()
    variants = []

    has_ff = bool(re.search(r"\bff\b", text))
    has_jb = "junction box" in text or bool(re.search(r"\bjb\b", text))
    has_vendor = bool(re.search(r"\b(vendor|vendors|supplier|suppliers|bidder|bidders|approved)\b", text))
    has_cpmsl = "cpmsl" in text or "supplier list" in text

    if has_ff and has_jb:
        variants.extend([
            "foundation fieldbus junction box",
            "foundation field bus ff jb",
        ])
        if has_vendor:
            variants.extend([
                "approved vendors foundation fieldbus junction box",
                "foundation field bus approved vendors",
                "6.122 foundation field bus",
            ])

    if has_cpmsl:
        variants.extend([
            "common project master supplier list",
            "section d instrumentation",
        ])

    return _dedupe_strings(variants, limit=6)


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if not hasattr(_get_embedding_model, "_cache"):
        _get_embedding_model._cache = {}
    cache = _get_embedding_model._cache
    if model_name not in cache:
        from sentence_transformers import SentenceTransformer
        cache[model_name] = SentenceTransformer(model_name)
    return cache[model_name]


def _thread_title_from_message(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return "New chat"
    return clean[:57] + "..." if len(clean) > 60 else clean


def _fetch_chat_messages(conn, thread_id: str):
    rows = conn.execute(
        "SELECT id, role, content, sources, created_at "
        "FROM chat_messages WHERE thread_id = ? ORDER BY id",
        (thread_id,)
    ).fetchall()
    messages = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item["sources"]) if item["sources"] else []
        messages.append(item)
    return messages


def _list_chat_threads(conn, folder: str):
    rows = conn.execute(
        """
        SELECT t.id, t.project, t.title, t.created_at, t.updated_at,
               (
                   SELECT content FROM chat_messages m
                   WHERE m.thread_id = t.id
                   ORDER BY m.id DESC LIMIT 1
               ) AS last_message,
               (
                   SELECT role FROM chat_messages m
                   WHERE m.thread_id = t.id
                   ORDER BY m.id DESC LIMIT 1
               ) AS last_role,
               (
                   SELECT COUNT(*) FROM chat_messages m
                   WHERE m.thread_id = t.id
               ) AS message_count
        FROM chat_threads t
        WHERE t.project = ?
        ORDER BY datetime(t.updated_at) DESC, t.id DESC
        """,
        (folder,)
    ).fetchall()
    threads = []
    for row in rows:
        item = dict(row)
        item["folder"] = item["project"]
        threads.append(item)
    return threads


def _create_chat_thread(conn, folder: str, title: Optional[str] = None) -> dict:
    thread_id = str(uuid4())
    clean_title = (title or "New chat").strip() or "New chat"
    conn.execute(
        "INSERT INTO chat_threads (id, project, title) VALUES (?, ?, ?)",
        (thread_id, folder, clean_title)
    )
    conn.commit()
    return {
        "id": thread_id,
        "project": folder,
        "folder": folder,
        "title": clean_title,
    }


def _touch_thread(conn, thread_id: str):
    conn.execute(
        "UPDATE chat_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (thread_id,)
    )


def _insert_chat_message(
    conn,
    thread_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
):
    conn.execute(
        "INSERT INTO chat_messages (thread_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (thread_id, role, content, json.dumps(sources or []))
    )
    _touch_thread(conn, thread_id)


def _fts_search_folder(conn, folder: str, query: str, limit: int = 5) -> list[dict]:
    doc_ids = _folder_doc_ids(conn, folder)
    if not doc_ids:
        return []

    clean = _sanitize_fts_query(query)
    if not clean:
        return []

    ph = ",".join("?" * len(doc_ids))
    queries = [clean]
    extra_terms = _query_expansion_terms(query)
    if extra_terms:
        expanded = " ".join([clean, _sanitize_fts_query(" ".join(extra_terms))]).strip()
        if expanded and expanded not in queries:
            queries.append(expanded)
    parts = clean.split()
    if extra_terms:
        parts.extend(_sanitize_fts_query(" ".join(extra_terms)).split())
    if len(parts) > 1:
        queries.append(" OR ".join(parts))

    seen = set()
    results = []
    for candidate in queries:
        try:
            rows = conn.execute(
                f"""
                SELECT p.doc_id, d.title AS doc_title, p.page_num, p.breadcrumb,
                       p.page_type, d.project AS folder,
                       snippet(pages_fts, 0, '>>>', '<<<', '...', 30) AS snippet,
                       rank
                FROM pages_fts
                JOIN pages p ON p.id = pages_fts.rowid
                JOIN documents d ON d.id = p.doc_id
                WHERE pages_fts MATCH ?
                  AND p.doc_id IN ({ph})
                  AND p.page_type != 'skipped'
                ORDER BY rank
                LIMIT ?
                """,
                [candidate, *doc_ids, limit]
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        for row in rows:
            key = (row["doc_id"], row["page_num"])
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "folder": row["folder"],
                "page_num": row["page_num"],
                "breadcrumb": row["breadcrumb"],
                "page_type": row["page_type"],
                "snippet": row["snippet"],
                "score": float(row["rank"]) if row["rank"] is not None else 0.0,
                "search_type": "fts",
            })
            if len(results) >= limit:
                return results
    return results


def _semantic_search_folder(conn, folder: str, query: str, limit: int = 5) -> list[dict]:
    doc_ids = _folder_doc_ids(conn, folder)
    if not doc_ids:
        return []

    ph = ",".join("?" * len(doc_ids))
    count = conn.execute(
        f"""
        SELECT COUNT(*) FROM page_embeddings pe
        JOIN pages p ON p.id = pe.page_id
        WHERE p.doc_id IN ({ph}) AND p.page_type != 'skipped'
        """,
        doc_ids
    ).fetchone()[0]
    if count == 0:
        return []

    try:
        import numpy as np
        model_name = conn.execute(
            f"""
            SELECT pe.model FROM page_embeddings pe
            JOIN pages p ON p.id = pe.page_id
            WHERE p.doc_id IN ({ph})
            LIMIT 1
            """,
            doc_ids
        ).fetchone()["model"]
        model = _get_embedding_model(model_name)
        query_emb = model.encode([query])[0]
    except Exception:
        return []

    rows = conn.execute(
        f"""
        SELECT p.doc_id, d.title AS doc_title, d.project AS folder,
               p.page_num, p.breadcrumb, p.page_type,
               pe.chunk_text, pe.embedding
        FROM page_embeddings pe
        JOIN pages p ON p.id = pe.page_id
        JOIN documents d ON d.id = p.doc_id
        WHERE p.doc_id IN ({ph}) AND p.page_type != 'skipped'
        """,
        doc_ids
    ).fetchall()

    scored = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if len(emb) != len(query_emb):
            continue
        sim = float(np.dot(query_emb, emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
        ))
        scored.append({
            "doc_id": row["doc_id"],
            "doc_title": row["doc_title"],
            "folder": row["folder"],
            "page_num": row["page_num"],
            "breadcrumb": row["breadcrumb"],
            "page_type": row["page_type"],
            "snippet": row["chunk_text"][:240],
            "score": sim,
            "search_type": "semantic",
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    deduped = []
    seen = set()
    for item in scored:
        key = (item["doc_id"], item["page_num"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _build_retrieval_query(history: list[dict], question: str) -> str:
    recent_user_turns = [
        m["content"].strip()
        for m in history
        if m["role"] == "user" and m["content"].strip()
    ][-2:]
    if not recent_user_turns:
        return question

    lower = question.lower()
    short_follow_up = len(question.split()) <= 10 or any(
        token in lower for token in ("it", "that", "those", "they", "same", "above", "there")
    )
    if short_follow_up:
        return " ".join(recent_user_turns + [question])
    return question


CHAT_TOOL_CATALOG = [
    {
        "name": "search_pages",
        "purpose": "keyword search across folder pages when exact terminology is likely to exist in the documents",
    },
    {
        "name": "semantic_search",
        "purpose": "vector search for conceptually similar passages when the wording may differ from the user's question",
    },
    {
        "name": "search_sections",
        "purpose": "search section headings and breadcrumbs to jump to relevant parts of documents",
    },
    {
        "name": "get_document_info",
        "purpose": "inspect document titles, metadata, and document types for relevant matches",
    },
    {
        "name": "get_toc",
        "purpose": "inspect a document outline before drilling into specific pages",
    },
    {
        "name": "get_page",
        "purpose": "read an exact page once a likely hit has been found",
    },
    {
        "name": "get_pages",
        "purpose": "expand to adjacent pages or a short page range for fuller context",
    },
]


def _invoke_chat_model(system_prompt: str, messages: list[dict], max_tokens: int = 1600) -> str:
    api_key = _chat_api_key()
    if not api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured on the server.")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": _chat_model_name(),
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"LLM request failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc

    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ).strip()
    if not text:
        raise HTTPException(502, "LLM returned an empty response.")
    return text


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])


def _dedupe_strings(values: list[str], limit: Optional[int] = None) -> list[str]:
    result = []
    seen = set()
    for value in values:
        clean = re.sub(r"\s+", " ", (value or "").strip())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
        if limit is not None and len(result) >= limit:
            break
    return result


def _fallback_investigation_plan(question: str, retrieval_query: str, attachment_excerpt: str = "") -> dict:
    domain_variants = _domain_query_variants(question, retrieval_query, attachment_excerpt)
    focus_terms = _dedupe_strings([
        *_query_focus_terms(retrieval_query, limit=8),
        *_query_focus_terms(attachment_excerpt, limit=4),
        *_query_focus_terms(" ".join(domain_variants), limit=4),
    ], limit=10)
    compact = " ".join(focus_terms[:5]) if focus_terms else retrieval_query
    return {
        "strategy_summary": (
            "Search exact terms first, shortlist likely documents and pages, "
            "open only the strongest matching pages, inspect likely section headings, "
            "then expand neighboring pages only if needed."
        ),
        "keyword_queries": _dedupe_strings([retrieval_query, compact, *domain_variants], limit=5),
        "semantic_queries": _dedupe_strings([question, retrieval_query, *domain_variants[:2]], limit=3),
        "section_queries": _dedupe_strings([compact, *domain_variants], limit=4),
        "focus_terms": focus_terms,
    }


def _plan_folder_investigation(
    folder: str,
    question: str,
    history: list[dict],
    retrieval_query: str,
    attachment_excerpt: str = "",
) -> dict:
    history_preview = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history[-6:]
        if item.get("content")
    ) or "[no prior messages]"
    tool_lines = "\n".join(
        f"- {tool['name']}: {tool['purpose']}"
        for tool in CHAT_TOOL_CATALOG
    )
    system_prompt = (
        "You are planning a folder-scoped document investigation.\n"
        "The backend can use the following tools:\n"
        f"{tool_lines}\n\n"
        "Your job is to propose a gradual investigation plan: start with precise keyword search, "
        "reword queries to match likely document terminology, use semantic search when wording may differ, "
        "and use section/page expansion to gather surrounding context before answering.\n"
        "Return JSON only with keys: strategy_summary, keyword_queries, semantic_queries, "
        "section_queries, focus_terms.\n"
        "Each query list should contain short, useful search strings, not full paragraphs."
    )
    attachment_preview = attachment_excerpt[:2500].strip() or "[none]"
    messages = [{
        "role": "user",
        "content": (
            f"Folder: {folder}\n"
            f"Conversation so far:\n{history_preview}\n\n"
            f"Latest question: {question}\n"
            f"Combined retrieval query: {retrieval_query}\n"
            f"Uploaded document excerpt:\n{attachment_preview}\n"
        ),
    }]
    try:
        raw = _invoke_chat_model(system_prompt, messages, max_tokens=700)
        plan = _extract_json_object(raw)
        return {
            "strategy_summary": str(plan.get("strategy_summary") or "").strip() or "Investigate with keyword and semantic search, then expand context.",
            "keyword_queries": _dedupe_strings([retrieval_query, question, *(plan.get("keyword_queries") or [])], limit=4),
            "semantic_queries": _dedupe_strings([question, *(plan.get("semantic_queries") or [])], limit=3),
            "section_queries": _dedupe_strings(plan.get("section_queries") or [], limit=3),
            "focus_terms": _dedupe_strings(plan.get("focus_terms") or [], limit=8),
        }
    except Exception:
        return _fallback_investigation_plan(question, retrieval_query, attachment_excerpt=attachment_excerpt)


def _search_sections_folder(conn, folder: str, query: str, limit: int = 4) -> list[dict]:
    if not (query or "").strip():
        return []
    rows = conn.execute(
        """
        SELECT s.doc_id, d.title AS doc_title, d.project AS folder,
               s.heading, s.level, s.page_start, COALESCE(s.page_end, s.page_start) AS page_end,
               s.breadcrumb
        FROM sections s
        JOIN documents d ON d.id = s.doc_id
        WHERE (d.project = ? OR d.project LIKE ?)
          AND (s.heading LIKE ? OR s.breadcrumb LIKE ?)
        ORDER BY s.level ASC, s.page_start ASC
        LIMIT ?
        """,
        (folder, f"{folder}/%", f"%{query}%", f"%{query}%", limit),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_page_range(conn, doc_id: int, start: int, end: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.doc_id, d.title AS doc_title, d.project AS folder,
               p.page_num, p.breadcrumb, p.page_type, p.content
        FROM pages p
        JOIN documents d ON d.id = p.doc_id
        WHERE p.doc_id = ? AND p.page_num BETWEEN ? AND ? AND p.page_type != 'skipped'
        ORDER BY p.page_num
        """,
        (doc_id, start, end),
    ).fetchall()
    return [dict(row) for row in rows]


def _term_overlap_score(text: str, focus_terms: list[str]) -> int:
    haystack = (text or "").lower()
    return sum(1 for term in focus_terms if term and term.lower() in haystack)


def _candidate_priority(item: dict, doc_scores: Counter, focus_terms: list[str]) -> tuple:
    searchable = " ".join([
        item.get("doc_title") or "",
        item.get("breadcrumb") or "",
        item.get("snippet") or "",
    ])
    return (
        _term_overlap_score(searchable, focus_terms),
        doc_scores.get(item["doc_id"], 0),
        1 if item.get("search_type") == "fts" else 0,
        1 if item.get("search_type") == "section" else 0,
        float(item.get("score") or 0.0),
    )


def _relevant_doc_sections(conn, doc_id: int, focus_terms: list[str], limit: int = 3) -> list[dict]:
    rows = conn.execute(
        """
        SELECT heading, level, page_start, COALESCE(page_end, page_start) AS page_end, breadcrumb
        FROM sections
        WHERE doc_id = ?
        ORDER BY seq
        """,
        (doc_id,),
    ).fetchall()
    scored = []
    for row in rows:
        item = dict(row)
        text = " ".join([item["heading"], item.get("breadcrumb") or ""])
        overlap = _term_overlap_score(text, focus_terms)
        if overlap <= 0:
            continue
        scored.append((
            overlap,
            -int(item["level"] or 0),
            -int(item["page_start"] or 0),
            item,
        ))
    scored.sort(key=lambda entry: entry[:-1], reverse=True)
    return [item for *_ignored, item in scored[:limit]]


def _open_page_window(
    conn,
    doc_id: int,
    start: int,
    end: int,
    *,
    search_type: str,
    score: float = 0.0,
    snippet: str = "",
) -> list[dict]:
    pages = _fetch_page_range(conn, doc_id, start, end)
    for page in pages:
        page["search_type"] = search_type
        page["score"] = score
        page["snippet"] = snippet or f"Opened pages {start}-{end}"
    return pages


def _focused_excerpt(body: str, snippet: str, max_chars: int) -> str:
    text = (body or "").strip()
    if len(text) <= max_chars:
        return text

    snippet_text = (snippet or "").strip()
    if not snippet_text:
        return text[:max_chars].rstrip() + "\n...[truncated]"

    lowered = text.lower()
    candidates = []

    clean_snippet = snippet_text.replace(">>>", "").replace("<<<", "")
    candidates.extend(
        segment.strip()
        for segment in clean_snippet.split("...")
        if len(segment.strip()) >= 10
    )
    candidates.extend(re.findall(r">>>(.*?)<<<", snippet_text))

    anchor_index = -1
    anchor_length = 0
    snippet_terms = [
        token for token in re.findall(r"[A-Za-z0-9]+", clean_snippet.lower())
        if len(token) >= 2 and token not in {"the", "and", "for", "with", "from", "page"}
    ]
    if snippet_terms:
        best_score = 0
        best_index = -1
        lines = text.splitlines()
        offset = 0
        for idx, line in enumerate(lines):
            window = "\n".join(lines[idx:idx + 3]).lower()
            score = sum(1 for term in snippet_terms if term in window)
            if score > best_score:
                best_score = score
                best_index = offset
            offset += len(line) + 1
        if best_score >= 2:
            anchor_index = best_index
            anchor_length = 0

    if anchor_index == -1:
        for candidate in sorted(candidates, key=len, reverse=True):
            idx = lowered.find(candidate.strip().lower())
            if idx != -1:
                anchor_index = idx
                anchor_length = len(candidate.strip())
                break

    if anchor_index == -1:
        return text[:max_chars].rstrip() + "\n...[truncated]"

    before = max_chars // 4
    start = max(0, anchor_index - before)
    end = min(len(text), max(anchor_index + anchor_length + (max_chars - before), start + max_chars))
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "...[truncated]\n" + excerpt
    if end < len(text):
        excerpt = excerpt.rstrip() + "\n...[truncated]"
    return excerpt


def _append_source(sources: list[dict], seen: set, item: dict):
    key = (item["doc_id"], item["page_num"])
    if key in seen:
        return
    seen.add(key)
    body = (item.get("content") or "").strip()
    if len(body) > CHAT_SOURCE_MAX_CHARS:
        body = _focused_excerpt(body, item.get("snippet") or "", CHAT_SOURCE_MAX_CHARS)
    sources.append({
        "doc_id": item["doc_id"],
        "doc_title": item["doc_title"],
        "folder": item["folder"],
        "page_num": item["page_num"],
        "breadcrumb": item.get("breadcrumb"),
        "page_type": item.get("page_type") or "text",
        "snippet": item.get("snippet") or body[:CHAT_SOURCE_SNIPPET_CHARS],
        "search_type": item.get("search_type") or "context",
        "score": round(float(item.get("score") or 0.0), 4),
        "content": body,
    })


def _retrieve_folder_context(
    conn,
    folder: str,
    question: str,
    history: list[dict],
    limit: int = 10,
    attachment_excerpt: str = "",
):
    retrieval_query = _build_retrieval_query(history, question)
    plan = _fallback_investigation_plan(question, retrieval_query, attachment_excerpt=attachment_excerpt)
    sources = []
    seen = set()
    investigation = []
    focus_terms = plan.get("focus_terms") or []

    fts_queries = _dedupe_strings([retrieval_query, *(plan.get("keyword_queries") or [])], limit=5)
    semantic_queries = _dedupe_strings([question, *(plan.get("semantic_queries") or [])], limit=3)
    section_queries = _dedupe_strings(plan.get("section_queries") or [], limit=4)

    page_candidates = []
    page_seen = set()
    for query in fts_queries:
        hits = _fts_search_folder(conn, folder, query, limit=4)
        investigation.append(f"search_pages('{query}') -> {len(hits)} hit(s)")
        for hit in hits:
            key = (hit["doc_id"], hit["page_num"])
            if key in page_seen:
                continue
            page_seen.add(key)
            page_candidates.append(hit)
            if len(page_candidates) >= CHAT_SEARCH_MAX_PAGE_CANDIDATES:
                break
        if len(page_candidates) >= CHAT_SEARCH_MAX_PAGE_CANDIDATES:
            break

    section_matches = []
    section_seen = set()
    for query in section_queries:
        matches = _search_sections_folder(conn, folder, query, limit=3)
        investigation.append(f"search_sections('{query}') -> {len(matches)} match(es)")
        for match in matches:
            key = (match["doc_id"], match["page_start"], match["heading"])
            if key in section_seen:
                continue
            section_seen.add(key)
            section_matches.append(match)
            if len(section_matches) >= CHAT_SEARCH_MAX_SECTION_CANDIDATES:
                break
        if len(section_matches) >= CHAT_SEARCH_MAX_SECTION_CANDIDATES:
            break

    if len(page_candidates) < 4:
        for query in semantic_queries:
            hits = _semantic_search_folder(conn, folder, query, limit=4)
            investigation.append(f"semantic_search('{query}') -> {len(hits)} hit(s)")
            for hit in hits:
                key = (hit["doc_id"], hit["page_num"])
                if key in page_seen:
                    continue
                page_seen.add(key)
                page_candidates.append(hit)
                if len(page_candidates) >= CHAT_SEARCH_MAX_PAGE_CANDIDATES:
                    break
            if len(page_candidates) >= CHAT_SEARCH_MAX_PAGE_CANDIDATES:
                break

    doc_scores = Counter()
    for hit in page_candidates:
        doc_scores[hit["doc_id"]] += 3 if hit.get("search_type") == "fts" else 1
        doc_scores[hit["doc_id"]] += max(1, _term_overlap_score(
            " ".join([hit.get("doc_title") or "", hit.get("breadcrumb") or "", hit.get("snippet") or ""]),
            focus_terms,
        ))
    for match in section_matches:
        doc_scores[match["doc_id"]] += 2 + max(1, _term_overlap_score(
            " ".join([match.get("heading") or "", match.get("breadcrumb") or ""]),
            focus_terms,
        ))

    page_candidates.sort(
        key=lambda item: _candidate_priority(item, doc_scores, focus_terms),
        reverse=True,
    )
    shortlisted_docs = [
        doc_id for doc_id, _score in doc_scores.most_common(CHAT_SEARCH_MAX_DOC_SHORTLIST)
    ]

    opened_pages = 0
    for hit in page_candidates[:CHAT_CONTEXT_TARGET_SOURCES]:
        if opened_pages >= CHAT_CONTEXT_MAX_OPEN_PAGES or len(sources) >= limit:
            break
        pages = _open_page_window(
            conn,
            hit["doc_id"],
            hit["page_num"],
            hit["page_num"],
            search_type=hit.get("search_type") or "fts",
            score=hit.get("score") or 0.0,
            snippet=hit.get("snippet") or f"Opened page {hit['page_num']}",
        )
        investigation.append(
            f"get_page(doc={hit['doc_id']}, page={hit['page_num']}) -> {len(pages)} page(s)"
        )
        for page in pages:
            before = len(sources)
            _append_source(sources, seen, page)
            if len(sources) > before:
                opened_pages += 1
                if opened_pages >= CHAT_CONTEXT_MAX_OPEN_PAGES or len(sources) >= limit:
                    break

    section_pages_opened = 0
    for doc_id in shortlisted_docs:
        if section_pages_opened >= CHAT_CONTEXT_MAX_SECTION_PAGES or len(sources) >= limit:
            break
        toc_matches = _relevant_doc_sections(conn, doc_id, focus_terms, limit=2)
        investigation.append(f"get_toc(doc={doc_id}) -> {len(toc_matches)} relevant heading(s)")
        for match in toc_matches:
            if section_pages_opened >= CHAT_CONTEXT_MAX_SECTION_PAGES or len(sources) >= limit:
                break
            page_end = min(match["page_end"], match["page_start"] + 1)
            pages = _open_page_window(
                conn,
                doc_id,
                match["page_start"],
                page_end,
                search_type="section",
                score=0.0,
                snippet=f"Section match: {match['heading']}",
            )
            investigation.append(
                f"get_pages(doc={doc_id}, start={match['page_start']}, end={page_end}) -> {len(pages)} page(s)"
            )
            for page in pages:
                before = len(sources)
                _append_source(sources, seen, page)
                if len(sources) > before:
                    section_pages_opened += 1
                    if section_pages_opened >= CHAT_CONTEXT_MAX_SECTION_PAGES or len(sources) >= limit:
                        break

    neighbor_pages_opened = 0
    for hit in page_candidates[:CHAT_CONTEXT_MAX_NEIGHBOR_PAGES]:
        if neighbor_pages_opened >= CHAT_CONTEXT_MAX_NEIGHBOR_PAGES or len(sources) >= limit:
            break
        start = max(1, hit["page_num"] - 1)
        end = hit["page_num"] + 1
        pages = _open_page_window(
            conn,
            hit["doc_id"],
            start,
            end,
            search_type="context",
            score=hit.get("score") or 0.0,
            snippet=f"Expanded around page {hit['page_num']}",
        )
        investigation.append(
            f"get_pages(doc={hit['doc_id']}, start={start}, end={end}) -> {len(pages)} page(s)"
        )
        for page in pages:
            before = len(sources)
            _append_source(sources, seen, page)
            if len(sources) > before:
                neighbor_pages_opened += 1
                if neighbor_pages_opened >= CHAT_CONTEXT_MAX_NEIGHBOR_PAGES or len(sources) >= limit:
                    break

    final = sources[:min(limit, CHAT_CONTEXT_MAX_SOURCES)]
    for idx, item in enumerate(final, start=1):
        item["id"] = idx

    return {
        "query": retrieval_query,
        "plan": plan,
        "investigation": investigation,
        "sources": final,
    }


def _chat_model_name() -> str:
    return os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514")


def _chat_api_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY")


def _generate_mcp_chat_answer(
    folder: str,
    question: str,
    history: list[dict],
    attachment: Optional[dict] = None,
):
    api_key = _chat_api_key()
    if not api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured on the server.")

    try:
        import mcp_server

        # Keep the web chat pinned to the same MCP tool definitions and database
        # as external clients, without depending on the local SSE hop.
        mcp_server.DB_PATH = DB_PATH
        return run_folder_chat(
            mcp_url=MCP_SERVER_URL,
            api_key=api_key,
            model=_chat_model_name(),
            project=folder,
            question=question,
            history=history,
            attachment=attachment,
            mcp_server=mcp_server.mcp,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Chat request failed: {exc}") from exc


def _generate_folder_answer(folder: str, question: str, history: list[dict], retrieval: dict):
    if not retrieval["sources"]:
        return (
            "I don't have enough information in this folder's documents to answer that.\n\n"
            "Try asking with more specific document terms, section names, tags, or page topics."
        )

    system_prompt = (
        "You are Esteem Folder Knowledge, a folder-scoped document assistant.\n"
        f"You are currently answering questions for the folder '{folder}'.\n"
        "The folder scope includes that folder and all of its sub-folders.\n"
        "You must answer ONLY from the provided source excerpts and the prior conversation.\n"
        "Available investigation tools in this system are: search_pages, semantic_search, search_sections, "
        "get_document_info, get_toc, get_page, and get_pages.\n"
        "The backend has already used these tools to investigate before answering.\n"
        "Treat this as a deliberate research workflow: start from the strongest direct evidence, then synthesize "
        "broader context from related sections and nearby pages.\n"
        "Never use outside knowledge. Never use information from any folder outside this scope.\n"
        "If the excerpts do not support a confident answer, say that you do not have enough "
        "information in the documents for this folder.\n"
        "Do not speculate or fill gaps.\n"
        "Respond in markdown.\n"
        "When you make factual claims, cite supporting sources inline like [1] or [2].\n"
        "Aim for a comprehensive answer, not a shallow summary. Compare documents when relevant, "
        "call out uncertainty or disagreement explicitly, and use brief headings when they help.\n"
        "If useful, end with a brief 'Sources' list that references the same source numbers."
    )

    excerpts = []
    for source in retrieval["sources"]:
        excerpts.append(
            f"[{source['id']}] {source['folder']} | {source['doc_title']} | page {source['page_num']} | "
            f"{source['breadcrumb'] or 'No section breadcrumb'}\n"
            f"{source['content']}"
        )

    prompt = (
        f"Folder: {folder}\n"
        f"Retrieved with query: {retrieval['query']}\n\n"
        f"Investigation strategy: {retrieval.get('plan', {}).get('strategy_summary', '')}\n"
        "Available tools:\n"
        + "\n".join(f"- {tool['name']}: {tool['purpose']}" for tool in CHAT_TOOL_CATALOG)
        + "\n\nInvestigation log:\n"
        + "\n".join(f"- {step}" for step in retrieval.get("investigation", []))
        + "\n\n"
        "Source excerpts:\n"
        "================\n"
        f"{chr(10).join(excerpts)}\n\n"
        "Answer the user's latest question using only those excerpts. First resolve the core question, "
        "then add the supporting detail that makes the answer useful."
    )

    messages = []
    for item in history[-8:]:
        messages.append({
            "role": item["role"],
            "content": item["content"],
        })
    messages.append({
        "role": "user",
        "content": f"{prompt}\n\nLatest user question:\n{question}",
    })
    return _invoke_chat_model(system_prompt, messages, max_tokens=2200)


def _truncate_text(text: str, max_chars: int, suffix: str = "\n...[truncated]") -> str:
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + suffix


def _pdf_page_count(file_path: str) -> Optional[int]:
    try:
        import fitz
    except Exception:
        return None

    try:
        doc = fitz.open(file_path)
        try:
            return int(getattr(doc, "page_count", len(doc)))
        finally:
            doc.close()
    except Exception:
        return None


def _default_attachment_review_question(filename: str) -> str:
    return (
        f"Review '{filename}' against the folder documents. "
        "Summarize matches, gaps, conflicts, and risks."
    )


def _prepare_chat_review_attachment(upload: UploadFile) -> dict:
    filename = Path(upload.filename or "document").name or "document"
    ext = Path(filename).suffix.lower()
    if not ext or ext not in CHAT_REVIEW_ALLOWED_EXTENSIONS or is_archive(filename):
        raise HTTPException(
            400,
            "Supported review files are PDF, image, Word, and Excel documents.",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="ocrrag_chat_review_"))
    tmp_path = tmp_dir / f"{uuid4().hex}{ext}"
    try:
        with tmp_path.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        if ext == ".pdf":
            pdf_pages = _pdf_page_count(str(tmp_path))
            if pdf_pages and pdf_pages > CHAT_REVIEW_MAX_PAGES:
                raise HTTPException(
                    400,
                    f"Review uploads are limited to {CHAT_REVIEW_MAX_PAGES} pages.",
                )

        pages, _sections = extract_file(str(tmp_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not read the uploaded document: {exc}") from exc
    finally:
        try:
            upload.file.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    normalized_pages = []
    for idx, page in enumerate(pages, start=1):
        content = _truncate_text(str(page.get("content") or "").strip(), CHAT_REVIEW_MAX_PAGE_CHARS)
        if not content:
            continue
        normalized_pages.append({
            "id": f"A{len(normalized_pages) + 1}",
            "page_num": int(page.get("page_num") or idx),
            "breadcrumb": str(page.get("breadcrumb") or "").strip(),
            "content": content,
        })

    if not normalized_pages:
        raise HTTPException(
            400,
            "No readable text could be extracted from that file. Try a clearer scan or a text-based file.",
        )
    if len(normalized_pages) > CHAT_REVIEW_MAX_PAGES:
        raise HTTPException(
            400,
            f"Review uploads are limited to {CHAT_REVIEW_MAX_PAGES} pages after extraction.",
        )

    excerpt_parts = []
    excerpt_chars = 0
    for page in normalized_pages:
        title = f"{page['id']} | page {page['page_num']}"
        if page["breadcrumb"]:
            title += f" | {page['breadcrumb']}"
        block = f"{title}\n{page['content']}"
        if excerpt_chars + len(block) > CHAT_REVIEW_MAX_EXCERPT_CHARS:
            remaining = CHAT_REVIEW_MAX_EXCERPT_CHARS - excerpt_chars
            if remaining > 200:
                excerpt_parts.append(_truncate_text(block, remaining, suffix=""))
            break
        excerpt_parts.append(block)
        excerpt_chars += len(block)

    return {
        "filename": filename,
        "page_count": len(normalized_pages),
        "excerpt": "\n\n".join(excerpt_parts),
        "pages": normalized_pages,
        "message_sources": [{
            "kind": "attachment_meta",
            "name": filename,
            "page_count": len(normalized_pages),
        }],
        "assistant_sources": [
            {
                "id": page["id"],
                "kind": "attachment",
                "name": filename,
                "page_num": page["page_num"],
                "breadcrumb": page["breadcrumb"],
                "snippet": _truncate_text(page["content"], 220, suffix="..."),
            }
            for page in normalized_pages
        ],
    }


def _generate_attachment_review(
    folder: str,
    question: str,
    history: list[dict],
    retrieval: dict,
    attachment: dict,
):
    attachment_excerpts = []
    for page in attachment["pages"]:
        attachment_excerpts.append(
            f"[{page['id']}] Uploaded file | {attachment['filename']} | page {page['page_num']} | "
            f"{page['breadcrumb'] or 'No section breadcrumb'}\n"
            f"{page['content']}"
        )

    folder_excerpts = []
    for source in retrieval["sources"]:
        folder_excerpts.append(
            f"[{source['id']}] {source['folder']} | {source['doc_title']} | page {source['page_num']} | "
            f"{source['breadcrumb'] or 'No section breadcrumb'}\n"
            f"{source['content']}"
        )

    system_prompt = (
        "You are Esteem Folder Knowledge, performing a document review.\n"
        f"You are currently comparing an uploaded document against the folder '{folder}'.\n"
        "You must answer ONLY from the uploaded document excerpts, the folder source excerpts, and the prior conversation.\n"
        "Never use outside knowledge. Do not speculate.\n"
        "Treat uploaded document pages as evidence labeled [A1], [A2], etc.\n"
        "Treat folder documents as evidence labeled [1], [2], etc.\n"
        "Respond in markdown.\n"
        "Use this structure unless a section would truly be empty:\n"
        "## Overall assessment\n"
        "## Matches\n"
        "## Gaps or missing evidence\n"
        "## Conflicts or risks\n"
        "## Suggested next checks\n"
        "## Sources\n"
        "When you make factual claims, cite the supporting uploaded page(s) and folder source(s) inline."
    )

    prompt = (
        f"Folder: {folder}\n"
        f"Latest question: {question}\n"
        f"Uploaded document: {attachment['filename']} ({attachment['page_count']} page(s))\n"
        f"Folder retrieval query: {retrieval['query']}\n\n"
        f"Investigation strategy: {retrieval.get('plan', {}).get('strategy_summary', '')}\n"
        "Investigation log:\n"
        + "\n".join(f"- {step}" for step in retrieval.get("investigation", []))
        + "\n\nUploaded document excerpts:\n"
        "==========================\n"
        + ("\n\n".join(attachment_excerpts) or "[No uploaded document excerpts available]")
        + "\n\nFolder source excerpts:\n"
        "======================\n"
        + ("\n\n".join(folder_excerpts) or "[No matching folder excerpts were found]")
        + "\n\nReview the uploaded document against the folder documents. "
        "Focus on what is supported, what is missing, and what appears inconsistent."
    )

    messages = []
    for item in history[-8:]:
        messages.append({
            "role": item["role"],
            "content": item["content"],
        })
    messages.append({
        "role": "user",
        "content": prompt,
    })
    return _invoke_chat_model(system_prompt, messages, max_tokens=2400)


# ---------------------------------------------------------------------------
# Ingestion tracker
# ---------------------------------------------------------------------------

class IngestionTracker:
    """Persists ingestion job state to SQLite so it survives page
    reloads and server restarts."""

    def _conn(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, filename, project, source_path=None):
        jid = str(uuid4())[:8]
        conn = self._conn()
        conn.execute(
            "INSERT INTO ingestion_jobs (id, filename, project, status, stage, source_path) "
            "VALUES (?, ?, ?, 'queued', '', ?)",
            (jid, filename, project, source_path)
        )
        conn.commit()
        conn.close()
        return jid

    def update(self, jid, **kw):
        if not kw:
            return
        conn = self._conn()
        sets = ", ".join(f"{k} = ?" for k in kw)
        vals = list(kw.values()) + [jid]
        conn.execute(f"UPDATE ingestion_jobs SET {sets} WHERE id = ?", vals)
        conn.commit()
        conn.close()

    def all(self):
        conn = self._conn()
        # Clean terminal jobs older than 1 hour
        conn.execute(
            "DELETE FROM ingestion_jobs WHERE status IN ('completed', 'failed', 'cancelled') "
            "AND started_at < datetime('now', '-1 hour')"
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM ingestion_jobs ORDER BY started_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def for_project(self, project):
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE project = ? "
            "AND (status IN ('queued', 'running') "
            "     OR started_at > datetime('now', '-1 hour')) "
            "ORDER BY started_at DESC",
            (project,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get(self, jid):
        conn = self._conn()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (jid,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def is_cancel_requested(self, jid):
        row = self.get(jid)
        return bool(row and row.get("cancel_requested"))


tracker = IngestionTracker()


class IngestionError(RuntimeError):
    """Raised when a file cannot be ingested cleanly."""


class IngestionCancelled(RuntimeError):
    """Raised when a user requests cancellation for an ingestion job."""


# ---------------------------------------------------------------------------
# Folder routes
# ---------------------------------------------------------------------------

@app.get("/api/projects")
@app.get("/api/folders")
def list_folders(scope: Optional[str] = Query(None), recursive: bool = Query(False)):
    conn = get_conn()
    try:
        folders = _all_known_folders(conn)
        exact_docs = {
            row["project"]: {
                "docs": row["docs"],
                "pages": row["pages"] or 0,
            }
            for row in conn.execute("""
                SELECT project, COUNT(*) AS docs, COALESCE(SUM(total_pages), 0) AS pages
                FROM documents
                GROUP BY project
            """).fetchall()
        }
        exact_pending = {}
        for known_folder in folders:
            exact_pending[known_folder] = len([
                item for item in _pending_uploads(conn, known_folder)
                if item["folder"] == known_folder
            ])

        scoped_folders = folders
        if scope:
            scope = _validate_folder_name(scope, "Folder name")
            prefix = scope + "/"
            if recursive:
                scoped_folders = [
                    folder for folder in folders
                    if folder == scope or folder.startswith(prefix)
                ]
            else:
                scoped_folders = [
                    folder for folder in folders
                    if folder.startswith(prefix) and "/" not in folder[len(prefix):]
                ]
        else:
            scoped_folders = folders

        results = []
        for folder in scoped_folders:
            scope_docs = 0
            scope_pages = 0
            scope_pending = 0
            prefix = folder + "/"
            for candidate, stats in exact_docs.items():
                if candidate == folder or candidate.startswith(prefix):
                    scope_docs += stats["docs"]
                    scope_pages += stats["pages"]
            for candidate, pending_count in exact_pending.items():
                if candidate == folder or candidate.startswith(prefix):
                    scope_pending += pending_count
            parent = folder.rsplit("/", 1)[0] if "/" in folder else None
            results.append({
                "folder": folder,
                "project": folder,
                "display_name": folder.rsplit("/", 1)[-1],
                "parent": parent,
                "depth": folder.count("/"),
                "docs": scope_docs,
                "pages": scope_pages,
                "pending": scope_pending,
            })

        return results
    finally:
        conn.close()


@app.post("/api/projects")
@app.post("/api/folders")
def create_folder(data: dict):
    folder = _validate_folder_name(data.get("name", "").strip(), "Folder name")
    folder_dir = _folder_disk_path(folder)
    folder_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "folder": folder, "project": folder}


@app.patch("/api/projects/{folder:path}")
@app.patch("/api/folders/{folder:path}")
def rename_folder(folder: str, data: dict):
    folder = _validate_folder_name(folder, "Folder name")
    new_folder = _validate_folder_name(data.get("name", "").strip(), "New folder name")
    if new_folder == folder:
        return {"status": "unchanged", "folder": folder}
    if new_folder.startswith(folder + "/"):
        raise HTTPException(400, "Cannot rename a folder inside itself")
    conn = get_conn()
    try:
        if new_folder in _all_known_folders(conn):
            raise HTTPException(400, f"Folder already exists: {new_folder}")
        _rename_folder_references(conn, "documents", folder, new_folder)
        _rename_folder_references(conn, "ingestion_jobs", folder, new_folder)
        _rename_folder_references(conn, "chat_threads", folder, new_folder)
        conn.commit()

        old_dir = _folder_disk_path(folder)
        new_dir = _folder_disk_path(new_folder)
        if old_dir.exists() and not new_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            old_dir.rename(new_dir)

        return {
            "status": "renamed",
            "old_name": folder,
            "new_name": new_folder,
            "folder": new_folder,
            "project": new_folder,
        }
    finally:
        conn.close()


@app.delete("/api/projects/{folder:path}")
@app.delete("/api/folders/{folder:path}")
def delete_folder(folder: str):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        doc_ids = [r["id"] for r in conn.execute(
            f"SELECT id FROM documents WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        ).fetchall()]
        for did in doc_ids:
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM sections WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM corrections WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM cross_references WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM quality_flags WHERE doc_id = ?", (did,))
        conn.execute(
            f"DELETE FROM chat_threads WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        )
        conn.execute(
            f"DELETE FROM ingestion_jobs WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        )
        conn.execute(
            f"DELETE FROM documents WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        )
        conn.commit()

        folder_dir = _folder_disk_path(folder)
        if folder_dir.exists():
            shutil.rmtree(folder_dir)
        _prune_empty_upload_ancestors(folder_dir)
        return {"status": "deleted", "folder": folder, "project": folder}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Document routes
# ---------------------------------------------------------------------------

@app.get("/api/projects/{folder:path}/documents")
@app.get("/api/folders/{folder:path}/documents")
def list_documents(folder: str):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        docs = conn.execute(
            f"SELECT * FROM documents WHERE {_folder_scope_sql('project')} ORDER BY project, id",
            _folder_scope_params(folder)
        ).fetchall()

        results = []
        for d in docs:
            meta = json.loads(d["metadata"]) if d["metadata"] else {}
            secs = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE doc_id = ?", (d["id"],)
            ).fetchone()[0]
            entry = {
                "id": d["id"], "title": d["title"],
                "folder": d["project"],
                "project": d["project"],
                "filename": d["filename"],
                "total_pages": d["total_pages"], "sections": secs,
                "document_type": meta.get("document_type"),
            }
            si = parse_split_info(d["filename"])
            if si:
                entry["split_info"] = si
            results.append(entry)
        return {"folder": folder, "project": folder, "documents": results, "pending": _pending_uploads(conn, folder)}
    finally:
        conn.close()


@app.get("/api/projects/{folder:path}/chats")
@app.get("/api/folders/{folder:path}/chats")
def list_folder_chats(folder: str):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        return _list_chat_threads(conn, folder)
    finally:
        conn.close()


@app.post("/api/projects/{folder:path}/chats")
@app.post("/api/folders/{folder:path}/chats")
def create_folder_chat(folder: str, data: Optional[dict] = None):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        created = _create_chat_thread(
            conn,
            folder,
            title=(data or {}).get("title"),
        )
        return created
    finally:
        conn.close()


@app.post("/api/projects/{folder:path}/upload")
@app.post("/api/folders/{folder:path}/upload")
async def upload_files(
    folder: str,
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
):
    """Upload files (or an entire folder) into *folder*.

    When *paths* is provided (one entry per file), each file is placed at
    the corresponding relative path inside the folder, preserving the
    original directory layout.  This is used by the browser folder-upload
    feature (``webkitdirectory``).
    """
    folder = _validate_folder_name(folder, "Folder name")
    folder_dir = _folder_disk_path(folder)
    folder_dir.mkdir(parents=True, exist_ok=True)

    use_paths = len(paths) == len(files)

    # Validate supplied paths
    if use_paths:
        for p in paths:
            normed = _normalize_folder_name(p)
            if '..' in normed.split('/'):
                raise HTTPException(400, "Invalid path component")

    uploaded = []
    for i, f in enumerate(files):
        ext = Path(f.filename).suffix.lower()
        content = await f.read()

        if is_archive(f.filename):
            # Determine extraction target: subfolder when a path is given
            if use_paths and paths[i]:
                rel_parent = str(Path(_normalize_folder_name(paths[i])).parent)
                target_dir = folder_dir / rel_parent if rel_parent != '.' else folder_dir
            else:
                target_dir = folder_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            import tempfile
            tmp = Path(tempfile.mktemp(suffix=ext))
            tmp.write_bytes(content)
            try:
                names = extract_archive(str(tmp), target_dir)
                uploaded.extend(names)
            finally:
                tmp.unlink(missing_ok=True)
        elif ext in INGESTABLE_EXTENSIONS:
            if use_paths and paths[i]:
                rel = _normalize_folder_name(paths[i])
                dest = folder_dir / rel
            else:
                dest = folder_dir / f.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            uploaded.append(dest.relative_to(folder_dir).as_posix())
        # else: silently skip unsupported types

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/api/projects/{folder:path}/pending/{filename:path}/discard")
@app.post("/api/folders/{folder:path}/pending/{filename:path}/discard")
def discard_pending_file(folder: str, filename: str, data: Optional[dict] = None):
    folder = _validate_folder_name(folder, "Folder name")
    relative_name = _normalize_folder_name(filename)
    upload_relative = _normalize_folder_name(f"{folder}/{relative_name}")
    _resolve_upload_path(upload_relative)
    reason = ((data or {}).get("reason") or "Removed from pending by user").strip()

    result = _discard_upload_path(
        upload_relative,
        reason=reason,
        action="discarded_pending",
    )

    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, status FROM ingestion_jobs
            WHERE source_path = ? AND status IN ('queued', 'running', 'failed')
            """,
            (upload_relative,),
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        if row["status"] == "running":
            tracker.update(
                row["id"],
                cancel_requested=1,
                cancel_reason=reason,
                stage="Cancellation requested",
            )
        else:
            tracker.update(
                row["id"],
                status="cancelled",
                stage="Removed from pending",
                error=reason,
                cancel_requested=0,
                delete_after_cancel=0,
                cancel_reason=reason,
            )

    return result


@app.patch("/api/documents/{doc_id}")
def rename_document(doc_id: int, data: dict):
    new_title = data.get("title", "").strip()
    if not new_title:
        raise HTTPException(400, "New title required")
    conn = get_conn()
    try:
        doc = conn.execute(
            "SELECT id FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        conn.execute(
            "UPDATE documents SET title = ? WHERE id = ?",
            (new_title, doc_id)
        )
        conn.commit()
        return {"status": "renamed", "doc_id": doc_id, "title": new_title}
    finally:
        conn.close()


@app.post("/api/documents/bulk-move")
def bulk_move_documents(data: dict):
    doc_ids = data.get("doc_ids") or []
    if not doc_ids:
        raise HTTPException(400, "Document ids required")

    try:
        normalized_ids = sorted({int(doc_id) for doc_id in doc_ids})
    except (TypeError, ValueError):
        raise HTTPException(400, "Document ids must be integers")

    target_folder = _validate_folder_name(data.get("target_folder", ""), "Target folder")
    target_root = _folder_root(target_folder)

    conn = get_conn()
    try:
        ph = ",".join("?" * len(normalized_ids))
        rows = conn.execute(
            f"SELECT * FROM documents WHERE id IN ({ph}) ORDER BY id",
            normalized_ids
        ).fetchall()
        if len(rows) != len(normalized_ids):
            raise HTTPException(404, "One or more documents were not found")

        moved = []
        for row in rows:
            current_root = _folder_root(row["project"])
            if current_root != target_root:
                raise HTTPException(
                    400,
                    f"Document {row['id']} cannot move outside root folder '{current_root}'"
                )

        for row in rows:
            new_pdf_path = _relocate_document_file(row, target_folder)
            conn.execute(
                "UPDATE documents SET project = ?, pdf_path = COALESCE(?, pdf_path) WHERE id = ?",
                (target_folder, new_pdf_path, row["id"])
            )
            moved.append({
                "id": row["id"],
                "title": row["title"],
                "old_folder": row["project"],
                "new_folder": target_folder,
            })

        conn.commit()
        return {"status": "moved", "count": len(moved), "documents": moved}
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/pdf")
def download_file(doc_id: int):
    conn = get_conn()
    try:
        doc = conn.execute(
            "SELECT id, project, title, filename, pdf_path FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        file_path = _resolve_document_source_path(conn, doc)
        if not file_path:
            raise HTTPException(404, "Source file not found on disk")
        conn.commit()
        download_name = doc["filename"] or f"{doc['title']}"
        return FileResponse(
            file_path,
            filename=download_name,
        )
    finally:
        conn.close()


@app.post("/api/documents/bulk-delete")
def bulk_delete_documents(data: dict):
    doc_ids = data.get("doc_ids") or []
    if not doc_ids:
        raise HTTPException(400, "Document ids required")

    try:
        normalized_ids = sorted({int(doc_id) for doc_id in doc_ids})
    except (TypeError, ValueError):
        raise HTTPException(400, "Document ids must be integers")

    conn = get_conn()
    try:
        ph = ",".join("?" * len(normalized_ids))
        rows = conn.execute(
            f"SELECT id, title FROM documents WHERE id IN ({ph}) ORDER BY id",
            normalized_ids
        ).fetchall()
        if len(rows) != len(normalized_ids):
            raise HTTPException(404, "One or more documents were not found")

        for row in rows:
            _delete_document_data(conn, row["id"])
        conn.commit()
        return {
            "status": "deleted",
            "count": len(rows),
            "documents": [dict(row) for row in rows],
        }
    finally:
        conn.close()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    conn = get_conn()
    try:
        doc = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")

        _delete_document_data(conn, doc_id)
        conn.commit()
        return {"status": "deleted", "doc_id": doc_id}
    finally:
        conn.close()


@app.get("/api/chats/{thread_id}")
def get_chat_thread(thread_id: str):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?",
            (thread_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Chat thread not found")
        result = dict(row)
        result["folder"] = result["project"]
        return result
    finally:
        conn.close()


@app.get("/api/chats/{thread_id}/messages")
def get_chat_messages(thread_id: str):
    conn = get_conn()
    try:
        thread = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?",
            (thread_id,)
        ).fetchone()
        if not thread:
            raise HTTPException(404, "Chat thread not found")
        return {
            "thread": {**dict(thread), "folder": thread["project"]},
            "messages": _fetch_chat_messages(conn, thread_id),
        }
    finally:
        conn.close()


@app.post("/api/chats/{thread_id}/messages")
def create_chat_message(thread_id: str, data: dict):
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "Message content required")

    conn = get_conn()
    try:
        thread = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?",
            (thread_id,)
        ).fetchone()
        if not thread:
            raise HTTPException(404, "Chat thread not found")

        _insert_chat_message(conn, thread_id, "user", content)

        existing_messages = _fetch_chat_messages(conn, thread_id)
        history = existing_messages[:-1]
        response = _generate_mcp_chat_answer(thread["project"], content, history)
        _insert_chat_message(conn, thread_id, "assistant", response.answer, sources=response.sources)

        if len(existing_messages) == 1 and thread["title"] == "New chat":
            conn.execute(
                "UPDATE chat_threads SET title = ? WHERE id = ?",
                (_thread_title_from_message(content), thread_id)
            )
            _touch_thread(conn, thread_id)

        conn.commit()
        return {
            "thread": {
                **dict(conn.execute(
                "SELECT * FROM chat_threads WHERE id = ?",
                (thread_id,)
            ).fetchone()),
                "folder": thread["project"],
            },
            "messages": _fetch_chat_messages(conn, thread_id),
            "retrieval": {
                "query": content,
                "source_count": len(response.sources),
            },
        }
    finally:
        conn.close()


@app.post("/api/chats/{thread_id}/review-document")
def review_chat_document(
    thread_id: str,
    file: UploadFile = File(...),
    content: str = Form(""),
):
    attachment = _prepare_chat_review_attachment(file)
    question = (content or "").strip() or _default_attachment_review_question(attachment["filename"])

    conn = get_conn()
    try:
        thread = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?",
            (thread_id,)
        ).fetchone()
        if not thread:
            raise HTTPException(404, "Chat thread not found")

        _insert_chat_message(
            conn,
            thread_id,
            "user",
            question,
            sources=attachment["message_sources"],
        )

        existing_messages = _fetch_chat_messages(conn, thread_id)
        history = existing_messages[:-1]
        response = _generate_mcp_chat_answer(
            thread["project"],
            question,
            history,
            attachment=attachment,
        )
        sources = [
            *attachment["assistant_sources"],
            *response.sources,
        ]
        _insert_chat_message(conn, thread_id, "assistant", response.answer, sources=sources)

        if len(existing_messages) == 1 and thread["title"] == "New chat":
            conn.execute(
                "UPDATE chat_threads SET title = ? WHERE id = ?",
                (_thread_title_from_message(question), thread_id)
            )
            _touch_thread(conn, thread_id)

        conn.commit()
        return {
            "thread": {
                **dict(conn.execute(
                    "SELECT * FROM chat_threads WHERE id = ?",
                    (thread_id,)
                ).fetchone()),
                "folder": thread["project"],
            },
            "messages": _fetch_chat_messages(conn, thread_id),
            "retrieval": {
                "query": question,
                "source_count": len(response.sources),
                "attachment_pages": attachment["page_count"],
            },
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Viewer routes
# ---------------------------------------------------------------------------

@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    conn = get_conn()
    try:
        d = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not d:
            raise HTTPException(404, "Document not found")
        meta = json.loads(d["metadata"]) if d["metadata"] else {}
        result = {
            "id": d["id"], "project": d["project"], "folder": d["project"], "title": d["title"],
            "filename": d["filename"], "total_pages": d["total_pages"],
            "document_number": meta.get("document_number"),
            "revision": meta.get("revision"),
            "document_type": meta.get("document_type"),
            "summary": meta.get("summary"),
            "keywords": meta.get("keywords"),
        }
        si = parse_split_info(d["filename"])
        if si:
            result["split_info"] = si
        return result
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/toc")
def get_toc(doc_id: int, max_level: int = Query(4)):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT heading, level, page_start, page_end, breadcrumb "
            "FROM sections WHERE doc_id = ? AND level <= ? ORDER BY seq",
            (doc_id, max_level)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/pages/{page_num}")
def get_page(doc_id: int, page_num: int):
    conn = get_conn()
    try:
        d = conn.execute(
            "SELECT title, total_pages FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if not d:
            raise HTTPException(404, "Document not found")
        p = conn.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_num = ?",
            (doc_id, page_num)
        ).fetchone()
        if not p:
            raise HTTPException(404, f"Page {page_num} not found")
        return {
            "doc_title": d["title"], "total_pages": d["total_pages"],
            "page_num": p["page_num"], "content": p["content"],
            "breadcrumb": p["breadcrumb"], "page_type": p["page_type"],
        }
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/pages")
def get_pages(doc_id: int, start: int = Query(1), end: int = Query(10)):
    conn = get_conn()
    try:
        end = min(end, start + 9)
        rows = conn.execute(
            "SELECT page_num, content, breadcrumb, page_type "
            "FROM pages WHERE doc_id = ? AND page_num BETWEEN ? AND ? "
            "ORDER BY page_num", (doc_id, start, end)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/search")
def search_in_doc(doc_id: int, q: str = Query(...)):
    conn = get_conn()
    try:
        words = q.strip().split()
        clean = ' '.join(
            w for w in words
            if w.upper() not in ('AND', 'OR', 'NOT', 'NEAR')
        )
        if not clean:
            return []
        rows = conn.execute("""
            SELECT p.page_num, p.breadcrumb, p.page_type,
                   snippet(pages_fts, 0, '>>>', '<<<', '...', 40) as snippet
            FROM pages_fts
            JOIN pages p ON p.id = pages_fts.rowid
            WHERE pages_fts MATCH ? AND p.doc_id = ?
            ORDER BY rank LIMIT 20
        """, (clean, doc_id)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ingestion routes
# ---------------------------------------------------------------------------

SPLIT_MIN_PAGES = 20       # only attempt splitting on PDFs with >= this many pages
SPLIT_SCORE_THRESHOLD = 3.0
SPLIT_MIN_DOC_PAGES = 4


def _split_pdf(pdf_path, project_dir):
    """Split a large PDF into sub-documents. Returns list of Path objects.
    If the PDF is small or the splitter finds no boundaries, returns [original]."""
    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count < SPLIT_MIN_PAGES:
        return [Path(pdf_path)]

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocrrag_split_"))
    splitter = PDFSplitter(
        pdf_path=str(pdf_path),
        output_dir=str(tmp_dir),
        threshold=SPLIT_SCORE_THRESHOLD,
        min_doc_pages=SPLIT_MIN_DOC_PAGES,
    )
    report = splitter.run()

    segments = report.get("segments", [])
    if report.get("status") == "skipped" or len(segments) <= 1:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [Path(pdf_path)]

    # Move split parts into the project upload directory
    parts = []
    for seg in segments:
        part_pattern = f"*_part{seg['segment']:03d}_*"
        matches = list(tmp_dir.glob(part_pattern))
        if matches:
            dest = project_dir / matches[0].name
            shutil.move(str(matches[0]), str(dest))
            parts.append(dest)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return parts if parts else [Path(pdf_path)]


def _is_duplicate(conn, project, filename, file_path):
    """Check if this file has already been ingested (by filename or content hash)."""
    # Check by filename
    row = conn.execute(
        "SELECT id FROM documents WHERE project = ? AND filename = ?",
        (project, filename)
    ).fetchone()
    if row:
        return row["id"]

    # Check by file content hash (catches renamed duplicates)
    import hashlib
    try:
        h = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        row = conn.execute(
            "SELECT d.id FROM documents d "
            "WHERE d.project = ? AND d.pdf_path LIKE '%' || ? || '%'",
            (project, filename)
        ).fetchone()
    except Exception:
        pass
    return None


def _ingest_one(jid, file_path, project, filename, stage_prefix=""):
    """Ingest a single file (PDF, image, DOCX, XLSX)."""
    prefix = f"{stage_prefix}" if stage_prefix else ""
    _raise_if_cancel_requested(jid)

    # Duplicate check
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    dup_id = _is_duplicate(conn, project, filename, str(file_path))
    conn.close()
    if dup_id:
        tracker.update(jid, status="completed",
                       stage=f"{prefix}Skipped (duplicate of doc {dup_id})",
                       doc_id=dup_id)
        return None  # return None so _run_ingestion doesn't overwrite stage

    tracker.update(jid, status="running", stage=f"{prefix}Extracting text & headings")
    pages, sections = extract_file(str(file_path))
    _raise_if_cancel_requested(jid)

    if not pages:
        raise IngestionError(f"No content extracted from {filename}")

    doc_id = None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ')
        tracker.update(jid, stage=f"{prefix}Ingesting into database")
        doc_id = ingest_document(
            conn, pages, sections, project, title,
            filename=filename,
            pdf_path=str(Path(file_path).resolve()),
        )
        _raise_if_cancel_requested(jid)

        tracker.update(jid, stage=f"{prefix}Replaying corrections")
        replay_corrections(conn, doc_id, str(Path(file_path).resolve()))
        _raise_if_cancel_requested(jid)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            tracker.update(jid, stage=f"{prefix}Extracting metadata (LLM)")
            meta = extract_metadata_llm(pages, api_key=api_key)
            if meta:
                apply_metadata(conn, doc_id, meta)

        tracker.update(jid, stage=f"{prefix}Computing embeddings")
        compute_embeddings(conn, doc_id)
        _raise_if_cancel_requested(jid)
        return doc_id
    except Exception as exc:
        conn.rollback()
        if doc_id is not None:
            try:
                _delete_document_data(conn, doc_id)
                conn.commit()
            except Exception:
                conn.rollback()
        raise IngestionError(f"{filename}: {exc}") from exc
    finally:
        conn.close()


def _run_ingestion(jid, file_path, project, filename):
    """Run in thread pool. Splits large PDFs first, then ingests each part.

    Non-PDF files (images, DOC/DOCX, XLS/XLSX) are ingested directly — splitting
    only applies to PDFs.
    """
    try:
        project_dir = Path(UPLOADS_DIR) / project
        ext = Path(filename).suffix.lower()
        _raise_if_cancel_requested(jid)

        # Only PDFs go through the splitting pipeline
        if ext != '.pdf':
            tracker.update(jid, status="running", stage="Extracting content")
            doc_id = _ingest_one(jid, file_path, project, filename)
            if doc_id is not None:
                tracker.update(jid, status="completed", stage="Done", doc_id=doc_id)
            return

        tracker.update(jid, status="running", stage="Checking for document boundaries")

        parts = _split_pdf(file_path, project_dir)
        _raise_if_cancel_requested(jid)

        if len(parts) == 1 and str(parts[0]) == file_path:
            # Single document — ingest directly
            doc_id = _ingest_one(jid, file_path, project, filename)
            if doc_id is not None:
                tracker.update(jid, status="completed", stage="Done", doc_id=doc_id)
        else:
            # Multiple sub-documents from split
            tracker.update(jid, stage=f"Split into {len(parts)} documents")

            # Move original PDF to _originals/ immediately so it won't
            # appear in pending uploads or get accidentally re-ingested
            originals_dir = project_dir / "_originals"
            originals_dir.mkdir(exist_ok=True)
            original = Path(file_path)
            archived_original = originals_dir / original.name
            if original.exists():
                shutil.move(str(original), str(archived_original))

            created_doc_ids = []
            ingested_parts = 0
            skipped_parts = 0
            for i, part in enumerate(parts, 1):
                prefix = f"[{i}/{len(parts)}] "
                _raise_if_cancel_requested(jid)
                try:
                    doc_id = _ingest_one(jid, str(part), project, part.name, stage_prefix=prefix)
                except IngestionError as exc:
                    _rollback_split_ingestion(original, archived_original, parts, created_doc_ids)
                    raise IngestionError(f"{prefix}{exc}") from exc
                if doc_id is None:
                    skipped_parts += 1
                else:
                    ingested_parts += 1
                    created_doc_ids.append(doc_id)

            summary = [f"{ingested_parts} ingested"]
            if skipped_parts:
                summary.append(f"{skipped_parts} skipped")
            tracker.update(
                jid,
                status="completed",
                stage=f"Done — processed {len(parts)} split documents ({', '.join(summary)})",
            )

    except IngestionCancelled as exc:
        job = tracker.get(jid) or {}
        tracker.update(
            jid,
            status="cancelled",
            stage="Cancelled by user",
            error=str(exc),
        )
        if job.get("delete_after_cancel") and job.get("source_path"):
            try:
                _discard_upload_path(
                    job["source_path"],
                    reason=job.get("cancel_reason") or str(exc),
                    job_id=jid,
                    action="cancelled_and_deleted",
                )
            except HTTPException:
                pass
    except Exception as e:
        import traceback
        print(f"Ingestion failed for {filename}: {e}", flush=True)
        traceback.print_exc()
        tracker.update(jid, status="failed", error=str(e))


@app.post("/api/projects/{folder:path}/ingest")
@app.post("/api/folders/{folder:path}/ingest")
async def ingest_all(folder: str):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        pending_files = _pending_uploads(conn, folder)
    finally:
        conn.close()

    if not pending_files:
        return {"status": "nothing_to_ingest", "jobs": []}

    loop = asyncio.get_event_loop()
    jobs = []
    for pending in pending_files:
        file_path = _folder_disk_path(pending["folder"]) / pending["filename"]
        jid = tracker.create(
            pending["relative_path"],
            pending["folder"],
            source_path=_folder_relative_path(file_path, Path(UPLOADS_DIR)),
        )
        loop.run_in_executor(
            executor, _run_ingestion, jid, str(file_path), pending["folder"], pending["filename"]
        )
        jobs.append(jid)
    return {"status": "started", "jobs": jobs, "count": len(jobs)}


@app.post("/api/projects/{folder:path}/ingest/{filename:path}")
@app.post("/api/folders/{folder:path}/ingest/{filename:path}")
async def ingest_single(folder: str, filename: str):
    folder = _validate_folder_name(folder, "Folder name")
    relative_name = _normalize_folder_name(filename)
    file_path = _folder_disk_path(folder) / relative_name
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    target_folder = _folder_relative_path(file_path.parent, Path(UPLOADS_DIR))
    jid = tracker.create(
        relative_name,
        target_folder,
        source_path=_folder_relative_path(file_path, Path(UPLOADS_DIR)),
    )
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_ingestion, jid, str(file_path), target_folder, file_path.name
    )
    return {"status": "started", "job_id": jid}


@app.get("/api/ingestion/jobs")
def get_jobs():
    return tracker.all()


@app.post("/api/ingestion/jobs/{job_id}/cancel")
def cancel_ingestion_job(job_id: str, data: Optional[dict] = None):
    job = tracker.get(job_id)
    if not job:
        raise HTTPException(404, "Ingestion job not found")

    reason = ((data or {}).get("reason") or "Cancelled by user").strip()
    delete_file = bool((data or {}).get("delete_file"))

    if job["status"] == "completed":
        raise HTTPException(400, "Completed ingestion jobs cannot be cancelled")
    if job["status"] == "cancelled":
        return {"status": "cancelled", "job_id": job_id}

    upload_relative = _job_upload_relative_path(job)

    if job["status"] in {"queued", "failed"}:
        if delete_file and upload_relative:
            _discard_upload_path(
                upload_relative,
                reason=reason,
                job_id=job_id,
                action="cancelled_and_deleted",
            )
        tracker.update(
            job_id,
            status="cancelled",
            stage="Cancelled by user",
            error=reason,
            cancel_requested=0,
            delete_after_cancel=0,
            cancel_reason=reason,
        )
        return {
            "status": "cancelled",
            "job_id": job_id,
            "deleted_file": bool(delete_file and upload_relative),
        }

    tracker.update(
        job_id,
        cancel_requested=1,
        delete_after_cancel=1 if delete_file else 0,
        cancel_reason=reason,
        stage="Cancellation requested",
    )
    return {
        "status": "cancelling",
        "job_id": job_id,
        "delete_file": delete_file,
    }


# ---------------------------------------------------------------------------
# Quality routes
# ---------------------------------------------------------------------------

@app.get("/api/projects/{folder:path}/quality")
@app.get("/api/folders/{folder:path}/quality")
def folder_quality(folder: str):
    folder = _validate_folder_name(folder, "Folder name")
    conn = get_conn()
    try:
        doc_ids = [r["id"] for r in conn.execute(
            f"SELECT id FROM documents WHERE {_folder_scope_sql('project')}",
            _folder_scope_params(folder)
        ).fetchall()]
        if not doc_ids:
            return {"folder": folder, "project": folder, "flags": [], "corrections": []}

        ph = ",".join("?" * len(doc_ids))
        flags = conn.execute(f"""
            SELECT qf.*, d.title as doc_title
            FROM quality_flags qf JOIN documents d ON d.id = qf.doc_id
            WHERE qf.doc_id IN ({ph}) ORDER BY qf.created_at DESC
        """, doc_ids).fetchall()
        corrections = conn.execute(f"""
            SELECT c.*, d.title as doc_title
            FROM corrections c JOIN documents d ON d.id = c.doc_id
            WHERE c.doc_id IN ({ph}) ORDER BY c.created_at DESC LIMIT 100
        """, doc_ids).fetchall()
        return {
            "folder": folder,
            "project": folder,
            "flags": [dict(r) for r in flags],
            "corrections": [dict(r) for r in corrections],
        }
    finally:
        conn.close()


@app.patch("/api/quality/{flag_id}/resolve")
def resolve_flag(flag_id: int):
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE quality_flags SET resolved = 1 WHERE id = ?",
            (flag_id,)
        )
        conn.commit()
        return {"status": "resolved", "id": flag_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Static files + entry point
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)))


def _start_mcp_server(db_path, port):
    """Start the MCP server in a background thread."""
    import mcp_server
    mcp_server.DB_PATH = db_path
    mcp_server.mcp.settings.port = port
    mcp_server.mcp.settings.host = "0.0.0.0"
    mcp_server.mcp.settings.transport_security.enable_dns_rebinding_protection = False
    mcp_server.mcp.run(transport="sse")


def main():
    global DB_PATH, MCP_SERVER_URL, UPLOADS_DIR

    p = argparse.ArgumentParser(description='Esteem Project Knowledge Server (Web GUI + MCP)')
    p.add_argument('--db', default='docs.db', help='SQLite database path')
    p.add_argument('--port', type=int, default=8201,
                   help='Web GUI port (default 8201)')
    p.add_argument('--mcp-port', type=int, default=8200,
                   help='MCP server port (default 8200)')
    p.add_argument('--uploads-dir', default='./uploads',
                   help='PDF uploads directory')
    args = p.parse_args()

    DB_PATH = args.db
    MCP_SERVER_URL = f"http://127.0.0.1:{args.mcp_port}/sse"
    UPLOADS_DIR = args.uploads_dir
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

    conn = init_db(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # Mark any jobs left running/queued from a previous process as failed
    stale = conn.execute(
        "UPDATE ingestion_jobs SET status = 'failed', "
        "error = 'Server restarted before completion' "
        "WHERE status IN ('running', 'queued')"
    ).rowcount
    conn.commit()
    if stale:
        print(f"  Cleaned {stale} stale job(s) from previous run")
    conn.close()

    # Start MCP server in background thread
    mcp_thread = threading.Thread(
        target=_start_mcp_server,
        args=(DB_PATH, args.mcp_port),
        daemon=True,
    )
    mcp_thread.start()

    print("Esteem Project Knowledge")
    print(f"  Database:  {DB_PATH} ({total} documents)")
    print(f"  Uploads:   {UPLOADS_DIR}")
    print(f"  Web GUI:   http://0.0.0.0:{args.port}")
    print(f"  MCP:       http://0.0.0.0:{args.mcp_port}/sse")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
