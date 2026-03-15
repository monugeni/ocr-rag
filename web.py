#!/usr/bin/env python3
"""
OCR-RAG Web GUI
================
Project & document management with PDF ingestion.

  python web.py --db docs.db --port 8201

Routes:
  /                                     Web GUI
  /api/projects                         List / create projects
  /api/projects/{name}/documents        List docs + pending uploads
  /api/projects/{name}/upload           Upload PDFs
  /api/projects/{name}/ingest           Ingest all pending PDFs
  /api/documents/{id}                   Document info
  /api/documents/{id}/toc               Section tree
  /api/documents/{id}/pages/{num}       Single page
  /api/documents/{id}/pages?start=&end= Page range
  /api/documents/{id}/search?q=         Search within doc
  /api/ingestion/jobs                   Active ingestion jobs
  /api/projects/{name}/quality          Quality flags + corrections
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="OCR-RAG Manager", docs_url="/api/docs")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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

    def create(self, filename, project):
        jid = str(uuid4())[:8]
        conn = self._conn()
        conn.execute(
            "INSERT INTO ingestion_jobs (id, filename, project, status, stage) "
            "VALUES (?, ?, ?, 'queued', '')",
            (jid, filename, project)
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
        # Clean completed/failed jobs older than 1 hour
        conn.execute(
            "DELETE FROM ingestion_jobs WHERE status IN ('completed', 'failed') "
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


tracker = IngestionTracker()


# ---------------------------------------------------------------------------
# Project routes
# ---------------------------------------------------------------------------

@app.get("/api/projects")
def list_projects():
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT project, COUNT(*) as docs,
                   COALESCE(SUM(total_pages), 0) as pages
            FROM documents GROUP BY project ORDER BY project
        """).fetchall()
        projects = [dict(r) for r in rows]

        # Include upload-only projects (dirs with PDFs but no ingested docs)
        # and empty projects (created but no files yet)
        uploads = Path(UPLOADS_DIR)
        if uploads.exists():
            existing = {p["project"] for p in projects}
            for d in sorted(uploads.iterdir()):
                if d.is_dir() and d.name not in existing and not d.name.startswith('_'):
                    n = sum(1 for f in d.iterdir()
                            if f.is_file() and f.suffix.lower() in INGESTABLE_EXTENSIONS
                            and not f.name.startswith('.'))
                    projects.append({
                        "project": d.name, "docs": 0,
                        "pages": 0, "pending": n,
                    })
        return projects
    finally:
        conn.close()


@app.post("/api/projects")
def create_project(data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Project name required")
    if re.search(r'[/\\<>:"|?*]', name):
        raise HTTPException(400, "Invalid characters in project name")
    project_dir = Path(UPLOADS_DIR) / name
    project_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "project": name}


@app.patch("/api/projects/{project}")
def rename_project(project: str, data: dict):
    new_name = data.get("name", "").strip()
    if not new_name:
        raise HTTPException(400, "New name required")
    if re.search(r'[/\\<>:"|?*]', new_name):
        raise HTTPException(400, "Invalid characters in project name")
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE documents SET project = ? WHERE project = ?",
            (new_name, project)
        )
        conn.execute(
            "UPDATE ingestion_jobs SET project = ? WHERE project = ?",
            (new_name, project)
        )
        conn.commit()

        # Rename uploads directory
        old_dir = Path(UPLOADS_DIR) / project
        new_dir = Path(UPLOADS_DIR) / new_name
        if old_dir.exists() and not new_dir.exists():
            old_dir.rename(new_dir)

        return {"status": "renamed", "old_name": project, "new_name": new_name}
    finally:
        conn.close()


@app.delete("/api/projects/{project}")
def delete_project(project: str):
    conn = get_conn()
    try:
        doc_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM documents WHERE project = ?", (project,)
        ).fetchall()]
        for did in doc_ids:
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM sections WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM corrections WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM cross_references WHERE doc_id = ?", (did,))
            conn.execute("DELETE FROM quality_flags WHERE doc_id = ?", (did,))
        conn.execute("DELETE FROM documents WHERE project = ?", (project,))
        conn.commit()

        project_dir = Path(UPLOADS_DIR) / project
        if project_dir.exists():
            shutil.rmtree(project_dir)
        return {"status": "deleted", "project": project}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Document routes
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project}/documents")
def list_documents(project: str):
    conn = get_conn()
    try:
        docs = conn.execute(
            "SELECT * FROM documents WHERE project = ? ORDER BY id",
            (project,)
        ).fetchall()

        results = []
        for d in docs:
            meta = json.loads(d["metadata"]) if d["metadata"] else {}
            secs = conn.execute(
                "SELECT COUNT(*) FROM sections WHERE doc_id = ?", (d["id"],)
            ).fetchone()[0]
            entry = {
                "id": d["id"], "title": d["title"],
                "filename": d["filename"],
                "total_pages": d["total_pages"], "sections": secs,
                "document_type": meta.get("document_type"),
            }
            si = parse_split_info(d["filename"])
            if si:
                entry["split_info"] = si
            results.append(entry)

        # Pending uploads (on disk but not ingested)
        project_dir = Path(UPLOADS_DIR) / project
        pending = []
        if project_dir.exists():
            ingested = {d["filename"] for d in results}
            for f in sorted(project_dir.iterdir()):
                if f.is_dir() or f.name.startswith('.') or f.name.startswith('_'):
                    continue
                if f.suffix.lower() not in INGESTABLE_EXTENSIONS:
                    continue
                if f.name not in ingested:
                    pending.append({
                        "filename": f.name,
                        "size_mb": round(f.stat().st_size / 1048576, 1),
                    })
        return {"documents": results, "pending": pending}
    finally:
        conn.close()


@app.post("/api/projects/{project}/upload")
async def upload_files(project: str, files: list[UploadFile] = File(...)):
    project_dir = Path(UPLOADS_DIR) / project
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        content = await f.read()

        if is_archive(f.filename):
            # Save archive to temp, extract supported files, delete archive
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=ext))
            tmp.write_bytes(content)
            try:
                names = extract_archive(str(tmp), project_dir)
                uploaded.extend(names)
            finally:
                tmp.unlink(missing_ok=True)
        elif ext in INGESTABLE_EXTENSIONS:
            dest = project_dir / f.filename
            dest.write_bytes(content)
            uploaded.append(f.filename)
        # else: silently skip unsupported types

    return {"uploaded": uploaded, "count": len(uploaded)}


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


@app.get("/api/documents/{doc_id}/pdf")
def download_file(doc_id: int):
    conn = get_conn()
    try:
        doc = conn.execute(
            "SELECT title, filename, pdf_path FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        file_path = doc["pdf_path"]
        if not file_path or not Path(file_path).exists():
            raise HTTPException(404, "Source file not found on disk")
        download_name = doc["filename"] or f"{doc['title']}"
        return FileResponse(
            file_path,
            filename=download_name,
        )
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

        conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM corrections WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM cross_references WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM quality_flags WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return {"status": "deleted", "doc_id": doc_id}
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
            "id": d["id"], "project": d["project"], "title": d["title"],
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

    if not pages:
        tracker.update(jid, status="failed", error=f"No content extracted from {filename}")
        return None

    tracker.update(jid, stage=f"{prefix}Ingesting into database")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    title = Path(filename).stem.replace('_', ' ').replace('-', ' ')
    doc_id = ingest_document(
        conn, pages, sections, project, title,
        filename=filename,
        pdf_path=str(Path(file_path).resolve()),
    )

    tracker.update(jid, stage=f"{prefix}Replaying corrections")
    replay_corrections(conn, doc_id, str(Path(file_path).resolve()))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        tracker.update(jid, stage=f"{prefix}Extracting metadata (LLM)")
        meta = extract_metadata_llm(pages, api_key=api_key)
        if meta:
            apply_metadata(conn, doc_id, meta)

    tracker.update(jid, stage=f"{prefix}Computing embeddings")
    compute_embeddings(conn, doc_id)

    conn.close()
    return doc_id


def _run_ingestion(jid, file_path, project, filename):
    """Run in thread pool. Splits large PDFs first, then ingests each part.

    Non-PDF files (images, DOCX, XLSX) are ingested directly — splitting
    only applies to PDFs.
    """
    try:
        project_dir = Path(UPLOADS_DIR) / project
        ext = Path(filename).suffix.lower()

        # Only PDFs go through the splitting pipeline
        if ext != '.pdf':
            tracker.update(jid, status="running", stage="Extracting content")
            doc_id = _ingest_one(jid, file_path, project, filename)
            if doc_id is not None:
                tracker.update(jid, status="completed", stage="Done", doc_id=doc_id)
            return

        tracker.update(jid, status="running", stage="Checking for document boundaries")

        parts = _split_pdf(file_path, project_dir)

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
            if original.exists():
                shutil.move(str(original), str(originals_dir / original.name))

            for i, part in enumerate(parts, 1):
                prefix = f"[{i}/{len(parts)}] "
                doc_id = _ingest_one(jid, str(part), project, part.name, stage_prefix=prefix)

            tracker.update(jid, status="completed",
                           stage=f"Done — split into {len(parts)} documents")

    except Exception as e:
        tracker.update(jid, status="failed", error=str(e))


@app.post("/api/projects/{project}/ingest")
async def ingest_all(project: str):
    project_dir = Path(UPLOADS_DIR) / project
    if not project_dir.exists():
        raise HTTPException(404, "No uploads for this project")

    conn = get_conn()
    try:
        ingested = {r["filename"] for r in conn.execute(
            "SELECT filename FROM documents WHERE project = ?", (project,)
        ).fetchall()}
    finally:
        conn.close()

    pending_files = []
    for f in sorted(project_dir.iterdir()):
        if f.is_dir() or f.name.startswith('.') or f.name.startswith('_'):
            continue
        if f.suffix.lower() in INGESTABLE_EXTENSIONS and f.name not in ingested:
            pending_files.append(f)

    if not pending_files:
        return {"status": "nothing_to_ingest", "jobs": []}

    loop = asyncio.get_event_loop()
    jobs = []
    for f in pending_files:
        jid = tracker.create(f.name, project)
        loop.run_in_executor(
            executor, _run_ingestion, jid, str(f), project, f.name
        )
        jobs.append(jid)
    return {"status": "started", "jobs": jobs, "count": len(jobs)}


@app.post("/api/projects/{project}/ingest/{filename:path}")
async def ingest_single(project: str, filename: str):
    file_path = Path(UPLOADS_DIR) / project / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    jid = tracker.create(filename, project)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_ingestion, jid, str(file_path), project, filename
    )
    return {"status": "started", "job_id": jid}


@app.get("/api/ingestion/jobs")
def get_jobs():
    return tracker.all()


# ---------------------------------------------------------------------------
# Quality routes
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project}/quality")
def project_quality(project: str):
    conn = get_conn()
    try:
        doc_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM documents WHERE project = ?", (project,)
        ).fetchall()]
        if not doc_ids:
            return {"flags": [], "corrections": []}

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
    global DB_PATH, UPLOADS_DIR

    p = argparse.ArgumentParser(description='OCR-RAG Server (Web GUI + MCP)')
    p.add_argument('--db', default='docs.db', help='SQLite database path')
    p.add_argument('--port', type=int, default=8201,
                   help='Web GUI port (default 8201)')
    p.add_argument('--mcp-port', type=int, default=8200,
                   help='MCP server port (default 8200)')
    p.add_argument('--uploads-dir', default='./uploads',
                   help='PDF uploads directory')
    args = p.parse_args()

    DB_PATH = args.db
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

    print(f"OCR-RAG Server")
    print(f"  Database:  {DB_PATH} ({total} documents)")
    print(f"  Uploads:   {UPLOADS_DIR}")
    print(f"  Web GUI:   http://0.0.0.0:{args.port}")
    print(f"  MCP:       http://0.0.0.0:{args.mcp_port}/sse")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
