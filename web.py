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
    extract_metadata_llm, apply_metadata,
)
from extractor import extract_pdf


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
                if d.is_dir() and d.name not in existing:
                    n = len(list(d.glob("*.pdf")))
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
            results.append({
                "id": d["id"], "title": d["title"],
                "filename": d["filename"],
                "total_pages": d["total_pages"], "sections": secs,
                "document_type": meta.get("document_type"),
            })

        # Pending uploads (on disk but not ingested)
        project_dir = Path(UPLOADS_DIR) / project
        pending = []
        if project_dir.exists():
            ingested = {d["filename"] for d in results}
            for pdf in sorted(project_dir.glob("*.pdf")):
                if pdf.name not in ingested:
                    pending.append({
                        "filename": pdf.name,
                        "size_mb": round(pdf.stat().st_size / 1048576, 1),
                    })
        return {"documents": results, "pending": pending}
    finally:
        conn.close()


@app.post("/api/projects/{project}/upload")
async def upload_pdfs(project: str, files: list[UploadFile] = File(...)):
    project_dir = Path(UPLOADS_DIR) / project
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = project_dir / f.filename
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        uploaded.append(f.filename)
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
def download_pdf(doc_id: int):
    conn = get_conn()
    try:
        doc = conn.execute(
            "SELECT title, filename, pdf_path FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        pdf_path = doc["pdf_path"]
        if not pdf_path or not Path(pdf_path).exists():
            raise HTTPException(404, "PDF file not found on disk")
        download_name = doc["filename"] or f"{doc['title']}.pdf"
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
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
        return {
            "id": d["id"], "project": d["project"], "title": d["title"],
            "filename": d["filename"], "total_pages": d["total_pages"],
            "document_number": meta.get("document_number"),
            "revision": meta.get("revision"),
            "document_type": meta.get("document_type"),
            "summary": meta.get("summary"),
            "keywords": meta.get("keywords"),
        }
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

def _run_ingestion(jid, pdf_path, project, filename):
    """Run in thread pool."""
    try:
        tracker.update(jid, status="running", stage="Extracting text & headings")
        pages, sections = extract_pdf(str(pdf_path))

        if not pages:
            tracker.update(jid, status="failed", error="No content extracted")
            return

        tracker.update(jid, stage="Ingesting into database")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")

        title = Path(filename).stem.replace('_', ' ').replace('-', ' ')
        doc_id = ingest_document(
            conn, pages, sections, project, title,
            filename=filename,
            pdf_path=str(Path(pdf_path).resolve()),
        )

        tracker.update(jid, stage="Replaying corrections")
        replay_corrections(conn, doc_id, str(Path(pdf_path).resolve()))

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            tracker.update(jid, stage="Extracting metadata (LLM)")
            meta = extract_metadata_llm(pages, api_key=api_key)
            if meta:
                apply_metadata(conn, doc_id, meta)

        conn.close()
        tracker.update(jid, status="completed", stage="Done", doc_id=doc_id)

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

    pdfs = [p for p in sorted(project_dir.glob("*.pdf"))
            if p.name not in ingested]
    if not pdfs:
        return {"status": "nothing_to_ingest", "jobs": []}

    loop = asyncio.get_event_loop()
    jobs = []
    for pdf in pdfs:
        jid = tracker.create(pdf.name, project)
        loop.run_in_executor(
            executor, _run_ingestion, jid, str(pdf), project, pdf.name
        )
        jobs.append(jid)
    return {"status": "started", "jobs": jobs, "count": len(jobs)}


@app.post("/api/projects/{project}/ingest/{filename}")
async def ingest_single(project: str, filename: str):
    pdf_path = Path(UPLOADS_DIR) / project / filename
    if not pdf_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    jid = tracker.create(filename, project)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_ingestion, jid, str(pdf_path), project, filename
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
