"""Ingestion: wrap the ocr-rag extractors and load documents into checker docs.db.

Every uploaded file (submitted / reference / prior_commented) is extracted to
``(pages, sections)`` via ocr-rag's ``file_extractors.extract_file`` and inserted
with ``ingest.ingest_document`` under the run's namespaced project
(``"{project_number}/{run_id}"``) so the ocr-rag MCP tools can search it.
"""
from __future__ import annotations

from pathlib import Path

from . import store
from .db import get_docs_conn


def ingest_upload(upload_id: int) -> int:
    """Extract + ingest one upload into docs.db. Returns the docs.db doc_id.

    Runs synchronously (called inside the ingestion executor). Updates the
    upload row's ingest_status as it progresses.
    """
    # Imported here so the heavy sibling deps load lazily / after path wiring.
    from file_extractors import extract_file
    from ingest import ingest_document

    up = store.get_upload(upload_id)
    if not up:
        raise ValueError(f"upload {upload_id} not found")
    run = store.get_run(up["run_id"])
    if not run:
        raise ValueError(f"run {up['run_id']} not found")

    project = run["ocrrag_project"]
    store.set_upload_ingest(upload_id, status="running")
    try:
        pages, sections = extract_file(up["disk_path"])
        if not pages:
            raise RuntimeError(f"No content extracted from {up['filename']}")

        title = Path(up["filename"]).stem.replace("_", " ").replace("-", " ")
        docs_conn = get_docs_conn()
        try:
            doc_id = ingest_document(
                docs_conn,
                pages,
                sections,
                project,
                title,
                filename=up["filename"],
                pdf_path=str(Path(up["disk_path"]).resolve()),
            )
        finally:
            docs_conn.close()

        store.set_upload_ingest(
            upload_id, status="done", doc_id=doc_id, page_count=len(pages)
        )
        return doc_id
    except Exception as exc:  # noqa: BLE001 — record and re-raise
        store.set_upload_ingest(upload_id, status="failed", error=str(exc))
        raise
