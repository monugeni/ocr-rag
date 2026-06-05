"""Background job execution.

A single-writer executor for ingestion (serialises writes to docs.db, matching
ocr-rag/web.py's ThreadPoolExecutor(max_workers=1) discipline). A separate small
pool runs check-run agents.
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from . import store
from .ingestion import ingest_upload

log = logging.getLogger("checker.jobs")

# Single writer to docs.db.
_ingest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
# Agent runs (network/CPU light locally); a couple in parallel is fine.
_run_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agentrun")


def submit_ingestion(upload_id: int) -> str:
    """Queue ingestion of an upload. Returns a job id."""
    job_id = uuid.uuid4().hex
    store.set_upload_ingest(upload_id, status="pending", job_id=job_id)

    def _task() -> None:
        try:
            ingest_upload(upload_id)
        except Exception:  # noqa: BLE001
            log.exception("ingestion failed for upload %s", upload_id)

    _ingest_executor.submit(_task)
    return job_id


def submit_run(fn, *args, **kwargs) -> Future:
    """Queue a check-run agent task (used in M4+)."""
    return _run_executor.submit(fn, *args, **kwargs)


def shutdown() -> None:
    _ingest_executor.shutdown(wait=False, cancel_futures=True)
    _run_executor.shutdown(wait=False, cancel_futures=True)
