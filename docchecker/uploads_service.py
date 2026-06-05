"""Shared helper to persist an uploaded file and queue its ingestion.

Used by both the JSON upload API and the server-rendered new-check form.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config, jobs, store

VALID_ROLES = {"submitted", "reference", "prior_commented"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    return _SAFE.sub("_", Path(name).name).strip("_") or "file"


def save_and_queue(run: dict, role: str, filename: str, data: bytes, content_type: str | None) -> int:
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role: {role}")
    dest_dir = Path(config.UPLOADS_DIR) / run["project_number"] / run["id"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(filename or "upload")
    dest = dest_dir / f"{role}__{safe}"
    dest.write_bytes(data)

    upload_id = store.add_upload(run["id"], role, filename or safe, str(dest), content_type)
    jobs.submit_ingestion(upload_id)
    return upload_id
