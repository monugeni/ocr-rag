"""Data-access helpers for check_runs / uploads / findings / comments.

Thin functions over checker.db so routers and the agent seam stay readable.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from .db import get_conn


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def list_templates(include_archived: bool = False) -> list[dict]:
    conn = get_conn()
    try:
        q = "SELECT * FROM templates"
        if not include_archived:
            q += " WHERE archived = 0"
        q += " ORDER BY name"
        return [dict(r) for r in conn.execute(q).fetchall()]
    finally:
        conn.close()


def get_template(template_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM templates WHERE id = ?", (template_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_template(data: dict, created_by: int | None) -> dict:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO templates
                 (name, description, instructions, default_doc_type, severity_scheme,
                  categories, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                data["name"],
                data.get("description"),
                data["instructions"],
                data.get("default_doc_type"),
                data.get("severity_scheme"),
                json.dumps(data["categories"]) if data.get("categories") else None,
                created_by,
            ),
        )
        conn.commit()
        return get_template(cur.lastrowid)
    finally:
        conn.close()


def update_template(template_id: int, fields: dict) -> Optional[dict]:
    allowed = {
        "name", "description", "instructions", "default_doc_type",
        "severity_scheme", "categories", "archived",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if "categories" in sets and sets["categories"] is not None:
        sets["categories"] = json.dumps(sets["categories"])
    if not sets:
        return get_template(template_id)
    sets["updated_at"] = "__now__"
    cols = []
    vals = []
    for k, v in sets.items():
        if v == "__now__":
            cols.append(f"{k} = CURRENT_TIMESTAMP")
        else:
            cols.append(f"{k} = ?")
            vals.append(v)
    conn = get_conn()
    try:
        conn.execute(
            f"UPDATE templates SET {', '.join(cols)} WHERE id = ?",
            (*vals, template_id),
        )
        conn.commit()
    finally:
        conn.close()
    return get_template(template_id)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def create_run(data: dict, created_by: int) -> dict:
    run_id = uuid.uuid4().hex
    ocrrag_project = f"{data['project_number']}/{run_id}"
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO check_runs
                 (id, status, project_number, document_type, originator, metadata,
                  template_id, guiding_prompt, is_revision, prior_run_id,
                  reference_mode, reference_project, ocrrag_project, created_by)
               VALUES (?, 'created', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                data["project_number"],
                data.get("document_type"),
                data.get("originator"),
                json.dumps(data.get("metadata") or {}),
                data.get("template_id"),
                data.get("guiding_prompt"),
                1 if data.get("is_revision") else 0,
                data.get("prior_run_id"),
                data.get("reference_mode", "fresh"),
                # Additive references stored as a JSON list in reference_project;
                # falls back to the legacy single value for back-compat.
                (json.dumps(data["reference_projects"])
                 if data.get("reference_projects")
                 else data.get("reference_project")),
                ocrrag_project,
                created_by,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_run(run_id)


def get_run(run_id: str) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM check_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["uploads"] = [
            dict(u)
            for u in conn.execute(
                "SELECT * FROM uploads WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        ]
        return run
    finally:
        conn.close()


def list_runs(
    *,
    project_number: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses, params = [], []
    if project_number:
        clauses.append("project_number = ?")
        params.append(project_number)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM check_runs{where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn = get_conn()
    try:
        conn.execute(
            f"UPDATE check_runs SET {cols} WHERE id = ?",
            (*fields.values(), run_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
def add_upload(run_id: str, role: str, filename: str, disk_path: str, mime: str | None) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO uploads (run_id, role, filename, disk_path, mime, ingest_status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (run_id, role, filename, disk_path, mime),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_upload(upload_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_upload_ingest(
    upload_id: int,
    *,
    status: str,
    doc_id: int | None = None,
    page_count: int | None = None,
    error: str | None = None,
    job_id: str | None = None,
) -> None:
    fields: dict[str, Any] = {"ingest_status": status}
    if doc_id is not None:
        fields["doc_id"] = doc_id
    if page_count is not None:
        fields["page_count"] = page_count
    if error is not None:
        fields["ingest_error"] = error
    if job_id is not None:
        fields["ingest_job_id"] = job_id
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn = get_conn()
    try:
        conn.execute(
            f"UPDATE uploads SET {cols} WHERE id = ?", (*fields.values(), upload_id)
        )
        conn.commit()
    finally:
        conn.close()


def add_finding(run_id: str, f: dict) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO findings
                 (run_id, doc_id, page_num, bbox, anchor_text, annotation_xref,
                  severity, category, title, detail, vendor_comment, citation, confidence, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (
                run_id,
                f.get("doc_id"),
                f.get("page_num"),
                json.dumps(f["bbox"]) if f.get("bbox") is not None else None,
                f.get("anchor_text"),
                f.get("annotation_xref"),
                f.get("severity"),
                f.get("category"),
                f.get("title"),
                f.get("detail"),
                f.get("vendor_comment"),
                json.dumps(f.get("citation") or {}),
                f.get("confidence", "medium"),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_findings(run_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["bbox"] = json.loads(d["bbox"]) if d.get("bbox") else None
            d["citation"] = json.loads(d["citation"]) if d.get("citation") else {}
            out.append(d)
        return out
    finally:
        conn.close()


def set_finding_status(finding_id: int, status: str) -> None:
    conn = get_conn()
    try:
        conn.execute("UPDATE findings SET status = ? WHERE id = ?", (status, finding_id))
        conn.commit()
    finally:
        conn.close()


def add_comment_result(run_id: str, c: dict) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO comment_incorporation
                 (run_id, prior_comment_ref, prior_comment_text, prior_page,
                  verdict, evidence, detail, annotation_xref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                c.get("prior_comment_ref"),
                c.get("prior_comment_text"),
                c.get("prior_page"),
                c.get("verdict"),
                json.dumps(c.get("evidence") or {}),
                c.get("detail"),
                c.get("annotation_xref"),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_comment_results(run_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM comment_incorporation WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"]) if d.get("evidence") else {}
            out.append(d)
        return out
    finally:
        conn.close()


def get_run_uploads(run_id: str, role: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        if role:
            rows = conn.execute(
                "SELECT * FROM uploads WHERE run_id = ? AND role = ? ORDER BY id",
                (run_id, role),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM uploads WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
