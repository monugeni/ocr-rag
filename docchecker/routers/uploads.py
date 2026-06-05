"""Upload routes: attach a file to a run (by role) and queue ingestion."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .. import auth, store, uploads_service

router = APIRouter(prefix="/api/runs", tags=["uploads"])


@router.post("/{run_id}/uploads")
async def upload_file(
    run_id: str,
    request: Request,
    role: str = Form(...),
    file: UploadFile = File(...),
):
    user = auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if role not in uploads_service.VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"invalid role: {role}")

    data = await file.read()
    upload_id = uploads_service.save_and_queue(
        run, role, file.filename or "upload", data, file.content_type
    )
    auth.record_audit(
        "upload_added",
        user_id=user["id"],
        run_id=run_id,
        payload={"role": role, "filename": file.filename, "upload_id": upload_id},
    )
    return {"upload_id": upload_id, "role": role, "status": "pending"}
