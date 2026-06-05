"""Reviewer actions on findings (accept / dismiss)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import auth, store

router = APIRouter(prefix="/api/findings", tags=["findings"])

VALID = {"open", "accepted", "dismissed"}


@router.post("/{finding_id}/status")
def set_status(finding_id: int, request: Request, status: str):
    user = auth.require_user(request)
    if status not in VALID:
        raise HTTPException(status_code=400, detail=f"invalid status: {status}")
    store.set_finding_status(finding_id, status)
    auth.record_audit(
        "finding_status_changed",
        user_id=user["id"],
        payload={"finding_id": finding_id, "status": status},
    )
    return {"finding_id": finding_id, "status": status}
