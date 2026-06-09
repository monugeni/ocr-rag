"""Check-run lifecycle routes (create / status). Start + results come later."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .. import agent_seam, auth, events, jobs, store
from ..models import RunCreate

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("")
def create_run(payload: RunCreate, request: Request):
    user = auth.require_user(request)
    run = store.create_run(payload.model_dump(), created_by=user["id"])
    auth.record_audit(
        "run_created",
        user_id=user["id"],
        run_id=run["id"],
        payload={"project_number": run["project_number"]},
    )
    return run


@router.get("")
def list_runs(request: Request, q: str | None = None, limit: int = 100):
    auth.require_user(request)
    return store.list_run_cards(q=q, limit=limit)


@router.get("/{run_id}")
def get_run(run_id: str, request: Request):
    auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.delete("/{run_id}")
def delete_run(run_id: str, request: Request):
    user = auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] in ("running", "queued"):
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete a run while it is {run['status']}",
        )
    from .. import config

    artifacts = store.delete_run(run_id)
    for p in artifacts["disk_paths"]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
    for d in (Path(config.ANNOTATED_DIR) / run_id, Path(config.EXPORTS_DIR) / run_id):
        shutil.rmtree(d, ignore_errors=True)
    auth.record_audit(
        "run_deleted",
        user_id=user["id"],
        run_id=run_id,
        payload={"project_number": run.get("project_number")},
    )
    return {"deleted": run_id}


@router.post("/{run_id}/start")
def start_run(run_id: str, request: Request):
    user = auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] in ("running", "queued"):
        raise HTTPException(status_code=409, detail=f"run already {run['status']}")

    submitted = [u for u in run["uploads"] if u["role"] == "submitted"]
    if not submitted:
        raise HTTPException(status_code=400, detail="no submitted document uploaded")
    if not any(u["ingest_status"] == "done" and u["doc_id"] for u in submitted):
        raise HTTPException(
            status_code=409,
            detail="submitted document not finished ingesting yet",
        )

    store.update_run(run_id, status="queued", stage="Queued", error=None)
    jobs.submit_run(agent_seam.run_check_for_run, run_id, user["id"])
    return {"run_id": run_id, "status": "queued"}


@router.get("/{run_id}/results")
def run_results(run_id: str, request: Request):
    auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run": run,
        "findings": store.get_findings(run_id),
        "comment_results": store.get_comment_results(run_id),
    }


@router.get("/{run_id}/stream")
async def run_stream(run_id: str, request: Request):
    auth.require_user(request)
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")

    async def gen():
        last = -1
        while True:
            if await request.is_disconnected():
                break
            for ev in events.get_since(run_id, last):
                last = ev["seq"]
                yield {"event": "progress", "data": json.dumps(ev)}
            run = store.get_run(run_id)
            if run and run["status"] in ("done", "failed"):
                # drain any final events then close
                for ev in events.get_since(run_id, last):
                    last = ev["seq"]
                    yield {"event": "progress", "data": json.dumps(ev)}
                yield {"event": "end", "data": json.dumps({"status": run["status"]})}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(gen())


@router.get("/{run_id}/trace")
def run_trace(run_id: str, request: Request):
    """Debug trace for a run: model reasoning, raw vs confirmed findings, what
    self-verification pruned (and why), and limits hit."""
    auth.require_user(request)
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    from .. import config

    path = Path(config.ANNOTATED_DIR) / run_id / "trace.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@router.get("/{run_id}/annotated.pdf")
def annotated_pdf(run_id: str, request: Request):
    auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    from .. import config

    out_dir = Path(config.ANNOTATED_DIR) / run_id
    pdfs = sorted(out_dir.glob("*.annotated.pdf")) if out_dir.is_dir() else []
    if not pdfs:
        raise HTTPException(status_code=404, detail="no annotated PDF for this run")
    # Serve inline so the in-page <iframe> viewer displays the PDF. Passing
    # filename= alone makes FileResponse send Content-Disposition: attachment,
    # which forced the browser to download the PDF every time the results
    # re-rendered (e.g. each time the user returned to the Check page).
    return FileResponse(
        str(pdfs[0]),
        media_type="application/pdf",
        filename=pdfs[0].name,
        content_disposition_type="inline",
    )
