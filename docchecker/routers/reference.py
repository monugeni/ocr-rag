"""Reference-library endpoints: list the company ocr-rag folders for selection."""
from __future__ import annotations

import json

from fastapi import APIRouter, Request

from .. import auth, config, mcp_clients

router = APIRouter(prefix="/api", tags=["reference"])


def _items(result) -> list[dict]:
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict) and isinstance(sc.get("result"), list):
        return sc["result"]
    if isinstance(sc, list):
        return sc
    out = []
    for b in getattr(result, "content", []) or []:
        t = getattr(b, "text", None)
        if not t:
            continue
        try:
            data = json.loads(t)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, list):
            out.extend(data)
        elif isinstance(data, dict):
            out.append(data)
    return out


@router.get("/reference-folders")
async def reference_folders(request: Request):
    """Top-level folders in the company ocr-rag KB (for the reference picker)."""
    auth.require_user(request)
    if not config.COMPANY_MCP_URL:
        return {"folders": [], "configured": False}
    try:
        result = await mcp_clients.call_tool(config.COMPANY_MCP_URL, "list_folders", {})
    except Exception as exc:  # noqa: BLE001
        return {"folders": [], "configured": True, "error": str(exc)}

    folders = []
    for it in _items(result):
        if not isinstance(it, dict):
            continue
        name = it.get("project") or it.get("folder")
        if not name:
            continue
        folders.append(
            {
                "value": name,
                "label": it.get("display_name") or name,
                "documents": it.get("documents"),
                "pages": it.get("total_pages"),
            }
        )
    folders.sort(key=lambda f: f["label"].lower())
    return {"folders": folders, "configured": True}
