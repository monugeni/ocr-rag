"""Retrieve reference context from a running company ocr-rag MCP server.

Used for ``reference_mode = existing | both``: instead of (or in addition to)
freshly-uploaded references, pull the governing requirements from the company
knowledge base via ``ranked_search`` over the chosen project/folder.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


def _parse_result(result: Any) -> dict:
    """Coerce an MCP CallToolResult into a dict (structuredContent or JSON text)."""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        # FastMCP sometimes wraps the payload under "result".
        return sc.get("result", sc) if "result" in sc else sc
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                data = json.loads(text)
                return data.get("result", data) if isinstance(data, dict) else {"items": data}
            except Exception:  # noqa: BLE001
                return {"text": text}
    return {}


async def _ranked_search(url: str, project: str, query: str, max_results: int) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=15, sse_read_timeout=60) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await session.call_tool(
                "ranked_search",
                {"project": project, "query": query, "max_results": max_results},
            )
            return _parse_result(res)


def fetch_reference_context(
    url: str,
    project: str,
    queries: list[str],
    *,
    max_results: int = 6,
    max_chars: int = 40_000,
) -> str:
    """Run each query against the company MCP and concatenate deduped chunks."""
    async def run() -> list[dict]:
        out = []
        for q in queries:
            if not q:
                continue
            try:
                out.append(await _ranked_search(url, project, q, max_results))
            except Exception:  # noqa: BLE001 — one bad query shouldn't sink the rest
                out.append({})
        return out

    payloads = asyncio.run(run())

    seen: set = set()
    blocks: list[str] = []
    total = 0
    for data in payloads:
        for item in data.get("results") or []:
            text = item.get("text") or item.get("snippet") or ""
            key = (item.get("doc_id"), item.get("page_num"), text[:60])
            if not text or key in seen:
                continue
            seen.add(key)
            header = (
                f"[{item.get('doc_title')} p{item.get('page_num')}"
                f" — {item.get('breadcrumb') or ''}]"
            )
            chunk = f"{header}\n{text}"
            if total + len(chunk) > max_chars:
                return "\n\n".join(blocks)
            blocks.append(chunk)
            total += len(chunk)
    return "\n\n".join(blocks)
