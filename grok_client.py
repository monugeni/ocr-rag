"""Minimal xAI (Grok) client — OpenAI-compatible chat completions over httpx.

This is the second LLM-provider path, run alongside the Anthropic one so the
folder chat and the document checker can be A/B-compared. Grok is fully
OpenAI-compatible (``tools``/``tool_calls``, ``response_format`` json_schema,
streaming), so we POST to ``{base}/chat/completions`` directly and keep the rest
of the app — MCP tools, source tracking, finding schema — unchanged.

Provider selection lives at the call sites (web.py for chat, docchecker config
for the checker); this module is just transport + small shape adapters.

Env:
- ``XAI_API_KEY`` (or ``GROK_API_KEY``) — xAI API key.
- ``XAI_BASE_URL`` — defaults to ``https://api.x.ai/v1``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

DEFAULT_GROK_MODEL = "grok-4.3"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

# grok-4.3 accepts reasoning_effort none|low|medium|high (defaults to "low" when
# omitted). The Anthropic side uses low|medium|high|xhigh|max; collapse the top
# three to "high" so an Anthropic-vs-Grok A/B compares like reasoning depth.
_EFFORT_TO_REASONING = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def reasoning_effort(effort: Optional[str]) -> str:
    """Map an Anthropic-style effort string to a valid grok reasoning_effort."""
    return _EFFORT_TO_REASONING.get((effort or "").strip().lower(), "low")


def new_conv_id() -> str:
    """A fresh conversation id for the ``x-grok-conv-id`` sticky-routing header."""
    import uuid

    return uuid.uuid4().hex


def xai_base_url() -> str:
    return (os.environ.get("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).rstrip("/")


def xai_api_key() -> Optional[str]:
    return os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or None


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal tool dicts ({name, description, input_schema}) into the
    OpenAI/xAI function-tool shape."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return converted


def normalize_usage(usage: Any) -> dict[str, int]:
    """Map an xAI/OpenAI ``usage`` object (dict) to our 4-key totals shape.

    xAI reports cached prompt tokens under ``prompt_tokens_details.cached_tokens``
    when its automatic prompt caching hits; we surface those as ``cache_read`` so
    the per-model spend totals stay comparable to the Anthropic path."""
    if not isinstance(usage, dict):
        return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    details = usage.get("prompt_tokens_details")
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    prompt = int(usage.get("prompt_tokens") or 0)
    return {
        "input": max(prompt - cached, 0),
        "output": int(usage.get("completion_tokens") or 0),
        "cache_read": cached,
        "cache_write": 0,
    }


def split_tool_content(content: Any) -> tuple[str, Optional[str]]:
    """A tool result from ``_format_tool_result`` is either a string or a list of
    ``[text-block, image-block]`` (Anthropic image shape). Return
    ``(text, image_data_url)`` where ``image_data_url`` is an OpenAI-style
    ``data:`` URL (or ``None``) so the Grok loops can re-attach page images as a
    follow-up vision message."""
    if isinstance(content, str):
        return content, None
    text = ""
    image_url: Optional[str] = None
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text") or text
            elif block.get("type") == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64" and src.get("data"):
                    media = src.get("media_type") or "image/png"
                    image_url = f"data:{media};base64,{src['data']}"
    return text, image_url


def _headers(api_key: str, conv_id: Optional[str] = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Sticky routing for xAI's automatic prompt caching — keeps a conversation's
    # requests on the same server so the growing prefix actually hits the cache.
    if conv_id:
        headers["x-grok-conv-id"] = conv_id
    return headers


def create(
    *,
    api_key: str,
    payload: dict[str, Any],
    base_url: Optional[str] = None,
    conv_id: Optional[str] = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Synchronous chat-completions call (used by the docchecker LLM protocol)."""
    import httpx

    url = f"{(base_url or xai_base_url()).rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=_headers(api_key, conv_id), json=payload)
        resp.raise_for_status()
        return resp.json()


async def acreate(
    *,
    api_key: str,
    payload: dict[str, Any],
    base_url: Optional[str] = None,
    conv_id: Optional[str] = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Async chat-completions call (used by the chat + agentic-sweep loops)."""
    import httpx

    url = f"{(base_url or xai_base_url()).rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=_headers(api_key, conv_id), json=payload)
        resp.raise_for_status()
        return resp.json()


def message_text(resp: dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()
