"""Grok (xAI) implementation of the checking-agent ``LLM`` protocol.

Drop-in alternative to ``AnthropicLLM`` so the document checker can run on Grok
for A/B comparison. Same two structured-output modes the pipeline expects:

- ``deep=True``  → ``response_format`` json_schema for guaranteed structure
  (Grok's reasoning is internal; ``effort`` has no Grok analogue, so it is
  accepted and ignored). Best-effort thinking is streamed via ``emit`` when the
  response exposes ``reasoning_content``.
- ``deep=False`` → forced function ``tool_choice`` for guaranteed structure with
  no reasoning surface — the verify/plan passes.

xAI is OpenAI-compatible, so this talks to ``{base}/chat/completions`` via the
shared ``grok_client``. Usage totals mirror ``AnthropicLLM`` (per-model dict)
so the spend + trace plumbing is unchanged.
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

import grok_client

from .llm import _loads_loose, _strictify


class GrokLLM:
    def __init__(self, api_key: str, default_model: str, base_url: Optional[str] = None):
        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url or grok_client.xai_base_url()
        # One conversation id per run: the verify/judge passes share a big stable
        # system prefix, so sticky routing lets xAI's auto-cache reuse it.
        self._conv_id = grok_client.new_conv_id()
        self.last_usage: dict = {}
        self.last_stop_reason: str | None = None
        self.any_truncated: bool = False
        self._usage_totals: dict[str, dict[str, int]] = {}
        self._usage_lock = threading.Lock()

    # ------------------------------------------------------------------ usage
    def _record_usage(self, usage, model: str = "") -> None:
        """Record one response's usage. Accepts an xAI/OpenAI ``usage`` dict
        (the agentic sweep and ``call_tool`` both pass the raw dict)."""
        rec = grok_client.normalize_usage(usage)
        self.last_usage = rec
        with self._usage_lock:
            tot = self._usage_totals.setdefault(
                model or self._default_model,
                {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            )
            for k, v in rec.items():
                tot[k] += v

    def drain_usage(self) -> dict[str, dict[str, int]]:
        with self._usage_lock:
            out = self._usage_totals
            self._usage_totals = {}
            return out

    # ------------------------------------------------------------------ call
    def call_tool(
        self,
        *,
        system: str,
        user_text: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        model: str | None = None,
        max_tokens: int = 8000,
        cache_context: str | None = None,
        deep: bool = False,
        effort: str = "high",
        emit: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        mdl = model or self._default_model
        sys_text = system if not cache_context else f"{system}\n\n{cache_context}"
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]
        if deep:
            return self._structured(mdl, messages, tool_name, input_schema, max_tokens, effort, emit)
        return self._forced_tool(mdl, messages, tool_name, tool_description, input_schema, max_tokens)

    def _forced_tool(self, model, messages, tool_name, tool_description, schema, max_tokens) -> dict:
        resp = grok_client.create(
            api_key=self._api_key,
            base_url=self._base_url,
            conv_id=self._conv_id,
            payload={
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": schema,
                    },
                }],
                "tool_choice": {"type": "function", "function": {"name": tool_name}},
                # Mirrors the Anthropic non-deep path (forced tool_use, no thinking).
                "reasoning_effort": "none",
            },
        )
        self._record_usage(resp.get("usage"), model)
        choice = (resp.get("choices") or [{}])[0]
        self.last_stop_reason = choice.get("finish_reason")
        if self.last_stop_reason == "length":
            self.any_truncated = True
        msg = choice.get("message") or {}
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") == tool_name:
                args = fn.get("arguments") or ""
                try:
                    return json.loads(args)
                except Exception:  # noqa: BLE001
                    return _loads_loose(args)
        # No tool call produced — fall back to parsing any text content.
        return _loads_loose(msg.get("content") or "")

    def _structured(self, model, messages, tool_name, schema, max_tokens, effort="high", emit=None) -> dict:
        resp = grok_client.create(
            api_key=self._api_key,
            base_url=self._base_url,
            conv_id=self._conv_id,
            payload={
                "model": model,
                "max_tokens": max(max_tokens, 16000),
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": tool_name or "result",
                        "schema": _strictify(schema),
                        "strict": True,
                    },
                },
                # Mirror the Anthropic deep pass's effort (low|medium|high|xhigh|max
                # → grok none|low|medium|high) so the A/B compares like depth.
                "reasoning_effort": grok_client.reasoning_effort(effort),
            },
        )
        self._record_usage(resp.get("usage"), model)
        choice = (resp.get("choices") or [{}])[0]
        self.last_stop_reason = choice.get("finish_reason")
        if self.last_stop_reason == "length":
            self.any_truncated = True
        msg = choice.get("message") or {}
        if emit is not None:
            reasoning = msg.get("reasoning_content")
            if reasoning:
                try:
                    emit({"type": "thinking", "delta": reasoning})
                except Exception:  # noqa: BLE001 — telemetry must never break a run
                    pass
        return _loads_loose(msg.get("content") or "")
