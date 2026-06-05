"""LLM client wrapper for the checking agent.

Two structured-output modes (both decoupled from the SDK so the pipeline is
unit-testable with a fake):

- ``deep=True``  → Opus-class reasoning: adaptive thinking + ``effort`` + a
  ``json_schema`` output format (forced tool_use is incompatible with thinking, so
  we constrain the format instead). Streamed, to avoid the SDK's large-max_tokens
  timeout guard. Used for the compare and revision-judge passes.
- ``deep=False`` → forced ``tool_use`` for guaranteed structure with no thinking.
  Cheap/fast — used for the verify pass and query planning.

Token reduction: prompt caching on the system prefix (and an optional large stable
``cache_context``, e.g. the submitted document reused across every revision-comment
call) so repeated calls read the big content at ~0.1x.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Callable, Optional, Protocol


class LLM(Protocol):
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
        """Return the structured result dict (tool input, or parsed json_schema output).

        ``emit``, when given, receives streamed reasoning as
        ``{"type": "thinking", "delta": <text>}`` during deep calls."""
        ...


def _strictify(schema: dict) -> dict:
    """Return a copy with additionalProperties:false on every object (json_schema requirement)."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


def _loads_loose(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {}


class AnthropicLLM:
    def __init__(self, api_key: str, default_model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = default_model
        self.last_usage: dict = {}

    @staticmethod
    def _system_blocks(system: str, cache_context: Optional[str]) -> list[dict]:
        blocks = [{"type": "text", "text": system}]
        if cache_context:
            blocks.append({"type": "text", "text": cache_context})
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def _record_usage(self, usage) -> None:
        self.last_usage = {
            "input": getattr(usage, "input_tokens", 0),
            "output": getattr(usage, "output_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0),
        }

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
        sys_blocks = self._system_blocks(system, cache_context)
        mdl = model or self._default_model
        if deep:
            return self._structured(mdl, sys_blocks, user_text, input_schema, effort, max_tokens, emit)

        resp = self._client.messages.create(
            model=mdl,
            max_tokens=max_tokens,
            system=sys_blocks,
            tools=[
                {"name": tool_name, "description": tool_description, "input_schema": input_schema}
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_text}],
        )
        self._record_usage(resp.usage)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return dict(block.input)
        return {}

    def _structured(self, model, sys_blocks, user_text, schema, effort, max_tokens, emit=None) -> dict:
        # Stream so large max_tokens (thinking + output) doesn't trip the SDK timeout guard.
        with self._client.messages.stream(
            model=model,
            max_tokens=max(max_tokens, 16000),
            system=sys_blocks,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": _strictify(schema)}},
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            if emit is not None:
                self._stream_thinking(stream, emit)
            msg = stream.get_final_message()
        self._record_usage(msg.usage)
        txt = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        return _loads_loose(txt) if txt else {}

    @staticmethod
    def _stream_thinking(stream, emit) -> None:
        """Forward the model's reasoning to ``emit`` as it streams. Deltas arrive
        in tiny pieces, so coalesce them into ~120-char chunks (or on newline) to
        avoid flooding the per-run event bus."""
        buf = ""

        def flush():
            nonlocal buf
            if buf.strip():
                try:
                    emit({"type": "thinking", "delta": buf})
                except Exception:  # noqa: BLE001 — never let telemetry break a run
                    pass
            buf = ""

        try:
            for event in stream:
                if getattr(event, "type", None) != "content_block_delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) != "thinking_delta":
                    continue
                buf += getattr(delta, "thinking", "") or ""
                if len(buf) >= 120 or "\n" in buf:
                    flush()
            flush()
        except Exception:  # noqa: BLE001 — fall back to non-streamed final message
            pass
