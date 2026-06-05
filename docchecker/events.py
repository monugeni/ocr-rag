"""In-memory per-run progress event bus (consumed by the SSE endpoint).

Events are appended by the agent seam's ``emit`` callback and read by index, so
an SSE client can resume from the last seen sequence number. Bounded per run.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

_MAX_EVENTS = 1000
_lock = threading.Lock()
_events: dict[str, list[dict[str, Any]]] = defaultdict(list)


def publish(run_id: str, event: dict[str, Any]) -> None:
    with _lock:
        buf = _events[run_id]
        seq = len(buf)
        buf.append({"seq": seq, **event})
        if len(buf) > _MAX_EVENTS:
            del buf[: len(buf) - _MAX_EVENTS]


def get_since(run_id: str, after_seq: int) -> list[dict[str, Any]]:
    with _lock:
        return [e for e in _events.get(run_id, []) if e["seq"] > after_seq]


def latest_seq(run_id: str) -> int:
    with _lock:
        buf = _events.get(run_id, [])
        return buf[-1]["seq"] if buf else -1


def clear(run_id: str) -> None:
    with _lock:
        _events.pop(run_id, None)
