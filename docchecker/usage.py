"""Per-user spend tracking for LLM usage (Ask chat + document Check).

A single ``usage_events`` table (in the checker DB) is written by both the chat
path (web.py) and the checker pipeline (agent_seam), keyed by the user's email so
spend aggregates per person across both features. Cost is an *estimate*: captured
token usage x per-model Anthropic pricing x a configurable USD->INR rate
(``OCR_RAG_USD_INR``, default 88.0). No exact billing data exists; this is a
transparent approximation the user asked for.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Optional

from . import config

# USD per 1M tokens, per model. Sourced from Anthropic pricing (Opus 4.8 5/25,
# Haiku 4.5 1/5, Sonnet 4.6 3/15); cache read ~0.1x input, cache write ~1.25x.
_PRICING: dict[str, dict[str, float]] = {
    "opus":   {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "sonnet": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "haiku":  {"input": 1.00, "output": 5.00,  "cache_read": 0.10, "cache_write": 1.25},
}
_DEFAULT_PRICE = _PRICING["opus"]  # unknown model -> price as the most expensive tier

_lock = threading.Lock()


def _grok_pricing() -> dict[str, float]:
    """xAI (Grok) USD/1M pricing. We don't hardcode exact rates — set the real
    numbers via OCR_RAG_GROK_PRICE_IN / _OUT (per 1M tokens). Defaults are a
    rough grok-4 family estimate; cache read ~0.25x input, write ~1.25x."""
    def _f(env: str, default: float) -> float:
        try:
            return float(os.environ.get(env, default))
        except (TypeError, ValueError):
            return default
    inp = _f("OCR_RAG_GROK_PRICE_IN", 3.00)
    out = _f("OCR_RAG_GROK_PRICE_OUT", 15.00)
    return {"input": inp, "output": out, "cache_read": inp * 0.25, "cache_write": inp * 1.25}


def _price_for(model: str) -> dict[str, float]:
    m = (model or "").lower()
    if "grok" in m:
        return _grok_pricing()
    for key, price in _PRICING.items():
        if key in m:
            return price
    return _DEFAULT_PRICE


def usd_to_inr() -> float:
    try:
        return float(os.environ.get("OCR_RAG_USD_INR", "88.0"))
    except (TypeError, ValueError):
        return 88.0


def cost_usd(model: str, usage: dict[str, Any]) -> float:
    """Estimate USD cost for one usage dict {input, output, cache_read, cache_write}."""
    p = _price_for(model)
    return (
        (usage.get("input", 0) or 0) * p["input"]
        + (usage.get("output", 0) or 0) * p["output"]
        + (usage.get("cache_read", 0) or 0) * p["cache_read"]
        + (usage.get("cache_write", 0) or 0) * p["cache_write"]
    ) / 1_000_000.0


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email    TEXT,
            kind          TEXT NOT NULL,          -- 'chat' | 'check'
            model         TEXT,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read    INTEGER NOT NULL DEFAULT 0,
            cache_write   INTEGER NOT NULL DEFAULT 0,
            usd           REAL NOT NULL DEFAULT 0,
            inr           REAL NOT NULL DEFAULT 0,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_events_user ON usage_events(user_email, created_at)"
    )
    conn.commit()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.CHECKER_DB, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def record(email: Optional[str], kind: str, model: str, usage: dict[str, Any]) -> None:
    """Record one usage event. Best-effort: never raise into a chat/check flow."""
    try:
        tokens = {
            "input": int(usage.get("input", 0) or 0),
            "output": int(usage.get("output", 0) or 0),
            "cache_read": int(usage.get("cache_read", 0) or 0),
            "cache_write": int(usage.get("cache_write", 0) or 0),
        }
        if not any(tokens.values()):
            return
        usd = cost_usd(model, tokens)
        inr = usd * usd_to_inr()
        with _lock:
            conn = _conn()
            try:
                ensure_schema(conn)
                conn.execute(
                    "INSERT INTO usage_events "
                    "(user_email, kind, model, input_tokens, output_tokens, cache_read, cache_write, usd, inr) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ((email or "").lower() or None, kind, model,
                     tokens["input"], tokens["output"], tokens["cache_read"], tokens["cache_write"],
                     usd, inr),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 — spend telemetry must never break the feature
        pass


def total_for_email(email: Optional[str]) -> dict[str, Any]:
    """Aggregate spend for a user: total INR/USD plus a per-kind breakdown."""
    out = {"inr": 0.0, "usd": 0.0, "tokens": 0, "by_kind": {}}
    if not email:
        return out
    try:
        conn = _conn()
        try:
            ensure_schema(conn)
            rows = conn.execute(
                "SELECT kind, "
                "SUM(usd) AS usd, SUM(inr) AS inr, "
                "SUM(input_tokens + output_tokens + cache_read + cache_write) AS tokens "
                "FROM usage_events WHERE user_email = ? GROUP BY kind",
                (email.lower(),),
            ).fetchall()
            for r in rows:
                out["usd"] += r["usd"] or 0.0
                out["inr"] += r["inr"] or 0.0
                out["tokens"] += int(r["tokens"] or 0)
                out["by_kind"][r["kind"]] = {
                    "usd": r["usd"] or 0.0,
                    "inr": r["inr"] or 0.0,
                    "tokens": int(r["tokens"] or 0),
                }
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        pass
    return out
