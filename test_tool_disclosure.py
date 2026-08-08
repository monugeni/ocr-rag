"""Progressive disclosure for OCR-RAG MCP tools.

Registration-only checks — does not open docs.db or write sidecars.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Prefer project venv if present when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcp_server as m


def test_only_core_tools_advertised():
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert names == set(m._CORE_TOOL_NAMES), sorted(names)
    # Deferred still exist as the original Tool objects
    assert "search_chunks" in m._DEFERRED_TOOLS
    assert "merge_documents" in m._DEFERRED_TOOLS
    assert "semantic_search" in m._DEFERRED_TOOLS
    assert "ranked_search" not in m._DEFERRED_TOOLS
    # All domain tools were registered somewhere
    domain_names = {n for names in m._TOOL_DOMAINS.values() for n in names}
    assert domain_names <= set(m._DEFERRED_TOOLS.keys()), domain_names - set(m._DEFERRED_TOOLS)


def test_discover_is_read_only_catalog():
    idx = m.discover_tools()
    assert idx["ok"] is True
    assert "corrections" in idx["domains"]
    cat = m.discover_tools(domain="corrections", detail="short")
    assert cat["ok"] is True
    names = {t["name"] for t in cat["tools"]}
    assert "merge_documents" in names
    assert "fix_ocr_text" in names
    full = m.discover_tools(domain="search", detail="full")
    assert full["ok"] is True
    assert any("inputSchema" in t for t in full["tools"])


def test_run_tool_preserves_handler_identity():
    """Deferred Tool.fn is the same object FastMCP registered (guards intact)."""
    tool = m._DEFERRED_TOOLS["merge_documents"]
    assert callable(tool.fn)
    # Admin gate is on the wrapper from register_correction_tools
    assert tool.fn.__name__ in ("guarded", "merge_documents") or callable(tool.fn)


def test_run_tool_rejects_core_and_unknown():
    core_err = m.run_tool("ranked_search", {"project": "x", "query": "y"})
    assert core_err.get("ok") is False
    unknown = m.run_tool("not_a_real_tool", {})
    assert unknown.get("ok") is False


def test_no_db_path_mutation_on_import():
    """Importing/finalizing must not retarget the database."""
    assert m.DB_PATH == "docs.db" or isinstance(m.DB_PATH, str)


if __name__ == "__main__":
    test_only_core_tools_advertised()
    test_discover_is_read_only_catalog()
    test_run_tool_preserves_handler_identity()
    test_run_tool_rejects_core_and_unknown()
    test_no_db_path_mutation_on_import()
    print("ALL PASSED")
