"""Launch and connect to the ocr-rag MCP server(s).

The ocr-rag MCP exposes search/read/render tools over a docs DB via
streamable-HTTP (``python mcp_server.py --db <db> --port <port>`` →
``http://127.0.0.1:<port>/mcp``). We run one instance over the checker docs DB,
and optionally a second (read-only) instance over the company docs DB for
``reference_mode=existing``.

This module both manages the subprocesses and offers async helpers
(``list_tools`` / ``call_tool``) built on the same client pattern as
``ocr-rag/chat_mcp_runner.py``.
"""
from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import config


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def mcp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


@dataclass
class McpProcess:
    proc: subprocess.Popen
    port: int
    db_path: str

    @property
    def url(self) -> str:
        return mcp_url(self.port)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_ocr_rag_mcp(db_path: str, port: int, *, wait: float = 25.0) -> McpProcess:
    """Launch an ocr-rag MCP server over ``db_path`` on ``port`` and wait for it."""
    if _port_open(port):
        raise RuntimeError(f"port {port} already in use")
    proc = subprocess.Popen(
        [config.OCR_RAG_PYTHON, config.OCR_RAG_MCP_SERVER, "--db", db_path, "--port", str(port)],
        cwd=str(config.OCR_RAG_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"ocr-rag MCP exited early:\n{out}")
        if _port_open(port):
            return McpProcess(proc, port, db_path)
        time.sleep(0.3)
    proc.terminate()
    raise TimeoutError(f"ocr-rag MCP on port {port} did not start within {wait}s")


# ---------------------------------------------------------------------------
# Process registry (lazy singletons)
# ---------------------------------------------------------------------------
_checker_mcp: Optional[McpProcess] = None
_company_mcp: Optional[McpProcess] = None


def ensure_checker_mcp() -> McpProcess:
    """Start (once) the MCP over the checker docs DB. Requires docs.db to exist."""
    global _checker_mcp
    if _checker_mcp is None or _checker_mcp.proc.poll() is not None:
        _checker_mcp = start_ocr_rag_mcp(config.DOCS_DB, config.CHECKER_MCP_PORT)
    return _checker_mcp


def ensure_company_mcp() -> Optional[McpProcess]:
    """Start (once) the MCP over the company docs DB, if configured."""
    global _company_mcp
    if not config.COMPANY_DOCS_DB:
        return None
    if _company_mcp is None or _company_mcp.proc.poll() is not None:
        _company_mcp = start_ocr_rag_mcp(config.COMPANY_DOCS_DB, config.COMPANY_MCP_PORT)
    return _company_mcp


def stop_all() -> None:
    global _checker_mcp, _company_mcp
    for m in (_checker_mcp, _company_mcp):
        if m is not None:
            m.stop()
    _checker_mcp = None
    _company_mcp = None


# ---------------------------------------------------------------------------
# Async client helpers (mirror ocr-rag/chat_mcp_runner.py)
# ---------------------------------------------------------------------------
async def list_tools(url: str) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=10, sse_read_timeout=120) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


async def call_tool(url: str, name: str, arguments: dict[str, Any]) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=30, sse_read_timeout=180) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)
