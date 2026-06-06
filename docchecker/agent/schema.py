"""Shared dataclasses for the checking agent and the backend seam.

These types are the contract between ``app.agent_seam`` (backend) and
``checker.agent.run_check`` (agent). Keep them dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Severity → annotation colour (also used by the UI legend).
SEVERITY_COLORS = {
    "critical": "#D7263D",
    "major": "#E8A33D",
    "minor": "#F4D35E",
    "observation": "#4C9BE8",
}

CATEGORIES = [
    "compliance",
    "completeness",
    "consistency",
    "correctness",
    "bom",
    "dimension",
    "deviation",
    "comment-incorporation",
]


@dataclass
class DocRef:
    doc_id: int
    project: str
    pdf_path: str
    role: str                       # submitted | reference | prior_commented
    title: Optional[str] = None
    document_type: Optional[str] = None
    page_count: Optional[int] = None


@dataclass
class CheckContext:
    run_id: str
    submitted: list[DocRef]
    references: list[DocRef] = field(default_factory=list)
    reference_project: Optional[str] = None       # legacy single company folder
    reference_projects: list[str] = field(default_factory=list)  # additive: one or more company folders
    company_mcp_url: Optional[str] = None          # running company ocr-rag MCP, if any
    old_commented: Optional[DocRef] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    template_instructions: Optional[str] = None
    guiding_prompt: Optional[str] = None
    is_revision: bool = False
    annotated_out_dir: str = ""
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    emit: Callable[[dict], None] = lambda e: None
    # Live-agent plumbing (M8). When live is False or api_key is None, run_check
    # uses the deterministic stub.
    docs_db_path: str = ""
    api_key: Optional[str] = None
    model: str = "claude-opus-4-8"
    fast_model: str = "claude-haiku-4-5"
    effort: str = "high"          # low | medium | high | xhigh | max (deep-pass thinking depth)
    live: bool = False


@dataclass
class Finding:
    doc_id: int
    page_num: int
    severity: str                    # critical|major|minor|observation
    category: str
    title: str
    detail: str
    anchor_text: Optional[str] = None
    bbox: Optional[list[float]] = None
    citation: dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"       # high|medium|low
    possible: bool = False           # kept-but-uncertain (recall-first verify)
    annotation_xref: Optional[int] = None


@dataclass
class CommentStatus:
    prior_comment_ref: Optional[str]
    prior_comment_text: str
    prior_page: Optional[int]
    verdict: str                     # incorporated|partially|not_incorporated|not_applicable
    evidence: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    annotation_xref: Optional[int] = None


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)
    comment_statuses: list[CommentStatus] = field(default_factory=list)
    annotated_pdfs: dict[int, str] = field(default_factory=dict)   # doc_id -> path
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)   # model -> token totals
    trace: dict[str, Any] = field(default_factory=dict)   # debug trace (thinking, raw/pruned findings, limits)
