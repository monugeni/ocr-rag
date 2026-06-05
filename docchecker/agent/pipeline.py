"""The real checking pipeline: gather → compare → self-verify → locate → annotate.

Driven by an injected ``LLM`` so it is fully unit-testable offline. Document text
is read from the ocr-rag docs DB; findings are written back as annotations via the
pdf-annotator library (see ``annotate.py``).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .annotate import annotate_finding
from .comments import read_prior_comments
from .company_refs import fetch_reference_context
from .docs_reader import document_text, page_text
from .llm import LLM
from .prompts import build_system_prompt
from .schema import CATEGORIES, CheckContext, CheckResult, CommentStatus, Finding

# ---------------------------------------------------------------------------
# Tool schemas (forced structured output)
# ---------------------------------------------------------------------------
_FINDINGS_TOOL = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "major", "minor", "observation"],
                    },
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "submitted_page": {"type": "integer"},
                    "submitted_anchor": {
                        "type": "string",
                        "description": "Verbatim substring copied from the submitted page, for annotation.",
                    },
                    "ref_doc_title": {"type": "string"},
                    "ref_page": {"type": "integer"},
                    "ref_quote": {"type": "string"},
                },
                "required": ["category", "severity", "title", "detail", "submitted_page", "submitted_anchor"],
            },
        }
    },
    "required": ["findings"],
}

_VERDICT_TOOL = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "string"},
    },
    "required": ["verified", "confidence", "evidence"],
}

_PLAN_TOOL = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Search queries to find governing requirements in the reference KB.",
        }
    },
    "required": ["queries"],
}

_INCORP_TOOL = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["incorporated", "partially", "not_incorporated", "not_applicable"],
        },
        "evidence": {"type": "string"},
        "new_page": {"type": "integer"},
    },
    "required": ["verdict", "evidence"],
}


def _plan_queries(ctx: CheckContext, llm: LLM, sub_text: str) -> list[str]:
    out = llm.call_tool(
        system="You plan retrieval queries to find the requirements that govern a submitted document.",
        user_text=(
            "Given the submitted document below, list up to 8 concise search queries to find the "
            "governing requirements (specs, clauses, quantities, materials, dimensions) in a "
            "reference knowledge base.\n\n" + sub_text[:8000]
        ),
        tool_name="plan_queries",
        tool_description="Emit search queries for the reference knowledge base.",
        input_schema=_PLAN_TOOL,
        model=ctx.fast_model,
        max_tokens=800,
    )
    return [q for q in (out.get("queries") or []) if q][:8]


def _reference_block(ctx: CheckContext, llm: LLM, sub_text: str) -> str:
    blocks = []
    # Freshly-uploaded references (full text).
    for ref in ctx.references:
        text = document_text(ctx.docs_db_path, ref.doc_id)
        blocks.append(f"=== REFERENCE (uploaded): {ref.title} (doc {ref.doc_id}) ===\n{text}")

    # Existing company knowledge base, via the running company MCP.
    if ctx.company_mcp_url and ctx.reference_project:
        ctx.emit({"stage": "Retrieving existing reference requirements", "type": "phase"})
        queries = _plan_queries(ctx, llm, sub_text)
        retrieved = fetch_reference_context(ctx.company_mcp_url, ctx.reference_project, queries)
        if retrieved:
            blocks.append(
                f"=== REFERENCE (existing KB: {ctx.reference_project}) ===\n{retrieved}"
            )

    return "\n\n".join(blocks) if blocks else "(no reference documents provided)"


def _compare(ctx: CheckContext, llm: LLM, submitted, sub_text: str) -> list[Finding]:
    system = build_system_prompt(ctx)
    user = (
        f"SUBMITTED DOCUMENT: {submitted.title} (doc {submitted.doc_id})\n{sub_text}\n\n"
        f"{_reference_block(ctx, llm, sub_text)}\n\n"
        "Compare the submitted document against the reference document(s) and report all "
        "candidate findings using the report_findings tool. Copy submitted_anchor verbatim "
        "from the relevant submitted page."
    )
    out = llm.call_tool(
        system=system,
        user_text=user,
        tool_name="report_findings",
        tool_description="Report all candidate findings from the comparison.",
        input_schema=_FINDINGS_TOOL,
        model=ctx.model,
        deep=True,
        effort=ctx.effort,
        max_tokens=16000,
    )
    findings = []
    for f in out.get("findings", []):
        findings.append(
            Finding(
                doc_id=submitted.doc_id,
                page_num=f.get("submitted_page") or 1,
                severity=f.get("severity", "observation"),
                category=f.get("category", "compliance"),
                title=f.get("title", "Finding"),
                detail=f.get("detail", ""),
                anchor_text=f.get("submitted_anchor"),
                citation={
                    "doc_title": f.get("ref_doc_title"),
                    "ref_page": f.get("ref_page"),
                    "snippet": f.get("ref_quote"),
                },
                confidence="medium",
            )
        )
    return findings


def _verify(ctx: CheckContext, llm: LLM, submitted, finding: Finding) -> bool:
    """Re-read the cited submitted page (fresh context) and confirm the finding."""
    sub_page = page_text(ctx.docs_db_path, submitted.doc_id, finding.page_num)
    ref_quote = (finding.citation or {}).get("snippet") or ""
    user = (
        "Verify this candidate finding against the evidence. Only confirm if the submitted "
        "page text genuinely supports it.\n\n"
        f"FINDING: [{finding.category}/{finding.severity}] {finding.title} — {finding.detail}\n"
        f"Reference quote: {ref_quote}\n\n"
        f"SUBMITTED PAGE {finding.page_num}:\n{sub_page}\n\n"
        "Use the verdict tool. Set verified=false if the page text does not support the finding."
    )
    out = llm.call_tool(
        system="You are a strict verifier. Default to verified=false when uncertain.",
        user_text=user,
        tool_name="verdict",
        tool_description="Confirm or reject the candidate finding.",
        input_schema=_VERDICT_TOOL,
        model=ctx.fast_model,
        max_tokens=1500,
    )
    finding.confidence = out.get("confidence", "low")
    if out.get("evidence"):
        finding.citation = {**(finding.citation or {}), "verify_evidence": out["evidence"]}
    return bool(out.get("verified")) and finding.confidence in ("high", "medium")


def _judge_comments(ctx: CheckContext, llm: LLM, submitted, sub_text: str) -> list[CommentStatus]:
    comments = read_prior_comments(ctx.old_commented.pdf_path)
    results = []
    # The new-revision text is identical across every comment → send it once as a
    # cached prefix so calls 2..N read it at ~0.1x cost instead of re-paying for it.
    judge_system = (
        "You judge whether prior review comments were addressed in the new revision. "
        "Quote evidence verbatim from the new-revision text. Use the incorporation tool.\n\n"
        f"NEW REVISION:\n{sub_text}"
    )
    for c in comments:
        ref = c.get("referenced_text")
        ref_line = f"\nThe comment marks/points at this text in the OLD revision: \"{ref}\"" if ref else ""
        user = (
            f"PRIOR COMMENT (old p{c.get('page')}): {c['content']}{ref_line}\n\n"
            "Has this been incorporated in the new revision above? Locate the corresponding "
            "content and judge."
        )
        out = llm.call_tool(
            system=judge_system,
            cache_context=None,
            user_text=user,
            tool_name="incorporation",
            tool_description="Judge incorporation of the prior comment.",
            input_schema=_INCORP_TOOL,
            model=ctx.model,
            deep=True,
            effort="high",
            max_tokens=16000,
        )
        results.append(
            CommentStatus(
                prior_comment_ref=str(c.get("xref")) if c.get("xref") is not None else None,
                prior_comment_text=c["content"],
                prior_page=c.get("page"),
                verdict=out.get("verdict", "not_applicable"),
                evidence={"text": out.get("evidence", ""), "new_page": out.get("new_page")},
                detail=out.get("evidence", ""),
            )
        )
    return results


def run_real_check(ctx: CheckContext, llm: LLM) -> CheckResult:
    result = CheckResult()
    if not ctx.submitted:
        result.warnings.append("No submitted document to check.")
        return result

    submitted = ctx.submitted[0]
    sub_text = document_text(ctx.docs_db_path, submitted.doc_id)

    ctx.emit({"stage": "Comparing against reference", "type": "phase"})
    candidates = _compare(ctx, llm, submitted, sub_text)
    ctx.emit({"stage": f"{len(candidates)} candidate finding(s)", "type": "phase"})

    ctx.emit({"stage": "Self-verifying findings", "type": "phase"})
    confirmed = []
    for f in candidates:
        try:
            if _verify(ctx, llm, submitted, f):
                confirmed.append(f)
            else:
                result.warnings.append(f"Dropped unverified finding: {f.title}")
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Verification error for '{f.title}': {exc}")
            confirmed.append(f)  # keep rather than silently lose

    # Annotate confirmed findings onto a copy of the submitted PDF.
    ctx.emit({"stage": f"Annotating {len(confirmed)} finding(s)", "type": "phase"})
    out_dir = Path(ctx.annotated_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = out_dir / f"{Path(submitted.pdf_path).stem}.annotated.pdf"
    shutil.copyfile(submitted.pdf_path, annotated)
    for f in confirmed:
        xref, _method = annotate_finding(str(annotated), f)
        f.annotation_xref = xref
    result.findings = confirmed
    result.annotated_pdfs[submitted.doc_id] = str(annotated)

    # Revision: comment-incorporation judging.
    if ctx.is_revision and ctx.old_commented and ctx.old_commented.pdf_path:
        ctx.emit({"stage": "Judging prior comment incorporation", "type": "phase"})
        result.comment_statuses = _judge_comments(ctx, llm, submitted, sub_text)

    result.stats = {
        "phase": "real",
        "candidates": len(candidates),
        "confirmed": len(confirmed),
        "comments": len(result.comment_statuses),
    }
    ctx.emit({"stage": "Done", "type": "phase", "findings": len(result.findings)})
    return result
