"""The real checking pipeline: gather → compare → self-verify → locate → annotate.

Driven by an injected ``LLM`` so it is fully unit-testable offline. Document text
is read from the ocr-rag docs DB; findings are written back as annotations via the
pdf-annotator library (see ``annotate.py``).
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
    """Plan reference-retrieval queries across the WHOLE submitted document.

    Previously this only saw the first 8k chars (~first 2-3 pages), so reference
    requirements governing later pages were never fetched and those pages went
    under-checked. We now plan over the full doc in windows (in parallel), then
    union + dedupe, so coverage scales with document length."""
    WINDOW = 7000          # chars per planning window
    PER_WINDOW = 6         # queries requested per window
    MAX_WINDOWS = 12       # bound cost on very large docs (~84k chars planned)
    MAX_TOTAL = 24         # cap total reference queries

    windows = [sub_text[i:i + WINDOW] for i in range(0, len(sub_text), WINDOW)][:MAX_WINDOWS]
    if not windows:
        windows = [sub_text[:WINDOW]]

    def plan(seg: str) -> list[str]:
        out = llm.call_tool(
            system="You plan retrieval queries to find the requirements that govern a submitted document.",
            user_text=(
                f"Given this section of a submitted document, list up to {PER_WINDOW} concise search "
                "queries to find the governing requirements (specs, clauses, quantities, materials, "
                "dimensions) in a reference knowledge base.\n\n" + seg
            ),
            tool_name="plan_queries",
            tool_description="Emit search queries for the reference knowledge base.",
            input_schema=_PLAN_TOOL,
            model=ctx.fast_model,
            max_tokens=800,
        )
        return [q for q in (out.get("queries") or []) if q][:PER_WINDOW]

    if len(windows) == 1:
        planned = plan(windows[0])
    else:
        with ThreadPoolExecutor(max_workers=min(6, len(windows)), thread_name_prefix="plan") as ex:
            planned = [q for lst in ex.map(plan, windows) for q in lst]

    seen: set[str] = set()
    uniq: list[str] = []
    for q in planned:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(q.strip())
    return uniq[:MAX_TOTAL]


def _reference_block(ctx: CheckContext, llm: LLM, sub_text: str) -> str:
    blocks = []
    # Freshly-uploaded references (full text).
    for ref in ctx.references:
        text = document_text(ctx.docs_db_path, ref.doc_id)
        blocks.append(f"=== REFERENCE (uploaded): {ref.title} (doc {ref.doc_id}) ===\n{text}")

    # Existing company knowledge base(s), via the running company MCP — one or
    # more reference folders (e.g. the tender folder + a Standards folder).
    ref_folders = ctx.reference_projects or ([ctx.reference_project] if ctx.reference_project else [])
    if ctx.company_mcp_url and ref_folders:
        ctx.emit({"stage": "Retrieving existing reference requirements", "type": "phase"})
        queries = _plan_queries(ctx, llm, sub_text)
        for folder in ref_folders:
            retrieved = fetch_reference_context(ctx.company_mcp_url, folder, queries)
            if retrieved:
                blocks.append(f"=== REFERENCE (existing KB: {folder}) ===\n{retrieved}")

    return "\n\n".join(blocks) if blocks else "(no reference documents provided)"


# One focused compare pass per dimension. A single all-categories pass has
# incomplete, run-to-run-varying recall (the model misses different things each
# single-shot run); running a dedicated pass per dimension and unioning the
# results dramatically improves recall and stability. Passes run in parallel.
_LENSES = [
    ("compliance", "Does the submission satisfy each requirement, spec and clause in the reference? Flag every deviation or unmet requirement."),
    ("completeness", "Does the submission cover everything the reference asks for? Flag every missing or unaddressed item."),
    ("consistency", "Do values, quantities, tags and units match across the submission and reference (and within the submission)? Flag every contradiction."),
    ("bom", "Bill of materials: do item lists, quantities and part numbers match the reference (PO/PR line items)? Flag every mismatch."),
    ("dimension", "Do dimensions, sizes and ratings match the reference and the datasheet? Flag every mismatch."),
    ("deviation", "Flag every explicit or implicit deviation from the reference — including notes, exceptions and qualifications."),
]


def _finding_from_dict(f: dict, submitted, category_default: str) -> Finding:
    return Finding(
        doc_id=submitted.doc_id,
        page_num=f.get("submitted_page") or 1,
        severity=f.get("severity", "observation"),
        category=f.get("category", category_default),
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


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Collapse the same issue raised by multiple lenses. Key on page + verbatim
    submitted anchor (strongest signal), falling back to page + title. Keep the
    most severe / most detailed instance."""
    _SEV = {"critical": 4, "major": 3, "minor": 2, "observation": 1}
    best: dict[tuple, Finding] = {}
    for f in findings:
        anchor = (f.anchor_text or "").strip().lower()
        key = (f.page_num, anchor[:80]) if anchor else (f.page_num, (f.title or "").strip().lower()[:60])
        cur = best.get(key)
        if cur is None or (_SEV.get(f.severity, 0), len(f.detail or "")) > (_SEV.get(cur.severity, 0), len(cur.detail or "")):
            best[key] = f
    return list(best.values())


def _lens_emit(ctx: CheckContext, key: str):
    """Tag a lens's streamed reasoning with its name so the parallel thinking
    streams stay attributable in the live panel and the trace."""
    def emit(ev):
        if isinstance(ev, dict) and ev.get("type") == "thinking":
            ev = {**ev, "delta": f"[{key}] " + (ev.get("delta") or "")}
        ctx.emit(ev)
    return emit


def _compare(ctx: CheckContext, llm: LLM, submitted, sub_text: str) -> list[Finding]:
    system = build_system_prompt(ctx)
    ref_block = _reference_block(ctx, llm, sub_text)  # retrieved once, reused by every lens
    base = (
        f"SUBMITTED DOCUMENT: {submitted.title} (doc {submitted.doc_id})\n{sub_text}\n\n"
        f"{ref_block}\n\n"
    )

    def run_lens(lens: tuple[str, str]) -> list[Finding]:
        key, desc = lens
        user = base + (
            f"Focus on {key.upper()}. {desc}\n"
            "Report EVERY such finding using the report_findings tool — be exhaustive and do not "
            "self-censor borderline ones (a separate step verifies them). Copy submitted_anchor "
            "verbatim from the relevant submitted page."
        )
        try:
            out = llm.call_tool(
                system=system, user_text=user,
                tool_name="report_findings",
                tool_description=f"Report all {key} findings from the comparison.",
                input_schema=_FINDINGS_TOOL,
                model=ctx.model, deep=True, effort=ctx.effort, max_tokens=24000,
                emit=_lens_emit(ctx, key),
            )
            fnds = [_finding_from_dict(f, submitted, key) for f in out.get("findings", [])]
        except Exception as exc:  # noqa: BLE001 — one lens must not sink the rest
            ctx.emit({"stage": f"Lens '{key}' failed: {exc}", "type": "phase"})
            fnds = []
        ctx.emit({"stage": f"Checked {key}: {len(fnds)} candidate(s)", "type": "phase"})
        return fnds

    with ThreadPoolExecutor(max_workers=len(_LENSES), thread_name_prefix="lens") as ex:
        per_lens = list(ex.map(run_lens, _LENSES))
    return _dedupe_findings([f for lst in per_lens for f in lst])


def _verify(ctx: CheckContext, llm: LLM, submitted, finding: Finding) -> bool:
    """Re-read the cited submitted page (fresh context) and confirm the finding."""
    sub_page = page_text(ctx.docs_db_path, submitted.doc_id, finding.page_num)
    ref_quote = (finding.citation or {}).get("snippet") or ""
    user = (
        "Verify this candidate finding against the evidence.\n\n"
        f"FINDING: [{finding.category}/{finding.severity}] {finding.title} — {finding.detail}\n"
        f"Reference quote: {ref_quote}\n\n"
        f"SUBMITTED PAGE {finding.page_num}:\n{sub_page}\n\n"
        "Use the verdict tool. Set verified=false ONLY if the page text clearly contradicts or "
        "clearly does not support the finding. If you are unsure, keep it: verified=true with "
        "confidence=low. Missing a genuine issue is worse than flagging a borderline one."
    )
    out = llm.call_tool(
        system=(
            "You are a recall-first verifier for engineering document review. Keep findings "
            "unless they are clearly wrong or unsupported by the submitted page. When unsure, "
            "keep them with low confidence rather than dropping them."
        ),
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
    # Keep unless the verifier explicitly rejected it (recall-first).
    return bool(out.get("verified"))


def _verify_all(ctx: CheckContext, llm: LLM, submitted, candidates, result):
    """Verify candidates concurrently (each is an independent fast-model call), then
    fold results back in candidate order. Returns (confirmed, verdicts) — verdicts is
    a per-candidate audit list (kept/dropped/error + confidence + evidence) for the
    debug trace, so users can see exactly what self-verification pruned and why."""
    if not candidates:
        return [], []

    def work(f: Finding):
        try:
            return f, ("keep" if _verify(ctx, llm, submitted, f) else "drop"), None
        except Exception as exc:  # noqa: BLE001
            return f, "error", str(exc)

    with ThreadPoolExecutor(max_workers=min(6, len(candidates)),
                            thread_name_prefix="verify") as ex:
        outcomes = list(ex.map(work, candidates))  # ex.map preserves input order

    confirmed: list[Finding] = []
    verdicts: list[dict] = []
    for f, verdict, err in outcomes:
        entry = {
            "title": f.title,
            "page": f.page_num,
            "confidence": getattr(f, "confidence", ""),
            "evidence": (f.citation or {}).get("verify_evidence", ""),
        }
        if verdict == "keep":
            # Kept-but-uncertain findings are surfaced as "possible" rather than dropped.
            f.possible = getattr(f, "confidence", "") == "low"
            confirmed.append(f)
            entry["verdict"] = "possible" if f.possible else "kept"
        elif verdict == "drop":
            result.warnings.append(f"Dropped (clearly unsupported): {f.title}")
            entry["verdict"] = "dropped"
        else:
            result.warnings.append(f"Verification error for '{f.title}': {err}")
            confirmed.append(f)  # keep rather than silently lose
            entry["verdict"] = "error"
            entry["reason"] = err
        verdicts.append(entry)
    return confirmed, verdicts


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
            emit=ctx.emit,
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

    # Capture the model's streamed reasoning for the persisted debug trace while
    # still forwarding it live to the SSE panel.
    thinking_buf: list[str] = []
    _orig_emit = ctx.emit

    def _capturing_emit(ev):
        if isinstance(ev, dict) and ev.get("type") == "thinking":
            thinking_buf.append(ev.get("delta", "") or "")
        _orig_emit(ev)

    ctx.emit = _capturing_emit

    sub_truncated = "[... truncated at" in sub_text
    if sub_truncated:
        result.warnings.append(
            "Submitted document was truncated for analysis (very long document) — later pages may be under-checked."
        )

    ctx.emit({"stage": "Comparing against reference (multi-lens)", "type": "phase"})
    candidates = _compare(ctx, llm, submitted, sub_text)
    compare_truncated = getattr(llm, "any_truncated", False)
    if compare_truncated:
        result.warnings.append(
            "A comparison pass hit its output budget — some findings may be missing. Re-run or split the document."
        )
    ctx.emit({"stage": f"{len(candidates)} unique candidate finding(s) across lenses", "type": "phase"})

    ctx.emit({"stage": "Self-verifying findings", "type": "phase"})
    confirmed, verdicts = _verify_all(ctx, llm, submitted, candidates, result)

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

    # Debug trace: what the model reasoned, what it raised, what verification
    # pruned, and which limits were hit — so a run can be understood/diffed.
    result.trace = {
        "model": ctx.model,
        "effort": getattr(ctx, "effort", ""),
        "thinking": "".join(thinking_buf).strip(),
        "raw_findings": [asdict(f) for f in candidates],
        "verify": verdicts,
        "lenses": [k for k, _ in _LENSES],
        "limits": {
            "submitted_chars": len(sub_text),
            "submitted_truncated": sub_truncated,
            "candidates": len(candidates),
            "confirmed": len(confirmed),
            "possible": sum(1 for v in verdicts if v.get("verdict") == "possible"),
            "dropped": sum(1 for v in verdicts if v.get("verdict") == "dropped"),
            "compare_max_output_tokens": 24000,
            "compare_output_truncated": compare_truncated,
        },
        "warnings": list(result.warnings),
    }
    ctx.emit({"stage": "Done", "type": "phase", "findings": len(result.findings)})
    return result
