"""Checking agent entry point: ``run_check(ctx) -> CheckResult``.

This is the integration seam the backend depends on. The full phased pipeline
(gather → plan → compare → verify → locate → annotate) lands in M8; for now this
is a deterministic STUB that exercises the whole path end-to-end: it reads the
submitted document's first page, emits one finding, and writes a real annotation
onto a copy of the submitted PDF.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .annotate import annotate_finding
from .comments import read_prior_comments, stub_comment_statuses
from .schema import CheckContext, CheckResult, Finding


def _first_text_line(pdf_path: str) -> tuple[int, str | None]:
    """Return (page_num, first non-trivial text line) from the PDF."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            for pno in range(doc.page_count):
                text = doc.load_page(pno).get_text("text")
                for line in (ln.strip() for ln in text.splitlines()):
                    if len(line) >= 4:
                        return pno + 1, line
        finally:
            doc.close()
    except Exception:  # noqa: BLE001
        pass
    return 1, None


def run_check(ctx: CheckContext) -> CheckResult:
    # Live pipeline when enabled and an API key is present; otherwise the stub.
    if ctx.live and ctx.api_key and ctx.docs_db_path:
        try:
            from .llm import AnthropicLLM
            from .pipeline import run_real_check

            llm = AnthropicLLM(ctx.api_key, ctx.model)
            result = run_real_check(ctx, llm)
            result.usage = llm.drain_usage()
            return result
        except Exception as exc:  # noqa: BLE001 — never fail the run; fall back to stub
            ctx.emit({"stage": f"Live agent failed ({exc}); using stub", "type": "phase"})
    return _run_stub(ctx)


def _run_stub(ctx: CheckContext) -> CheckResult:
    result = CheckResult()
    ctx.emit({"stage": "Starting check", "type": "phase"})

    if not ctx.submitted:
        result.warnings.append("No submitted document to check.")
        return result

    sub = ctx.submitted[0]
    ctx.emit({"stage": "Reading submitted document", "type": "phase",
              "doc": sub.title, "doc_id": sub.doc_id})

    page_num, anchor = _first_text_line(sub.pdf_path)

    ref = ctx.references[0] if ctx.references else None
    citation = {}
    if ref:
        citation = {"doc_title": ref.title, "heading": None, "ref_page": 1,
                    "snippet": "(stub) reference comparison pending real agent"}

    finding = Finding(
        doc_id=sub.doc_id,
        page_num=page_num,
        severity="observation",
        category="compliance",
        title="Stub finding",
        detail=(
            "Placeholder finding produced by the stub agent to validate the "
            "end-to-end pipeline. The real agent replaces this in M8."
        ),
        anchor_text=anchor,
        citation=citation,
        confidence="low",
    )

    # Copy the submitted PDF into the run's annotated dir, then annotate in place.
    ctx.emit({"stage": "Writing annotations", "type": "phase"})
    out_dir = Path(ctx.annotated_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / f"{Path(sub.pdf_path).stem}.annotated.pdf"
    shutil.copyfile(sub.pdf_path, annotated_path)

    xref, method = annotate_finding(str(annotated_path), finding)
    finding.annotation_xref = xref
    if method == "failed":
        result.warnings.append(f"Could not place annotation for finding on page {page_num}.")

    result.findings.append(finding)
    result.annotated_pdfs[sub.doc_id] = str(annotated_path)

    # Revision mode: read prior review comments from the old commented PDF.
    if ctx.is_revision and ctx.old_commented and ctx.old_commented.pdf_path:
        ctx.emit({"stage": "Reading prior review comments", "type": "phase"})
        comments = read_prior_comments(ctx.old_commented.pdf_path)
        result.comment_statuses = stub_comment_statuses(comments)
        ctx.emit(
            {
                "stage": f"{len(result.comment_statuses)} prior comment(s) found",
                "type": "phase",
            }
        )

    result.stats = {
        "phase": "stub",
        "findings": len(result.findings),
        "anchor_method": method,
        "comments": len(result.comment_statuses),
    }
    ctx.emit({"stage": "Done", "type": "phase", "findings": len(result.findings)})
    return result
