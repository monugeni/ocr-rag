"""Backend ↔ agent boundary.

Builds a ``CheckContext`` from a stored run, invokes ``checker.agent.run_check``,
and persists the results (findings, comment statuses, annotated PDFs, run status).
The backend depends only on this module — not on the agent internals.
"""
from __future__ import annotations

import datetime
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from .agent.agent import run_check as agent_run_check
from .agent.schema import CheckContext, DocRef

from . import auth, config, events, store


def _emit(run_id: str):
    def emit(event: dict) -> None:
        events.publish(run_id, event)
        if "stage" in event:
            store.update_run(run_id, stage=str(event["stage"]))

    return emit


def _docrefs(uploads: list[dict], project: str, default_doc_type: str | None) -> list[DocRef]:
    refs = []
    for u in uploads:
        if u.get("doc_id") is None:
            continue
        refs.append(
            DocRef(
                doc_id=u["doc_id"],
                project=project,
                pdf_path=u["disk_path"],
                role=u["role"],
                title=Path(u["filename"]).stem,
                document_type=default_doc_type,
                page_count=u.get("page_count"),
            )
        )
    return refs


def build_context(run_id: str) -> CheckContext:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run {run_id} not found")
    ups = run["uploads"]
    project = run["ocrrag_project"]

    submitted = _docrefs(
        [u for u in ups if u["role"] == "submitted"], project, run.get("document_type")
    )
    references = _docrefs([u for u in ups if u["role"] == "reference"], project, None)

    old_commented = None
    prior = [u for u in ups if u["role"] == "prior_commented"]
    if prior:
        p = prior[0]
        old_commented = DocRef(
            doc_id=p.get("doc_id") or -1,
            project=project,
            pdf_path=p["disk_path"],
            role="prior_commented",
            title=Path(p["filename"]).stem,
        )

    # Reference folders: a JSON list (additive) or a legacy single string.
    ref_raw = run.get("reference_project")
    reference_projects: list[str] = []
    if ref_raw:
        try:
            parsed = json.loads(ref_raw)
            reference_projects = parsed if isinstance(parsed, list) else [ref_raw]
        except (ValueError, TypeError):
            reference_projects = [ref_raw]
    reference_projects = [p for p in reference_projects if p]

    annotated_dir = Path(config.ANNOTATED_DIR) / run_id

    run_meta = json.loads(run["metadata"]) if run.get("metadata") else {}
    # Default to high: the checker is recall-critical and medium was shown to
    # under-report (grok-4.3 found 1 vs Opus 10 on the same doc at medium). For
    # grok this maps to reasoning_effort "high" (its max).
    effort = run_meta.get("effort") or "high"
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        effort = "high"

    # Provider selection (anthropic | grok). Per-run metadata wins over the
    # server default so two runs can be compared head-to-head.
    provider = (run_meta.get("provider") or config.CHECKER_PROVIDER or "anthropic").strip().lower()
    if provider not in ("anthropic", "grok"):
        provider = "anthropic"
    if provider == "grok":
        api_key = config.XAI_API_KEY or None
        model = config.CHECKER_GROK_MODEL
        fast_model = config.CHECKER_GROK_FAST_MODEL
        base_url = config.XAI_BASE_URL
    else:
        api_key = config.ANTHROPIC_API_KEY or None
        model = config.CHECKER_MODEL
        fast_model = config.CHECKER_FAST_MODEL
        base_url = None

    template_instructions = None
    if run.get("template_id"):
        tmpl = store.get_template(run["template_id"])
        if tmpl:
            template_instructions = tmpl["instructions"]

    return CheckContext(
        run_id=run_id,
        submitted=submitted,
        references=references,
        reference_project=reference_projects[0] if reference_projects else None,
        reference_projects=reference_projects,
        company_mcp_url=config.COMPANY_MCP_URL,
        old_commented=old_commented,
        metadata={
            "project_number": run["project_number"],
            "document_type": run.get("document_type"),
            "originator": run.get("originator"),
            **run_meta,
        },
        template_instructions=template_instructions,
        guiding_prompt=run.get("guiding_prompt"),
        is_revision=bool(run.get("is_revision")),
        annotated_out_dir=str(annotated_dir),
        mcp_servers={},
        emit=_emit(run_id),
        docs_db_path=config.DOCS_DB,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        model=model,
        fast_model=fast_model,
        effort=effort,
        live=config.CHECKER_LIVE_AGENT,
    )


def run_check_for_run(run_id: str, user_id: int | None = None) -> None:
    """Execute a check run synchronously (called inside the run executor)."""
    store.update_run(run_id, status="running", started_at=_now())
    auth.record_audit("run_started", user_id=user_id, run_id=run_id)
    try:
        ctx = build_context(run_id)
        result = agent_run_check(ctx)

        # Record per-user spend for this run's LLM usage (best-effort).
        try:
            from . import usage as usage_mod
            email = auth.get_user_email(user_id)
            for model, toks in (result.usage or {}).items():
                usage_mod.record(email, "check", model, toks)
        except Exception:  # noqa: BLE001 — telemetry must never fail a run
            pass

        # Persist the debug trace next to the annotated PDF (best-effort).
        try:
            trace_dir = Path(config.ANNOTATED_DIR) / run_id
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "trace.json").write_text(
                json.dumps(result.trace or {}, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

        for f in result.findings:
            store.add_finding(run_id, asdict(f))
        for c in result.comment_statuses:
            store.add_comment_result(run_id, asdict(c))

        store.update_run(
            run_id,
            status="done",
            stage="Done",
            finished_at=_now(),
            error=None,
        )
        events.publish(run_id, {"type": "complete", "findings": len(result.findings)})
        auth.record_audit(
            "run_completed",
            user_id=user_id,
            run_id=run_id,
            payload={"findings": len(result.findings), "warnings": result.warnings},
        )
    except Exception as exc:  # noqa: BLE001
        store.update_run(run_id, status="failed", finished_at=_now(), error=str(exc))
        events.publish(run_id, {"type": "error", "error": str(exc)})
        auth.record_audit(
            "run_failed",
            user_id=user_id,
            run_id=run_id,
            payload={"error": str(exc), "traceback": traceback.format_exc()[-2000:]},
        )


def _now() -> str:
    # Stored as text; SQLite CURRENT_TIMESTAMP-compatible.
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
