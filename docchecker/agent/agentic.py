"""Agentic, page-by-page document checker.

Sweeps the submitted document one page at a time in a SINGLE cumulative,
prompt-cached conversation. For each page the model searches the reference KB
(the company ocr-rag MCP) on demand for the governing requirements, checks the
page, and reports that page's findings via a ``report_findings`` tool. Because
it is one growing conversation, requirements retrieved on an earlier page stay
in context (read from cache) when a later page refers back.

Mirrors the proven cached tool-use loop in ``chat_mcp_runner`` (same MCP
connection, tool dispatch, and prompt-cache placement).

v1 scope: core sweep + caching. Server-side compaction (for 100s of pages) and a
submission re-lookup tool are deferred — see the design spec. Relies on Opus's
1M context window for now.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import chat_mcp_runner as chat  # reuse the proven cached tool-loop helpers

from .docs_reader import read_pages
from .prompts import build_system_prompt
from .schema import CheckContext, Finding

# Per-page cap on tool rounds so a single page can't loop forever.
MAX_ROUNDS_PER_PAGE = 14
MAX_TOKENS = 12000  # mirrors chat_mcp_runner (non-streaming create + adaptive thinking)


def _report_findings_tool(findings_schema: dict) -> dict:
    return {
        "name": "report_findings",
        "description": (
            "Report every finding for the CURRENT page. Call exactly once per page after you "
            "have checked it against the tender — with an empty list if the page has no issues."
        ),
        "input_schema": findings_schema,
    }


def _system_prompt(ctx: CheckContext, uploaded_ref_text: str) -> str:
    base = build_system_prompt(ctx)
    agentic = (
        "\n\n# How this review works\n"
        "You are checking a SUBMITTED document against the company's tender / reference "
        "requirements. You will receive the submitted document ONE PAGE AT A TIME. For each page:\n"
        "1. Use the search tools (ranked_search, search_chunks, search_pages, get_page, "
        "read_document_chunks, …) to find the governing requirements in the reference knowledge "
        "base for what is on this page — specs, clauses, quantities, materials, dimensions, "
        "ratings, tags. Search with several phrasings; a missed requirement is a costly error.\n"
        "2. Check the page against those requirements.\n"
        "3. Call report_findings ONCE with every deviation, unmet requirement, missing item, "
        "inconsistency, BOM/quantity mismatch and dimensional mismatch on this page (empty list "
        "if none). Be exhaustive — a later step verifies findings, so do not self-censor "
        "borderline ones. Copy submitted_anchor verbatim from the page.\n"
        "\n"
        "CRITICAL — report what you notice. If, while reasoning, you spot ANY anomaly, you MUST put "
        "it in report_findings; never leave a problem only in your private reasoning. This includes "
        "issues you can judge WITHOUT finding a reference clause:\n"
        "  - internal inconsistencies: a cited standard that doesn't match the product (e.g. IS:7098 "
        "is for XLPE-insulated cable — flag it if the cable is PVC/other), a type/designation that "
        "contradicts the description (e.g. AYY vs 2XWY vs A2XFY), a value that contradicts another on "
        "the page or an earlier page;\n"
        "  - typos / printing errors in standards, dates, grades, ratings, tag or document numbers "
        "(e.g. 'IS:7098 (PT-1)/2025' where the real standard year differs) — report each occurrence;\n"
        "  - test reports/certificates: a missing required test, a result outside the stated limit, a "
        "wrong unit, an unsigned/undated certificate, a sample that doesn't match the ordered item.\n"
        "If you could not find the governing reference for an item, still report the anomaly as a "
        "finding with confidence 'low' and say the reference wasn't located — do NOT drop it. A single "
        "exhaustive page often yields several findings; reporting only one or none means you are "
        "under-reporting.\n"
        "Earlier pages and the requirements you already retrieved remain in this conversation — "
        "use them when a later page refers back to an earlier value.\n"
        "Then I will send the next page."
    )
    if uploaded_ref_text:
        agentic += (
            "\n\n# Uploaded reference document(s)\n"
            "These were supplied for this run in addition to the reference knowledge base:\n"
            + uploaded_ref_text
        )
    return base + agentic


def _page_message(page_num: int, total: int, title: str, text: str) -> str:
    return (
        f"PAGE {page_num} of {total} of the submitted document '{title}':\n{text}\n\n"
        "Search the reference KB for the requirements governing this page, then call "
        "report_findings with this page's findings (empty list if none)."
    )


def _collect_thinking(content: Any) -> str:
    parts = []
    for b in content:
        if getattr(b, "type", None) == "thinking":
            parts.append(getattr(b, "thinking", "") or "")
    return "".join(parts).strip()


async def _page_loop(client, ctx, llm, submitted, pages, system, tools, session, ref_scope, findings, tracker):
    """Sweep every page in one cumulative cached conversation. ``session`` is the
    MCP client session for reference search (or None for an uploaded-refs-only
    run, where the only tool is report_findings)."""
    from .pipeline import _finding_from_dict

    messages: list[dict] = []
    total = len(pages)
    for idx, page in enumerate(pages, start=1):
        page_num = page["page_num"]
        ctx.emit({"stage": f"Checking page {page_num} ({idx}/{total})…", "type": "phase"})
        messages.append({
            "role": "user",
            "content": _page_message(page_num, total, submitted.title, page.get("content") or ""),
        })
        before = len(findings)
        reported = False
        for _ in range(MAX_ROUNDS_PER_PAGE):
            resp = await client.messages.create(
                model=ctx.model,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": ctx.effort},
                cache_control={"type": "ephemeral"},  # cache the growing conversation prefix
            )
            llm._record_usage(resp.usage, ctx.model)  # feed spend + trace usage
            think = _collect_thinking(resp.content)
            if think:
                ctx.emit({"type": "thinking", "delta": f"[page {page_num}] {think}\n"})
            messages.append({"role": "assistant", "content": [chat._dump_block(b) for b in resp.content]})

            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                break  # model answered without a tool call → page done
            tool_results = []
            for b in tool_uses:
                if b.name == "report_findings":
                    for f in (b.input.get("findings") or []):
                        findings.append(_finding_from_dict(f, submitted, "compliance"))
                    reported = True
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": b.id,
                        "content": "Recorded. Continue to the next page.",
                    })
                elif session is not None:
                    args = chat._normalize_tool_arguments(b.name, b.input, ref_scope)
                    raw = await session.call_tool(b.name, args)
                    payload = chat._coerce_tool_payload(raw)
                    content = chat._format_tool_result(
                        tool_name=b.name, payload=payload, tracker=tracker, project=ref_scope,
                    )
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": b.id, "content": content,
                        "is_error": bool(getattr(raw, "isError", False)),
                    })
                else:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": b.id,
                        "content": "Search is unavailable for this run; rely on the provided references.",
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})
            if reported:
                break
        ctx.emit({"stage": f"Page {page_num}: {len(findings) - before} finding(s)", "type": "phase"})


async def _sweep_async(ctx: CheckContext, llm, submitted, pages, uploaded_ref_text: str) -> list[Finding]:
    from .pipeline import _dedupe_findings, _FINDINGS_TOOL

    client = anthropic.AsyncAnthropic(api_key=ctx.api_key)
    ref_projects = ctx.reference_projects or ([ctx.reference_project] if ctx.reference_project else [])
    ref_scope = ref_projects[0] if ref_projects else ""
    system = _system_prompt(ctx, uploaded_ref_text)
    findings: list[Finding] = []
    tracker = chat.SourceTracker()
    report_tool = _report_findings_tool(_FINDINGS_TOOL)
    have_mcp = bool(ctx.company_mcp_url and ref_projects)

    try:
        if have_mcp:
            async with streamablehttp_client(ctx.company_mcp_url, timeout=15, sse_read_timeout=120) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    tools = (await chat._list_allowed_tools(session)) + [report_tool]
                    await _page_loop(client, ctx, llm, submitted, pages, system, tools, session, ref_scope, findings, tracker)
        else:
            # No reference KB to search — check each page against the uploaded
            # reference text carried in the (cached) system prompt.
            await _page_loop(client, ctx, llm, submitted, pages, system, [report_tool], None, "", findings, tracker)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass

    return _dedupe_findings(findings)


def agentic_sweep(ctx: CheckContext, llm, submitted, uploaded_ref_text: str = "") -> list[Finding]:
    """Run the page-by-page agentic sweep and return deduped candidate findings.

    Falls back to an empty list (the run continues, recording a warning upstream)
    if the reference MCP is unavailable — the caller already isolates failures."""
    pages = read_pages(ctx.docs_db_path, submitted.doc_id)
    if not pages:
        return []
    if (ctx.provider or "anthropic").lower() == "grok":
        return asyncio.run(_sweep_async_grok(ctx, llm, submitted, pages, uploaded_ref_text))
    return asyncio.run(_sweep_async(ctx, llm, submitted, pages, uploaded_ref_text))


# ---------------------------------------------------------------------------
# Grok (xAI) sweep — OpenAI-compatible mirror of the Anthropic path above.
# Same single growing conversation (xAI auto-caches the prefix), same MCP search
# tools and report_findings interception; only the tool-call dialect differs.
# ---------------------------------------------------------------------------
def _report_findings_function(findings_schema: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": (
                "Report every finding for the CURRENT page. Call exactly once per page after you "
                "have checked it against the tender — with an empty list if the page has no issues."
            ),
            "parameters": findings_schema,
        },
    }


async def _page_loop_grok(ctx, llm, submitted, pages, system, tools, session, ref_scope, findings, tracker, conv_id):
    """Grok page-by-page sweep in one cumulative conversation. ``session`` is the
    MCP client session for reference search (or None for an uploaded-refs-only run).
    ``conv_id`` pins xAI sticky routing so the growing prefix hits the cache."""
    import grok_client

    from .pipeline import _finding_from_dict

    effort = grok_client.reasoning_effort(ctx.effort)
    messages: list[dict] = [{"role": "system", "content": system}]
    total = len(pages)
    for idx, page in enumerate(pages, start=1):
        page_num = page["page_num"]
        ctx.emit({"stage": f"Checking page {page_num} ({idx}/{total})…", "type": "phase"})
        messages.append({
            "role": "user",
            "content": _page_message(page_num, total, submitted.title, page.get("content") or ""),
        })
        before = len(findings)
        reported = False
        for _ in range(MAX_ROUNDS_PER_PAGE):
            resp = await grok_client.acreate(
                api_key=ctx.api_key,
                base_url=ctx.base_url,
                conv_id=conv_id,
                payload={
                    "model": ctx.model,
                    "max_tokens": MAX_TOKENS,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "reasoning_effort": effort,
                },
            )
            llm._record_usage(resp.get("usage"), ctx.model)  # feed spend + trace usage
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            reasoning = msg.get("reasoning_content")
            if reasoning:
                ctx.emit({"type": "thinking", "delta": f"[page {page_num}] {reasoning}\n"})

            tool_calls = msg.get("tool_calls") or []
            assistant_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                break  # model answered without a tool call → page done

            tool_messages: list[dict] = []
            image_messages: list[dict] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                if name == "report_findings":
                    for f in (args.get("findings") or []):
                        findings.append(_finding_from_dict(f, submitted, "compliance"))
                    reported = True
                    tool_messages.append({
                        "role": "tool", "tool_call_id": tc.get("id"),
                        "content": "Recorded. Continue to the next page.",
                    })
                elif session is not None:
                    arguments = chat._normalize_tool_arguments(name, args, ref_scope)
                    try:
                        raw = await session.call_tool(name, arguments)
                        payload = chat._coerce_tool_payload(raw)
                    except Exception as exc:  # noqa: BLE001
                        payload = {"error": str(exc)}
                    content = chat._format_tool_result(
                        tool_name=name, payload=payload, tracker=tracker, project=ref_scope,
                    )
                    text, image_url = grok_client.split_tool_content(content)
                    tool_messages.append({
                        "role": "tool", "tool_call_id": tc.get("id"),
                        "content": text or "(no result)",
                    })
                    if image_url:
                        image_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Image returned by {name}:"},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        })
                else:
                    tool_messages.append({
                        "role": "tool", "tool_call_id": tc.get("id"),
                        "content": "Search is unavailable for this run; rely on the provided references.",
                    })
            messages.extend(tool_messages)
            messages.extend(image_messages)
            if reported:
                break
        ctx.emit({"stage": f"Page {page_num}: {len(findings) - before} finding(s)", "type": "phase"})


async def _sweep_async_grok(ctx: CheckContext, llm, submitted, pages, uploaded_ref_text: str) -> list[Finding]:
    import grok_client

    from .pipeline import _dedupe_findings, _FINDINGS_TOOL

    ref_projects = ctx.reference_projects or ([ctx.reference_project] if ctx.reference_project else [])
    ref_scope = ref_projects[0] if ref_projects else ""
    system = _system_prompt(ctx, uploaded_ref_text)
    findings: list[Finding] = []
    tracker = chat.SourceTracker()
    report_tool = _report_findings_function(_FINDINGS_TOOL)
    have_mcp = bool(ctx.company_mcp_url and ref_projects)
    conv_id = grok_client.new_conv_id()  # one id for the whole cumulative sweep

    if have_mcp:
        async with streamablehttp_client(ctx.company_mcp_url, timeout=15, sse_read_timeout=120) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = grok_client.to_openai_tools(await chat._list_allowed_tools(session)) + [report_tool]
                await _page_loop_grok(ctx, llm, submitted, pages, system, tools, session, ref_scope, findings, tracker, conv_id)
    else:
        # No reference KB to search — check each page against the uploaded
        # reference text carried in the system prompt.
        await _page_loop_grok(ctx, llm, submitted, pages, system, [report_tool], None, "", findings, tracker, conv_id)

    return _dedupe_findings(findings)
