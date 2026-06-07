from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


MAX_HISTORY_MESSAGES = 8
MAX_TOOL_ROUNDS = 20
MAX_TOOL_RESULT_CHARS = 12000
MAX_STRING_CHARS = 4000
SOURCE_SNIPPET_CHARS = 280
MCP_CONNECT_RETRIES = 5
MCP_CONNECT_DELAY_SECONDS = 0.35
MAX_REPEATED_TOOL_CALLS = 2
MAX_STALE_TOOL_ROUNDS = 3


def _semantic_tools_enabled() -> bool:
    return os.environ.get("OCR_RAG_ENABLE_SEMANTIC_FALLBACK", "").lower() in {
        "1", "true", "yes", "on",
    }

PROJECT_SCOPED_TOOLS = {
    "list_folder_entries",
    "list_documents",
    "ranked_search",
    "search_chunks",
    "search_pages",
    "search_sections",
    "semantic_search",
}

ALLOWED_TOOLS = {
    "list_folder_entries",
    "list_documents",
    "get_document_info",
    "get_toc",
    "ranked_search",
    "search_chunks",
    "search_pages",
    "search_sections",
    "semantic_search",
    "get_section",
    "get_page",
    "get_pages",
    "read_document",
    "read_document_chunks",
    "get_adjacent",
    "render_page_image",
    "reextract_page",
    "reextract_table",
}


@dataclass
class ChatRunResult:
    answer: str
    sources: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


def _acc_usage(acc: dict[str, int], usage: Any) -> None:
    """Accumulate one Anthropic response.usage into a running per-turn total."""
    if not usage:
        return
    acc["input"] = acc.get("input", 0) + (getattr(usage, "input_tokens", 0) or 0)
    acc["output"] = acc.get("output", 0) + (getattr(usage, "output_tokens", 0) or 0)
    acc["cache_read"] = acc.get("cache_read", 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    acc["cache_write"] = acc.get("cache_write", 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)


def _acc_usage_oai(acc: dict[str, int], usage: Any) -> None:
    """Accumulate one xAI/OpenAI ``usage`` dict into the running per-turn total."""
    if not usage:
        return
    import grok_client

    for key, value in grok_client.normalize_usage(usage).items():
        acc[key] = acc.get(key, 0) + value


@dataclass
class SourceTracker:
    next_id: int = 1
    by_key: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    order: list[tuple[int, int]] = field(default_factory=list)

    def add(self, candidate: dict[str, Any]) -> dict[str, Any]:
        doc_id = int(candidate["doc_id"])
        page_num = int(candidate["page_num"])
        key = (doc_id, page_num)

        existing = self.by_key.get(key)
        if existing is None:
            existing = {
                "id": str(self.next_id),
                "doc_id": doc_id,
                "doc_title": candidate.get("doc_title") or f"Document {doc_id}",
                "folder": candidate.get("folder") or "",
                "page_num": page_num,
                "breadcrumb": candidate.get("breadcrumb") or "",
                "page_type": candidate.get("page_type") or "text",
                "snippet": candidate.get("snippet") or "",
                "search_type": candidate.get("search_type") or "mcp",
                "score": float(candidate.get("score") or 0.0),
            }
            self.by_key[key] = existing
            self.order.append(key)
            self.next_id += 1
        else:
            existing["doc_title"] = candidate.get("doc_title") or existing["doc_title"]
            existing["folder"] = candidate.get("folder") or existing["folder"]
            existing["breadcrumb"] = candidate.get("breadcrumb") or existing["breadcrumb"]
            existing["page_type"] = candidate.get("page_type") or existing["page_type"]
            existing["search_type"] = candidate.get("search_type") or existing["search_type"]
            existing["score"] = max(float(candidate.get("score") or 0.0), float(existing.get("score") or 0.0))

            snippet = candidate.get("snippet") or ""
            if snippet and (not existing.get("snippet") or len(snippet) > len(existing["snippet"])):
                existing["snippet"] = snippet

        return existing

    def as_list(self) -> list[dict[str, Any]]:
        return [self.by_key[key] for key in self.order]


def run_folder_chat(
    *,
    mcp_url: Optional[str],
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]] = None,
    mcp_server: Any = None,
    provider: str = "anthropic",
    base_url: Optional[str] = None,
) -> ChatRunResult:
    return asyncio.run(
        _run_folder_chat(
            mcp_url=mcp_url,
            api_key=api_key,
            model=model,
            project=project,
            question=question,
            history=history,
            attachment=attachment,
            mcp_server=mcp_server,
            provider=provider,
            base_url=base_url,
        )
    )


async def arun_folder_chat(
    *,
    mcp_url: Optional[str],
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]] = None,
    mcp_server: Any = None,
    provider: str = "anthropic",
    base_url: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> ChatRunResult:
    """Async entrypoint for the streaming chat endpoint. Identical to
    ``run_folder_chat`` but awaitable (so the caller can drain ``on_event``
    progress events on the same loop) and with no inner ``asyncio.run``."""
    return await _run_folder_chat(
        mcp_url=mcp_url,
        api_key=api_key,
        model=model,
        project=project,
        question=question,
        history=history,
        attachment=attachment,
        mcp_server=mcp_server,
        provider=provider,
        base_url=base_url,
        on_event=on_event,
    )


async def _dispatch_chat(
    provider: str,
    *,
    session: Any,
    api_key: str,
    model: str,
    base_url: Optional[str],
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    tools: list[dict[str, Any]],
    tracker: SourceTracker,
    usage: dict[str, int],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Run one chat turn against the selected provider. The MCP session, tool
    list, source tracking and result formatting are provider-agnostic; only the
    model call and its tool-block dialect differ."""
    if (provider or "anthropic").lower() == "grok":
        return await _chat_with_tools_grok(
            session=session, api_key=api_key, model=model, base_url=base_url,
            project=project, question=question, history=history,
            attachment=attachment, tools=tools, tracker=tracker, usage=usage,
            on_event=on_event,
        )
    return await _chat_with_tools(
        session=session, api_key=api_key, model=model, project=project,
        question=question, history=history, attachment=attachment,
        tools=tools, tracker=tracker, usage=usage, on_event=on_event,
    )


async def _run_folder_chat(
    *,
    mcp_url: Optional[str],
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    mcp_server: Any = None,
    provider: str = "anthropic",
    base_url: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> ChatRunResult:
    tracker = SourceTracker()
    usage: dict[str, int] = {}

    if mcp_server is not None:
        tools = await _list_allowed_tools(mcp_server)
        response = await _dispatch_chat(
            provider,
            session=mcp_server,
            api_key=api_key,
            model=model,
            base_url=base_url,
            project=project,
            question=question,
            history=history,
            attachment=attachment,
            tools=tools,
            tracker=tracker,
            usage=usage,
            on_event=on_event,
        )
        return ChatRunResult(answer=response, sources=tracker.as_list(), usage=usage, model=model)

    last_error: Optional[Exception] = None
    if not mcp_url:
        raise RuntimeError("No MCP server or MCP URL configured for chat.")

    for attempt in range(MCP_CONNECT_RETRIES):
        try:
            async with streamablehttp_client(mcp_url, timeout=10, sse_read_timeout=120) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await _initialize_session(session)
                    tools = await _list_allowed_tools(session)
                    response = await _dispatch_chat(
                        provider,
                        session=session,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                        project=project,
                        question=question,
                        history=history,
                        attachment=attachment,
                        tools=tools,
                        tracker=tracker,
                        usage=usage,
                        on_event=on_event,
                    )
            return ChatRunResult(answer=response, sources=tracker.as_list(), usage=usage, model=model)
        except Exception as exc:
            last_error = exc
            if attempt == MCP_CONNECT_RETRIES - 1:
                break
            await asyncio.sleep(MCP_CONNECT_DELAY_SECONDS)

    raise RuntimeError(f"Could not connect to MCP server: {last_error}") from last_error


async def _initialize_session(session: ClientSession) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(MCP_CONNECT_RETRIES):
        try:
            await session.initialize()
            return
        except Exception as exc:  # pragma: no cover - retry behavior
            last_error = exc
            if attempt == MCP_CONNECT_RETRIES - 1:
                break
            await asyncio.sleep(MCP_CONNECT_DELAY_SECONDS)
    raise RuntimeError(f"Could not connect to MCP server: {last_error}") from last_error


async def _list_allowed_tools(session: Any) -> list[dict[str, Any]]:
    selected = []
    result = await session.list_tools()
    tools = result.tools if hasattr(result, "tools") else result
    for tool in tools:
        if tool.name == "semantic_search" and not _semantic_tools_enabled():
            continue
        if tool.name not in ALLOWED_TOOLS:
            continue
        selected.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        })
    return selected


async def _chat_with_tools(
    *,
    session: Any,
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    tools: list[dict[str, Any]],
    tracker: SourceTracker,
    usage: Optional[dict[str, int]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    if usage is None:
        usage = {}
    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages = _build_messages(question=question, history=history, attachment=attachment)
    system_prompt = _build_system_prompt(project=project, attachment=attachment)
    seen_tool_calls: dict[tuple[str, str], int] = {}
    stale_rounds = 0
    previous_round_signature: Optional[tuple[tuple[str, str], ...]] = None

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.messages.create(
                model=model,
                max_tokens=12000,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=tools,
                # display:"summarized" so the thinking blocks carry text we can
                # stream live to the chat UI (Opus/Sonnet 4.x omit it otherwise).
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},
            )
            _acc_usage(usage, getattr(response, "usage", None))

            thinking = _collect_thinking(response.content)
            if thinking:
                _emit(on_event, {"type": "thinking", "text": thinking})

            assistant_blocks = [_dump_block(block) for block in response.content]
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [block for block in response.content if getattr(block, "type", "") == "tool_use"]
            for block in tool_uses:
                _emit(on_event, {"type": "tool", "name": block.name,
                                 "summary": _tool_call_summary(block.name, block.input)})
            if not tool_uses:
                answer = _collect_text(response.content)
                if not answer:
                    raise RuntimeError("LLM returned an empty response.")
                return answer

            tool_results = []
            repeated_loop = True
            tool_errors = 0
            sources_before = len(tracker.order)
            round_signature: list[tuple[str, str]] = []
            for block in tool_uses:
                arguments = _normalize_tool_arguments(block.name, block.input, project)
                call_key = (
                    block.name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=True, default=str),
                )
                round_signature.append(call_key)
                seen_tool_calls[call_key] = seen_tool_calls.get(call_key, 0) + 1
                if seen_tool_calls[call_key] <= MAX_REPEATED_TOOL_CALLS:
                    repeated_loop = False

                raw_result = await session.call_tool(block.name, arguments)
                payload = _coerce_tool_payload(raw_result)
                is_error = bool(getattr(raw_result, "isError", False))
                if is_error:
                    tool_errors += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _format_tool_result(
                        tool_name=block.name,
                        payload=payload,
                        tracker=tracker,
                        project=project,
                    ),
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})
            round_signature_key = tuple(round_signature)
            new_sources = len(tracker.order) - sources_before
            no_progress = new_sources <= 0 and (
                repeated_loop
                or tool_errors == len(tool_uses)
                or round_signature_key == previous_round_signature
            )
            stale_rounds = stale_rounds + 1 if no_progress else 0
            previous_round_signature = round_signature_key

            if repeated_loop or stale_rounds >= MAX_STALE_TOOL_ROUNDS:
                break
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    return await _force_final_answer(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        attachment=attachment,
        usage=usage,
    )


def _build_system_prompt(*, project: str, attachment: Optional[dict[str, Any]]) -> str:
    base = (
        "You are Esteem Tender Intelligence, a requirements analyst for complex EPC "
        "(engineering, procurement, construction) projects. Engineers rely on you to find, "
        "extract and verify the exact requirements stated in a client's tender and reference "
        "documents.\n"
        f"You work only within the folder '{project}' and its sub-folders. Never use outside "
        "knowledge or documents from any other folder.\n"
        "\n"
        "Scope of use: answer only official, work-related questions about these project/tender "
        "documents and their engineering content. Politely decline personal, casual, or off-topic "
        "requests (general knowledge, chit-chat, coding, opinions, anything unrelated to the folder's "
        "documents) with one short sentence redirecting the user to ask about the project documents. "
        "Do not answer the off-topic question even partially.\n"
        "\n"
        "How to answer:\n"
        "- Be exhaustive about requirements. Tenders state requirements across the scope of work, "
        "technical specifications, datasheets, particular and general conditions, standards, and "
        "annexures — look across all of them, not just the obvious section.\n"
        "- Quote the governing requirement verbatim and give its exact location: document, page, and "
        "clause/section number, plus the source IDs [1], [2] from this turn's tool results.\n"
        "- Distinguish mandatory requirements ('shall', 'must', 'minimum', 'not less than') from "
        "guidance or options ('should', 'may', 'preferred'). Say plainly where the tender is silent "
        "or ambiguous.\n"
        "- Context and applicability are decisive. Establish WHAT a clause governs before applying "
        "it. A requirement stated for one item, system or service does NOT automatically apply to "
        "another: stringent tests or materials for heater-coil pipes inside a fired heater do not "
        "govern external utility piping; a grade specified for dampers does not govern on-off valves; "
        "a thickness for one line class does not govern another. BUT honour genuinely general "
        "requirements — when the tender says 'all piping' or covers a whole class/service, it applies "
        "across that class unless a more specific clause overrides it (apply the stated order of "
        "precedence; flag conflicts rather than choosing). Read clause and document context, not "
        "keywords: a pipe 'bend' (a fitting) is not a road 'bend' nor a 'bend test' on a plate or "
        "weld coupon — match an item to a requirement only when subject, service, size/class and "
        "component type genuinely correspond. State what a requirement governs before asserting it.\n"
        "- Surface cross-references and conflicts: tenders refer out to codes/standards (IS, ASME, "
        "API, etc.) and to other documents. Where requirements conflict, point it out, cite both, and "
        "apply the stated order of precedence; if none is stated, flag it rather than choosing.\n"
        "- Be exact with numbers, units, ratings, tag numbers, quantities and material grades — never "
        "round or paraphrase a specification value.\n"
        "- If the documents do not support a confident answer, say so clearly, do not speculate, and "
        "suggest where that requirement would typically be found.\n"
        "- Respond in markdown. Ignore any numeric citations from prior chat history; only this turn's "
        "tool source IDs are valid.\n"
    )
    if not attachment:
        prompt = (
            "\nInvestigate with the MCP tools before answering.\n"
            "- Start with ranked_search for clauses, specifications, equipment and material names, tag "
            "numbers and exact phrases. Try multiple phrasings of the same requirement — synonyms, "
            "abbreviations and unit variants (e.g. 'barg' vs 'bar(g)', 'NPS' vs 'inch').\n"
            "- Use next_offset to page through the same ranked query when has_more=true; issue a new "
            "query when better terms become visible.\n"
            "- After a hit, read the page and nearby pages (get_page / get_pages) to capture the full "
            "clause and all its conditions, exceptions and provisos.\n"
            "- Use read_document_chunks with next_offset pagination to read or summarise a whole "
            "specification or scope document.\n"
            "- Use render_page_image for drawings, scanned pages, forms, datasheets, title blocks, or "
            "when the extracted text looks incomplete.\n"
            "- A missed requirement is a costly error, so before concluding a requirement is absent, "
            "search a few alternative terms first.\n"
        )
        if _semantic_tools_enabled():
            prompt += (
                "- Semantic search is a secondary recall aid only; do not treat it as the source of "
                "truth for technical values.\n"
            )
        return base + prompt + "Stop calling tools once you have enough evidence to answer completely.\n"

    return base + (
        "You are also reviewing an uploaded document excerpt against the folder documents.\n"
        "Uploaded document pages are already provided in the conversation as evidence IDs like [A1], [A2], etc.\n"
        "Use this structure unless a section would truly be empty:\n"
        "## Overall assessment\n"
        "## Matches\n"
        "## Gaps or missing evidence\n"
        "## Conflicts or risks\n"
        "## Suggested next checks\n"
        "## Sources\n"
        "Cite uploaded pages with [A1], [A2], etc. and folder documents with [1], [2], etc.\n"
        "Do not keep calling tools once you have enough evidence to complete the review.\n"
    )


async def _force_final_answer(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    usage: Optional[dict[str, int]] = None,
) -> str:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    final_messages = list(messages)
    final_messages.append({
        "role": "user",
        "content": (
            "Stop using tools and answer now using only the evidence already gathered in the conversation. "
            "If the gathered evidence is insufficient, say that clearly instead of speculating."
        ),
    })

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=12000,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=final_messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            cache_control={"type": "ephemeral"},
        )
        if usage is not None:
            _acc_usage(usage, getattr(response, "usage", None))
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    answer = _collect_text(response.content)
    if answer:
        return answer

    if attachment:
        return (
            "I gathered some evidence from the folder and the uploaded document, but I still could not "
            "produce a stable final review. Please narrow the question or ask about a smaller section."
        )

    return (
        "I gathered evidence from the folder documents, but I still could not produce a stable final answer. "
        "Please narrow the question or ask about a smaller section."
    )


async def _chat_with_tools_grok(
    *,
    session: Any,
    api_key: str,
    model: str,
    base_url: Optional[str],
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    tools: list[dict[str, Any]],
    tracker: SourceTracker,
    usage: Optional[dict[str, int]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Grok (xAI) equivalent of ``_chat_with_tools`` — same MCP tools, source
    tracking and stale-loop guards, but the OpenAI tool-calling dialect.

    The single growing ``messages`` list lets xAI's automatic prompt caching
    reuse the prefix across rounds, mirroring the Anthropic path's cache_control."""
    import grok_client

    if usage is None:
        usage = {}
    system_prompt = _build_system_prompt(project=project, attachment=attachment)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_build_messages(question=question, history=history, attachment=attachment))
    oai_tools = grok_client.to_openai_tools(tools)
    # One conversation id for the whole turn so xAI's automatic prompt caching
    # keeps the growing prefix on the same server across rounds. Reasoning at
    # "medium" mirrors the Anthropic chat's effort=medium for a fair A/B.
    conv_id = grok_client.new_conv_id()

    seen_tool_calls: dict[tuple[str, str], int] = {}
    stale_rounds = 0
    previous_round_signature: Optional[tuple[tuple[str, str], ...]] = None

    for _ in range(MAX_TOOL_ROUNDS):
        resp = await grok_client.acreate(
            api_key=api_key,
            base_url=base_url,
            conv_id=conv_id,
            payload={
                "model": model,
                "max_tokens": 12000,
                "messages": messages,
                "tools": oai_tools,
                "tool_choice": "auto",
                "reasoning_effort": "medium",
            },
        )
        _acc_usage_oai(usage, resp.get("usage"))
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning:
            _emit(on_event, {"type": "thinking", "text": reasoning})
        for block in tool_calls:
            fn = block.get("function") or {}
            try:
                _args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                _args = {}
            _emit(on_event, {"type": "tool", "name": fn.get("name") or "",
                             "summary": _tool_call_summary(fn.get("name") or "", _args)})

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            if not answer:
                raise RuntimeError("LLM returned an empty response.")
            return answer

        # Every tool_call must get a matching tool message BEFORE any other
        # message type, so collect tool results first and append image follow-ups
        # (vision) only after all tool messages are in place.
        tool_messages: list[dict[str, Any]] = []
        image_messages: list[dict[str, Any]] = []
        round_signature: list[tuple[str, str]] = []
        repeated_loop = True
        tool_errors = 0
        sources_before = len(tracker.order)

        for block in tool_calls:
            fn = block.get("function") or {}
            name = fn.get("name") or ""
            try:
                raw_args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                raw_args = {}
            arguments = _normalize_tool_arguments(name, raw_args, project)
            call_key = (
                name,
                json.dumps(arguments, sort_keys=True, ensure_ascii=True, default=str),
            )
            round_signature.append(call_key)
            seen_tool_calls[call_key] = seen_tool_calls.get(call_key, 0) + 1
            if seen_tool_calls[call_key] <= MAX_REPEATED_TOOL_CALLS:
                repeated_loop = False

            try:
                raw_result = await session.call_tool(name, arguments)
                payload = _coerce_tool_payload(raw_result)
                is_error = bool(getattr(raw_result, "isError", False))
            except Exception as exc:  # noqa: BLE001
                payload = {"error": str(exc)}
                is_error = True
            if is_error:
                tool_errors += 1

            content = _format_tool_result(
                tool_name=name, payload=payload, tracker=tracker, project=project,
            )
            text, image_url = grok_client.split_tool_content(content)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("id"),
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

        messages.extend(tool_messages)
        messages.extend(image_messages)

        round_signature_key = tuple(round_signature)
        new_sources = len(tracker.order) - sources_before
        no_progress = new_sources <= 0 and (
            repeated_loop
            or tool_errors == len(tool_calls)
            or round_signature_key == previous_round_signature
        )
        stale_rounds = stale_rounds + 1 if no_progress else 0
        previous_round_signature = round_signature_key

        if repeated_loop or stale_rounds >= MAX_STALE_TOOL_ROUNDS:
            break

    return await _force_final_answer_grok(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        attachment=attachment,
        usage=usage,
        conv_id=conv_id,
    )


async def _force_final_answer_grok(
    *,
    api_key: str,
    base_url: Optional[str],
    model: str,
    messages: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    usage: Optional[dict[str, int]] = None,
    conv_id: Optional[str] = None,
) -> str:
    import grok_client

    final_messages = list(messages)
    final_messages.append({
        "role": "user",
        "content": (
            "Stop using tools and answer now using only the evidence already gathered in the conversation. "
            "If the gathered evidence is insufficient, say that clearly instead of speculating."
        ),
    })
    resp = await grok_client.acreate(
        api_key=api_key,
        base_url=base_url,
        conv_id=conv_id,
        payload={
            "model": model,
            "max_tokens": 12000,
            "messages": final_messages,
            "reasoning_effort": "medium",
        },
    )
    if usage is not None:
        _acc_usage_oai(usage, resp.get("usage"))

    answer = grok_client.message_text(resp)
    if answer:
        return answer

    if attachment:
        return (
            "I gathered some evidence from the folder and the uploaded document, but I still could not "
            "produce a stable final review. Please narrow the question or ask about a smaller section."
        )
    return (
        "I gathered evidence from the folder documents, but I still could not produce a stable final answer. "
        "Please narrow the question or ask about a smaller section."
    )


def _build_messages(
    *,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    user_parts = [f"Latest user question:\n{question.strip()}"]
    if attachment:
        excerpt = _attachment_excerpt(attachment)
        user_parts.append(
            "Uploaded document excerpt for comparison:\n"
            f"{excerpt}"
        )
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def _attachment_excerpt(attachment: dict[str, Any]) -> str:
    parts = [
        f"Filename: {attachment.get('filename', 'document')}",
        f"Page count: {attachment.get('page_count', 0)}",
    ]
    pages = attachment.get("pages") or []
    if pages:
        page_blocks = []
        for page in pages:
            title = f"[{page['id']}] page {page['page_num']}"
            if page.get("breadcrumb"):
                title += f" | {page['breadcrumb']}"
            page_blocks.append(f"{title}\n{page['content']}")
        parts.append("\n\n".join(page_blocks))
    return "\n".join(parts).strip()


def _normalize_tool_arguments(name: str, arguments: Any, project: str) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        arguments = {}
    normalized = dict(arguments)
    if name in PROJECT_SCOPED_TOOLS:
        normalized["project"] = project
    return normalized


def _coerce_tool_payload(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        _content, structured = result
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    structured = getattr(result, "structuredContent", None)
    if structured not in (None, {}):
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    text_blocks = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text_blocks.append(getattr(block, "text", ""))

    text = "\n".join(part for part in text_blocks if part).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


def _format_tool_result(
    *,
    tool_name: str,
    payload: Any,
    tracker: SourceTracker,
    project: str,
) -> Any:
    annotations = []
    for source in _extract_sources(payload, tool_name=tool_name, project=project):
        tracked = tracker.add(source)
        annotations.append(
            f"[{tracked['id']}] {tracked['doc_title']} | {tracked['folder'] or project} | "
            f"page {tracked['page_num']}"
            + (f" | {tracked['breadcrumb']}" if tracked.get("breadcrumb") else "")
        )

    trimmed = _trim_payload(payload)
    body = json.dumps(trimmed, indent=2, ensure_ascii=True)
    if len(body) > MAX_TOOL_RESULT_CHARS:
        body = body[:MAX_TOOL_RESULT_CHARS].rstrip() + "\n...[truncated]"

    if annotations:
        text = (
            f"Tool: {tool_name}\n"
            "Page-based source IDs from this tool result:\n"
            + "\n".join(f"- {line}" for line in annotations)
            + "\n\nStructured result:\n"
            + body
        )
    else:
        text = f"Tool: {tool_name}\nStructured result:\n{body}"

    image_block = _image_block_from_payload(payload)
    if image_block:
        return [
            {"type": "text", "text": text},
            image_block,
        ]
    return text


def _image_block_from_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    image_base64 = payload.get("image_base64")
    mime_type = payload.get("mime_type") or "image/png"
    if not image_base64:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": image_base64,
        },
    }


def _extract_sources(payload: Any, *, tool_name: str, project: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("error"):
        return []

    sources = []
    top_level = _page_source_from_record(payload, payload, tool_name=tool_name, project=project)
    if top_level:
        sources.append(top_level)

    if isinstance(payload.get("pages"), list):
        for page in payload["pages"]:
            source = _page_source_from_record(page, payload, tool_name=tool_name, project=project)
            if source:
                sources.append(source)

    if isinstance(payload.get("chunks"), list):
        for chunk in payload["chunks"]:
            source = _page_source_from_record(chunk, payload, tool_name=tool_name, project=project)
            if source:
                sources.append(source)

    if isinstance(payload.get("results"), list):
        for row in payload["results"]:
            source = _page_source_from_record(row, payload, tool_name=tool_name, project=project)
            if source:
                sources.append(source)

    for neighbor_name in ("prev_page", "next_page"):
        neighbor = payload.get(neighbor_name)
        if isinstance(neighbor, dict):
            neighbor_source = _page_source_from_record(neighbor, payload, tool_name=tool_name, project=project)
            if neighbor_source:
                sources.append(neighbor_source)

    deduped = {}
    for source in sources:
        deduped[(int(source["doc_id"]), int(source["page_num"]))] = source
    return list(deduped.values())


def _page_source_from_record(
    record: dict[str, Any],
    container: dict[str, Any],
    *,
    tool_name: str,
    project: str,
) -> Optional[dict[str, Any]]:
    doc_id = record.get("doc_id", container.get("doc_id"))
    doc_title = record.get("doc_title", container.get("doc_title"))
    if doc_title is None:
        doc_title = container.get("title")

    page_num = record.get("page_num")
    if page_num is None and "page_start" in record:
        page_num = record.get("page_start")

    if doc_id is None or doc_title is None or page_num is None:
        return None

    snippet = (
        record.get("snippet")
        or record.get("chunk_preview")
        or record.get("heading")
        or record.get("markdown")
        or record.get("content")
        or ""
    )
    snippet = _snippet(snippet)

    score = record.get("rank")
    if score is None:
        score = record.get("similarity")

    breadcrumb = record.get("breadcrumb")
    if not breadcrumb and "section" in container and isinstance(container["section"], dict):
        breadcrumb = container["section"].get("breadcrumb")

    return {
        "doc_id": int(doc_id),
        "doc_title": str(doc_title),
        "folder": str(record.get("project") or container.get("project") or project or ""),
        "page_num": int(page_num),
        "breadcrumb": str(breadcrumb or ""),
        "page_type": str(record.get("page_type") or "text"),
        "snippet": snippet,
        "search_type": tool_name,
        "score": float(score or 0.0),
    }


def _snippet(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= SOURCE_SNIPPET_CHARS:
        return text
    return text[: SOURCE_SNIPPET_CHARS - 3].rstrip() + "..."


def _trim_payload(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= MAX_STRING_CHARS:
            return value
        return value[:MAX_STRING_CHARS].rstrip() + "\n...[truncated]"
    if isinstance(value, list):
        return [_trim_payload(item) for item in value]
    if isinstance(value, dict):
        trimmed = {}
        for key, item in value.items():
            if key == "image_base64":
                trimmed[key] = "[base64 omitted from text; image attached to tool result]"
            else:
                trimmed[key] = _trim_payload(item)
        return trimmed
    return value


def _dump_block(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if isinstance(block, dict):
        return block
    raise TypeError(f"Unsupported content block: {type(block)!r}")


def _collect_text(blocks: list[Any]) -> str:
    parts = []
    for block in blocks:
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def _collect_thinking(blocks: list[Any]) -> str:
    """Pull summarized thinking text out of an Anthropic response's content blocks.

    Only populated when the request opts in with ``thinking.display = "summarized"``
    (Opus/Sonnet 4.x omit thinking text by default)."""
    parts = []
    for block in blocks:
        if getattr(block, "type", "") == "thinking":
            parts.append(getattr(block, "thinking", "") or "")
    return "".join(parts).strip()


def _tool_call_summary(name: str, arguments: Any) -> str:
    """A short 'tool: key argument' line for the live chat progress trace."""
    args = arguments if isinstance(arguments, dict) else {}
    for key in ("query", "q", "text", "title", "name"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return f"{name}: {val.strip()[:80]}"
    for key in ("page_num", "page", "doc_id", "document_id", "section_id"):
        if args.get(key) is not None:
            return f"{name}: {key}={args[key]}"
    return name


def _emit(on_event: Optional[Callable[[dict], None]], event: dict) -> None:
    """Fire a progress event for the streaming chat UI; never let it break a run."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:  # noqa: BLE001 — progress is best-effort
        pass
