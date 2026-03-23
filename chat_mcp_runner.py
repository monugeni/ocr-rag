from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client


MAX_HISTORY_MESSAGES = 8
MAX_TOOL_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 12000
MAX_STRING_CHARS = 4000
SOURCE_SNIPPET_CHARS = 280
MCP_CONNECT_RETRIES = 5
MCP_CONNECT_DELAY_SECONDS = 0.35

PROJECT_SCOPED_TOOLS = {
    "list_folder_entries",
    "list_documents",
    "search_pages",
    "search_sections",
    "semantic_search",
}

ALLOWED_TOOLS = {
    "list_folder_entries",
    "list_documents",
    "get_document_info",
    "get_toc",
    "search_pages",
    "search_sections",
    "semantic_search",
    "get_section",
    "get_page",
    "get_pages",
    "get_adjacent",
    "reextract_page",
    "reextract_table",
}


@dataclass
class ChatRunResult:
    answer: str
    sources: list[dict[str, Any]]


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
    mcp_url: str,
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]] = None,
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
        )
    )


async def _run_folder_chat(
    *,
    mcp_url: str,
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
) -> ChatRunResult:
    tracker = SourceTracker()
    last_error: Optional[Exception] = None

    for attempt in range(MCP_CONNECT_RETRIES):
        try:
            async with sse_client(mcp_url, timeout=10, sse_read_timeout=120) as streams:
                async with ClientSession(*streams) as session:
                    await _initialize_session(session)
                    tools = await _list_allowed_tools(session)
                    response = await _chat_with_tools(
                        session=session,
                        api_key=api_key,
                        model=model,
                        project=project,
                        question=question,
                        history=history,
                        attachment=attachment,
                        tools=tools,
                        tracker=tracker,
                    )
            return ChatRunResult(answer=response, sources=tracker.as_list())
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


async def _list_allowed_tools(session: ClientSession) -> list[dict[str, Any]]:
    cursor: Optional[str] = None
    selected = []
    while True:
        result = await session.list_tools(cursor=cursor)
        for tool in result.tools:
            if tool.name not in ALLOWED_TOOLS:
                continue
            selected.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            })
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            break
    return selected


async def _chat_with_tools(
    *,
    session: ClientSession,
    api_key: str,
    model: str,
    project: str,
    question: str,
    history: list[dict[str, Any]],
    attachment: Optional[dict[str, Any]],
    tools: list[dict[str, Any]],
    tracker: SourceTracker,
) -> str:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages = _build_messages(question=question, history=history, attachment=attachment)
    system_prompt = _build_system_prompt(project=project, attachment=attachment)

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.messages.create(
                model=model,
                max_tokens=2200,
                system=system_prompt,
                messages=messages,
                tools=tools,
                temperature=0.1,
            )

            assistant_blocks = [_dump_block(block) for block in response.content]
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [block for block in response.content if getattr(block, "type", "") == "tool_use"]
            if not tool_uses:
                answer = _collect_text(response.content)
                if not answer:
                    raise RuntimeError("LLM returned an empty response.")
                return answer

            tool_results = []
            for block in tool_uses:
                arguments = _normalize_tool_arguments(block.name, block.input, project)
                raw_result = await session.call_tool(block.name, arguments)
                payload = _coerce_tool_payload(raw_result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _format_tool_result(
                        tool_name=block.name,
                        payload=payload,
                        tracker=tracker,
                        project=project,
                    ),
                    "is_error": bool(getattr(raw_result, "isError", False)),
                })

            messages.append({"role": "user", "content": tool_results})
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    raise RuntimeError("LLM exceeded the tool-use limit before producing a final answer.")


def _build_system_prompt(*, project: str, attachment: Optional[dict[str, Any]]) -> str:
    base = (
        "You are Esteem Folder Knowledge, a folder-scoped document assistant.\n"
        f"You are working only within the folder '{project}' and its sub-folders.\n"
        "Use the provided MCP tools to inspect the folder documents directly.\n"
        "Never use outside knowledge or documents from any other folder.\n"
        "Ignore any numeric citations in prior chat history. Only source IDs provided in this turn's tool results are valid for your new answer.\n"
        "When you make factual claims from folder documents, cite the supporting source IDs inline like [1] or [2].\n"
        "If the documents do not support a confident answer, say that clearly and do not speculate.\n"
        "Respond in markdown.\n"
    )
    if not attachment:
        return base + (
            "Investigate with the MCP tools before answering when the question requires document evidence.\n"
            "Prefer precise search first, then expand to page reads or nearby pages when needed.\n"
        )

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
) -> str:
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
        return (
            f"Tool: {tool_name}\n"
            "Page-based source IDs from this tool result:\n"
            + "\n".join(f"- {line}" for line in annotations)
            + "\n\nStructured result:\n"
            + body
        )

    return f"Tool: {tool_name}\nStructured result:\n{body}"


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
        return {key: _trim_payload(item) for key, item in value.items()}
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
