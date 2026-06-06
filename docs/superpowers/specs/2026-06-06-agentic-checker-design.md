# Agentic Document Checker — Design

**Date:** 2026-06-06
**Scope:** Replace the document-checker's "retrieve-then-check" compare stage with a
sequential, page-by-page **agentic sweep** that searches the tender on demand, in one
cumulative cached conversation. Verification, dedupe, PDF annotation, and the debug
trace are reused unchanged.

## Goal

The current checker retrieves reference context **once, upfront** (plan queries → search
the company KB → build a reference block), then checks the whole submission against that
fixed block. If the upfront retrieval misses a requirement, the check never sees it.

The user wants the checker to work the way a human reviewer does: **go through the
submission a bite at a time, and for each bite search the tender for the governing
requirements and check that bite against them — then move on.** This is an *agentic*
loop (the model issues its own reference searches as it works) rather than fixed RAG.
Caching is to be maximized so the cost of one long, accumulating conversation stays low.

## Decisions (locked during brainstorming)

- **Bite size:** one **page** per bite.
- **Sweep style:** one **cumulative conversation** — sequential, the model accumulates
  tender knowledge as it sweeps; maximum caching.
- **Long documents:** **server-side compaction** (Anthropic beta) so the single
  conversation survives 100s of pages.
- **Mode:** agentic **fully replaces** the existing retrieve-then-check compare. One path.
- Reuses: recall-first verification, dedupe, annotate-on-PDF, debug trace (unchanged).

## Architecture

```
run_real_check (pipeline.py)
  └─► agentic_sweep(ctx, llm)            ← NEW, replaces _compare + _reference_block
        • connect to company MCP (reference KB), build reference-search tool list
        • ONE conversation; system = checker instructions (+ uploaded ref text), cached
        • for each submitted page (1..N):
            user turn: "PAGE n:\n<page text>\n\nCheck this page against the tender.
                        Search the reference KB as needed, then report findings."
            agentic loop until the model reports findings for this page:
              - model → tool_use (ranked_search / search_chunks / get_page / …)
              - harness → tool_result (search hits)
              - model → report_findings(tool)  → harness records findings, acks
            cache_control on the latest turn (growing conversation read at ~0.1×)
            compaction edit enabled so old context is summarized when large
        • return all findings collected across pages
  └─► _verify_all (recall-first, unchanged) → confirmed + "possible"
  └─► annotate + trace (unchanged)
```

The agentic loop is adapted from `chat_mcp_runner.py`, which already implements a cached
Anthropic tool-use loop over MCP tools (`MAX_TOOL_ROUNDS`, ephemeral `cache_control`,
`ALLOWED_TOOLS`). We reuse that machinery rather than build a new loop.

## Components

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `agentic_sweep(ctx, llm)` (new, in `docchecker/agent/agentic.py`) | Drive the page-by-page sweep; own the single conversation, caching, compaction; return `list[Finding]`. | MCP client, the checker system prompt, `read_pages`, `report_findings` schema |
| `report_findings` tool handler | Intercept the model's per-page findings tool calls, convert each to a `Finding` (via the existing `_finding_from_dict`), ack so the model advances. | `_finding_from_dict`, `_dedupe_findings` |
| Reference-search tool list | The read-only ocr-rag search tools scoped to `ctx.reference_projects` (the tender/KB). | reuse `chat_mcp_runner.ALLOWED_TOOLS` / `PROJECT_SCOPED_TOOLS` |
| Submission re-lookup tools | Read-only search/`get_page` over the **submitted** doc's folder, so the model can re-fetch an earlier page on demand (cross-page context under compaction). | same MCP tool set, scoped to the submission's project |
| System prompt (checker, agentic) | Instructions: check each page against the tender, search the KB on demand, be exhaustive, report via `report_findings`; includes any **uploaded** reference text inline (cached). | `build_system_prompt` + uploaded ref text |
| `run_real_check` integration | Call `agentic_sweep` instead of `_compare`; keep verify/annotate/trace. | above |

`_compare`, `_reference_block`, `_plan_queries`, and the upfront `fetch_reference_context`
call are **removed from the check path** (the model now searches on demand).
`fetch_reference_context`/`company_refs.py` may still be used by the tool layer or removed
if the model calls the MCP tools directly — see Open Questions.

## Data flow

1. `run_real_check` builds `ctx` (submitted DocRef, reference folders, uploaded refs).
2. `agentic_sweep`:
   - Pages = `read_pages(docs_db, submitted.doc_id)` → bite list (one page each).
   - Open ONE Anthropic conversation. System block (cached) = checker instructions +
     uploaded-reference full text (if any).
   - Tools = reference-search MCP tools over `ctx.reference_projects`.
   - For each page: append a user turn with the page text; run the tool-use loop until the
     model calls `report_findings` (or yields text with no tool call). Capture findings.
     Put `cache_control: ephemeral` on the last block of the latest turn each round.
     Enable the compaction context edit so earlier turns are summarized when large.
   - Emit progress per page (`Checked page n: k finding(s)`) and stream the model's
     reasoning to the live panel + trace (existing `emit`/thinking mechanism).
3. Collect findings → `_dedupe_findings` → `_verify_all` (recall-first) → annotate → trace.

## Caching (maximized)

- **System + tools** carry `cache_control: ephemeral` — stable for the whole sweep.
- **Growing conversation:** a cache breakpoint on the latest turn each round, so the entire
  prior conversation (pages already checked + tender search results already retrieved) is
  read from cache (~0.1×) rather than reprocessed. This is the core win of the cumulative
  design: accumulated tender knowledge is carried cheaply.
- **Verification & usage:** unchanged; the recall-first verify uses the fast model.
- Verify with `usage.cache_read_input_tokens` in the trace that caching is actually hitting.

## Cross-page context

A value or spec stated on an earlier page must inform the check of a later page. Two
mechanisms guarantee this:

1. **It's one cumulative conversation.** Every prior page's text, its findings, and the
   tender requirements already retrieved remain in the message history, so when the model
   reaches page N it already holds pages 1..N-1 and carries them forward. Caching makes
   re-reading that history cheap — it does **not** drop it.
2. **On-demand re-lookup.** The model also has search/`get_page` tools over the *submitted*
   document, so it can re-fetch a specific earlier page when a later page references it —
   robust even when compaction has summarized early turns.

**Caveat:** on very long documents where compaction trims early context, fine-grained recall
of a specific early value from memory degrades; the re-lookup tool (mechanism 2) is the
safety net, and the system prompt instructs the model to re-fetch earlier pages rather than
guess when a later page depends on earlier data.

## Compaction (long docs)

Enable the compaction beta — `context_management: {edits: [{type: "compact_20260112"}]}`
with beta header `compact-2026-01-12` (supported on Opus 4.8/4.7/4.6 and Sonnet 4.6). When
the conversation approaches the context limit, the API summarizes earlier
turns server-side. **Critical:** append the full `response.content` (including compaction
blocks) back to the message list each turn — not just text — or compaction state is lost.

## Error handling

- A page whose sweep errors (tool failure, model error) is logged to the run warnings and
  the sweep continues to the next page (one page must not sink the run) — mirrors the
  current per-lens/per-chunk error isolation.
- MCP connection retries reuse `chat_mcp_runner`'s retry/timeout settings.
- If the model never calls `report_findings` for a page (e.g. no issues), that's a valid
  "no findings on this page" — advance.
- All telemetry (trace, usage) is best-effort and never fails a run.

## Testing

- **Unit (offline, stubbed):** page segmentation; `report_findings` interception →
  `Finding` conversion; loop control (advance per page, stop at end, error isolation);
  findings dedupe. Stub the MCP client + Anthropic client.
- **Integration (real, paid):** one real check on a multi-page submission; confirm via the
  debug trace that (a) findings are produced per page, (b) `cache_read_input_tokens` is
  non-zero (caching hitting), (c) recall-first verify keeps borderline findings as
  "possible". This is the true acceptance test — it cannot be fully simulated offline.

## Out of scope / YAGNI

- No change to verification, dedupe, annotation, or the trace UI.
- No new MCP tools — reuse the existing read-only search tools.
- No parallel sweep (the cumulative cached conversation is deliberately sequential).
- No UI mode toggle (agentic fully replaces the old mode).

## Risks & open questions

- **Cost/latency:** sequential page sweep with tool round-trips is the most expensive mode;
  caching mitigates token cost but not the sequential latency. Accepted tradeoff.
- **Tool wiring:** the cleanest reuse is `chat_mcp_runner`'s async streamable-HTTP MCP loop.
  The checker runs in a background thread (`agent_seam`), and `chat_mcp_runner` uses
  `asyncio.run` — confirm the threaded/async boundary is clean (it already runs the chat
  loop this way from a request handler).
- **Uploaded references in agentic mode:** included as cached system text (they're already
  extracted). If an uploaded ref is very large it inflates the cached system block — bound
  it with the existing 120k-char cap.
- **Open:** whether to keep `company_refs.fetch_reference_context` at all once the model
  searches via MCP directly (likely removable from the check path).
- **Compaction availability:** depends on the model + beta header; if unavailable, fall back
  to segmenting the sweep (the deferred long-doc option) — flagged, not built here.

## Reference points
- Reuse: `chat_mcp_runner.py` (cached MCP tool-use loop), `ALLOWED_TOOLS`,
  `_build_system_prompt`.
- Replace in `docchecker/agent/pipeline.py`: `_compare`, `_reference_block`,
  `_plan_queries`.
- Keep in `pipeline.py`: `_finding_from_dict`, `_dedupe_findings`, `_verify`, `_verify_all`,
  `run_real_check` (wiring), trace assembly.
