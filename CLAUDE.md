# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

OCR-RAG: a pipeline that ingests large engineering PDFs (tenders, specs, drawings) into SQLite + FTS5, exposes search/navigation as MCP tools, and ships a FastAPI web UI whose chat answers by calling those same MCP tools. There is no separate retrieval path — everything goes through the MCP server.

## Common commands

```bash
# Setup
./setup.sh                                      # installs requirements.txt
./setup_embeddings.sh [db_path]                 # installs sentence-transformers + computes embeddings for every project in the DB

# Ingest
python ingest.py --project samples/split_output --db docs.db
python ingest.py --pdf input.pdf --db docs.db --project-name MyProject
python ingest.py --project ... --skip-llm       # skip Anthropic metadata extraction

# Reingest existing docs through the new fast Poppler pipeline (in-place upgrade,
# auto-backs up docs.db first to docs.db.before-fast-reingest-<stamp>.bak)
python reingest_fast.py --db docs.db

# Servers (run both — web UI calls the local MCP server over SSE)
python mcp_server.py --db docs.db --port 8200
python web.py --db docs.db --port 8201 --mcp-port 8200

# Splitter / extractors (standalone, no DB writes)
python splitter.py tender_package.pdf --threshold 0.6 --dry-run
python extractor.py input.pdf --output out.json
python fast_pipeline.py input.pdf --out-dir diag/   # diagnostic-first Poppler pipeline

# Production (Ubuntu)
sudo bash deploy.sh                # first-time install -> /btrfs/ocr-rag, systemd unit ocr-rag-mcp
sudo bash deploy.sh --update       # pull + restart
```

There is no test suite in this repo. To smoke-test a change, prefer ingesting a small file from `samples/` into a throwaway DB (e.g. `--db /tmp/scratch.db`) and exercising the resulting MCP server with `curl` or via the web UI.

Marker is **not** a default dependency. The fast Poppler pipeline (`pdftohtml -xml` + `pdftotext --layout`) is the production path; Marker is only used when you explicitly pre-convert PDFs with `marker_single`/`marker` and then point `ingest.py` at the JSON output. The fast pipeline needs `poppler-utils` installed.

## Big-picture architecture

```
PDF ─┬─► splitter.py (optional)  ──► smaller PDFs by detected document boundary
     │
     ├─► extractor.py (pdfplumber + heuristics)       ──┐
     ├─► fast_pipeline.py (Poppler XML)               ──┼─► (pages, sections)
     └─► Marker JSON (external) parsed by ingest.py   ──┘        │
                                                                 ▼
                                                  ingest.py ── SQLite (FTS5 + embeddings)
                                                                 │
                                                  mcp_server.py ─┴──► web.py chat / external MCP clients
```

Three things are worth internalising before editing anything:

**1. Every extractor produces the same `(pages, sections)` shape.**
`extractor.py`, `fast_pipeline.py`, `file_extractors.py` (docx/xlsx/images/archives) and the Marker JSON parser inside `ingest.py` all emit the same two-list structure that `ingest_document()` consumes. New extractors must keep this contract; the ingestion, embeddings, corrections, and MCP layers all depend on it.

**2. SQLite schema (defined in `ingest.py` `SCHEMA`).**
- `documents` → `sections` (tree via `parent_id`, with `breadcrumb`) → `pages` → `chunks` (paragraph/clause level — the **primary retrieval unit** for tender Q&A).
- Two FTS5 virtual tables: `pages_fts` and `chunks_fts`, both kept in sync via triggers. Chunk-level FTS is preferred because it preserves precise breadcrumbs at hit level.
- `page_embeddings` for semantic search (sentence-transformers, optional — only populated if `setup_embeddings.sh` has been run).
- `corrections` + sidecar `{stem}_corrections.json` next to the PDF: dual persistence so LLM-supplied fixes survive re-ingestion. `replay_corrections()` reapplies them after every ingest.
- `quality_flags`, `cross_references`, `ingestion_jobs`, `upload_events` track operational state for the web UI.

**3. Web chat → MCP tools, not a private path.**
`web.py` does **not** query SQLite directly for chat. It spawns `chat_mcp_runner.run_folder_chat()`, which connects to the local MCP server over SSE and lets Claude (Anthropic API) call the same tools external MCP clients see. The whitelist of tools the chat is allowed to use lives in `chat_mcp_runner.ALLOWED_TOOLS` / `PROJECT_SCOPED_TOOLS` — if you add an MCP tool that chat should use, register it there too.

## Key modules

| File | Role |
|------|------|
| `ingest.py` | Schema, `ingest_document()`, Marker JSON parser, LLM metadata, embeddings, corrections replay. The schema source of truth. |
| `mcp_server.py` | FastMCP server. Discovery / Search / Navigation / Fallback tool groups. `sanitize_fts()` is the FTS5 query escaper — non-obvious for section numbers like `3.2` and `B31.3`. |
| `corrections.py` | 25 MCP tools letting the LLM fix heuristic mistakes (heading levels, OCR garbage, metadata, doc boundaries). Registered into the MCP server via `register_correction_tools()`. |
| `extractor.py` | Heuristic born-digital extractor (pdfplumber + font/layout signals). Falls back to `ocrmypdf` for scanned pages. |
| `fast_pipeline.py` | Production CPU-first extractor using Poppler (`pdftohtml -xml`, `pdftotext --layout`). Emits headings + chunks + diagnostics. |
| `splitter.py` | 10-engine heuristic splitter for concatenated tender packs (bookmarks, geometry, font fingerprinting, xref, content-stream dialects). No LLM, no ML. |
| `file_extractors.py` | Non-PDF inputs (DOC/DOCX/XLS/XLSX/images) and archives (ZIP/TAR/GZ). Same `(pages, sections)` contract. |
| `chat_mcp_runner.py` | Anthropic tool-use loop that drives MCP tools for folder chat (caps: `MAX_TOOL_ROUNDS`, `MAX_TOOL_RESULT_CHARS`, repeated-call detection). |
| `web.py` | FastAPI app + static UI. Manages folders/uploads/ingestion jobs, document and page viewers, and chat. Knobs at the top: `CHAT_CONTEXT_*`, `CHAT_REVIEW_*`. |
| `reingest_fast.py` | One-shot operational tool to upgrade an existing DB to the fast pipeline. Backs up the DB before touching it. |
| `deploy.sh` | Ubuntu install/update + systemd `ocr-rag-mcp.service` on ports 8200/8201, app at `/btrfs/ocr-rag`. |

## Conventions and gotchas

- **FTS5 queries must go through `sanitize_fts()` in `mcp_server.py`.** It strips boolean operators, quotes section numbers, and splits delimited tokens. Bypassing it will let user input crash queries on punctuation.
- **Sidecar corrections live next to the PDF**, suffix `_corrections.json` (`CORRECTIONS_SUFFIX` in `corrections.py`). Re-ingestion replays them; deleting the PDF means deleting the corrections too.
- **Embeddings are optional** and gated by env: chat semantic fallback only runs when `OCR_RAG_ENABLE_SEMANTIC_FALLBACK=1`. Fast-pipeline OCR is gated by `OCR_RAG_FAST_PIPELINE_OCR=1`. The MCP tool `semantic_search` is only useful if `page_embeddings` has been populated.
- **`ANTHROPIC_API_KEY` is required for two things**: ingestion-time metadata extraction (skippable with `--skip-llm`) and the web chat (`chat_mcp_runner`). Loaded from `.env` via `python-dotenv`.
- **Split-document filenames are parsed**, not stored as metadata: see `_SPLIT_RE` in `web.py` — `<parent>_part<NNN>_p<start>-<end>[_<label>].pdf`. The UI groups parts back together based on this pattern.
- **Production paths differ from dev paths**: deploy.sh installs to `/btrfs/ocr-rag` with data under `/btrfs/ocr-rag/data`. Don't hardcode dev-relative paths in code that runs under the systemd unit.
- **Backups before destructive ops**: anything that rewrites `docs.db` in place (notably `reingest_fast.py`) must snapshot the file first. Existing `*.before-fast-reingest-*.bak` files in the repo are evidence of this convention.
