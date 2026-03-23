# OCR-RAG: Document Retrieval Agent

A pipeline for ingesting large engineering PDFs (tenders, specifications, etc.) into a searchable SQLite FTS5 database, exposed via an MCP server for LLM-powered retrieval.

## What This Does

1. **PDF Splitting** — Large PDFs (1000+ pages) are split into logical sections using a PDF splitter
2. **OCR & Extraction** — [Marker](https://github.com/VikParuchuri/marker) converts each PDF into structured JSON with OCR, layout detection, and section hierarchy
3. **Table Enhancement** — pdfplumber extracts accurate tables to replace Marker's table output where possible
4. **Ingestion** — Extracted content is stored page-by-page in SQLite with FTS5 full-text search, section breadcrumbs, and LLM-extracted metadata
5. **MCP Server** — An MCP server provides search, navigation, and re-extraction tools for LLM agents
6. **Web Chat** — The built-in web chat now answers by using that same MCP tool surface instead of a separate private retrieval path

## Current Status

- PDF splitter working — splits large tenders into logical parts by section boundaries
- Marker OCR pipeline set up (marker-pdf v1.10.2 with surya-ocr)
- Ingestion script (`ingest.py`) complete — Marker JSON parsing, pdfplumber table enhancement, LLM metadata extraction via Anthropic API
- MCP server (`mcp_server.py`) complete — full-text search, section search, page navigation, re-extraction fallbacks
- Tested with a 1567-page tender document split into 36 parts

## Prerequisites

- Python 3.10+
- [marker-pdf](https://pypi.org/project/marker-pdf/) (`pip install marker-pdf`)
- An `ANTHROPIC_API_KEY` environment variable (for LLM metadata extraction and MCP-backed web chat)

## Setup

```bash
./setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
pip install marker-pdf
```

## Commands

### 1. Split a large PDF

Use the PDF splitter to break a large document into sections.

### 2. Run Marker OCR

**Single file:**
```bash
marker_single input.pdf --output_format json --output_dir output/
```

**Batch (folder of PDFs):**
```bash
# Use --workers 1 to avoid excessive RAM usage (each worker loads ~4-6GB of ML models)
marker samples/split_output --workers 1 --output_format json --output_dir samples/split_output
```

> **Note:** With `--workers 2`, Marker loads 5 ML models (layout, recognition, table, detection, OCR error) per worker process. This can easily consume 10-12GB+ of RAM. Use `--workers 1` for machines with limited memory.

### 3. Ingest into SQLite

```bash
# Ingest a project folder
python ingest.py --project samples/split_output --db docs.db

# Ingest a single PDF
python ingest.py --pdf input.pdf --db docs.db --project-name MyProject

# Skip LLM metadata extraction
python ingest.py --project samples/split_output --db docs.db --skip-llm
```

### 4. Start the MCP Server

```bash
python mcp_server.py --db docs.db --port 8200
```

### 5. Start the Web App

```bash
python web.py --db docs.db --port 8201 --mcp-port 8200
```

The web UI chat uses the same MCP tools exposed on the local MCP server, so the in-app assistant and external MCP clients investigate folder documents through the same interface.

## MCP Server Tools

| Tool | Description |
|------|-------------|
| `list_projects` | List all projects |
| `list_documents` | List documents in a project |
| `get_document_info` | Full metadata for a document |
| `get_toc` | Section tree for a document |
| `search_pages` | Full-text keyword search |
| `search_sections` | Search section headings |
| `get_page` | Get full page content with context |
| `get_adjacent` | Get next/previous page |
| `reextract_page` | Re-extract page from original PDF via pdfplumber |
| `reextract_table` | Re-extract table from original PDF via pdfplumber |

## Architecture

```
PDF ──► PDF Splitter ──► Marker (OCR/JSON) ──► ingest.py ──► SQLite (FTS5)
                                                                  │
                                                           MCP Server ──► LLM Agent
```
