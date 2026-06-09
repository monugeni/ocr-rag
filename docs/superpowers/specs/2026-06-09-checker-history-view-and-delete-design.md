# Checker History view + per-run delete — design

- **Date:** 2026-06-09
- **Status:** Approved (design); pending spec review
- **Area:** docchecker (embedded in ocr-rag `web.py`), `static/app.js`

## Goal

Today, past checking sessions live in one cramped panel at the bottom of the Check
page (`static/app.js:1574`, `loadRecentChecks()` at 1617): flat rows of
`status · document_type · created_at · open`, scoped to the **current folder only**.
There is **no delete** — neither a UI control nor a `DELETE /api/runs/{id}` route, and
nothing cleans up annotated PDFs, uploads, traces, or ingested docs on disk.

This work gives past sessions their own home (retrieval) and adds a real delete that
cleans up everything a run created.

## Decisions (locked)

| Question | Choice |
|---|---|
| Placement | **Dedicated "History" view** — a new top-level nav item. The Check page keeps only the run form. |
| Retrieval scope | **All folders**, newest-first, **search only** (no status/date/folder filters). |
| Deletion | **Hard delete, single run at a time**, with a confirm dialog. No bulk multi-select. |
| Purge ingested docs | **Yes** — deleting a run also purges that run's ingested docs (pages/chunks/FTS) from `check_docs.db`. |

## Out of scope (deliberately not built)

- Bulk multi-select delete.
- Status / date / folder filter dropdowns.
- Soft-archive / restore.
- Any change to how a check is *run* or its results rendered.

## Backend changes (`docchecker/`)

### 1. List enrichment — `store.list_run_cards(q=None, limit=100)`

The current `check_runs` row lacks the document name and a findings count. Add one
store function returning, per run, the existing run fields plus:

- `submitted_name` — `uploads.filename` where `role='submitted'` (first match).
- `finding_count` — `COUNT(*)` from `findings` for the run.

Single query with correlated subqueries (no N+1), ordered `created_at DESC`, `LIMIT ?`.
When `q` is provided, filter with `LIKE` (case-insensitive) across `project_number`,
`document_type`, and the submitted filename.

### 2. Route — list cards

`GET /api/runs?q=<text>` in `routers/runs.py` returns `store.list_run_cards(q, limit)`.
(Extends the existing `list_runs` route with an optional `q` param; the cards shape is
a superset of the current rows, so the Check page's removal of the inline panel is the
only caller affected.)

### 3. Route — delete

New `DELETE /api/runs/{run_id}` in `routers/runs.py`:

1. `require_user`.
2. Load the run; 404 if missing.
3. **Guard:** if `status in ("running", "queued")` → 409 `"cannot delete a run while it is <status>"`.
4. Call `store.delete_run(run_id)` (DB + cross-DB purge; returns the disk paths to remove).
5. Remove on-disk artifacts (see cleanup matrix).
6. `auth.record_audit("run_deleted", user_id=..., run_id=run_id, payload={...})`.
7. Return `{"deleted": run_id}`.

### 4. `store.delete_run(run_id)`

1. Read, **before deleting**: all `uploads` rows for the run (`disk_path`, `doc_id`,
   `role`).
2. Purge ingested docs from `check_docs.db`: for each distinct `doc_id`, run the same
   FTS-safe delete sequence as `web.py:_delete_document_data` (page_embeddings → chunks
   → pages → sections → corrections → cross_references → quality_flags → ingestion_jobs
   → documents). Implemented as a thin local helper `purge_document(docs_db_path, doc_id)`
   in docchecker (copy of the sequence) rather than importing `web.py`, whose import has
   app-level side effects. Opens its own `check_docs.db` connection.
3. Delete the `check_runs` row from `checks.db`. FK cascade (`PRAGMA foreign_keys=ON`,
   db.py:139) clears `uploads`, `findings`, `comment_incorporation` automatically.
4. Return the gathered upload `disk_path`s so the route can unlink files.

`audit_events.run_id` has no FK, so the audit trail (including the new `run_deleted`
event) survives deletion.

## Cleanup matrix — what a delete removes

| Artifact | Location | Mechanism |
|---|---|---|
| `check_runs` row | `checks.db` | explicit `DELETE` |
| `uploads`, `findings`, `comment_incorporation` rows | `checks.db` | FK `ON DELETE CASCADE` |
| Ingested submitted + fresh-reference docs (pages/chunks/sections/FTS/embeddings) | `check_docs.db` | `purge_document()` per `doc_id` |
| Uploaded source files | `UPLOADS_DIR/...` (`uploads.disk_path`) | `Path.unlink(missing_ok=True)` |
| Annotated PDF + `trace.json` | `ANNOTATED_DIR/{run_id}/` | `shutil.rmtree(..., ignore_errors=True)` |
| Per-run exports, if any | `EXPORTS_DIR/{run_id}/` | `shutil.rmtree(..., ignore_errors=True)` |
| Audit events | `checks.db` `audit_events` | **kept** (no FK); new `run_deleted` appended |

All filesystem removals are best-effort (`missing_ok` / `ignore_errors`) so a missing
artifact never blocks the delete; the DB delete is the source of truth.

## Frontend changes (`static/app.js`, `static/index.html`)

### 1. Navigation

Add a `History` nav item alongside Folders / Check / Jobs. Route hash `#history`
renders `renderHistory()`.

### 2. Check page

Remove the inline "Recent checks" panel (`app.js:1574`) and `loadRecentChecks()`. Add a
small `See past checks →` link to `#history` where the panel was.

### 3. `renderHistory()`

- A search input (debounced) that calls `GET /api/runs?q=...`.
- A list of run cards, newest-first, across all folders. Each card shows:
  status pill · `submitted_name` · folder (`project_number`) · `created_at` ·
  `finding_count` findings · `[open]` · `[🗑]`.
- Empty state: "No checks yet." Search empty state: "No checks match '<q>'."

### 4. Open

`[open]` calls the existing `openCheckRun(runId)`, which switches to the Check results
split-view. (Behaviour unchanged; just invoked from History.)

### 5. Delete

- `[🗑]` is hidden/disabled when `status` is `running` or `queued`.
- Clicking it opens a confirm dialog: *"Permanently delete this check and its annotated
  PDF? This cannot be undone."* with Cancel / Delete.
- On confirm → `DELETE /api/runs/{id}` → on success remove the card from the DOM and
  toast "Check deleted"; on 409/other → toast the error and leave the card.

## Data flow

```
History view ──GET /api/runs?q──► routers.runs.list_runs ──► store.list_run_cards ──► cards
   [🗑] ──confirm──► DELETE /api/runs/{id} ──► store.delete_run ─┬─► checks.db (cascade)
                                                                 ├─► check_docs.db (purge_document)
                                                                 └─► return disk paths
                                              route ──► unlink uploads + rmtree run dirs
                                              route ──► record_audit("run_deleted")
```

## Edge cases / errors

- **Delete while running/queued:** blocked server-side (409) and disabled in the UI.
- **Missing files on disk:** best-effort removal; never blocks the DB delete.
- **`doc_id` shared across runs:** not a concern — each run ingests its own copies into
  `check_docs.db`; purging is run-local.
- **Concurrent delete + open:** opening a just-deleted run returns 404; the UI shows the
  existing not-found handling.
- **`q` with FTS/SQL metacharacters:** parameterised `LIKE`, so input is inert; `%`/`_`
  in the query are treated literally enough for this list (no escaping needed for a
  bounded LIKE filter).

## Manual test plan

No test suite in this repo (per CLAUDE.md). Smoke against a throwaway run:

1. Run a check on a small `samples/` PDF; confirm it appears in History across folders.
2. Search by part of the document name; confirm filtering.
3. Open from History; confirm the results split-view loads.
4. Delete a `done` run; confirm: card disappears; `check_runs`/`uploads`/`findings` rows
   gone; `ANNOTATED_DIR/{run_id}` gone; the submitted `doc_id` no longer in
   `check_docs.db` (`SELECT … FROM documents WHERE id=?` returns nothing); a
   `run_deleted` audit row exists.
5. Attempt to delete a `running` run; confirm the button is disabled and the API returns
   409.
