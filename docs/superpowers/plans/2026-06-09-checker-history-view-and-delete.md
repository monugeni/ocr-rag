# Checker History view + per-run delete — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give past checking sessions a dedicated, searchable History view across all folders, and add a hard single-run delete that cleans up DB rows, ingested docs, and on-disk artifacts.

**Architecture:** Backend adds an enriched list query + a `DELETE /api/runs/{id}` route in the docchecker FastAPI routers. Frontend adds a `history` tab to the existing single-page app (auto-wired by `VALID_TABS`), removes the inline "Recent checks" panel from the Check page, and reuses the existing modal pattern for a delete confirm.

**Tech Stack:** FastAPI + SQLite (docchecker), vanilla JS SPA (`static/app.js` + `static/index.html`). No test suite (per CLAUDE.md) — verification is manual smoke testing against a throwaway run.

**Spec:** `docs/superpowers/specs/2026-06-09-checker-history-view-and-delete-design.md`

---

### Task 1: Backend — store functions (`list_run_cards`, purge, `delete_run`)

**Files:**
- Modify: `docchecker/store.py` (add `import sqlite3`; add three functions)

- [ ] **Step 1: Add `import sqlite3`** at the top of `store.py` (alongside `import json`, `import uuid`).

- [ ] **Step 2: Add `list_run_cards`** (enriched, all-folders, optional search):

```python
def list_run_cards(q: str | None = None, limit: int = 100) -> list[dict]:
    """Runs across ALL folders, newest-first, each enriched with the submitted
    document name and a findings count — backs the History view. Optional `q`
    filters (LIKE) over folder, document type, and submitted filename."""
    where, params = "", []
    if q:
        like = f"%{q}%"
        where = (
            " WHERE (r.project_number LIKE ? OR r.document_type LIKE ? "
            "OR EXISTS (SELECT 1 FROM uploads u2 WHERE u2.run_id = r.id "
            "AND u2.role = 'submitted' AND u2.filename LIKE ?))"
        )
        params = [like, like, like]
    conn = get_conn()
    try:
        rows = conn.execute(
            f"""SELECT r.*,
                  (SELECT u.filename FROM uploads u
                     WHERE u.run_id = r.id AND u.role = 'submitted'
                     ORDER BY u.id LIMIT 1) AS submitted_name,
                  (SELECT COUNT(*) FROM findings f WHERE f.run_id = r.id) AS finding_count
                FROM check_runs r{where}
                ORDER BY r.created_at DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [dict(x) for x in rows]
    finally:
        conn.close()
```

- [ ] **Step 3: Add `_purge_ingested_doc`** (FTS-safe, mirrors `web.py:_delete_document_data`):

```python
def _purge_ingested_doc(docs_db_path: str, doc_id: int) -> None:
    """Remove one document and all derived rows from a docs.db (ocr-rag schema).
    Mirrors web.py:_delete_document_data; the *_fts tables stay in sync via the
    triggers on the documents/pages/chunks deletes. Best-effort per table so a
    schema that predates a column never blocks the run delete."""
    conn = sqlite3.connect(docs_db_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute(
                "DELETE FROM page_embeddings WHERE page_id IN "
                "(SELECT id FROM pages WHERE doc_id = ?)",
                (doc_id,),
            )
        except sqlite3.OperationalError:
            pass  # embeddings table absent in this DB
        for tbl in ("chunks", "pages", "sections", "corrections",
                    "cross_references", "quality_flags", "ingestion_jobs"):
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE doc_id = ?", (doc_id,))
            except sqlite3.OperationalError:
                pass
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Add `delete_run`** (DB cascade + cross-DB purge; returns disk paths):

```python
def delete_run(run_id: str) -> dict:
    """Hard-delete a run. Purges its ingested docs from check_docs.db, deletes the
    check_runs row (FK cascade clears uploads/findings/comment_incorporation), and
    returns the upload disk paths so the caller can unlink them."""
    from . import config
    conn = get_conn()
    try:
        ups = conn.execute(
            "SELECT disk_path, doc_id FROM uploads WHERE run_id = ?", (run_id,)
        ).fetchall()
        disk_paths = [u["disk_path"] for u in ups if u["disk_path"]]
        doc_ids = {u["doc_id"] for u in ups if u["doc_id"] is not None}
        conn.execute("DELETE FROM check_runs WHERE id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()
    for did in doc_ids:
        try:
            _purge_ingested_doc(config.DOCS_DB, did)
        except Exception:  # noqa: BLE001 — best-effort; the run row is already gone
            pass
    return {"disk_paths": disk_paths, "doc_ids": list(doc_ids)}
```

- [ ] **Step 5: Commit**

```bash
git add docchecker/store.py
git commit -m "feat(checker): store helpers for run cards + hard delete with doc purge"
```

---

### Task 2: Backend — routes (cards list `q` param + DELETE)

**Files:**
- Modify: `docchecker/routers/runs.py` (add `import shutil`; change list route; add delete route)

- [ ] **Step 1: Add `import shutil`** near the top (next to `import json`).

- [ ] **Step 2: Replace the existing list route** (`@router.get("")`) so it returns enriched cards across all folders with optional search:

```python
@router.get("")
def list_runs(request: Request, q: str | None = None, limit: int = 100):
    auth.require_user(request)
    return store.list_run_cards(q=q, limit=limit)
```

- [ ] **Step 3: Add the delete route.** Place it AFTER `get_run` so the static segments are unaffected (FastAPI matches `DELETE` separately from `GET /{run_id}`, so ordering is safe):

```python
@router.delete("/{run_id}")
def delete_run(run_id: str, request: Request):
    user = auth.require_user(request)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] in ("running", "queued"):
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete a run while it is {run['status']}",
        )
    from .. import config

    artifacts = store.delete_run(run_id)
    for p in artifacts["disk_paths"]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
    for d in (Path(config.ANNOTATED_DIR) / run_id, Path(config.EXPORTS_DIR) / run_id):
        shutil.rmtree(d, ignore_errors=True)
    auth.record_audit(
        "run_deleted",
        user_id=user["id"],
        run_id=run_id,
        payload={"project_number": run.get("project_number")},
    )
    return {"deleted": run_id}
```

- [ ] **Step 4: Smoke the backend.** Start the app, sign in, then:

```bash
curl -s -b cookies.txt 'http://localhost:8201/api/runs?q=' | head -c 400   # cards JSON with submitted_name + finding_count
curl -s -b cookies.txt -X DELETE 'http://localhost:8201/api/runs/<known-run-id>'  # {"deleted": "..."}
```
Expected: list returns objects containing `submitted_name` and `finding_count`; delete returns `{"deleted": ...}` and a follow-up GET of that run id returns 404.

- [ ] **Step 5: Commit**

```bash
git add docchecker/routers/runs.py
git commit -m "feat(checker): /api/runs cards search + DELETE /api/runs/{id}"
```

---

### Task 3: Frontend — History tab + view (retrieval)

**Files:**
- Modify: `static/index.html` (nav button + view section)
- Modify: `static/app.js` (`VALID_TABS`, `renderActiveView`, history state + render/load functions)

- [ ] **Step 1: Add the nav button** in `static/index.html` after the check-tab (line 46):

```html
          <button id="history-tab" class="mode-tab">History</button>
```

- [ ] **Step 2: Add the view container** after `check-view` (line 60):

```html
      <section id="history-view" class="view page-view hidden"></section>
```

- [ ] **Step 3: Register the tab** — add `'history'` to `VALID_TABS` (app.js:24):

```javascript
const VALID_TABS = new Set(['ask', 'check', 'history', 'documents', 'ingest', 'jobs', 'inspect']);
```

- [ ] **Step 4: Route the view** — in `renderActiveView()` (app.js ~316), after the `check` line add:

```javascript
  if (state.tab === 'history') renderHistory();
```

- [ ] **Step 5: Add history state + view functions** (place near `renderCheck`, e.g. after `loadRecentChecks` is removed in Task 5; for now append after `renderCheckNew`'s helpers):

```javascript
const historyState = { q: '', runs: [] };

function renderHistory() {
  const view = $('history-view');
  if (!view) return;
  view.innerHTML = `
    ${pageHeader('History', 'Past checking sessions across all folders.', '')}
    <div class="history-toolbar">
      <input id="hist-search" type="search" placeholder="Search by document, type or folder…" value="${escapeHtml(historyState.q)}">
    </div>
    <div id="hist-list" class="list"><div class="muted">Loading…</div></div>
  `;
  const search = $('hist-search');
  let t;
  search.addEventListener('input', () => {
    clearTimeout(t);
    t = setTimeout(() => { historyState.q = search.value.trim(); loadHistory(); }, 250);
  });
  loadHistory();
}

async function loadHistory() {
  const el = $('hist-list');
  if (!el) return;
  try {
    const qs = historyState.q ? `?q=${enc(historyState.q)}` : '';
    const runs = await api(`/api/runs${qs}`);
    historyState.runs = runs;
    if (!runs.length) {
      el.innerHTML = `<div class="muted">${historyState.q
        ? `No checks match “${escapeHtml(historyState.q)}”.`
        : 'No checks yet.'}</div>`;
      return;
    }
    el.innerHTML = runs.map(renderHistoryRow).join('');
    el.querySelectorAll('.hist-open').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault();
      setTab('check');
      openCheckRun(a.dataset.runid);
    }));
    el.querySelectorAll('.hist-del').forEach((b) => b.addEventListener('click', () => confirmDeleteRun(b.dataset.runid)));
  } catch (err) {
    el.innerHTML = `<div class="muted">Could not load history: ${escapeHtml(err.message)}</div>`;
  }
}

function renderHistoryRow(r) {
  const name = escapeHtml(r.submitted_name || r.document_type || 'document');
  const folder = escapeHtml(r.project_number || '');
  const when = escapeHtml(r.created_at || '');
  const n = r.finding_count || 0;
  const count = r.status === 'done' ? `${n} finding${n === 1 ? '' : 's'}` : '';
  const busy = r.status === 'running' || r.status === 'queued';
  const del = busy ? '' : `<button class="ghost small hist-del" data-runid="${r.id}" title="Delete">🗑</button>`;
  return `<div class="recent-row hist-row">
    <span class="status-pill" data-s="${r.status}">${escapeHtml(r.status)}</span>
    <span class="hist-name">${name}</span>
    <span class="muted">${folder}</span>
    <span class="muted">${when}</span>
    <span class="muted">${count}</span>
    <a href="#" class="hist-open" data-runid="${r.id}">open</a>
    ${del}
  </div>`;
}
```

- [ ] **Step 6: Smoke** — reload the UI, click the new **History** tab. Expect runs from all folders, newest first; `open` loads the results view; typing in search filters.

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/app.js
git commit -m "feat(checker): dedicated History view (all folders, search)"
```

---

### Task 4: Frontend — delete confirm dialog

**Files:**
- Modify: `static/app.js` (`confirmDialog` helper + `confirmDeleteRun`)

- [ ] **Step 1: Add a reusable confirm dialog** (mirrors the existing `.modal-overlay`/`.modal` pattern from `renderMoveDialog`):

```javascript
function confirmDialog({ title, body, confirmLabel = 'Confirm', cancelLabel = 'Cancel' }) {
  return new Promise((resolve) => {
    const existing = $('confirm-dialog');
    if (existing) existing.remove();
    const dlg = document.createElement('div');
    dlg.id = 'confirm-dialog';
    dlg.className = 'modal-overlay';
    dlg.innerHTML = `
      <div class="modal">
        <div class="modal-head">
          <h3>${escapeHtml(title)}</h3>
          <button class="ghost small" type="button" data-act="cancel" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
          <div class="muted">${escapeHtml(body)}</div>
          <div class="modal-actions">
            <button class="ghost" type="button" data-act="cancel">${escapeHtml(cancelLabel)}</button>
            <button class="primary" type="button" data-act="ok">${escapeHtml(confirmLabel)}</button>
          </div>
        </div>
      </div>`;
    const done = (val) => { dlg.remove(); resolve(val); };
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg) return done(false);
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'cancel') done(false);
      if (act === 'ok') done(true);
    });
    document.body.appendChild(dlg);
  });
}

async function confirmDeleteRun(runId) {
  const ok = await confirmDialog({
    title: 'Delete check',
    body: 'Permanently delete this check and its annotated PDF? This cannot be undone.',
    confirmLabel: 'Delete',
  });
  if (!ok) return;
  try {
    await api(`/api/runs/${enc(runId)}`, { method: 'DELETE' });
    showToast('Check deleted');
    historyState.runs = historyState.runs.filter((r) => r.id !== runId);
    loadHistory();
  } catch (err) {
    showToast(err.message, 'error');
  }
}
```

- [ ] **Step 2: Smoke** — in History, click 🗑 on a `done` run → confirm → row disappears, toast shows. Cancel leaves it. The 🗑 is absent on running/queued rows.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat(checker): confirm-and-delete a past check from History"
```

---

### Task 5: Frontend — retire the inline "Recent checks" panel

**Files:**
- Modify: `static/app.js` (`renderCheckNew` panel → link; remove `loadRecentChecks`)

- [ ] **Step 1: Replace the recent panel** in `renderCheckNew` (app.js:1574). Change:

```javascript
    <div class="check-recent"><h3>Recent checks</h3><div id="chk-recent" class="muted">Loading…</div></div>
```
to:
```javascript
    <div class="check-recent"><a href="#" id="chk-see-history">See past checks →</a></div>
```

- [ ] **Step 2: Replace the `loadRecentChecks()` call** (app.js:1586) with the link binding:

```javascript
  $('chk-see-history').addEventListener('click', (e) => { e.preventDefault(); setTab('history'); });
```

- [ ] **Step 3: Delete the `loadRecentChecks` function** (app.js:1617–1625) entirely — it is now unused.

- [ ] **Step 4: Smoke** — open Check page: no inline list, just the "See past checks →" link, which switches to History.

- [ ] **Step 5: Commit**

```bash
git add static/app.js
git commit -m "refactor(checker): Check page links to History instead of inline recent list"
```

---

### Task 6: End-to-end smoke (matches spec test plan)

- [ ] Run a check on a small `samples/` PDF; confirm it appears in History across folders.
- [ ] Search by part of the document name; confirm filtering.
- [ ] Open from History; confirm the results split-view loads.
- [ ] Delete a `done` run; confirm: card disappears; `check_runs`/`uploads`/`findings` rows gone; `ANNOTATED_DIR/{run_id}` gone; the submitted `doc_id` no longer in `check_docs.db` (`SELECT … FROM documents WHERE id=?` returns nothing); a `run_deleted` audit row exists.
- [ ] Confirm a `running` run shows no delete button and `DELETE` returns 409.

---

## Self-Review

- **Spec coverage:** History view (T3), all-folders + search (T1/T2/T3), hard single delete + confirm (T2/T4), full cleanup incl. doc purge (T1/T2), retire inline panel (T5), guards/edge cases (T2/T4), manual test plan (T6). ✓
- **Placeholders:** none — every step has concrete code/commands. ✓
- **Type/name consistency:** `list_run_cards`, `delete_run`, `_purge_ingested_doc` used identically across store/routes; frontend `historyState`, `renderHistory`, `loadHistory`, `renderHistoryRow`, `confirmDialog`, `confirmDeleteRun` consistent; `config.DOCS_DB`/`ANNOTATED_DIR`/`EXPORTS_DIR` match `config.py`. ✓
