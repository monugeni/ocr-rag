"""Database layer.

Two SQLite databases (both WAL):

* ``checker.db`` — app/audit tables defined by ``CHECKER_SCHEMA`` below.
* ``docs.db``    — the *exact ocr-rag schema*, created via ``ingest.init_db`` so
                   the ocr-rag MCP tools operate against it unchanged. Holds the
                   ingested submitted + fresh-reference documents.

Cross-DB ``doc_id`` references are *logical* — never JOIN across the two DBs in
SQL; resolve in Python.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from . import config

# ---------------------------------------------------------------------------
# checker.db schema
# ---------------------------------------------------------------------------
CHECKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY,
    oidc_sub     TEXT UNIQUE NOT NULL,
    email        TEXT,
    display_name TEXT,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login   DATETIME
);

CREATE TABLE IF NOT EXISTS templates (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT,
    instructions     TEXT NOT NULL,
    default_doc_type TEXT,
    severity_scheme  TEXT,
    categories       TEXT,                 -- JSON list, optional restriction
    created_by       INTEGER REFERENCES users(id),
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    archived         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS check_runs (
    id                TEXT PRIMARY KEY,        -- uuid
    status            TEXT NOT NULL DEFAULT 'created',
    stage             TEXT DEFAULT '',
    project_number    TEXT NOT NULL,
    document_type     TEXT,
    originator        TEXT,
    metadata          TEXT,                    -- JSON: arbitrary extra fields
    template_id       INTEGER REFERENCES templates(id),
    guiding_prompt    TEXT,
    is_revision       INTEGER NOT NULL DEFAULT 0,
    prior_run_id      TEXT REFERENCES check_runs(id),
    reference_mode    TEXT,                    -- existing | fresh | both
    reference_project TEXT,                    -- company ocr-rag folder, if any
    ocrrag_project    TEXT,                    -- folder used in checker docs.db
    error             TEXT,
    created_by        INTEGER REFERENCES users(id),
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at        DATETIME,
    finished_at       DATETIME
);

CREATE TABLE IF NOT EXISTS uploads (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES check_runs(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,               -- submitted | reference | prior_commented
    filename      TEXT NOT NULL,
    disk_path     TEXT NOT NULL,
    doc_id        INTEGER,                     -- logical FK into docs.db
    ingest_job_id TEXT,
    ingest_status TEXT DEFAULT 'pending',      -- pending | running | done | failed
    ingest_error  TEXT,
    page_count    INTEGER,
    mime          TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS findings (
    id              INTEGER PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES check_runs(id) ON DELETE CASCADE,
    doc_id          INTEGER,                   -- submitted doc in docs.db
    page_num        INTEGER,
    bbox            TEXT,                       -- JSON [x0,y0,x1,y1] or null
    anchor_text     TEXT,
    annotation_xref INTEGER,                    -- id from pdf-annotator annotate()
    severity        TEXT,                       -- critical|major|minor|observation
    category        TEXT,
    title           TEXT,
    detail          TEXT,
    citation        TEXT,                       -- JSON {ref_doc, ref_page, heading, snippet}
    confidence      TEXT,                       -- high|medium|low
    status          TEXT DEFAULT 'open',        -- open|accepted|dismissed
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comment_incorporation (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES check_runs(id) ON DELETE CASCADE,
    prior_comment_ref TEXT,                     -- annotation xref in old PDF
    prior_comment_text TEXT,
    prior_page        INTEGER,
    verdict           TEXT,                     -- incorporated|partially|not_incorporated|not_applicable
    evidence          TEXT,                     -- JSON citation in new revision
    detail            TEXT,
    annotation_xref   INTEGER,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_events (
    id         INTEGER PRIMARY KEY,
    run_id     TEXT,
    user_id    INTEGER REFERENCES users(id),
    event_type TEXT NOT NULL,
    payload    TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON check_runs(project_number, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_created ON check_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_uploads_run ON uploads(run_id);
CREATE INDEX IF NOT EXISTS idx_comments_run ON comment_incorporation(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_events(run_id);
"""


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_conn() -> sqlite3.Connection:
    """Connection to the app DB (checker.db)."""
    return _connect(config.CHECKER_DB)


@contextmanager
def conn_ctx():
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def get_docs_conn() -> sqlite3.Connection:
    """Connection to the ocr-rag-schema docs DB (docs.db)."""
    return _connect(config.DOCS_DB)


def init_databases() -> None:
    """Create both databases if missing. Safe to call on every startup."""
    config.ensure_dirs()

    # checker.db — our schema
    conn = get_conn()
    try:
        conn.executescript(CHECKER_SCHEMA)
        # Additive column migrations for pre-existing DBs.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        if "is_admin" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    finally:
        conn.close()

    # docs.db — ocr-rag schema (import is wired via config.wire_sibling_paths())
    import ingest  # noqa: E402  (sibling library, path injected in config)

    docs_conn = ingest.init_db(config.DOCS_DB)
    docs_conn.close()
