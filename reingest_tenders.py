#!/usr/bin/env python3
"""Re-ingest tender-folder documents through the fixed fast pipeline.

Use this after deploying the extraction fixes (super/subscript + degree sign +
bordered-table reconstruction) so EXISTING documents pick up the corrected text
— new ingestions already use it.

It selects every document whose folder matches a SQL LIKE pattern
(default ``%tender%``, case-insensitive), takes ONE backup of the DB, then
re-ingests each in place (``replace=True``). LLM metadata is skipped by default
(the corrections sidecars are replayed by ingest); pass ``--with-embeddings`` to
recompute semantic vectors.

Examples
--------
    # dev, against a scratch copy
    python reingest_tenders.py --db data/docs.db --root .

    # production (run as the service user, ideally with the web service stopped
    # so the re-ingest writes don't contend with the live server):
    sudo systemctl stop ocr-rag-mcp
    sudo -u ocrrag /btrfs/ocr-rag/venv/bin/python \
        /btrfs/ocr-rag/reingest_tenders.py \
        --db /btrfs/ocr-rag/data/docs.db --root /btrfs/ocr-rag
    sudo systemctl start ocr-rag-mcp

    # preview only — list the folders/docs that WOULD be re-ingested
    python reingest_tenders.py --db data/docs.db --root . --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest import init_db, ingest_fast_pdf
from reingest_fast import _backup_db, _resolve_pdf_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="SQLite database path (the company KB, e.g. docs.db).")
    ap.add_argument("--pattern", default="%tender%",
                    help="SQL LIKE on the folder name, case-insensitive. Default '%%tender%%'.")
    ap.add_argument("--root", default=".", help="App root for fallback PDF lookup. Default: cwd.")
    ap.add_argument("--no-backup", action="store_true", help="Skip the one-time DB backup.")
    ap.add_argument("--with-embeddings", action="store_true",
                    help="Recompute sentence-transformer embeddings (slow; needs setup_embeddings deps).")
    ap.add_argument("--limit", type=int, help="Cap the number of documents (for a trial batch).")
    ap.add_argument("--dry-run", action="store_true", help="List what would be re-ingested, change nothing.")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()

    conn = init_db(str(db_path))
    rows = conn.execute(
        "SELECT id, project, filename, pdf_path FROM documents "
        "WHERE project LIKE ? COLLATE NOCASE ORDER BY project, id",
        (args.pattern,),
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    folders = sorted({r["project"] for r in rows})
    print(f"Pattern: {args.pattern!r}")
    print(f"Matching folders ({len(folders)}):")
    for f in folders:
        print(f"  - {f}")
    print(f"Documents selected: {len(rows)}")
    if not rows:
        conn.close()
        return 0

    plan, missing = [], []
    for row in rows:
        path = _resolve_pdf_path(row, root)
        (plan if path.exists() else missing).append((row, path))

    if missing:
        print(f"\nMissing source PDFs ({len(missing)}) — aborting so nothing is half-rebuilt:")
        for row, path in missing:
            print(f"  id={row['id']} {row['project']} :: {row['filename']} -> {path}")
        conn.close()
        return 3

    if args.dry_run:
        print("\nDry run — nothing re-ingested.")
        for row, path in plan:
            print(f"  would reingest: {row['project']} :: {row['filename']}")
        conn.close()
        return 0

    if not args.no_backup:
        print(f"Backup: {_backup_db(db_path)}")

    ok, failed = 0, []
    for i, (row, path) in enumerate(plan, start=1):
        print(f"[{i}/{len(plan)}] {row['project']} :: {row['filename']}")
        try:
            ingest_fast_pdf(
                conn, path, row["project"],
                replace=True, skip_llm=True,
                skip_embeddings=not args.with_embeddings,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001 — one bad PDF must not abort the batch
            conn.rollback()
            failed.append((row, exc))
            print(f"  FAILED: {exc}")

    stats = conn.execute(
        "SELECT (SELECT COUNT(*) FROM documents) docs, "
        "(SELECT COUNT(*) FROM pages) pages, "
        "(SELECT COUNT(*) FROM chunks) chunks, "
        "(SELECT COUNT(*) FROM chunks_fts) chunks_fts"
    ).fetchone()
    conn.close()

    print(f"\nDone. Successful: {ok}  Failed: {len(failed)}")
    print(dict(stats))
    if failed:
        print("Failures:")
        for row, exc in failed:
            print(f"  id={row['id']} {row['project']} :: {row['filename']} :: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
