#!/usr/bin/env python3
"""
Reingest existing database documents with the fast PDF pipeline.

This is intended for production upgrades where documents already exist in the
database and their source PDFs are still available at documents.pdf_path.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from ingest import init_db, ingest_fast_pdf


def _candidate_paths(root: Path, project: str, filename: str) -> list[Path]:
    return [
        root / "uploads" / project / filename,
        root / "samples" / filename,
    ]


def _resolve_pdf_path(row: sqlite3.Row, root: Path) -> Path:
    pdf_path = Path(row["pdf_path"] or "")
    if pdf_path.exists():
        return pdf_path

    for candidate in _candidate_paths(root, row["project"], row["filename"]):
        if candidate.exists():
            return candidate

    for base in (root / "uploads", root / "samples"):
        if base.exists():
            matches = list(base.glob(f"**/{row['filename']}"))
            for match in matches:
                if match.exists():
                    return match

    return pdf_path


def _backup_db(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db_path.with_name(f"{db_path.name}.before-fast-reingest-{stamp}.bak")
    shutil.copy2(db_path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reingest existing DB documents with fast PDF chunk FTS."
    )
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument(
        "--project",
        help="Optional project/folder to reingest. Defaults to all projects.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Application root used for fallback PDF lookup. Default: current directory.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy the DB before reingesting.",
    )
    parser.add_argument(
        "--with-llm-metadata",
        action="store_true",
        help="Call the metadata LLM for each document. Requires ANTHROPIC_API_KEY or --api-key.",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Compute sentence-transformer embeddings after chunk FTS ingestion.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run ocrmypdf --skip-text before fast extraction to cover scanned pages.",
    )
    parser.add_argument(
        "--ocr-jobs",
        type=int,
        default=4,
        help="ocrmypdf worker count for --ocr.",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key for metadata extraction. Defaults to ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-4-20250514",
        help="Metadata extraction model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for dry production batches.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    if not args.no_backup:
        backup = _backup_db(db_path)
        print(f"Backup: {backup}")

    conn = init_db(str(db_path))
    where = ""
    params: list[str] = []
    if args.project:
        where = "WHERE project = ? OR project LIKE ?"
        params = [args.project, f"{args.project}/%"]

    rows = conn.execute(
        f"SELECT id, project, filename, pdf_path FROM documents {where} ORDER BY id",
        params,
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    plan = []
    missing = []
    for row in rows:
        path = _resolve_pdf_path(row, root)
        if not path.exists():
            missing.append((row["id"], row["project"], row["filename"], str(path)))
        else:
            plan.append((row, path))

    print(f"Documents selected: {len(rows)}")
    print(f"Documents with source PDFs: {len(plan)}")
    if missing:
        print("Missing source PDFs:")
        for item in missing:
            print(f"  id={item[0]} project={item[1]} filename={item[2]} path={item[3]}")
        conn.close()
        return 3

    print(f"LLM metadata: {'enabled' if args.with_llm_metadata else 'skipped'}")
    print(f"Embeddings: {'enabled' if args.with_embeddings else 'skipped'}")
    print(f"OCR preparation: {'enabled' if args.ocr else 'skipped'}")

    ok = 0
    failed = []
    for index, (row, path) in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {row['project']} :: {row['filename']}")
        try:
            ingest_fast_pdf(
                conn,
                path,
                row["project"],
                replace=True,
                skip_llm=not args.with_llm_metadata,
                skip_embeddings=not args.with_embeddings,
                api_key=args.api_key,
                llm_model=args.llm_model,
                ocr=args.ocr,
                ocr_jobs=args.ocr_jobs,
            )
            ok += 1
        except Exception as exc:
            conn.rollback()
            failed.append((row["id"], row["project"], row["filename"], str(exc)))
            print(f"  FAILED: {exc}")

    stats = conn.execute(
        "SELECT "
        "(SELECT COUNT(*) FROM documents) docs, "
        "(SELECT COUNT(*) FROM pages) pages, "
        "(SELECT COUNT(*) FROM pages_fts) pages_fts, "
        "(SELECT COUNT(*) FROM chunks) chunks, "
        "(SELECT COUNT(*) FROM chunks_fts) chunks_fts, "
        "(SELECT COUNT(*) FROM page_embeddings) embeddings"
    ).fetchone()
    conn.close()

    print("Done.")
    print(f"Successful: {ok}")
    print(f"Failed: {len(failed)}")
    print(dict(stats))
    if failed:
        print("Failures:")
        for item in failed:
            print(f"  id={item[0]} project={item[1]} filename={item[2]} error={item[3]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
