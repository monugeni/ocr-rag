"""
Survey LLM heading-correction data already accumulated in the corpus.

Reads:
  - SQLite docs.db -> documents, sections, corrections tables
  - Sidecar {stem}_corrections.json files next to each PDF (if discoverable)

Reports counts and per-document breakdown so we can decide whether
there is enough labelled data to train a classifier.

Usage:
  python scripts/survey_heading_data.py --db docs.db
  python scripts/survey_heading_data.py --db docs.db --pdf-root samples/
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def survey_db(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    out: dict = {}

    out["documents"] = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    out["pages"] = cur.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    out["sections"] = cur.execute("SELECT COUNT(*) FROM sections").fetchone()[0]

    by_level = Counter()
    for row in cur.execute("SELECT level, COUNT(*) c FROM sections GROUP BY level"):
        by_level[row["level"]] = row["c"]
    out["sections_by_level"] = dict(by_level)

    by_cat: Counter = Counter()
    by_action: Counter = Counter()
    rows = cur.execute(
        "SELECT category, action, payload FROM corrections"
    ).fetchall()
    out["corrections_total"] = len(rows)

    heading_pos = 0
    heading_neg = 0
    heading_level_change = 0
    new_headings = 0
    removed_headings = 0
    docs_with_heading_corrections: set[int] = set()

    for r in rows:
        by_cat[r["category"]] += 1
        by_action[(r["category"], r["action"])] += 1
        if r["category"] not in ("heading", "headings"):
            continue
        try:
            payload = json.loads(r["payload"])
        except (TypeError, ValueError):
            continue
        action = r["action"]
        if action in ("override", "update"):
            if isinstance(payload, dict):
                if payload.get("is_heading") is False:
                    heading_neg += 1
                else:
                    heading_pos += 1
                    if "level" in payload:
                        heading_level_change += 1
        elif action in ("add", "new"):
            heading_pos += 1
            new_headings += 1
        elif action in ("remove", "delete"):
            heading_neg += 1
            removed_headings += 1

    out["corrections_by_category"] = dict(by_cat)
    out["heading_positive_db"] = heading_pos
    out["heading_negative_db"] = heading_neg
    out["heading_level_changes_db"] = heading_level_change
    out["new_headings_db"] = new_headings
    out["removed_headings_db"] = removed_headings

    return out


def find_sidecars(pdf_roots: list[Path]) -> list[Path]:
    sidecars: list[Path] = []
    for root in pdf_roots:
        if not root.exists():
            continue
        sidecars.extend(root.rglob("*_corrections.json"))
    return sidecars


def survey_sidecars(sidecars: list[Path]) -> dict:
    pos = 0
    neg = 0
    level_changes = 0
    new_h = 0
    removed_h = 0
    by_doc: dict[str, dict] = defaultdict(lambda: {"pos": 0, "neg": 0})

    for path in sidecars:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stem = path.stem.replace("_corrections", "")
        for key, override in (data.get("heading_overrides") or {}).items():
            if override.get("is_heading") is False:
                neg += 1
                by_doc[stem]["neg"] += 1
            else:
                pos += 1
                by_doc[stem]["pos"] += 1
                if "level" in override:
                    level_changes += 1
        for h in data.get("new_headings") or []:
            pos += 1
            new_h += 1
            by_doc[stem]["pos"] += 1
        for h in data.get("removed_headings") or []:
            neg += 1
            removed_h += 1
            by_doc[stem]["neg"] += 1

    return {
        "sidecar_files": len(sidecars),
        "heading_positive_sidecar": pos,
        "heading_negative_sidecar": neg,
        "heading_level_changes_sidecar": level_changes,
        "new_headings_sidecar": new_h,
        "removed_headings_sidecar": removed_h,
        "docs_with_corrections": len(by_doc),
        "top_corrected_docs": sorted(
            ((s, d["pos"] + d["neg"], d) for s, d in by_doc.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:10],
    }


def verdict(total_pos: int, total_neg: int, docs_corrected: int, docs_total: int) -> str:
    """Heuristic recommendation."""
    if total_pos < 200 and total_neg < 50:
        return (
            "INSUFFICIENT for training. Use rule-based extractor_v2 (DocLayout-YOLO + "
            "numbering parser) and accumulate corrections first. Re-run this in a few weeks."
        )
    if total_pos < 1000 or total_neg < 200:
        return (
            "MARGINAL. Train a binary heading-vs-not classifier only (skip the level model). "
            "Expect modest gains over the existing heuristic. Use cross-doc CV to avoid overfit."
        )
    if docs_corrected < max(20, docs_total // 20):
        return (
            "Volume is OK but corrections concentrate in too few documents. Risk of overfitting "
            "to those tenders. Train with grouped CV by document; ensure held-out docs."
        )
    return (
        "READY. Train both binary heading classifier and level (1-4) classifier. "
        "Use grouped CV by document. Target: beat extractor.py F1 by 5-10pp."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="Path to docs.db")
    ap.add_argument(
        "--pdf-root",
        action="append",
        default=[],
        help="Folder to scan for *_corrections.json sidecars (repeatable)",
    )
    ap.add_argument("--out", help="Write report JSON to this path")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: db not found: {args.db}", file=sys.stderr)
        sys.exit(2)

    conn = connect(args.db)
    db_report = survey_db(conn)

    sidecar_paths = find_sidecars([Path(p) for p in args.pdf_root])
    sc_report = survey_sidecars(sidecar_paths)

    total_pos = db_report["heading_positive_db"] + sc_report["heading_positive_sidecar"]
    total_neg = db_report["heading_negative_db"] + sc_report["heading_negative_sidecar"]
    rec = verdict(
        total_pos,
        total_neg,
        sc_report["docs_with_corrections"],
        db_report["documents"],
    )

    print("=" * 70)
    print("OCR-RAG heading corpus survey")
    print("=" * 70)
    print(f"Database: {args.db}")
    print(f"  documents: {db_report['documents']}")
    print(f"  pages:     {db_report['pages']}")
    print(f"  sections:  {db_report['sections']}")
    print(f"  sections_by_level: {db_report['sections_by_level']}")
    print()
    print(f"Corrections rows: {db_report['corrections_total']}")
    for cat, c in sorted(db_report["corrections_by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} {c}")
    print()
    print(f"Sidecar files scanned: {sc_report['sidecar_files']}")
    print(f"Documents with sidecar corrections: {sc_report['docs_with_corrections']}")
    print()
    print("Heading labels available for training:")
    print(f"  positive (is-heading) DB:      {db_report['heading_positive_db']}")
    print(f"  positive (is-heading) sidecar: {sc_report['heading_positive_sidecar']}")
    print(f"  negative (not-heading) DB:     {db_report['heading_negative_db']}")
    print(f"  negative (not-heading) sidecar:{sc_report['heading_negative_sidecar']}")
    print(f"  level reassignments:           "
          f"{db_report['heading_level_changes_db'] + sc_report['heading_level_changes_sidecar']}")
    print(f"  brand-new headings (LLM-added):"
          f"{db_report['new_headings_db'] + sc_report['new_headings_sidecar']}")
    print(f"  removed headings (LLM-removed):"
          f"{db_report['removed_headings_db'] + sc_report['removed_headings_sidecar']}")
    print()
    print("Top corrected documents:")
    for stem, total, breakdown in sc_report["top_corrected_docs"]:
        print(f"  {total:5d}  {stem}  (+{breakdown['pos']} / -{breakdown['neg']})")
    print()
    print("VERDICT:")
    print(f"  {rec}")
    print()

    if args.out:
        full = {
            "db": db_report,
            "sidecar": {k: v for k, v in sc_report.items() if k != "top_corrected_docs"},
            "verdict": rec,
            "totals": {"positive": total_pos, "negative": total_neg},
        }
        Path(args.out).write_text(json.dumps(full, indent=2, default=str))
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
