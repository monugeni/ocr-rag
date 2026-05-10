"""
A/B benchmark: extractor_v2 vs Marker JSON output.

For one or many PDFs, compare:
  - wall-clock time
  - per-page text similarity (token overlap)
  - heading recall/precision against Marker (treating Marker as one signal,
    NOT ground truth — useful for spotting drift)
  - heading recall/precision against the post-correction docs.db sections
    (treating those as the closest thing to ground truth)
  - level distribution and breadcrumb agreement

Usage:
  # Compare extractor_v2 vs an existing Marker JSON
  python scripts/bench_extractors.py --pdf file.pdf --marker file.json

  # Batch: for every doc in docs.db, compare v2 vs the DB sections
  python scripts/bench_extractors.py --db docs.db --pdf-root samples/ --limit 5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extractor_v2 import extract  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


# ---------------------------------------------------------------------------
# Marker parsing (mirrors ingest.parse_marker_json briefly)
# ---------------------------------------------------------------------------

def parse_marker(marker_json_path: Path) -> dict:
    data = json.loads(marker_json_path.read_text())
    pages = []
    sections = []
    seen = set()

    def walk(block, page_num: int, header_index: dict):
        bt = block.get("block_type", "")
        if bt.startswith("SectionHeader") or bt == "SectionHeader":
            text = block.get("html") or block.get("text") or ""
            text = text.strip()
            level = block.get("section_hierarchy_level") or block.get("level") or 1
            key = (page_num, normalize(text))
            if text and key not in seen:
                seen.add(key)
                sections.append({
                    "heading": text,
                    "level": int(level),
                    "page_num": page_num,
                })
        for child in block.get("children", []) or []:
            walk(child, page_num, header_index)

    for page in data.get("children", []) or [data]:
        page_num = page.get("page", page.get("page_num"))
        if not isinstance(page_num, int):
            continue
        text = page.get("html") or page.get("text") or ""
        pages.append({"page_num": page_num, "content": text})
        for child in page.get("children", []) or []:
            walk(child, page_num, {})

    return {"pages": pages, "sections": sections}


# ---------------------------------------------------------------------------
# DB ground truth
# ---------------------------------------------------------------------------

def db_sections(conn: sqlite3.Connection, doc_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT heading, level, page_start FROM sections WHERE doc_id=? ORDER BY page_start, id",
        (doc_id,),
    ).fetchall()
    return [
        {"heading": r["heading"], "level": int(r["level"] or 0),
         "page_num": int(r["page_start"] or 0)}
        for r in rows
    ]


def find_doc_by_filename(conn: sqlite3.Connection, name: str):
    return conn.execute(
        "SELECT id, filename, pdf_path FROM documents WHERE filename=? OR pdf_path LIKE ?",
        (name, f"%{name}"),
    ).fetchone()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def heading_set(sections: list[dict]) -> set[tuple[int, str]]:
    return {(s["page_num"], normalize(s["heading"])[:80]) for s in sections if s.get("heading")}


def fuzzy_match_set(a: set[tuple[int, str]], b: set[tuple[int, str]],
                    page_window: int = 1) -> int:
    """Count items in b matched by some item in a within +/- page_window."""
    matched = 0
    a_by_page = defaultdict(set)
    for pn, t in a:
        a_by_page[pn].add(t)
    for pn, t in b:
        for d in range(-page_window, page_window + 1):
            for cand in a_by_page.get(pn + d, ()):
                if cand == t or cand.startswith(t[:30]) or t.startswith(cand[:30]):
                    matched += 1
                    break
            else:
                continue
            break
    return matched


def prf(pred: set, truth: set) -> dict:
    if not pred and not truth:
        return {"p": 1.0, "r": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = fuzzy_match_set(pred, truth)
    fp = max(0, len(pred) - tp)
    fn = max(0, len(truth) - tp)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def compare_one(pdf_path: Path, model: str | None = None,
                marker_path: Path | None = None,
                truth: list[dict] | None = None) -> dict:
    t0 = time.time()
    v2 = extract(str(pdf_path), model_path=model)
    v2_sec = v2["sections"]
    out = {
        "pdf": pdf_path.name,
        "v2": {
            "elapsed_sec": v2["stats"]["elapsed_sec"],
            "sec_per_page": v2["stats"]["sec_per_page"],
            "n_sections": len(v2_sec),
            "by_level": dict(Counter(s["level"] for s in v2_sec)),
        },
    }
    v2_set = heading_set(v2_sec)

    if marker_path and marker_path.exists():
        mt0 = time.time()
        m = parse_marker(marker_path)
        m_sec = m["sections"]
        out["marker"] = {
            "n_sections": len(m_sec),
            "by_level": dict(Counter(s["level"] for s in m_sec)),
            "parse_sec": round(time.time() - mt0, 3),
        }
        out["v2_vs_marker"] = prf(v2_set, heading_set(m_sec))

    if truth:
        out["truth_db"] = {
            "n_sections": len(truth),
            "by_level": dict(Counter(s["level"] for s in truth)),
        }
        out["v2_vs_truth"] = prf(v2_set, heading_set(truth))
        if marker_path and marker_path.exists():
            out["marker_vs_truth"] = prf(heading_set(m_sec), heading_set(truth))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", help="Single PDF to test")
    ap.add_argument("--marker", help="Existing Marker JSON for comparison")
    ap.add_argument("--db", help="docs.db for batch + ground truth")
    ap.add_argument("--pdf-root", action="append", default=[],
                    help="Folder(s) to find PDFs referenced by docs.db")
    ap.add_argument("--model", help="Trained heading model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="Write all results to this JSON")
    args = ap.parse_args()

    results = []

    if args.pdf:
        truth = None
        marker_path = Path(args.marker) if args.marker else None
        results.append(compare_one(Path(args.pdf), args.model, marker_path, truth))

    if args.db:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        roots = [Path(p) for p in args.pdf_root]
        docs = list(conn.execute(
            "SELECT id, filename, pdf_path FROM documents ORDER BY id"
        ))
        if args.limit:
            docs = docs[: args.limit]
        for d in docs:
            hint = d["pdf_path"] or d["filename"]
            pdf_path = None
            p = Path(hint or "")
            if p.is_absolute() and p.exists():
                pdf_path = p
            else:
                for root in roots:
                    for cand in root.rglob(p.name):
                        pdf_path = cand
                        break
                    if pdf_path:
                        break
            if not pdf_path:
                print(f"skip {hint!r}: PDF not found")
                continue
            truth = db_sections(conn, d["id"])
            results.append(compare_one(pdf_path, args.model, None, truth))
            r = results[-1]
            print(f"{pdf_path.name}: v2={r['v2']['n_sections']} secs in "
                  f"{r['v2']['elapsed_sec']}s "
                  f"({r['v2']['sec_per_page']}s/page); "
                  f"vs_truth={r.get('v2_vs_truth')}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote: {args.out}")

    # Aggregate
    if len(results) > 1:
        v2_f1 = [r["v2_vs_truth"]["f1"] for r in results if "v2_vs_truth" in r]
        v2_sec_per_page = [r["v2"]["sec_per_page"] for r in results]
        print()
        print("=" * 60)
        print(f"docs:                {len(results)}")
        print(f"v2 sec/page (median): {statistics.median(v2_sec_per_page):.3f}")
        if v2_f1:
            print(f"v2 vs truth F1 (median): {statistics.median(v2_f1):.3f}")
        m_f1 = [r["marker_vs_truth"]["f1"] for r in results if "marker_vs_truth" in r]
        if m_f1:
            print(f"marker vs truth F1 (median): {statistics.median(m_f1):.3f}")


if __name__ == "__main__":
    main()
