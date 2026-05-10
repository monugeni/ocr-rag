"""
Build a (features, label) dataset for training a heading classifier.

Strategy
--------
For every document in docs.db:
  1. Re-extract per-line features from the original PDF using extractor.py's
     extract_lines_from_page() (same features the existing scorer uses).
  2. Build the label set from three sources, in priority order:
       a) Sidecar JSON corrections (most authoritative).
       b) corrections table rows in docs.db.
       c) Final 'sections' rows in docs.db (post-correction truth).
  3. For each line:
       - is_heading = True/False
       - level = 1..4 if heading
       - source = 'sidecar_override' | 'sidecar_new' | 'sidecar_removed'
                | 'db_correction' | 'db_section' | 'unlabelled'
  4. Subsample unlabelled negatives to keep class balance reasonable.

Output: JSONL, one row per line, ready for the trainer.

Usage:
  python scripts/build_heading_dataset.py \
      --db docs.db --pdf-root samples/ --out heading_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Reuse extractor.py helpers (must be importable)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

from extractor import (  # noqa: E402
    ExtractedLine,
    cluster_font_sizes,
    detect_running_headers,
    extract_lines_from_page,
    parse_decimal_numbering,
)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def label_index_from_sidecar(sidecar: dict) -> dict[tuple[int, str], dict]:
    """Build {(page_num, normalized_text_prefix): label_dict}."""
    idx: dict[tuple[int, str], dict] = {}
    for key, override in (sidecar.get("heading_overrides") or {}).items():
        try:
            page_str, text_prefix = key.split(":", 1)
            pn = int(page_str)
        except (ValueError, AttributeError):
            continue
        idx[(pn, _norm(text_prefix))] = {
            "is_heading": override.get("is_heading", True),
            "level": override.get("level"),
            "source": "sidecar_override",
        }
    for h in sidecar.get("new_headings") or []:
        idx[(int(h["page_num"]), _norm(h["text"][:50]))] = {
            "is_heading": True,
            "level": int(h.get("level", 2)),
            "source": "sidecar_new",
        }
    for h in sidecar.get("removed_headings") or []:
        idx[(int(h["page_num"]), _norm(h["text_prefix"]))] = {
            "is_heading": False,
            "level": None,
            "source": "sidecar_removed",
        }
    return idx


def label_index_from_db(conn: sqlite3.Connection, doc_id: int) -> dict[tuple[int, str], dict]:
    idx: dict[tuple[int, str], dict] = {}
    cur = conn.cursor()
    # Final sections (post-correction). These are positives at known level.
    for row in cur.execute(
        "SELECT page_start, heading, level FROM sections WHERE doc_id=?",
        (doc_id,),
    ):
        if row["page_start"] is None or row["heading"] is None:
            continue
        idx[(int(row["page_start"]), _norm(row["heading"][:50]))] = {
            "is_heading": True,
            "level": int(row["level"]) if row["level"] is not None else 2,
            "source": "db_section",
        }
    # Explicit corrections (override final sections if conflicting).
    for row in cur.execute(
        "SELECT category, action, payload FROM corrections WHERE doc_id=? "
        "AND category IN ('heading','headings')",
        (doc_id,),
    ):
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            continue
        action = row["action"]
        if action in ("override", "update") and isinstance(payload, dict):
            pn = payload.get("page_num")
            text = payload.get("text") or payload.get("text_prefix") or ""
            if pn is None or not text:
                continue
            idx[(int(pn), _norm(str(text)[:50]))] = {
                "is_heading": payload.get("is_heading", True),
                "level": payload.get("level"),
                "source": "db_correction",
            }
        elif action in ("add", "new") and isinstance(payload, dict):
            pn = payload.get("page_num")
            text = payload.get("text") or ""
            if pn is None or not text:
                continue
            idx[(int(pn), _norm(text[:50]))] = {
                "is_heading": True,
                "level": int(payload.get("level", 2)),
                "source": "db_correction",
            }
        elif action in ("remove", "delete") and isinstance(payload, dict):
            pn = payload.get("page_num")
            text = payload.get("text") or payload.get("text_prefix") or ""
            if pn is None or not text:
                continue
            idx[(int(pn), _norm(str(text)[:50]))] = {
                "is_heading": False,
                "level": None,
                "source": "db_correction",
            }
    return idx


def find_label(idx: dict, page_num: int, text: str):
    norm = _norm(text[:50])
    if (page_num, norm) in idx:
        return idx[(page_num, norm)]
    # Loose prefix match — labels keyed on first N chars
    for (pn, prefix), label in idx.items():
        if pn != page_num:
            continue
        if norm.startswith(prefix) or prefix.startswith(norm):
            return label
    return None


def featurize(
    line: ExtractedLine,
    body_size: float,
    heading_sizes: list[float],
    median_gap: float,
    left_margin: float,
    page_width: float,
) -> dict:
    num_str, num_depth, _ = parse_decimal_numbering(line.text)
    text = line.text.strip()
    alpha = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))
    return {
        "font_size": line.font_size,
        "font_ratio": (line.font_size / body_size) if body_size else 1.0,
        "is_bold": int(line.is_bold),
        "is_upper": int(line.is_upper),
        "caps_ratio": caps_ratio,
        "word_count": line.word_count,
        "char_count": len(text),
        "x0": line.x0,
        "x0_rel_margin": line.x0 - left_margin if left_margin else 0.0,
        "x0_rel_width": (line.x0 / page_width) if page_width else 0.0,
        "gap_above": line.gap_above,
        "gap_ratio": (line.gap_above / median_gap) if median_gap else 0.0,
        "ends_colon": int(text.endswith(":")),
        "ends_period": int(text.endswith(".")),
        "num_depth": num_depth,
        "has_decimal_num": int(bool(num_str)),
        "heading_size_rank": (
            next((i + 1 for i, t in enumerate(heading_sizes) if line.font_size >= t * 0.95), 0)
        ),
    }


def find_pdf_path(pdf_root: list[Path], path_hint: str) -> Path | None:
    p = Path(path_hint)
    if p.is_absolute() and p.exists():
        return p
    name = p.name
    for root in pdf_root:
        for cand in root.rglob(name):
            return cand
    return None


def find_sidecar(pdf_path: Path) -> dict:
    candidates = [
        pdf_path.with_name(pdf_path.stem + "_corrections.json"),
        pdf_path.parent / pdf_path.stem / "_corrections.json",
        pdf_path.parent / pdf_path.stem / f"{pdf_path.stem}_corrections.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def process_document(
    conn: sqlite3.Connection,
    doc_row: sqlite3.Row,
    pdf_path: Path,
    out_fh,
    neg_per_pos: float = 5.0,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Returns (positives_emitted, negatives_emitted)."""
    rng = rng or random.Random(0)
    sidecar = find_sidecar(pdf_path)
    sidecar_idx = label_index_from_sidecar(sidecar)
    db_idx = label_index_from_db(conn, doc_row["id"])
    # sidecar wins
    label_idx = {**db_idx, **sidecar_idx}

    pos = 0
    neg = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_lines: dict[int, list[ExtractedLine]] = {}
        all_sizes: list[float] = []
        all_gaps: list[float] = []
        all_x0: list[float] = []
        page_widths: dict[int, float] = {}

        for i, page in enumerate(pdf.pages, start=1):
            try:
                lines = extract_lines_from_page(page, i)
            except Exception:  # noqa: BLE001
                continue
            page_lines[i] = lines
            page_widths[i] = float(getattr(page, "width", 0.0))
            for ln in lines:
                if ln.font_size > 0:
                    all_sizes.append(ln.font_size)
                if ln.gap_above > 0:
                    all_gaps.append(ln.gap_above)
                all_x0.append(ln.x0)

        if not page_lines:
            return 0, 0

        body_size = (
            sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0
        )
        heading_sizes = cluster_font_sizes(all_sizes)
        median_gap = sorted(all_gaps)[len(all_gaps) // 2] if all_gaps else 0.0
        left_margin = sorted(all_x0)[len(all_x0) // 20] if all_x0 else 0.0  # 5th percentile
        running = detect_running_headers(page_lines)

        unlabelled_negatives = []

        for pn, lines in page_lines.items():
            for ln in lines:
                norm = " ".join(ln.text.strip().lower().split())
                if norm in running:
                    continue
                label = find_label(label_idx, pn, ln.text)
                feats = featurize(
                    ln,
                    body_size,
                    heading_sizes,
                    median_gap,
                    left_margin,
                    page_widths.get(pn, 0.0),
                )
                row = {
                    "doc_id": doc_row["id"],
                    "doc_filename": pdf_path.name,
                    "page_num": pn,
                    "text": ln.text.strip()[:200],
                    "features": feats,
                }
                if label is not None:
                    row["label"] = int(bool(label["is_heading"]))
                    row["level"] = label.get("level")
                    row["source"] = label["source"]
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if row["label"]:
                        pos += 1
                    else:
                        neg += 1
                else:
                    # candidate unlabelled negative; subsampled later
                    row["label"] = 0
                    row["level"] = None
                    row["source"] = "unlabelled"
                    unlabelled_negatives.append(row)

        # subsample unlabelled negatives proportional to positives in this doc
        budget = int(max(50, pos * neg_per_pos)) if pos else 50
        if len(unlabelled_negatives) > budget:
            unlabelled_negatives = rng.sample(unlabelled_negatives, budget)
        for row in unlabelled_negatives:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            neg += 1

    return pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument(
        "--pdf-root",
        action="append",
        required=True,
        help="Folder(s) holding original PDFs. Repeatable.",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--limit", type=int, default=0, help="Process at most N documents (debug)"
    )
    ap.add_argument(
        "--neg-per-pos",
        type=float,
        default=5.0,
        help="Negative-to-positive ratio for subsampling unlabelled lines",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    pdf_roots = [Path(p) for p in args.pdf_root]

    docs = list(
        conn.execute(
            "SELECT id, filename, pdf_path FROM documents ORDER BY id"
        )
    )
    if args.limit:
        docs = docs[: args.limit]

    total_pos = total_neg = ok = skipped = 0
    by_source: dict[str, int] = defaultdict(int)

    with open(args.out, "w") as out_fh:
        for i, d in enumerate(docs, start=1):
            hint = d["pdf_path"] or d["filename"] or ""
            pdf_path = find_pdf_path(pdf_roots, hint)
            if pdf_path is None:
                skipped += 1
                print(f"[{i}/{len(docs)}] SKIP: PDF not found for {hint}")
                continue
            try:
                pos, neg = process_document(conn, d, pdf_path, out_fh)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(docs)}] ERROR {pdf_path.name}: {e}")
                skipped += 1
                continue
            total_pos += pos
            total_neg += neg
            ok += 1
            print(f"[{i}/{len(docs)}] {pdf_path.name}: +{pos} / -{neg}")

    print()
    print(f"Done. {ok} docs processed, {skipped} skipped.")
    print(f"Positives: {total_pos}   Negatives: {total_neg}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
