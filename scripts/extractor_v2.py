"""
Fast CPU heading + breadcrumb extractor.

Pipeline:
  PDF
   -> pymupdf (fast text + spans + bboxes)
   -> per-line features (same shape as extractor.py)
   -> heading classifier:
        * trained LightGBM if --model present, else
        * falls back to extractor.py's score_heading() heuristic
   -> level resolver:
        * decimal numbering (1.2.3 -> depth 3) is always primary
        * trained level model OR font-size buckets as fallback
   -> stack-based breadcrumb builder

Optional layout refinement (off by default):
  --doclayout-yolo PATH   ONNX model that emits "title" boxes per page;
                          spans inside those boxes get a heading-prior boost.

Output: same (pages, sections) tuple shape as ingest.parse_marker_json(),
so it can be plugged into ingest.py with a one-line swap.

Usage:
  python scripts/extractor_v2.py path/to/file.pdf --out out.json
  python scripts/extractor_v2.py file.pdf --model models/heading/heading_clf.joblib
  python scripts/extractor_v2.py file.pdf --doclayout-yolo doclayout_yolo.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor import (  # noqa: E402
    ExtractedLine,
    Heading,
    cluster_font_sizes,
    detect_running_headers,
    parse_decimal_numbering,
    score_heading,
    build_breadcrumbs,
)


# ---------------------------------------------------------------------------
# pymupdf-based line extraction (fast)
# ---------------------------------------------------------------------------

def extract_lines_pymupdf(doc: fitz.Document) -> tuple[dict[int, list[ExtractedLine]], dict[int, float]]:
    """Return {page_num: [ExtractedLine]} and {page_num: page_width}."""
    page_lines: dict[int, list[ExtractedLine]] = {}
    page_widths: dict[int, float] = {}

    for page_num, page in enumerate(doc, start=1):
        page_widths[page_num] = float(page.rect.width)
        d = page.get_text("dict")
        prev_bottom = 0.0
        lines: list[ExtractedLine] = []
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue  # image block
            for ln in block.get("lines", []):
                spans = ln.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                sizes = [s.get("size", 0) for s in spans if s.get("size")]
                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                fonts = {s.get("font", "") for s in spans}
                is_bold = any(
                    "Bold" in f or "bold" in f or s.get("flags", 0) & 16
                    for f, s in zip(fonts, spans)
                )
                bbox = ln.get("bbox", [0, 0, 0, 0])
                x0, top, x1, bottom = bbox
                gap = top - prev_bottom if prev_bottom > 0 else 0.0
                is_upper = (
                    text == text.upper()
                    and len(text) > 3
                    and any(c.isalpha() for c in text)
                )
                lines.append(ExtractedLine(
                    page_num=page_num,
                    text=text,
                    font_size=round(avg_size, 1),
                    is_bold=is_bold,
                    is_upper=is_upper,
                    word_count=len(text.split()),
                    top=top,
                    bottom=bottom,
                    x0=x0,
                    gap_above=gap,
                    fonts=fonts,
                ))
                prev_bottom = bottom
        page_lines[page_num] = lines
    return page_lines, page_widths


# ---------------------------------------------------------------------------
# Feature builder (must match build_heading_dataset.featurize)
# ---------------------------------------------------------------------------

FEATURE_KEYS = [
    "font_size", "font_ratio", "is_bold", "is_upper", "caps_ratio",
    "word_count", "char_count", "x0", "x0_rel_margin", "x0_rel_width",
    "gap_above", "gap_ratio", "ends_colon", "ends_period",
    "num_depth", "has_decimal_num", "heading_size_rank",
]


def featurize(line, body_size, heading_sizes, median_gap, left_margin, page_width):
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
            next((i + 1 for i, t in enumerate(heading_sizes)
                  if line.font_size >= t * 0.95), 0)
        ),
    }


# ---------------------------------------------------------------------------
# DocLayout-YOLO prior (optional)
# ---------------------------------------------------------------------------

class DocLayoutYOLOPrior:
    """Run a DocLayout-YOLO ONNX model and return per-page 'title' boxes."""

    def __init__(self, model_path: str, conf: float = 0.25):
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            raise SystemExit(
                "onnxruntime not installed. pip install onnxruntime"
            )
        from doclayout_yolo import YOLOv10  # type: ignore  # noqa: E402

        self.model = YOLOv10(model_path)
        self.conf = conf

    def title_boxes(self, page_image) -> list[tuple[float, float, float, float]]:
        res = self.model.predict(page_image, conf=self.conf, verbose=False)
        boxes = []
        for r in res:
            names = r.names
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                cls_name = names.get(int(c), "")
                if cls_name in ("title", "section_header", "Title", "Section-header"):
                    boxes.append(tuple(b))
        return boxes


def render_page_image(page: fitz.Page, dpi: int = 96):
    """Render a PyMuPDF page to a numpy image for YOLO inference."""
    import numpy as np
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return img


def line_in_any_box(line: ExtractedLine, boxes) -> bool:
    cx = (line.x0 + line.x0 + 100) / 2  # rough center; line bbox not stored fully
    cy = (line.top + line.bottom) / 2
    for x0, y0, x1, y1 in boxes:
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


# ---------------------------------------------------------------------------
# Classifier-based detection
# ---------------------------------------------------------------------------

def detect_headings_with_model(
    page_lines, model_bundle, body_size, heading_sizes, median_gap,
    left_margin, page_widths, running, yolo_titles_by_page=None, threshold=None
) -> list[Heading]:
    import numpy as np
    bin_clf = model_bundle["binary"]
    lvl_clf = model_bundle.get("level")
    feature_keys = model_bundle["feature_keys"]
    thr = threshold if threshold is not None else model_bundle.get("threshold", 0.5)

    # Build feature matrix in one pass
    rows = []
    refs = []
    for pn, lines in page_lines.items():
        for ln in lines:
            norm = " ".join(ln.text.lower().split())
            if norm in running:
                continue
            if len(ln.text.strip()) < 4:
                continue
            f = featurize(ln, body_size, heading_sizes, median_gap,
                          left_margin, page_widths.get(pn, 0.0))
            rows.append([f.get(k, 0) or 0 for k in feature_keys])
            refs.append((pn, ln, f))
    if not rows:
        return []
    X = np.array(rows, dtype=np.float32)
    proba = bin_clf.predict_proba(X)[:, 1]

    # YOLO title boxes act as a prior boost
    if yolo_titles_by_page:
        for i, (pn, ln, _) in enumerate(refs):
            if line_in_any_box(ln, yolo_titles_by_page.get(pn, [])):
                proba[i] = max(proba[i], 0.8)

    headings: list[Heading] = []
    Xh = []
    Xh_idx = []
    for i, p in enumerate(proba):
        if p < thr:
            continue
        pn, ln, _ = refs[i]
        Xh.append(rows[i])
        Xh_idx.append(i)
    if not Xh_idx:
        return []

    if lvl_clf is not None:
        levels = lvl_clf.predict(np.array(Xh, dtype=np.float32)) + 1
    else:
        levels = [None] * len(Xh_idx)

    for j, i in enumerate(Xh_idx):
        pn, ln, f = refs[i]
        num_str, num_depth, remaining = parse_decimal_numbering(ln.text)
        if num_depth > 0:
            level = min(num_depth, 4)
        elif levels[j] is not None:
            level = int(levels[j])
        else:
            level = 2
        headings.append(Heading(
            text=ln.text.strip(),
            level=level,
            page_num=pn,
            confidence=float(proba[i]),
            signals=["clf"] + (["yolo"] if yolo_titles_by_page else []),
            numbering=num_str,
        ))
    return headings


# ---------------------------------------------------------------------------
# Heuristic fallback (uses existing extractor.py)
# ---------------------------------------------------------------------------

def detect_headings_heuristic(page_lines, body_size, heading_sizes, median_gap,
                              left_margin, running) -> list[Heading]:
    out: list[Heading] = []
    for pn, lines in page_lines.items():
        for ln in lines:
            h = score_heading(
                ln, body_size, heading_sizes, median_gap, left_margin, running
            )
            if h:
                out.append(h)
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def extract(pdf_path: str, model_path: Optional[str] = None,
            doclayout_yolo: Optional[str] = None,
            threshold: Optional[float] = None) -> dict:
    t0 = time.time()
    doc = fitz.open(pdf_path)
    page_lines, page_widths = extract_lines_pymupdf(doc)
    t_extract = time.time() - t0

    # Page text for output
    page_text: dict[int, str] = {}
    for pn, lines in page_lines.items():
        page_text[pn] = "\n".join(ln.text for ln in lines)

    # Stats for features
    all_sizes = [ln.font_size for lines in page_lines.values() for ln in lines if ln.font_size]
    all_gaps = [ln.gap_above for lines in page_lines.values() for ln in lines if ln.gap_above > 0]
    all_x0 = [ln.x0 for lines in page_lines.values() for ln in lines]
    body_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0
    heading_sizes = cluster_font_sizes(all_sizes)
    median_gap = sorted(all_gaps)[len(all_gaps) // 2] if all_gaps else 0.0
    left_margin = sorted(all_x0)[len(all_x0) // 20] if all_x0 else 0.0
    running = detect_running_headers(page_lines)

    # Optional YOLO prior
    yolo_titles_by_page = None
    if doclayout_yolo:
        prior = DocLayoutYOLOPrior(doclayout_yolo)
        yolo_titles_by_page = {}
        for page_num in page_lines.keys():
            img = render_page_image(doc[page_num - 1])
            yolo_titles_by_page[page_num] = prior.title_boxes(img)

    if model_path:
        import joblib
        bundle = joblib.load(model_path)
        headings = detect_headings_with_model(
            page_lines, bundle, body_size, heading_sizes, median_gap,
            left_margin, page_widths, running, yolo_titles_by_page, threshold
        )
        method = "lightgbm"
    else:
        headings = detect_headings_heuristic(
            page_lines, body_size, heading_sizes, median_gap, left_margin, running
        )
        method = "heuristic"

    breadcrumbs = build_breadcrumbs(headings)

    pages = []
    for pn in sorted(page_text):
        pages.append({
            "page_num": pn,
            "content": page_text[pn],
            "breadcrumb": breadcrumbs.get(pn, ""),
        })
    sections = []
    for h in sorted(headings, key=lambda x: (x.page_num, -x.confidence)):
        sections.append({
            "heading": h.text,
            "level": h.level,
            "page_num": h.page_num,
            "breadcrumb": breadcrumbs.get(h.page_num, ""),
            "numbering": h.numbering,
            "confidence": h.confidence,
            "signals": h.signals,
        })

    elapsed = time.time() - t0
    return {
        "pages": pages,
        "sections": sections,
        "stats": {
            "method": method,
            "n_pages": len(pages),
            "n_sections": len(sections),
            "n_lines": sum(len(v) for v in page_lines.values()),
            "elapsed_sec": round(elapsed, 3),
            "extract_sec": round(t_extract, 3),
            "sec_per_page": round(elapsed / max(1, len(pages)), 4),
            "body_font_size": body_size,
            "heading_font_sizes": heading_sizes,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--model", help="Path to trained heading_clf.joblib")
    ap.add_argument("--doclayout-yolo", help="Path to DocLayout-YOLO ONNX model")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", help="Write full JSON output here")
    args = ap.parse_args()

    result = extract(args.pdf, args.model, args.doclayout_yolo, args.threshold)
    s = result["stats"]
    print(f"method:    {s['method']}")
    print(f"pages:     {s['n_pages']}")
    print(f"sections:  {s['n_sections']}")
    print(f"lines:     {s['n_lines']}")
    print(f"elapsed:   {s['elapsed_sec']}s  ({s['sec_per_page']}s/page)")
    print(f"extract:   {s['extract_sec']}s")
    print(f"body font: {s['body_font_size']}  heading sizes: {s['heading_font_sizes']}")
    print()
    print("First 30 sections:")
    for sec in result["sections"][:30]:
        print(f"  L{sec['level']} p{sec['page_num']:>4} "
              f"[{sec['confidence']:.2f}] {sec['heading'][:80]}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
