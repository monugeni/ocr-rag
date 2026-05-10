# ocr-rag/scripts — heading extractor R&D

Tools for replacing the Marker + font-size pipeline with a CPU-fast pipeline
that uses **pymupdf for text** + **trained classifier (or DocLayout-YOLO) for
headings** + the existing numbering parser for level resolution.

## Workflow

### 1. Survey what training data you already have

```bash
python scripts/survey_heading_data.py \
    --db docs.db \
    --pdf-root samples/split_output \
    --out heading_survey.json
```

Reports counts of LLM-confirmed positives, LLM-removed negatives, level
reassignments, and prints a verdict: `READY`, `MARGINAL`, or `INSUFFICIENT`.

### 2. If READY/MARGINAL: build a labelled dataset

```bash
python scripts/build_heading_dataset.py \
    --db docs.db \
    --pdf-root samples/split_output \
    --out heading_dataset.jsonl
```

For each document in `docs.db`:
- Re-extracts per-line features from the original PDF (pdfplumber, same
  features as `extractor.py` already uses).
- Joins each line against:
  1. sidecar `*_corrections.json` (sidecar wins),
  2. rows in the `corrections` table,
  3. final `sections` rows (post-correction truth).
- Subsamples unlabelled negatives to ~5× positives per doc.

### 3. Train

```bash
pip install lightgbm scikit-learn joblib
python scripts/train_heading_classifier.py \
    --dataset heading_dataset.jsonl \
    --out models/heading
```

GroupKFold-CV by document (no leakage). Reports binary P/R/F1 and per-level
confusion matrix. Saves `heading_clf.joblib` containing both binary and
level-1..4 LightGBM models.

### 4. Run the new extractor

```bash
# pure heuristic (no model yet) — already faster than Marker
python scripts/extractor_v2.py samples/split_output/part_01.pdf

# with trained classifier
python scripts/extractor_v2.py samples/split_output/part_01.pdf \
    --model models/heading/heading_clf.joblib \
    --out part_01_v2.json

# with DocLayout-YOLO as a vision prior (optional, needs onnxruntime + doclayout-yolo)
python scripts/extractor_v2.py samples/split_output/part_01.pdf \
    --model models/heading/heading_clf.joblib \
    --doclayout-yolo doclayout_yolo.onnx
```

Output JSON has the same `(pages, sections)` shape as
`ingest.parse_marker_json()`, so swapping it into `ingest.py` is a one-liner.

### 5. A/B benchmark

```bash
# Single PDF: v2 vs an existing Marker output
python scripts/bench_extractors.py \
    --pdf samples/split_output/part_01.pdf \
    --marker samples/split_output/part_01.json \
    --model models/heading/heading_clf.joblib

# Batch over the whole corpus, vs final docs.db sections as ground truth
python scripts/bench_extractors.py \
    --db docs.db \
    --pdf-root samples/split_output \
    --model models/heading/heading_clf.joblib \
    --limit 20 \
    --out bench.json
```

Reports per-doc precision/recall/F1 against:
- Marker output (drift signal, not truth)
- post-correction `docs.db` sections (closest to truth)

Plus median seconds-per-page.

## What you'd swap in `ingest.py`

`ingest.parse_marker_json()` at line 360 currently builds `(pages, sections)`
from Marker JSON. After validation, replace its body with:

```python
from scripts.extractor_v2 import extract
result = extract(str(pdf_path), model_path="models/heading/heading_clf.joblib")
return result["pages"], result["sections"]
```

Marker (and `marker_single` in the README) becomes optional — only fire it for
pages where pymupdf returns no text (scanned). Keep `corrections.py` and the
LLM polish layer as-is; they sit on top of whatever extractor produces the
sections.

## Dependencies

Add to `requirements.txt` if/when you keep this:

```
pymupdf>=1.24       # pymupdf4llm uses this; we use raw fitz for speed
lightgbm>=4         # only needed for training/inference of heading clf
scikit-learn>=1.3   # for GroupKFold + classification_report
joblib>=1.3         # model serialization
# optional vision prior
onnxruntime>=1.18
doclayout-yolo>=0.0.4
```

## Why this stack

- `pymupdf` extracts text + spans + bboxes natively in C; no model load cost,
  sub-second per 1000 pages on CPU.
- The trained LightGBM classifier replaces font-size guessing in
  `extractor.py:336-454` with a model trained on **your** corrections — it
  learns what looks like a heading in **your** tenders.
- Decimal numbering (`1.2.3`) stays as the primary level signal; the model
  only resolves levels when numbering is absent or ambiguous.
- DocLayout-YOLO is an optional vision prior for pages where text-only signals
  are weak (scanned headers, decorative title pages). Skip it for clean
  born-digital tenders.
