"""
Train a heading-vs-not classifier (and optional level model) from
the JSONL produced by build_heading_dataset.py.

Two models, both LightGBM:
  - heading_clf: binary, line is/isn't a heading.
  - level_clf:   1..4 multiclass, conditional on heading=True.

Evaluation uses GroupKFold by document so we don't leak text style across
splits. Reports per-class precision/recall/F1, level confusion matrix, and
the false-positive / false-negative lines for inspection.

Usage:
  python scripts/train_heading_classifier.py \
      --dataset heading_dataset.jsonl --out models/heading
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # noqa: F841
    raise SystemExit(
        "lightgbm not installed. pip install lightgbm scikit-learn joblib"
    )

import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


FEATURE_KEYS = [
    "font_size",
    "font_ratio",
    "is_bold",
    "is_upper",
    "caps_ratio",
    "word_count",
    "char_count",
    "x0",
    "x0_rel_margin",
    "x0_rel_width",
    "gap_above",
    "gap_ratio",
    "ends_colon",
    "ends_period",
    "num_depth",
    "has_decimal_num",
    "heading_size_rank",
]


def load_dataset(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("dataset empty")
    return rows


def to_matrix(rows):
    X = np.array(
        [[r["features"].get(k, 0) or 0 for k in FEATURE_KEYS] for r in rows],
        dtype=np.float32,
    )
    y_bin = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    y_lvl = np.array(
        [int(r["level"]) if r.get("level") else 0 for r in rows], dtype=np.int32
    )
    groups = np.array([r["doc_id"] for r in rows])
    return X, y_bin, y_lvl, groups


def train_binary_cv(X, y, groups, n_splits=5, seed=0):
    """Cross-validated binary classifier; returns OOF preds and final model."""
    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        # Fall back to in-sample evaluation if only one document
        params = dict(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        )
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X, y)
        oof = clf.predict_proba(X)[:, 1]
        return clf, oof

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=np.float32)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        params = dict(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        )
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof[va] = clf.predict_proba(X[va])[:, 1]
        print(f"  fold {fold + 1}: best_iter={clf.best_iteration_}")

    # Final model trained on all data
    final = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(np.mean([c.best_iteration_ or 400 for c in [clf]])),
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        class_weight="balanced",
        verbose=-1,
    )
    final.fit(X, y)
    return final, oof


def train_level_cv(X, y_lvl, groups, n_splits=5, seed=0):
    """Multiclass level model trained on heading rows only."""
    mask = y_lvl > 0
    if mask.sum() < 100:
        print("  (level model skipped: <100 heading rows)")
        return None, None
    Xh = X[mask]
    yh = y_lvl[mask] - 1  # 0..3
    gh = groups[mask]
    n_splits = min(n_splits, len(set(gh)))
    if n_splits < 2:
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(Xh, yh)
        oof = clf.predict(Xh)
        return clf, (yh, oof)

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(yh, dtype=np.int32)
    for fold, (tr, va) in enumerate(gkf.split(Xh, yh, gh)):
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(
            Xh[tr], yh[tr],
            eval_set=[(Xh[va], yh[va])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof[va] = clf.predict(Xh[va])
    final = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    final.fit(Xh, yh)
    return final, (yh, oof)


def report_binary(y_true, oof_proba, threshold=0.5):
    pred = (oof_proba >= threshold).astype(int)
    print()
    print(f"BINARY heading-vs-not @ threshold={threshold}")
    print(classification_report(y_true, pred, target_names=["not_heading", "heading"]))


def report_level(y_true, y_pred):
    if y_true is None:
        return
    print()
    print("LEVEL classifier (1-4)")
    labels = [0, 1, 2, 3]
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=["L1", "L2", "L3", "L4"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))


def show_errors(rows, oof, y, n=20):
    print()
    print(f"Top {n} false positives (predicted heading, label=0):")
    for i in np.argsort(-oof)[:200]:
        if y[i] == 0:
            r = rows[i]
            print(f"  [p{r['page_num']:>4} prob={oof[i]:.2f}] "
                  f"{r['doc_filename'][:40]:40s} | {r['text'][:80]}")
            n -= 1
            if n <= 0:
                break
    print()
    print("Top false negatives (low prob but label=1):")
    n = 20
    for i in np.argsort(oof):
        if y[i] == 1:
            r = rows[i]
            print(f"  [p{r['page_num']:>4} prob={oof[i]:.2f}] "
                  f"{r['doc_filename'][:40]:40s} | {r['text'][:80]}")
            n -= 1
            if n <= 0:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="Output dir for trained models")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    X, y_bin, y_lvl, groups = to_matrix(rows)
    print(f"Loaded {len(rows)} rows. Positives: {int(y_bin.sum())}, "
          f"Negatives: {int((1 - y_bin).sum())}, Docs: {len(set(groups))}")
    print(f"Source mix: {Counter(r['source'] for r in rows)}")

    print("\n== Binary heading classifier ==")
    bin_clf, oof = train_binary_cv(X, y_bin, groups, n_splits=args.folds)
    report_binary(y_bin, oof, threshold=args.threshold)

    print("\n== Level classifier ==")
    lvl_clf, lvl_eval = train_level_cv(X, y_lvl, groups, n_splits=args.folds)
    if lvl_eval is not None:
        report_level(*lvl_eval)

    show_errors(rows, oof, y_bin)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "binary": bin_clf,
            "level": lvl_clf,
            "feature_keys": FEATURE_KEYS,
            "threshold": args.threshold,
        },
        out_dir / "heading_clf.joblib",
    )
    print(f"\nSaved: {out_dir / 'heading_clf.joblib'}")


if __name__ == "__main__":
    main()
