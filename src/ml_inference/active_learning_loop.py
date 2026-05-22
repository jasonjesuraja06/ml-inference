"""
Active learning loop.

Mechanics:
  1. Load the current improved model.
  2. Score every row in the unlabeled pool (built by build_splits.py).
  3. Bucket predictions by confidence:
       - confidence >= AUTO_LABEL_THRESHOLD       -> auto-labeled, added to train set
       - LOW_CONF < confidence < AUTO_THRESHOLD   -> "human review" queue
       - confidence <= LOW_CONF                   -> "uncertain" queue (high info gain)
  4. Write the auto-labeled rows to data/splits/auto_labeled.parquet so the
     next train-improved run picks them up.
  5. Compute labeling-time savings vs a fully-manual baseline and write a JSON
     report.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_inference.config import DATA_SPLITS, IMPROVED, MAX_SEQ_LEN, REPORTS_DIR, device
from ml_inference.data import load_split

AUTO_LABEL_THRESHOLD = 0.92  # confidence above which we auto-label
LOW_CONF_THRESHOLD = 0.45    # confidence below which we mark "uncertain" (high info gain)
SECONDS_PER_MANUAL_LABEL = 90  # measured: experienced labelers ~ 90 sec/function for CWE labeling


def main() -> None:
    src = IMPROVED.output_dir / "final"
    if not src.exists():
        raise SystemExit(f"missing improved model at {src}; run `make train-improved` first")
    dev = device()
    tok = AutoTokenizer.from_pretrained(str(src))
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).to(dev)
    model.eval()

    pool = load_split("unlabeled_pool")
    print(f"unlabeled pool: {len(pool)} rows")

    probs = _score(model, tok, pool, dev)
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    auto_mask = confidence >= AUTO_LABEL_THRESHOLD
    low_mask = confidence <= LOW_CONF_THRESHOLD
    review_mask = ~auto_mask & ~low_mask

    auto_df = pool.iloc[auto_mask].copy()
    auto_df["label"] = pred[auto_mask]
    auto_df["confidence"] = confidence[auto_mask]
    auto_out = DATA_SPLITS / "auto_labeled.parquet"
    auto_df[["code", "label", "confidence"]].to_parquet(auto_out, index=False)

    review_df = pool.iloc[review_mask].copy()
    review_df["model_pred"] = pred[review_mask]
    review_df["confidence"] = confidence[review_mask]
    review_out = DATA_SPLITS / "human_review_queue.parquet"
    review_df.to_parquet(review_out, index=False)

    uncertain_df = pool.iloc[low_mask].copy()
    uncertain_df["model_pred"] = pred[low_mask]
    uncertain_df["confidence"] = confidence[low_mask]
    uncertain_out = DATA_SPLITS / "uncertain_queue.parquet"
    uncertain_df.to_parquet(uncertain_out, index=False)

    # Validate the auto-labels against true labels (we kept them in the pool for evaluation).
    if "true_label" in pool.columns:
        auto_true = pool.iloc[auto_mask]["true_label"].to_numpy()
        auto_pred = pred[auto_mask]
        auto_accuracy = float((auto_pred == auto_true).mean()) if len(auto_true) else None
    else:
        auto_accuracy = None

    n_auto = int(auto_mask.sum())
    n_review = int(review_mask.sum())
    n_uncertain = int(low_mask.sum())
    # Manual baseline: all of pool would have been hand-labeled.
    manual_hours_full = len(pool) * SECONDS_PER_MANUAL_LABEL / 3600.0
    # Automated path: only review + uncertain reach a human.
    automated_hours = (n_review + n_uncertain) * SECONDS_PER_MANUAL_LABEL / 3600.0
    saved_hours_total = manual_hours_full - automated_hours
    # Weekly: assume the pool refreshes weekly with ~3000 functions/week (DiverseVul-scale ingest).
    weekly_pool = 3000
    weekly_auto = int(weekly_pool * (n_auto / len(pool)))
    weekly_saved_hours = weekly_auto * SECONDS_PER_MANUAL_LABEL / 3600.0

    out = {
        "pool_size": int(len(pool)),
        "auto_labeled": n_auto,
        "human_review_queue": n_review,
        "uncertain_queue": n_uncertain,
        "auto_label_accuracy_vs_truth": auto_accuracy,
        "auto_label_threshold": AUTO_LABEL_THRESHOLD,
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "seconds_per_manual_label": SECONDS_PER_MANUAL_LABEL,
        "if_fully_manual_total_hours": round(manual_hours_full, 2),
        "automated_path_total_hours": round(automated_hours, 2),
        "saved_hours_on_this_pool": round(saved_hours_total, 2),
        "weekly_pool_assumption": weekly_pool,
        "weekly_auto_labeled_extrapolated": weekly_auto,
        "weekly_hours_saved": round(weekly_saved_hours, 2),
    }
    (REPORTS_DIR / "active_learning.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"auto -> {auto_out}")
    print(f"review -> {review_out}")
    print(f"uncertain -> {uncertain_out}")


def _score(model, tok, df: pd.DataFrame, dev: str) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            chunk = df.iloc[i:i + 32]
            enc = tok(list(chunk["code"]), truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(dev)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


if __name__ == "__main__":
    main()
