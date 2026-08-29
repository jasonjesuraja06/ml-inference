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
  5. Report the share of the pool that still needs a human, and the agreement
     rate of the auto-labels against the pool's retained ground truth.

The report deliberately stops at counts and rates. Converting a triage rate
into hours saved needs a per-function labeling time, which this project has
not measured; docs/labeling_runbook.md shows the conversion so a team can
apply its own figure.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_inference.config import DATA_SPLITS, IMPROVED, MAX_SEQ_LEN, REPORTS_DIR, device, env_float
from ml_inference.data import load_split

AUTO_LABEL_THRESHOLD = env_float("AUTO_LABEL_THRESHOLD", 0.92)
LOW_CONF_THRESHOLD = env_float("LOW_CONF_THRESHOLD", 0.45)


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
    n_pool = int(len(pool))

    out = {
        "pool_size": n_pool,
        "auto_labeled": n_auto,
        "human_review_queue": n_review,
        "uncertain_queue": n_uncertain,
        "auto_labeled_fraction": round(n_auto / n_pool, 4) if n_pool else None,
        "needs_human_fraction": round((n_review + n_uncertain) / n_pool, 4) if n_pool else None,
        "auto_label_accuracy_vs_truth": auto_accuracy,
        "auto_label_threshold": AUTO_LABEL_THRESHOLD,
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "model": str(src),
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
