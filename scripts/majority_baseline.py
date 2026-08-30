#!/usr/bin/env python3
"""
Score the trivial predictor on the top-10 CWE holdout: always answer the
majority class of the training split.

A macro F1 of 0.24 means nothing on its own. This is the floor it has to be
read against, and it costs no training. It also makes the accuracy column
honest: on a split where `__OTHER__` is the largest class, a model can look
respectable on accuracy while having learned nothing, and this run shows
exactly how respectable.

The majority class is taken from the training split, never from the holdout,
so the predictor uses no information the models do not also have.

Writes results/majority_baseline/ through the same collector the trained runs
use, so its artifacts have the same shape as theirs.

Usage:
    PYTHONPATH=src:. python scripts/majority_baseline.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ml_inference.config import MAX_SEQ_LEN, REPO_ROOT, SEED  # noqa: E402
from ml_inference.data import load_label_map, load_split  # noqa: E402
from ml_inference.metrics import compute_classification_metrics  # noqa: E402

COMMAND = "PYTHONPATH=src:. python scripts/majority_baseline.py"


def main() -> None:
    label_map = load_label_map()
    inv = {v: k for k, v in label_map.items()}
    train = load_split("train")
    holdout = load_split("holdout")

    majority = int(train["label"].value_counts().idxmax())
    y_true = holdout["label"].to_numpy().astype(int)
    y_pred = np.full_like(y_true, majority)

    metrics = compute_classification_metrics(
        y_true, y_pred, average="macro", target_names=[inv[i] for i in range(len(label_map))]
    )
    metrics["scope"] = {
        "config_name": "majority-class",
        "model_name": "none, constant predictor",
        "predicted_class": inv[majority],
        "epochs": 0,
        "batch_size": 0,
        "grad_accum_steps": 0,
        "learning_rate": 0.0,
        "max_seq_len": MAX_SEQ_LEN,
        "train_rows": int(len(train)),
        "eval_rows": int(len(holdout)),
        "device": "none",
        "host": "any",
        "seed": SEED,
    }
    metrics["train_wall_clock_seconds"] = 0.0

    out_dir = REPO_ROOT / "results" / "majority_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "metrics.json"
    report.write_text(json.dumps(metrics, indent=2))

    print(f"majority class in the training split: {inv[majority]}")
    print(f"holdout rows: {len(holdout)}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"macro F1:  {metrics['f1_macro']:.4f}")
    print(f"weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"wrote {report}")
    print()
    print(f"Freeze the rest of the artifacts with:\n  python scripts/collect_result.py "
          f"--name majority_baseline --report {report.relative_to(REPO_ROOT)} "
          f"--command '{COMMAND}'")


if __name__ == "__main__":
    main()
