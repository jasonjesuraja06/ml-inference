"""Shared metric helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    target_names: list[str] | None = None,
) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(acc),
        f"precision_{average}": float(p),
        f"recall_{average}": float(r),
        f"f1_{average}": float(f1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
        "confusion_matrix": cm,
    }
