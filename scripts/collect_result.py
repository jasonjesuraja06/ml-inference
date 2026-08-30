#!/usr/bin/env python3
"""
Freeze one training run into results/<name>/ so its numbers stay checkable.

A metrics JSON in bench/reports/ is overwritten by the next run at a different
scope. This script copies one into results/, next to the exact command that
produced it, a rendered per-class table, a rendered confusion matrix, and the
run's stdout. Nothing here recomputes a metric; it only re-renders what the
training script already wrote.

Usage:
    PYTHONPATH=src:. python scripts/collect_result.py \
        --name cwe_baseline \
        --report bench/reports/baseline_holdout_metrics.json \
        --command 'EPOCHS=3 make train-baseline' \
        --log runlogs/baseline.log
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ml_inference.config import REPO_ROOT  # noqa: E402

AGGREGATE_ROWS = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}


def per_class_markdown(metrics: dict) -> str:
    rows = metrics.get("per_class", {})
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    classes = [(k, v) for k, v in rows.items() if k not in AGGREGATE_ROWS]
    classes.sort(key=lambda kv: -kv[1].get("support", 0))
    for name, v in classes:
        lines.append(
            f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | "
            f"{v['f1-score']:.4f} | {int(v['support'])} |"
        )
    for name, v in rows.items():
        if name in AGGREGATE_ROWS and isinstance(v, dict):
            lines.append(
                f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | "
                f"{v['f1-score']:.4f} | {int(v['support'])} |"
            )
    return "\n".join(lines) + "\n"


def confusion_matrix_text(metrics: dict) -> str:
    cm = metrics.get("confusion_matrix", [])
    names = [k for k in metrics.get("per_class", {}) if k not in AGGREGATE_ROWS]
    if len(names) != len(cm):
        names = [str(i) for i in range(len(cm))]
    width = max([len(n) for n in names] + [5])
    head = " " * (width + 2) + " ".join(f"{n[:7]:>7}" for n in names)
    lines = ["rows = true class, columns = predicted class", "", head]
    for name, row in zip(names, cm, strict=False):
        lines.append(f"{name:<{width}}  " + " ".join(f"{v:>7}" for v in row))
    return "\n".join(lines) + "\n"


def summary_text(metrics: dict, command: str) -> str:
    scope = metrics.get("scope", {})
    out = [f"command: {command}", ""]
    for k in sorted(scope):
        out.append(f"{k}: {scope[k]}")
    out.append("")
    for k in (
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_binary",
        "precision_macro",
        "recall_macro",
        "precision_binary",
        "recall_binary",
        "best_val_f1_macro",
        "val_f1_macro_per_epoch",
        "train_wall_clock_seconds",
    ):
        if k in metrics:
            out.append(f"{k}: {metrics[k]}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>/ directory to write")
    ap.add_argument("--report", required=True, help="metrics JSON written by a training script")
    ap.add_argument("--command", required=True, help="the exact command that produced the report")
    ap.add_argument("--log", default=None, help="stdout log of the run, copied verbatim")
    args = ap.parse_args()

    report = pathlib.Path(args.report)
    if not report.is_absolute():
        report = REPO_ROOT / report
    metrics = json.loads(report.read_text())

    out_dir = REPO_ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "command.txt").write_text(args.command.strip() + "\n")
    (out_dir / "summary.txt").write_text(summary_text(metrics, args.command.strip()))
    (out_dir / "per_class.md").write_text(per_class_markdown(metrics))
    (out_dir / "confusion_matrix.txt").write_text(confusion_matrix_text(metrics))

    if args.log:
        log = pathlib.Path(args.log)
        if not log.is_absolute():
            log = REPO_ROOT / log
        if log.exists():
            shutil.copyfile(log, out_dir / "stdout.log")

    print(f"wrote {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
