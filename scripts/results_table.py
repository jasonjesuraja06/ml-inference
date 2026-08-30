#!/usr/bin/env python3
"""
Render the measured results in results/ as markdown, next to published numbers.

Reads only files already on disk: the frozen metrics under results/<run>/ and
the transcribed literature numbers in results/published_baselines.json. It
computes nothing, so the tables it prints and the tables in the README carry
the same digits as the committed run artifacts.

Used by the README and printed at the end of notebooks/train_gpu.ipynb, so a
GPU rerun produces a table in the same shape as the CPU one.

Usage:
    PYTHONPATH=src:. python scripts/results_table.py
    PYTHONPATH=src:. python scripts/results_table.py --results-dir results
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ml_inference.config import REPO_ROOT  # noqa: E402

AGGREGATE_ROWS = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}


def load(results_dir: pathlib.Path, name: str) -> dict | None:
    p = results_dir / name / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def hhmm(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def cwe_table(baseline: dict | None, improved: dict | None) -> str:
    lines = [
        "| Config | Train rows | Epochs | Macro F1 | Weighted F1 | Accuracy | Macro recall | Train wall clock |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, m in (("codebert-base, plain cross-entropy", baseline),
                     ("unixcoder-base, focal + class weights + augmentation", improved)):
        if m is None:
            lines.append(f"| `{label}` | not run | | | | | | |")
            continue
        s = m["scope"]
        lines.append(
            f"| `{label}` | {s['train_rows']} | {s['epochs']} | {m['f1_macro']:.4f} | "
            f"{m['f1_weighted']:.4f} | {m['accuracy']:.4f} | {m['recall_macro']:.4f} | "
            f"{hhmm(m.get('train_wall_clock_seconds'))} |"
        )
    return "\n".join(lines)


def per_class_table(baseline: dict | None, improved: dict | None) -> str:
    if improved is None and baseline is None:
        return "_no run found_"
    ref = improved or baseline
    names = [k for k in ref["per_class"] if k not in AGGREGATE_ROWS]
    names.sort(key=lambda n: -ref["per_class"][n]["support"])
    lines = [
        "| Class | Holdout support | Baseline F1 | Improved F1 |",
        "|---|---|---|---|",
    ]
    for n in names:
        b = f"{baseline['per_class'][n]['f1-score']:.4f}" if baseline and n in baseline["per_class"] else "n/a"
        i = f"{improved['per_class'][n]['f1-score']:.4f}" if improved and n in improved["per_class"] else "n/a"
        lines.append(f"| {n} | {int(ref['per_class'][n]['support'])} | {b} | {i} |")
    b = f"{baseline['f1_macro']:.4f}" if baseline else "n/a"
    i = f"{improved['f1_macro']:.4f}" if improved else "n/a"
    lines.append(f"| **macro average** | {int(ref['per_class']['macro avg']['support'])} | {b} | {i} |")
    return "\n".join(lines)


def devign_table(measured: dict | None, published: dict) -> str:
    block = published["codexglue_defect_detection"]
    lines = [
        "| System | Accuracy | Binary F1 | Source |",
        "|---|---|---|---|",
    ]
    for r in block["results"]:
        lines.append(f"| {r['model']} | {r['accuracy'] / 100:.4f} | not reported | CodeXGLUE leaderboard |")
    if measured is None:
        lines.append("| this repository | not run | not run | |")
    else:
        s = measured["scope"]
        lines.append(
            f"| **this repository, {s['model_name'].split('/')[-1]}, {s['epochs']} epochs, "
            f"{s['max_seq_len']} tokens** | **{measured['accuracy']:.4f}** | "
            f"**{measured['f1_binary']:.4f}** | `results/devign_codebert/metrics.json` |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    rd = pathlib.Path(args.results_dir)

    baseline = load(rd, "cwe_baseline")
    improved = load(rd, "cwe_improved")
    devign = load(rd, "devign_codebert")
    published = json.loads((rd / "published_baselines.json").read_text())

    print("## Task A: top-10 CWE multiclass on DiverseVul (11 classes with __OTHER__)")
    print()
    print(cwe_table(baseline, improved))
    print()
    print("### Per-class F1 on the holdout")
    print()
    print(per_class_table(baseline, improved))
    print()
    print("## Task B: binary vulnerable/benign on CodeXGLUE Defect Detection (Devign)")
    print()
    print(devign_table(devign, published))
    print()
    dv = published["diversevul_paper"]
    print("## Published context")
    print()
    print(f"- {dv['source']['title']} ({dv['source']['venue']}): " + "; ".join(
        f"{r['model']} F1 {r['f1']}" for r in dv["results"]
    ) + ". " + dv["comparability_note"])
    pv = published["devign_paper"]
    best = pv["results"][-1]
    print(
        f"- {pv['source']['title']} ({pv['source']['venue']}), {pv['source']['table']}: "
        f"{best['model']} accuracy {best['accuracy']}, F1 {best['f1']}. {pv['comparability_note']}"
    )


if __name__ == "__main__":
    main()
