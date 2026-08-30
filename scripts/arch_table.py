"""
Render the cross-architecture INT8 table from the committed latency reports.

Every row is read out of a bench/reports/**/arch_latency_*.json written by
`ml_inference.arch_bench`, so the table in the README cannot drift away from
the runs behind it: change a report and re-run this, or the table is wrong in a
way that shows up immediately.

  .venv/bin/python scripts/arch_table.py            one row per run
  .venv/bin/python scripts/arch_table.py --by-cpu   one row per CPU model
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "bench" / "reports"


def load_reports() -> list[dict]:
    rows = []
    for path in sorted(REPORTS.rglob("arch_latency_*.json")):
        d = json.loads(path.read_text())
        d["_path"] = str(path.relative_to(REPO_ROOT))
        rows.append(d)
    return rows


def features(d: dict) -> str:
    present = [k for k, v in d["cpu"]["int8_features"].items() if v]
    return ", ".join(present) if present else "none"


def provenance(d: dict) -> str:
    """Where the run happened, taken from the report's path."""
    parts = pathlib.Path(d["_path"]).parts
    return "CI " + "/".join(parts[2:-1]) if "ci" in parts else "local"


def by_cpu(rows: list[dict]) -> None:
    """One row per CPU model, medians across that model's runs.

    Several jobs landing on the same CPU model are repeats of one measurement,
    so they collapse; jobs on different models do not.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for d in rows:
        groups[(d["cpu"]["model"] or "unknown", features(d), d["scope"]["weights"])].append(d)

    print(
        "| CPU | INT8 instructions present | Preset | Runs | FP32 P50 | INT8 P50 "
        "| INT8 speedup P50 | INT8 speedup P95 |"
    )
    print("|---" * 8 + "|")
    for (model, feats, weights), group in sorted(
        groups.items(), key=lambda kv: statistics.median(d["int8_speedup_vs_fp32"]["p50"] for d in kv[1])
    ):
        def med(f, g=group):
            return round(statistics.median(f(d) for d in g), 2)

        label = model if weights != "fine-tuned" else f"{model} (fine-tuned weights)"
        print(
            f"| {label} | {feats} | `{group[0]['scope']['quant_preset']}` | {len(group)} "
            f"| {med(lambda d: d['onnx_fp32']['p50_ms'])} ms "
            f"| {med(lambda d: d['onnx_int8']['p50_ms'])} ms "
            f"| {med(lambda d: d['int8_speedup_vs_fp32']['p50'])}x "
            f"| {med(lambda d: d['int8_speedup_vs_fp32']['p95'])}x |"
        )


def main() -> int:
    rows = load_reports()
    if not rows:
        print(f"no arch_latency_*.json under {REPORTS}", file=sys.stderr)
        return 1

    if "--by-cpu" in sys.argv:
        by_cpu(rows)
        return 0

    header = (
        "| CPU | INT8 instructions present | Preset | Weights | FP32 P50 | INT8 P50 "
        "| FP32 P95 | INT8 P95 | INT8 speedup P50 / P95 |"
    )
    print(header)
    print("|---" * 9 + "|")
    for d in sorted(rows, key=lambda r: (r["cpu"]["machine"], r["cpu"]["model"] or "")):
        f32, i8, sp = d["onnx_fp32"], d["onnx_int8"], d["int8_speedup_vs_fp32"]
        print(
            f"| {d['cpu']['model']} | {features(d)} | `{d['scope']['quant_preset']}` "
            f"| {d['scope']['weights']} | {f32['p50_ms']} ms | {i8['p50_ms']} ms "
            f"| {f32['p95_ms']} ms | {i8['p95_ms']} ms | {sp['p50']}x / {sp['p95']}x |"
        )

    print()
    print("Sources, one row each:")
    for d in sorted(rows, key=lambda r: r["_path"]):
        print(
            f"  {d['_path']}  ({provenance(d)}, {d['scope']['latency_samples_per_variant']} samples, "
            f"{d['scope']['intra_op_threads']} intra-op threads, ORT {d['scope']['onnxruntime_version']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
