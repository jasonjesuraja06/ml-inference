#!/usr/bin/env python3
"""
Print the CWE frequency distribution of the usable DiverseVul rows.

This exists so the choice of N in the top-N CWE task is checkable rather than
asserted. `config.TOP_K_CWES` fixes N = 10 before any model is trained; this
script shows the support behind that cut: how many rows each CWE has, where
rank 10 falls, and how much of the labeled data the tail below it holds.

Writes results/dataset/cwe_support.json and prints a markdown table of the
top 20 CWEs by frequency.

Usage:
    PYTHONPATH=src:. python scripts/cwe_support.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from build_splits import diversevul_to_df  # noqa: E402

from ml_inference.config import REPO_ROOT, TOP_K_CWES  # noqa: E402


def main() -> None:
    df = diversevul_to_df()
    counts = Counter(df["cwe"])
    total = sum(counts.values())
    ranked = counts.most_common()

    top = ranked[:TOP_K_CWES]
    tail = ranked[TOP_K_CWES:]
    tail_rows = sum(c for _, c in tail)

    print(f"usable CWE-labeled rows: {total}")
    print(f"distinct CWEs: {len(ranked)}")
    print()
    print("| Rank | CWE | Rows | Share of labeled rows |")
    print("|---|---|---|---|")
    for i, (cwe, c) in enumerate(ranked[:20], start=1):
        mark = "" if i <= TOP_K_CWES else " (below the cut)"
        print(f"| {i} | {cwe}{mark} | {c} | {100 * c / total:.1f}% |")
    print()
    print(
        f"N = {TOP_K_CWES}: rank {TOP_K_CWES} has {top[-1][1]} rows "
        f"({100 * top[-1][1] / total:.1f}%), rank {TOP_K_CWES + 1} has {tail[0][1]} rows "
        f"({100 * tail[0][1] / total:.1f}%)."
    )
    print(
        f"The {len(tail)} CWEs below the cut hold {tail_rows} rows "
        f"({100 * tail_rows / total:.1f}%) and are collapsed into __OTHER__."
    )

    out_dir = REPO_ROOT / "results" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "usable_cwe_labeled_rows": total,
        "distinct_cwes": len(ranked),
        "top_k": TOP_K_CWES,
        "counts_by_cwe": dict(ranked),
        "rank_k_rows": top[-1][1],
        "rank_k_plus_1_rows": tail[0][1],
        "other_class_rows": tail_rows,
        "other_class_share": round(tail_rows / total, 4),
    }
    out = out_dir / "cwe_support.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
