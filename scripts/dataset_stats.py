"""Dataset statistics behind the DiverseVul numbers quoted in the README.

Writes bench/reports/dataset_stats.json. Run after `make download` and
`make splits`:

    .venv/bin/python scripts/dataset_stats.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_from_disk

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "diversevul" / "hf_dataset"
SPLITS = REPO / "data" / "splits"
OUT = REPO / "bench" / "reports" / "dataset_stats.json"

REQUEST_CAP_BEFORE_FIX = 20_000


def main() -> None:
    dd = load_from_disk(str(RAW))
    full = concatenate_datasets([dd[k] for k in dd])
    lengths = np.array([len(f) for f in full["func"]])
    target = np.array(full["target"])
    cwe = full["cwe"]

    vuln = target == 1
    labeled = [i for i in np.where(vuln)[0] if len(cwe[i]) > 0]
    multi = [i for i in labeled if len(cwe[i]) > 1]

    split_frames = [
        pd.read_parquet(SPLITS / f"{n}.parquet")
        for n in ("train", "val", "test", "holdout", "unlabeled_pool")
    ]
    split_lengths = pd.concat(split_frames, ignore_index=True)["code"].astype(str).str.len()

    stats = {
        "source": "DiverseVul mirror as downloaded by scripts/download_data.py",
        "rows_total": int(len(full)),
        "rows_target_1_vulnerable": int(vuln.sum()),
        "vulnerable_rows_with_a_cwe": len(labeled),
        "vulnerable_rows_citing_more_than_one_cwe": len(multi),
        "pct_labeled_rows_multi_cwe": round(len(multi) / len(labeled) * 100, 1),
        "rows_kept_in_splits": int(len(split_lengths)),
        "request_cap_before_fix_chars": REQUEST_CAP_BEFORE_FIX,
        "pct_all_rows_over_cap": round(float((lengths > REQUEST_CAP_BEFORE_FIX).mean() * 100), 2),
        "pct_vulnerable_rows_over_cap": round(
            float((lengths[vuln] > REQUEST_CAP_BEFORE_FIX).mean() * 100), 2
        ),
        "pct_split_rows_over_cap": round(
            float((split_lengths > REQUEST_CAP_BEFORE_FIX).mean() * 100), 2
        ),
        "split_rows_p99_chars": int(np.percentile(split_lengths, 99)),
        "split_rows_max_chars": int(split_lengths.max()),
    }
    OUT.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
