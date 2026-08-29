#!/usr/bin/env python3
"""
Build the top-K CWE multi-class splits from DiverseVul.

Two details of the source data drive this script:

1. DiverseVul ships both the vulnerable and the patched version of every
   function touched by a vulnerability-fixing commit, and it attaches the
   commit's CWE to both. The CWE is therefore a property of the commit, not of
   the function, and it is only a valid label for the rows with `target == 1`.
   This script keeps those rows and drops the rest. On the mirror used here
   that is 18,945 of 330,492 rows, which matches the vulnerable-function count
   reported in the DiverseVul paper.
2. The `cwe` field is an array, because a commit can reference several CWEs.
   About a quarter of rows carry more than one. This script takes the first
   entry and ignores the rest, so a multi-CWE function is trained as a
   single-label example.

Output:
  data/splits/train.parquet
  data/splits/val.parquet
  data/splits/test.parquet
  data/splits/holdout.parquet          evaluation holdout, never trained on
  data/splits/unlabeled_pool.parquet   for active-learning iterations
  data/splits/label_map.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from ml_inference.config import (  # noqa: E402
    DATA_RAW,
    DATA_SPLITS,
    HOLDOUT_FRACTION,
    SEED,
    TOP_K_CWES,
    env_int,
)


def _first_cwe(v: object) -> str | None:
    """Normalize the `cwe` field to a single `CWE-<n>` string, or None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return None
        v = v[0]
    s = str(v).strip().upper()
    if not s or s in {"NONE", "NULL", "NAN", "OTHER", "[]"}:
        return None
    if not s.startswith("CWE-"):
        if s.startswith("CWE"):
            s = "CWE-" + s[3:]
        elif s.isdigit():
            s = f"CWE-{s}"
    return s


def diversevul_to_df() -> pd.DataFrame:
    hf_path = DATA_RAW / "diversevul" / "hf_dataset"
    if not hf_path.exists():
        j = DATA_RAW / "diversevul" / "diversevul.json"
        if not j.exists():
            raise SystemExit(
                "DiverseVul not found. Run `make download` first, or place diversevul.json "
                "in data/raw/diversevul/."
            )
        with j.open() as fh:
            df = pd.DataFrame([json.loads(line) for line in fh])
    else:
        ds = load_from_disk(str(hf_path))
        split = "train" if "train" in ds else next(iter(ds))
        df = ds[split].to_pandas()

    rename = {}
    for src in ("func", "function", "code"):
        if src in df.columns:
            rename[src] = "code"
    for src in ("cwe", "cwe_id", "vul_type"):
        if src in df.columns:
            rename[src] = "cwe"
    df = df.rename(columns=rename)
    if "code" not in df.columns or "cwe" not in df.columns:
        raise SystemExit(f"unexpected DiverseVul columns: {df.columns.tolist()}")

    n_all = len(df)
    if "target" in df.columns:
        df = df[df["target"] == 1]
        print(f"kept {len(df)} vulnerable rows of {n_all} (target == 1)")

    df["code"] = df["code"].astype(str)
    df["cwe"] = df["cwe"].apply(_first_cwe)
    df = df.dropna(subset=["code", "cwe"])
    df = df[df["code"].str.len() > 20]
    return df[["code", "cwe"]].reset_index(drop=True)


def main() -> None:
    df = diversevul_to_df()
    print(f"usable CWE-labeled rows: {len(df)}")

    counts = Counter(df["cwe"])
    top = [cwe for cwe, _ in counts.most_common(TOP_K_CWES)]
    print(f"top-{TOP_K_CWES} CWEs: {top}")

    label_map = {cwe: i for i, cwe in enumerate(top)}
    label_map["__OTHER__"] = len(label_map)
    df["label"] = df["cwe"].map(lambda c: label_map.get(c, label_map["__OTHER__"]))

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    holdout_rows = env_int("HOLDOUT_ROWS", int(HOLDOUT_FRACTION * len(df)))
    if len(df) < holdout_rows + 2000:
        raise SystemExit(f"too few rows ({len(df)}) for a {holdout_rows}-row holdout")
    holdout = df.iloc[:holdout_rows].copy()
    rest = df.iloc[holdout_rows:].copy()

    # 10% of the remainder becomes the active-learning pool. Its true labels are
    # kept in a separate column so auto-labels can be scored against them.
    pool_size = int(0.10 * len(rest))
    pool = rest.iloc[:pool_size].copy()
    pool["true_label"] = pool["label"].to_numpy()
    pool["label"] = -1

    labeled = rest.iloc[pool_size:]
    n_train = int(0.80 * len(labeled))
    n_val = int(0.10 * len(labeled))
    parts = [
        ("train", labeled.iloc[:n_train]),
        ("val", labeled.iloc[n_train:n_train + n_val]),
        ("test", labeled.iloc[n_train + n_val:]),
        ("holdout", holdout),
        ("unlabeled_pool", pool),
    ]
    manifest = {}
    for name, part in parts:
        out = DATA_SPLITS / f"{name}.parquet"
        part.to_parquet(out, index=False)
        manifest[name] = int(len(part))
        print(f"wrote {out} rows={len(part)}")

    with (DATA_SPLITS / "label_map.json").open("w") as fh:
        json.dump(label_map, fh, indent=2)

    inv = {v: k for k, v in label_map.items()}
    dist = {inv[k]: int(v) for k, v in sorted(parts[0][1]["label"].value_counts().items())}
    manifest["train_label_distribution"] = dist
    with (DATA_SPLITS / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print("train label distribution:", dist)
    print(f"wrote {DATA_SPLITS / 'manifest.json'}")


if __name__ == "__main__":
    main()
