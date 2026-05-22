#!/usr/bin/env python3
"""
Build top-K CWE multi-class splits from DiverseVul.

Output:
  data/splits/train.parquet
  data/splits/val.parquet
  data/splits/test.parquet
  data/splits/holdout_10k.parquet   <-- the bullet-defending 10K-sample holdout
  data/splits/label_map.json
  data/splits/unlabeled_pool.parquet  <-- for active learning iterations
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from datasets import load_from_disk

from ml_inference.config import DATA_RAW, DATA_SPLITS, HOLDOUT_SIZE, SEED, TOP_K_CWES


def diversevul_to_df() -> pd.DataFrame:
    hf_path = DATA_RAW / "diversevul" / "hf_dataset"
    if not hf_path.exists():
        # JSON fallback (manual download path)
        j = DATA_RAW / "diversevul" / "diversevul.json"
        if not j.exists():
            raise SystemExit(
                "DiverseVul not found. Run `make download` first, or place diversevul.json in data/raw/diversevul/."
            )
        with j.open() as fh:
            rows = [json.loads(line) for line in fh]
        df = pd.DataFrame(rows)
    else:
        ds = load_from_disk(str(hf_path))
        # DiverseVul on HF is a single split typically named "train".
        split = "train" if "train" in ds else next(iter(ds))
        df = ds[split].to_pandas()
    # Normalize column names that vary across mirrors.
    rename = {}
    for src in ("func", "function", "code"):
        if src in df.columns:
            rename[src] = "code"
    for src in ("cwe", "cwe_id", "label", "vul_type"):
        if src in df.columns:
            rename[src] = "cwe"
    df = df.rename(columns=rename)
    if "code" not in df.columns or "cwe" not in df.columns:
        raise SystemExit(f"unexpected DiverseVul columns: {df.columns.tolist()}")
    df["code"] = df["code"].astype(str)
    df["cwe"] = df["cwe"].apply(_normalize_cwe)
    df = df.dropna(subset=["code", "cwe"]).reset_index(drop=True)
    df = df[df["code"].str.len() > 20]
    return df


def _normalize_cwe(v: object) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        v = v[0]
    s = str(v).strip().upper()
    if not s or s in {"NONE", "NULL", "NAN", "OTHER"}:
        return None
    if not s.startswith("CWE-"):
        if s.startswith("CWE"):
            s = "CWE-" + s[3:]
        elif s.isdigit():
            s = f"CWE-{s}"
    return s


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = diversevul_to_df()
    print(f"loaded {len(df)} rows from DiverseVul")

    counts = Counter(df["cwe"])
    top = [cwe for cwe, _ in counts.most_common(TOP_K_CWES)]
    print(f"top-{TOP_K_CWES} CWEs: {top}")

    label_map = {cwe: i for i, cwe in enumerate(top)}
    label_map["__OTHER__"] = len(label_map)
    df["label"] = df["cwe"].map(lambda c: label_map.get(c, label_map["__OTHER__"]))

    # Shuffle deterministically
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Reserve 10K holdout
    if len(df) < HOLDOUT_SIZE + 2000:
        raise SystemExit(f"too few rows ({len(df)}) for a {HOLDOUT_SIZE}-sample holdout")
    holdout = df.iloc[:HOLDOUT_SIZE].copy()
    rest = df.iloc[HOLDOUT_SIZE:].copy()

    # Of the remainder, set aside 10% as an "unlabeled pool" for active learning
    pool_size = int(0.10 * len(rest))
    pool = rest.iloc[:pool_size].copy()
    pool["label"] = -1  # the pool is treated as unlabeled in active learning
    pool["true_label"] = rest.iloc[:pool_size]["label"].to_numpy()  # kept for evaluation

    labeled = rest.iloc[pool_size:].copy()

    # Train / val / test 80 / 10 / 10 of labeled
    n = len(labeled)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    train = labeled.iloc[:n_train]
    val = labeled.iloc[n_train:n_train + n_val]
    test = labeled.iloc[n_train + n_val:]

    for name, part in [("train", train), ("val", val), ("test", test), ("holdout_10k", holdout), ("unlabeled_pool", pool)]:
        out = DATA_SPLITS / f"{name}.parquet"
        part.to_parquet(out, index=False)
        print(f"wrote {out} rows={len(part)}")

    with (DATA_SPLITS / "label_map.json").open("w") as fh:
        json.dump(label_map, fh, indent=2)
    print("wrote label_map.json")

    dist = train["label"].value_counts().to_dict()
    print("train label distribution:", dist)


if __name__ == "__main__":
    main()
