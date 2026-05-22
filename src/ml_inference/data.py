"""Dataset + tokenization utilities shared by training scripts."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_inference.config import DATA_SPLITS, MAX_SEQ_LEN


def load_label_map() -> dict[str, int]:
    with (DATA_SPLITS / "label_map.json").open() as fh:
        return json.load(fh)


def n_labels() -> int:
    return len(load_label_map())


def load_split(name: str) -> pd.DataFrame:
    path = DATA_SPLITS / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing split {path}; run `make splits`")
    return pd.read_parquet(path)


class CodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        enc = self.tok(
            row["code"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if "label" in row and row["label"] != -1:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


def class_weights(df: pd.DataFrame, num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights for multi-class CE loss."""
    counts = np.bincount(df["label"].to_numpy(), minlength=num_labels).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_labels * counts)
    return torch.tensor(w, dtype=torch.float32)
