"""
Improved classifier for the top-10 CWE multi-class task.

UniXcoder backbone with the following changes against the baseline trainer:
  - Class-weighted focal loss (gamma 2.0) instead of plain cross-entropy
  - Label smoothing 0.1
  - Augmentation of minority classes (see `augment.py`)
  - LR warmup, weight decay, longer training, early stopping on val macro F1
  - Larger effective batch via gradient accumulation
  - Rows auto-labeled by a previous `active_learning_loop.py` run are folded
    into the training set when `data/splits/auto_labeled.parquet` exists

Scope is set by the environment (MODEL_NAME, EPOCHS, BATCH_SIZE,
LEARNING_RATE, MAX_TRAIN_ROWS, MAX_EVAL_ROWS, MAX_SEQ_LEN) and recorded in the
report this script writes.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml_inference.augment import augment
from ml_inference.config import IMPROVED, REPORTS_DIR, SEED, device, run_scope, scoped_config
from ml_inference.data import (
    CodeDataset,
    class_weights,
    load_auto_labeled,
    load_label_map,
    load_split,
    n_labels,
    subsample,
)
from ml_inference.loss import FocalLoss
from ml_inference.metrics import compute_classification_metrics


def maybe_augment_minority(df: pd.DataFrame, label_counts: dict[int, int], rng: random.Random):
    """Duplicate minority-class rows with a surface-form perturbation applied.

    A class counts as minority when its training count is below 70% of the
    median per-class count.
    """
    median = sorted(label_counts.values())[len(label_counts) // 2]
    minority = {k for k, v in label_counts.items() if v < median * 0.7}
    if not minority:
        return df
    aug_rows = []
    for _, row in df.iterrows():
        if int(row["label"]) in minority:
            new = row.copy()
            new["code"] = augment(row["code"], rng)
            aug_rows.append(new)
    if not aug_rows:
        return df
    return pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    cfg = scoped_config(IMPROVED)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_df = subsample(load_split("train"), cfg.max_train_rows)
    val_df = subsample(load_split("val"), cfg.max_eval_rows)
    holdout_df = subsample(load_split("holdout"), cfg.max_eval_rows)

    n_auto = 0
    if os.environ.get("NO_AUTO_LABELS", "0") != "1":
        auto = load_auto_labeled()
        if auto is not None:
            n_auto = len(auto)
            train_df = pd.concat([train_df[["code", "label"]], auto], ignore_index=True)
            print(f"folded in {n_auto} auto-labeled rows from a previous active-learning run")

    if cfg.use_augmentation:
        counts = dict(Counter(train_df["label"].tolist()))
        train_df = maybe_augment_minority(train_df, counts, rng)
    print(f"train={len(train_df)} val={len(val_df)} holdout={len(holdout_df)}")

    n = n_labels()
    inv = {v: k for k, v in load_label_map().items()}

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=n)
    model.to(dev)

    train_dl = DataLoader(CodeDataset(train_df, tok), batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(CodeDataset(val_df, tok), batch_size=cfg.batch_size * 2)
    holdout_dl = DataLoader(CodeDataset(holdout_df, tok), batch_size=cfg.batch_size * 2)

    optim = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, (len(train_dl) // cfg.grad_accum_steps) * cfg.epochs)
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * cfg.warmup_ratio), total_steps)

    w = class_weights(train_df, n).to(dev) if cfg.use_class_weights else None
    if cfg.use_focal_loss:
        loss_fn = FocalLoss(alpha=w, gamma=2.0, label_smoothing=cfg.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    best_f1 = -1.0
    val_history = []
    bad_epochs = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            loss = loss_fn(out.logits, labels) / cfg.grad_accum_steps
            loss.backward()
            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_dl)} loss={loss.item():.4f}", flush=True)

        val_metrics = evaluate(model, val_dl, dev)
        val_history.append(round(val_metrics["f1_macro"], 4))
        print(f"epoch={epoch} val_f1_macro={val_metrics['f1_macro']:.4f}", flush=True)
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            bad_epochs = 0
            save_model(model, tok, cfg.output_dir / "final")
        else:
            bad_epochs += 1
            if cfg.early_stopping_patience and bad_epochs > cfg.early_stopping_patience:
                print(f"early stopping at epoch {epoch}")
                break
    train_seconds = time.perf_counter() - t0

    model = AutoModelForSequenceClassification.from_pretrained(str(cfg.output_dir / "final")).to(dev)
    metrics = evaluate(model, holdout_dl, dev, target_names=[inv[i] for i in range(n)])
    metrics["scope"] = run_scope(cfg, len(train_df), len(holdout_df))
    metrics["scope"]["auto_labeled_rows_folded_in"] = n_auto
    metrics["best_val_f1_macro"] = round(best_f1, 4)
    metrics["val_f1_macro_per_epoch"] = val_history
    metrics["train_wall_clock_seconds"] = round(train_seconds, 1)

    out = REPORTS_DIR / "improved_holdout_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"wrote {out}")


def evaluate(model, dl, dev, target_names: list[str] | None = None) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            preds = model(**batch).logits.argmax(-1).cpu().numpy()
            ps.extend(preds.tolist())
            if labels is not None:
                ys.extend(labels.cpu().numpy().tolist())
    return compute_classification_metrics(np.array(ys), np.array(ps), target_names=target_names)


def save_model(model, tok, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))


if __name__ == "__main__":
    main()
