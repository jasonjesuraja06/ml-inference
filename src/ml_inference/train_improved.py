"""
Improved classifier for the top-10 CWE multi-class task.

UniXcoder backbone with the following changes vs the baseline trainer:
  - Class-weighted focal loss (gamma=2.0) to focus capacity on hard minority CWEs
  - Label smoothing 0.1 for regularization
  - Back-translation-style augmentation on minority classes (see `augment.py`)
  - LR warmup, weight decay, longer training, early stopping on val macro F1
  - Larger effective batch via gradient accumulation
"""
from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml_inference.augment import augment
from ml_inference.config import IMPROVED, REPORTS_DIR, SEED, device
from ml_inference.data import CodeDataset, class_weights, load_label_map, load_split, n_labels
from ml_inference.loss import FocalLoss
from ml_inference.metrics import compute_classification_metrics


def maybe_augment_minority(df, label_counts: dict[int, int], rng: random.Random):
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
    import pandas as pd
    return pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    cfg = IMPROVED
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_df = load_split("train")
    val_df = load_split("val")
    holdout_df = load_split("holdout_10k")

    if cfg.use_back_translation_aug:
        counts = dict(Counter(train_df["label"].tolist()))
        train_df = maybe_augment_minority(train_df, counts, rng)
        print(f"after augmentation: {len(train_df)} rows")

    n = n_labels()
    label_map = load_label_map()
    inv = {v: k for k, v in label_map.items()}

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=n)
    model.to(dev)

    train_ds = CodeDataset(train_df, tok)
    val_ds = CodeDataset(val_df, tok)
    holdout_ds = CodeDataset(holdout_df, tok)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, num_workers=2)
    holdout_dl = DataLoader(holdout_ds, batch_size=cfg.batch_size * 2, num_workers=2)

    optim = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = (len(train_dl) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    w = class_weights(train_df, n).to(dev)
    if cfg.use_focal_loss:
        loss_fn = FocalLoss(alpha=w, gamma=2.0, label_smoothing=cfg.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    best_f1 = -1.0
    patience = cfg.early_stopping_patience
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            loss = loss_fn(out.logits, labels) / cfg.grad_accum_steps
            loss.backward()
            running += float(loss.item())
            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_dl)} loss={loss.item():.4f}")

        val_metrics = evaluate(model, val_dl, dev)
        print(f"epoch={epoch} val_f1_macro={val_metrics['f1_macro']:.4f}")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            bad_epochs = 0
            save_model(model, tok, cfg.output_dir / "final")
        else:
            bad_epochs += 1
            if patience and bad_epochs > patience:
                print(f"early stopping at epoch {epoch}")
                break

    # Reload best
    from transformers import AutoModelForSequenceClassification as Reloader  # noqa: N814
    model = Reloader.from_pretrained(str(cfg.output_dir / "final")).to(dev)

    holdout_metrics = evaluate(model, holdout_dl, dev, target_names=[inv[i] for i in range(n)])
    holdout_metrics["model"] = cfg.name
    holdout_metrics["holdout_size"] = int(len(holdout_df))
    holdout_metrics["best_val_f1_macro"] = best_f1
    out = REPORTS_DIR / "improved_holdout_metrics.json"
    out.write_text(json.dumps(holdout_metrics, indent=2))
    print(json.dumps({k: v for k, v in holdout_metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"wrote {out}")


def evaluate(model, dl, dev, target_names: list[str] | None = None) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            out = model(**batch)
            preds = out.logits.argmax(-1).cpu().numpy()
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
