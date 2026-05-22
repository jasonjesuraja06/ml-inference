"""
Train the baseline classifier on the top-10 CWE multi-class task.

CodeBERT, intentionally minimal config:
  - 2 epochs
  - no class weights (severe imbalance on minority CWEs is left untreated)
  - no warmup, no weight decay
  - no label smoothing, no augmentation
  - no early stopping

The point of this configuration is to produce a v1-quality model on this task,
the kind a team usually ships first before iterating. The improved trainer
(`train_improved.py`) layers in the techniques that lift macro F1 substantially.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ml_inference.config import BASELINE, REPORTS_DIR, SEED
from ml_inference.data import CodeDataset, load_label_map, load_split, n_labels
from ml_inference.metrics import compute_classification_metrics


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = BASELINE
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    holdout_df = load_split("holdout_10k")

    n = n_labels()
    label_map = load_label_map()
    inv = {v: k for k, v in label_map.items()}

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=n)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        seed=SEED,
        report_to=[],
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
    )

    from sklearn.metrics import f1_score

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0))}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CodeDataset(train_df, tok),
        eval_dataset=CodeDataset(val_df, tok),
        tokenizer=tok,
        compute_metrics=hf_metrics,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final"))
    tok.save_pretrained(str(cfg.output_dir / "final"))

    # Eval on 10K holdout (this is the bullet number)
    eval_ds = CodeDataset(holdout_df, tok)
    pred = trainer.predict(eval_ds)
    y_true = np.array([int(holdout_df.iloc[i]["label"]) for i in range(len(holdout_df))])
    y_pred = pred.predictions.argmax(-1)
    target_names = [inv[i] for i in range(n)]
    metrics = compute_classification_metrics(y_true, y_pred, average="macro", target_names=target_names)
    metrics["model"] = cfg.name
    metrics["holdout_size"] = int(len(holdout_df))

    out = REPORTS_DIR / "baseline_holdout_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
