"""
Baseline classifier for the top-10 CWE multi-class task.

CodeBERT with plain cross-entropy: no class weights, no warmup, no weight
decay, no label smoothing, no augmentation, no early stopping. It is the
control arm for `train_improved.py`, which adds those techniques and is
evaluated on the same holdout.

Scope is set by the environment (MODEL_NAME, EPOCHS, BATCH_SIZE,
LEARNING_RATE, MAX_TRAIN_ROWS, MAX_EVAL_ROWS, MAX_SEQ_LEN). Whatever scope a
run used is recorded in the report it writes, so a score is never separable
from the run that produced it.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ml_inference.config import BASELINE, REPORTS_DIR, SEED, run_scope, scoped_config
from ml_inference.data import CodeDataset, load_label_map, load_split, n_labels, subsample
from ml_inference.metrics import compute_classification_metrics


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = scoped_config(BASELINE)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_df = subsample(load_split("train"), cfg.max_train_rows)
    val_df = subsample(load_split("val"), cfg.max_eval_rows)
    holdout_df = subsample(load_split("holdout"), cfg.max_eval_rows)
    print(f"train={len(train_df)} val={len(val_df)} holdout={len(holdout_df)}")

    n = n_labels()
    inv = {v: k for k, v in load_label_map().items()}

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
        save_strategy="no",
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
        processing_class=tok,
        compute_metrics=hf_metrics,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0
    trainer.save_model(str(cfg.output_dir / "final"))
    tok.save_pretrained(str(cfg.output_dir / "final"))

    pred = trainer.predict(CodeDataset(holdout_df, tok))
    y_true = holdout_df["label"].to_numpy().astype(int)
    y_pred = pred.predictions.argmax(-1)
    target_names = [inv[i] for i in range(n)]
    metrics = compute_classification_metrics(y_true, y_pred, average="macro", target_names=target_names)
    metrics["scope"] = run_scope(cfg, len(train_df), len(holdout_df))
    metrics["train_wall_clock_seconds"] = round(train_seconds, 1)

    out = REPORTS_DIR / "baseline_holdout_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
