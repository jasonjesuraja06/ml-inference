"""
Train and evaluate on CodeXGLUE Defect Detection (Devign).

Devign is a binary vulnerability-detection benchmark over C functions from
FFmpeg and QEMU. It is included as a second, independently published task, so
the pipeline can be checked against something other than the DiverseVul splits
built by this repository.

Scope is set by the environment (EPOCHS, BATCH_SIZE, LEARNING_RATE,
MAX_TRAIN_ROWS, MAX_EVAL_ROWS, MAX_SEQ_LEN) and written into the report, so a
reduced run is never mistaken for a full one. A reduced run is not comparable
to published Devign numbers, which fine-tune on all 21,854 training rows.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ml_inference.config import (
    DATA_RAW,
    MAX_SEQ_LEN,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    device,
    env_float,
    env_int,
    env_str,
)
from ml_inference.metrics import compute_classification_metrics

MODEL_NAME = env_str("MODEL_NAME", "microsoft/codebert-base")


class DevignDataset(Dataset):
    def __init__(self, split, tok):
        self.split = split
        self.tok = tok

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        enc = self.tok(row["func"], truncation=True, padding="max_length",
                       max_length=MAX_SEQ_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["target"]), dtype=torch.long)
        return item


def take(split, max_rows: int):
    """Deterministically shrink a HF split. 0 means the whole split."""
    if not max_rows or max_rows >= len(split):
        return split
    return split.shuffle(seed=SEED).select(range(max_rows))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = load_from_disk(str(DATA_RAW / "codexglue_devign" / "hf_dataset"))
    train_split = take(ds["train"], env_int("MAX_TRAIN_ROWS", 0))
    val_split = take(ds["validation"], env_int("MAX_EVAL_ROWS", 0))
    test_split = take(ds["test"], env_int("MAX_EVAL_ROWS", 0))
    print(f"train={len(train_split)} val={len(val_split)} test={len(test_split)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    out_dir = MODELS_DIR / "devign"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = env_int("EPOCHS", 3)
    batch_size = env_int("BATCH_SIZE", 16)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=env_float("LEARNING_RATE", 2e-5),
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        report_to=[],
        dataloader_pin_memory=False,
    )

    def hf_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=DevignDataset(train_split, tok),
        eval_dataset=DevignDataset(val_split, tok),
        compute_metrics=hf_metrics,
        processing_class=tok,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0

    pred = trainer.predict(DevignDataset(test_split, tok))
    y_true = np.array(test_split["target"], dtype=int)
    y_pred = pred.predictions.argmax(-1)
    metrics = compute_classification_metrics(
        y_true, y_pred, average="binary", target_names=["benign", "vulnerable"]
    )
    metrics["benchmark"] = "CodeXGLUE Defect Detection (Devign)"
    metrics["scope"] = {
        "model_name": MODEL_NAME,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_seq_len": MAX_SEQ_LEN,
        "train_rows": len(train_split),
        "test_rows": len(test_split),
        "full_train_rows_available": len(ds["train"]),
        "device": device(),
        "seed": SEED,
    }
    metrics["train_wall_clock_seconds"] = round(train_seconds, 1)

    out = REPORTS_DIR / "devign_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
