"""
Train + evaluate on CodeXGLUE Defect Detection (Devign).

Devign is a binary vulnerability detection benchmark (vulnerable vs benign
C/C++ functions). Reported SOTA accuracy hovers around ~65%; the included
training config is a straightforward CodeBERT fine-tune for reproducibility
against the published benchmark.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ml_inference.config import DATA_RAW, MAX_SEQ_LEN, MODELS_DIR, REPORTS_DIR, SEED, device
from ml_inference.metrics import compute_classification_metrics


class DevignDataset(Dataset):
    def __init__(self, split, tok):
        self.split = split
        self.tok = tok

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        enc = self.tok(row["func"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["target"]), dtype=torch.long)
        return item


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()

    ds = load_from_disk(str(DATA_RAW / "codexglue_devign" / "hf_dataset"))
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    out_dir = MODELS_DIR / "devign"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        report_to=[],
        dataloader_pin_memory=False,
    )

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=DevignDataset(ds["train"], tok),
        eval_dataset=DevignDataset(ds["validation"], tok),
        compute_metrics=hf_metrics,
        tokenizer=tok,
    )
    trainer.train()

    pred = trainer.predict(DevignDataset(ds["test"], tok))
    y_true = np.array([int(ds["test"][i]["target"]) for i in range(len(ds["test"]))])
    y_pred = pred.predictions.argmax(-1)
    metrics = compute_classification_metrics(y_true, y_pred, average="binary", target_names=["benign", "vulnerable"])
    metrics["benchmark"] = "CodeXGLUE Defect Detection (Devign)"
    metrics["note"] = "Reported SOTA accuracy on this benchmark is around 65%."
    (REPORTS_DIR / "devign_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))


if __name__ == "__main__":
    main()
