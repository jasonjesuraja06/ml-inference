"""
Benchmark inference latency and 10K-holdout accuracy across model variants.

For each of (PyTorch FP32 reference, ONNX FP32, ONNX INT8):
  - mean / P50 / P95 / P99 single-input latency
  - macro F1 on the 10K holdout
  - INT8 vs PyTorch FP32 accuracy delta (so quantization regressions are caught)
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ml_inference.config import IMPROVED, MAX_SEQ_LEN, ONNX_DIR, REPORTS_DIR
from ml_inference.data import load_label_map, load_split, n_labels
from ml_inference.metrics import compute_classification_metrics


def time_model_single(infer_fn, samples: list[str], warmup: int = 10) -> dict[str, float]:
    """Measure single-input latency (batch=1). Mirrors a worst-case mobile path."""
    for s in samples[:warmup]:
        infer_fn(s)
    lats = []
    for s in samples:
        t0 = time.perf_counter()
        infer_fn(s)
        lats.append((time.perf_counter() - t0) * 1000)  # ms
    lats.sort()
    return {
        "mean_ms": statistics.mean(lats),
        "p50_ms": lats[len(lats) // 2],
        "p95_ms": lats[int(len(lats) * 0.95)],
        "p99_ms": lats[int(len(lats) * 0.99)],
        "min_ms": lats[0],
        "max_ms": lats[-1],
        "n": len(lats),
    }


def predict_all_torch(model, tok, df, device: str) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            chunk = df.iloc[i:i + 32]
            enc = tok(list(chunk["code"]), truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
            out = model(**enc)
            preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())
    return np.array(preds)


def predict_all_onnx(ort_model, tok, df) -> np.ndarray:
    preds = []
    for i in range(0, len(df), 32):
        chunk = df.iloc[i:i + 32]
        enc = tok(list(chunk["code"]), truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        out = ort_model(**{k: v for k, v in enc.items()})
        preds.extend(out.logits.argmax(-1).numpy().tolist())
    return np.array(preds)


def main() -> None:
    holdout = load_split("holdout_10k")
    n = n_labels()
    inv = {v: k for k, v in load_label_map().items()}
    target_names = [inv[i] for i in range(n)]
    y_true = holdout["label"].to_numpy()

    # FP32 PyTorch (reference)
    src = IMPROVED.output_dir / "final"
    tok = AutoTokenizer.from_pretrained(str(src))
    torch_model = AutoModelForSequenceClassification.from_pretrained(str(src))
    torch_model.to("cpu")

    # ONNX FP32
    fp32_dir = ONNX_DIR / "improved-fp32"
    fp32_model = ORTModelForSequenceClassification.from_pretrained(str(fp32_dir))

    # ONNX INT8
    int8_dir = ONNX_DIR / "improved-int8"
    int8_model = ORTModelForSequenceClassification.from_pretrained(str(int8_dir), file_name="model_quantized.onnx")

    # Sample 200 codes for single-input latency (avoid pathological outliers)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(holdout), size=min(200, len(holdout)), replace=False)
    samples = [holdout.iloc[int(i)]["code"] for i in sample_idx]

    def call_torch(s: str):
        enc = tok(s, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            torch_model(**enc)

    def call_ort(m):
        def _c(s: str):
            enc = tok(s, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt")
            m(**{k: v for k, v in enc.items()})
        return _c

    torch_lat = time_model_single(call_torch, samples)
    fp32_lat = time_model_single(call_ort(fp32_model), samples)
    int8_lat = time_model_single(call_ort(int8_model), samples)

    # Accuracy on full 10K holdout (this is the bullet number)
    y_torch = predict_all_torch(torch_model, tok, holdout, "cpu")
    y_fp32 = predict_all_onnx(fp32_model, tok, holdout)
    y_int8 = predict_all_onnx(int8_model, tok, holdout)

    m_torch = compute_classification_metrics(y_true, y_torch, target_names=target_names)
    m_fp32 = compute_classification_metrics(y_true, y_fp32, target_names=target_names)
    m_int8 = compute_classification_metrics(y_true, y_int8, target_names=target_names)

    delta = (m_torch["f1_macro"] - m_int8["f1_macro"]) / max(m_torch["f1_macro"], 1e-9) * 100
    speedup = torch_lat["p95_ms"] / int8_lat["p95_ms"]

    out = {
        "holdout_size": int(len(holdout)),
        "torch_fp32": {"f1_macro": m_torch["f1_macro"], **{f"latency_{k}": v for k, v in torch_lat.items()}},
        "onnx_fp32":  {"f1_macro": m_fp32["f1_macro"],  **{f"latency_{k}": v for k, v in fp32_lat.items()}},
        "onnx_int8":  {"f1_macro": m_int8["f1_macro"],  **{f"latency_{k}": v for k, v in int8_lat.items()}},
        "f1_macro_drop_pct_int8_vs_torch": round(delta, 2),
        "p95_speedup_int8_vs_torch": round(speedup, 2),
        "p95_int8_ms": int8_lat["p95_ms"],
        "p95_torch_ms": torch_lat["p95_ms"],
    }
    (REPORTS_DIR / "inference_bench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
