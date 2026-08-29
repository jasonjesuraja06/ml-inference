"""
Benchmark single-input latency and holdout macro F1 across model variants.

For each of (PyTorch FP32, ONNX FP32, ONNX INT8):
  - mean / P50 / P95 / P99 single-input latency, batch size 1, on CPU
  - macro F1 on the holdout split
  - the INT8-versus-FP32 accuracy delta, so a quantization regression is visible

All three variants run on CPU so the comparison is like for like; ONNX Runtime
has no MPS execution provider, and comparing a CPU ONNX graph against an
MPS PyTorch model would measure the accelerator, not the quantization.

Scope (both defaults keep the run inside a few minutes on a laptop):
  BENCH_LATENCY_SAMPLES   inputs timed per variant   (default 100)
  BENCH_ACCURACY_ROWS     holdout rows scored        (default 1000, 0 = all)
"""
from __future__ import annotations

import json
import math
import platform
import statistics
import time

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_inference.config import IMPROVED, MAX_SEQ_LEN, ONNX_DIR, REPORTS_DIR, env_int
from ml_inference.data import load_label_map, load_split, n_labels, subsample
from ml_inference.metrics import compute_classification_metrics


def percentile(sorted_lats: list[float], q: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_lats:
        return float("nan")
    idx = min(len(sorted_lats) - 1, max(0, math.ceil(q * len(sorted_lats)) - 1))
    return sorted_lats[idx]


def time_model_single(infer_fn, samples: list[str], warmup: int = 10) -> dict[str, float]:
    """Measure single-input latency at batch size 1."""
    for s in samples[:warmup]:
        infer_fn(s)
    lats = []
    for s in samples:
        t0 = time.perf_counter()
        infer_fn(s)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    return {
        "mean_ms": round(statistics.mean(lats), 2),
        "p50_ms": round(percentile(lats, 0.50), 2),
        "p95_ms": round(percentile(lats, 0.95), 2),
        "p99_ms": round(percentile(lats, 0.99), 2),
        "min_ms": round(lats[0], 2),
        "max_ms": round(lats[-1], 2),
        "n": len(lats),
    }


def predict_all_torch(model, tok, df) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            chunk = df.iloc[i:i + 32]
            enc = tok(list(chunk["code"]), truncation=True, padding=True,
                      max_length=MAX_SEQ_LEN, return_tensors="pt")
            preds.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
    return np.array(preds)


def predict_all_onnx(ort_model, tok, df) -> np.ndarray:
    preds = []
    for i in range(0, len(df), 32):
        chunk = df.iloc[i:i + 32]
        enc = tok(list(chunk["code"]), truncation=True, padding=True,
                  max_length=MAX_SEQ_LEN, return_tensors="pt")
        preds.extend(ort_model(**dict(enc)).logits.argmax(-1).numpy().tolist())
    return np.array(preds)


def main() -> None:
    n_latency = env_int("BENCH_LATENCY_SAMPLES", 100)
    n_accuracy = env_int("BENCH_ACCURACY_ROWS", 1000)

    holdout_full = load_split("holdout")
    holdout = subsample(holdout_full, n_accuracy)
    n = n_labels()
    inv = {v: k for k, v in load_label_map().items()}
    target_names = [inv[i] for i in range(n)]
    y_true = holdout["label"].to_numpy().astype(int)

    src = IMPROVED.output_dir / "final"
    if not src.exists():
        raise SystemExit(f"missing improved model at {src}; run `make train-improved` first")
    tok = AutoTokenizer.from_pretrained(str(src))
    torch_model = AutoModelForSequenceClassification.from_pretrained(str(src)).to("cpu")

    fp32_model = ORTModelForSequenceClassification.from_pretrained(str(ONNX_DIR / "improved-fp32"))
    int8_model = ORTModelForSequenceClassification.from_pretrained(
        str(ONNX_DIR / "improved-int8"), file_name="model_quantized.onnx"
    )

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(holdout_full), size=min(n_latency, len(holdout_full)), replace=False)
    samples = [holdout_full.iloc[int(i)]["code"] for i in sample_idx]

    def call_torch(s: str):
        enc = tok(s, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            torch_model(**enc)

    def call_ort(m):
        def _c(s: str):
            enc = tok(s, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt")
            m(**dict(enc))
        return _c

    print(f"timing {len(samples)} single inputs per variant on CPU")
    torch_lat = time_model_single(call_torch, samples)
    fp32_lat = time_model_single(call_ort(fp32_model), samples)
    int8_lat = time_model_single(call_ort(int8_model), samples)

    print(f"scoring {len(holdout)} holdout rows per variant")
    m_torch = compute_classification_metrics(y_true, predict_all_torch(torch_model, tok, holdout), target_names=target_names)
    m_fp32 = compute_classification_metrics(y_true, predict_all_onnx(fp32_model, tok, holdout), target_names=target_names)
    m_int8 = compute_classification_metrics(y_true, predict_all_onnx(int8_model, tok, holdout), target_names=target_names)

    quant_meta = {}
    quant_path = ONNX_DIR / "improved-int8" / "quantization.json"
    if quant_path.exists():
        quant_meta = json.loads(quant_path.read_text())

    delta = (m_torch["f1_macro"] - m_int8["f1_macro"]) / max(m_torch["f1_macro"], 1e-9) * 100
    out = {
        "scope": {
            "latency_samples_per_variant": len(samples),
            "accuracy_rows": int(len(holdout)),
            "holdout_rows_total": int(len(holdout_full)),
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": 1,
            "execution_provider": "CPUExecutionProvider",
            "host_machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "quantization": quant_meta,
        "torch_fp32": {"f1_macro": round(m_torch["f1_macro"], 4), **{f"latency_{k}": v for k, v in torch_lat.items()}},
        "onnx_fp32": {"f1_macro": round(m_fp32["f1_macro"], 4), **{f"latency_{k}": v for k, v in fp32_lat.items()}},
        "onnx_int8": {"f1_macro": round(m_int8["f1_macro"], 4), **{f"latency_{k}": v for k, v in int8_lat.items()}},
        "f1_macro_drop_pct_int8_vs_torch_fp32": round(delta, 2),
        "p95_speedup_int8_vs_torch_fp32": round(torch_lat["p95_ms"] / int8_lat["p95_ms"], 2),
        "p95_speedup_int8_vs_onnx_fp32": round(fp32_lat["p95_ms"] / int8_lat["p95_ms"], 2),
    }
    (REPORTS_DIR / "inference_bench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {REPORTS_DIR / 'inference_bench.json'}")


if __name__ == "__main__":
    main()
