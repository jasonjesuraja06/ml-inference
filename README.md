# ml-inference

A code-vulnerability classification pipeline. Trains transformer-based classifiers on the [DiverseVul](https://github.com/wagner-group/diversevul) and [CodeXGLUE Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) datasets, exports to ONNX with INT8 quantization, and serves predictions over a FastAPI HTTP service with an embedding cache and dynamic micro-batching.

## Overview

Classifying a function by which CWE (Common Weakness Enumeration) class it represents is useful for triage in security review, SAST tool ranking, and prioritization of remediation work. The challenge is that public CWE-labeled datasets are heavily imbalanced (a small number of CWE classes account for most examples) and labeling new functions by hand is expensive.

This project provides:

- A reproducible training pipeline for two models — a baseline and an improved variant — on the top-10 most-frequent CWE classes plus an "other" bucket.
- A benchmark on CodeXGLUE Defect Detection (Devign) for direct comparison against published numbers.
- ONNX export and INT8 dynamic quantization for low-latency inference.
- A FastAPI service that wraps the quantized model with an LRU embedding cache and dynamic micro-batching.
- A load test harness that drives sustained traffic against the service.
- An active-learning loop that auto-labels high-confidence predictions from an unlabeled pool, leaving only low-confidence cases for human review.

## Features

- **Multi-class CWE classifier.** Top-10 CWE classes + "other"; macro-F1 reported on a held-out 10K-sample test set.
- **Improved-vs-baseline training.** Class-weighted focal loss, label smoothing, minority-class augmentation, LR warmup, early stopping on validation macro F1.
- **CodeXGLUE Defect Detection benchmark.** CodeBERT fine-tune on Devign for reproducibility against the published binary-classification task.
- **ONNX + INT8 quantization.** AVX-512-VNNI dynamic quantization via Optimum and ONNX Runtime.
- **FastAPI inference service.** `POST /predict` (single, cached, batched) and `POST /predict/batch` (pre-batched). `GET /stats` exposes live cache hit rate.
- **LRU embedding cache.** xxhash-keyed cache of logits per code snippet.
- **Dynamic micro-batching.** 8ms collection window, configurable max batch size; reduces per-request overhead under concurrency.
- **Load test.** Locust scenario with configurable cache-repeat rate to mimic real scanner traffic.
- **Active-learning loop.** Confidence-bucketed routing of an unlabeled pool to auto-label / review / uncertain queues, plus an audited labeling-time delta vs the fully-manual baseline.

## Architecture

```
   datasets (DiverseVul, CodeXGLUE Devign)
                 |
                 v
   build_splits.py  ->  train / val / test / 10K holdout / unlabeled pool
                 |
                 v
   train_baseline.py        train_improved.py        train_devign.py
   (CodeBERT, minimal)      (UniXcoder + focal +     (CodeBERT on Devign
                            class weights +           binary benchmark)
                            aug + early stop)
                 |
                 v
   export_onnx.py  ->  quantize_onnx.py  (INT8 dynamic, AVX-512 VNNI)
                 |
                 v
   bench_inference.py  (FP32 vs INT8 latency + 10K-holdout accuracy)
                 |
                 v
   api/main.py (FastAPI)
        Engine = ORT session + LRUEmbeddingCache + dynamic micro-batching
        Endpoints: /predict, /predict/batch, /stats, /healthz
                 |
                 v
   bench/locustfile.py  (sustained load, configurable cache-repeat rate)

   active_learning_loop.py
        scores unlabeled pool -> auto_labeled / review / uncertain queues,
        reports labeling-time savings vs fully-manual baseline
```

See [docs/architecture.md](docs/architecture.md) for the component-by-component breakdown.

## Getting started

Requirements: Python 3.11+ (3.13 supported), GNU Make.

```bash
make install     # creates .venv and installs dependencies
make download    # DiverseVul + CodeXGLUE Devign from the Hugging Face Hub
make splits      # build train/val/test/holdout/unlabeled-pool parquets
```

The download step uses the Hugging Face Hub. DiverseVul is fetched from a community mirror (the script tries several candidates).

## Training

```bash
make train-baseline      # baseline CodeBERT on top-10 CWE multi-class
make train-improved      # improved UniXcoder with focal loss + aug + early stop
make train-devign        # CodeBERT on CodeXGLUE Defect Detection (Devign binary)
```

On Apple Silicon, training uses the MPS device automatically (set `device()` in `src/ml_inference/config.py`). On CUDA-capable hosts, it uses CUDA. Falls back to CPU otherwise. Training times on M3:

| Stage | Approx. time |
|---|---|
| `train-baseline` | 25 min |
| `train-improved` | 90 min |
| `train-devign` | 60 min |

## Inference and serving

```bash
make export-onnx        # FP32 ONNX
make quantize           # INT8 dynamic ONNX (AVX-512 VNNI)
make bench-inference    # latency + accuracy comparison across FP32 / INT8

make serve              # FastAPI on :8000 with cache + batching
make bench-api          # locust 1h sustained load test
```

### API

```http
POST /predict
Content-Type: application/json
{ "code": "void foo(char *buf) { strcpy(buf, user_input); }", "return_probs": false }

200 OK
{
  "label": "CWE-119",
  "label_id": 0,
  "confidence": 0.91,
  "cached": false,
  "inference_ms": 14.7
}
```

`POST /predict/batch` takes up to 64 codes in one call and bypasses the dynamic-batch queue.

`GET /stats` exposes cache hit rate and model variant.

`GET /healthz` for liveness probes.

## Active learning

```bash
make active-learn        # one iteration: score pool, route, report
```

Confidence thresholds are configured in `src/ml_inference/active_learning_loop.py`. Auto-labeled rows land in `data/splits/auto_labeled.parquet` and are picked up by subsequent `train-improved` runs. Labeling-time accounting is documented in [docs/labeling_runbook.md](docs/labeling_runbook.md).

## Benchmarks

Reference numbers (Apple Silicon M3, single-host CPU inference, 2026-05 dry-run):

| Metric | Value |
|---|---|
| Improved-model macro F1 on 10K holdout (top-10 CWE + other) | ~0.84 |
| Baseline macro F1 on same holdout | ~0.72 |
| PyTorch FP32 single-input P95 latency | ~800ms |
| ONNX INT8 single-input P95 latency | <200ms |
| INT8 vs FP32 macro-F1 drop | <3% |
| FastAPI P95 latency under 50-concurrent-user load | <150ms |
| Daily request capacity extrapolated from load test | 10K+ |

Reproduce on your own hardware with `make train-* && make export-onnx && make quantize && make bench-inference && make bench-api`. Results land in `bench/reports/*.json`.

## Project layout

```
src/ml_inference/
  config.py                Paths + hyperparams. BASELINE and IMPROVED dataclasses.
  data.py                  Dataset class, split loader, class weights
  metrics.py               classification_report wrapper
  loss.py                  Focal loss with class weights + label smoothing
  augment.py               Code augmentation for minority CWEs
  train_baseline.py        Baseline trainer
  train_improved.py        Improved trainer
  train_devign.py          CodeXGLUE Defect Detection benchmark
  export_onnx.py           FP32 ONNX export
  quantize_onnx.py         INT8 dynamic quantization
  bench_inference.py       FP32 vs INT8 latency + accuracy
  active_learning_loop.py  Pool scoring + queue routing + time accounting

api/
  main.py                  FastAPI app
  inference.py             ORT session + cache + batching
  schemas.py               Pydantic request/response models

bench/
  locustfile.py            Load test
  reports/                 JSON reports + locust HTML output

scripts/
  download_data.py
  build_splits.py

docs/
  architecture.md
  labeling_runbook.md
  limitations.md
```

## Scope and roadmap

See [docs/limitations.md](docs/limitations.md).

## License

MIT (see LICENSE).
