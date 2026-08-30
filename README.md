# ml-inference

[![CI](https://github.com/jasonjesuraja06/ml-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonjesuraja06/ml-inference/actions/workflows/ci.yml)

Classifies a C/C++ function by CWE class, then serves the model over HTTP with ONNX Runtime, an LRU cache, and dynamic micro-batching.

## Why

Triaging security findings means asking which weakness class a function represents, not just whether it is suspicious. Public CWE-labeled data is small and heavily skewed, so the interesting engineering questions are what class imbalance does to a fine-tuned code model, and what quantization and serving choices actually buy at inference time. This repository measures both rather than asserting them.

## Architecture

```
DiverseVul --> build_splits.py --> train / val / test / holdout / pool
                                        |
     +----------------------------------+
     +--> train_baseline.py   CodeBERT, plain cross-entropy
     +--> train_improved.py   UniXcoder, class-weighted focal loss,
               |              label smoothing, augmentation, early stop
               v
        export_onnx.py --> quantize_onnx.py    FP32 ONNX --> INT8 dynamic
               +--> bench_inference.py   latency + macro F1 per variant
               +--> api/main.py          ORT session + LRU cache + 8ms
               |         |               micro-batch window
               |         v
               |     bench/locustfile.py    load test
               v
        active_learning_loop.py   pool -> auto / review / uncertain

CodeXGLUE Devign --> train_devign.py   independent binary benchmark
```

`docs/architecture.md` covers the components.

## Measured results

All numbers below were produced on an **Apple M4 Pro, 14 cores, 48 GB RAM, macOS arm64, no CUDA GPU**, at the reduced scope printed beside each. Every run writes a JSON report to `bench/reports/` carrying its own `scope` block, and those reports are committed. Training used MPS; all inference measurements are CPU-only, because ONNX Runtime has no MPS execution provider and timing an MPS PyTorch model against a CPU ONNX graph would measure the accelerator instead of the quantization.

**Classifier quality.** Top-10 CWE classes plus `__OTHER__`, 4,000 training rows, 2 epochs, 256 tokens, evaluated on the full 2,415-row holdout. This is a small-scope run, not a converged one.

| Config | Macro F1 | Accuracy | Macro recall | Train wall clock |
|---|---|---|---|---|
| `codebert-base`, plain cross-entropy | 0.160 | 0.438 | 0.176 | 391 s |
| `unixcoder-base`, focal + class weights + augmentation | 0.239 | 0.260 | 0.283 | 357 s |

The second config trades accuracy for macro F1, which is what class weighting is supposed to do: it stops the model from collapsing onto `__OTHER__`, the largest class, and raises macro recall from 0.176 to 0.283. Reporting only macro F1 would hide that the accuracy went down.

```
MAX_TRAIN_ROWS=4000 EPOCHS=2 make train-baseline train-improved
```

**Quantization and inference latency.** Batch size 1, CPU, 100 timed inputs per variant, macro F1 over 1,000 holdout rows.

| Variant | P50 | P95 | P99 | Macro F1 | Model size |
|---|---|---|---|---|---|
| PyTorch FP32 | 30.7 ms | 37.1 ms | 43.5 ms | 0.258 | n/a |
| ONNX FP32 | 26.2 ms | 32.3 ms | 39.1 ms | 0.258 | 504 MB |
| ONNX INT8 dynamic | 27.2 ms | 33.2 ms | 40.4 ms | 0.255 | 127 MB |

INT8 shrinks the model 3.96x and costs 1.3% of macro F1, but it is **3% slower than ONNX FP32** at P95 on this host. Dynamic INT8 pays off on x86 with AVX-512 VNNI; on Apple Silicon through ONNX Runtime the arm64 kernels give back the win. The size reduction is real, the speedup is not. `quantize_onnx.py` picks its Optimum preset from the host ISA rather than hard-coding the x86 one.

```
make export-onnx && make quantize && make bench-inference
```

**Serving throughput.** 50 concurrent locust users, 60 s, 30% payload repeat rate, single uvicorn worker.

| Service config | Requests/s | P50 | P95 | P99 | Failures |
|---|---|---|---|---|---|
| INT8 + LRU cache + micro-batching | 108.4 | 70 ms | 550 ms | 730 ms | 0 / 6477 |
| FP32, no cache, no batching | 44.2 | 790 ms | 1000 ms | 1200 ms | 0 / 2625 |

2.5x throughput and an 11x lower median. The cache did most of the work, reaching a 66% hit rate at a 30% repeat rate; that hit rate is a property of the synthetic traffic, not a claim about real scanners. Raising the request-size cap also mattered: at the original 20,000-character limit, 2.4% of real DiverseVul functions were rejected with a 422.

```
scripts/run_load_test.sh 50 60s optimized && PORT=8001 scripts/run_load_test.sh 50 60s baseline
```

**CodeXGLUE Devign** (`MAX_TRAIN_ROWS=2000 MAX_EVAL_ROWS=1000 EPOCHS=1 make train-devign`). Binary vulnerable/benign, 2,000 of 21,854 training rows, 1 epoch, 1,000 test rows: accuracy 0.533, binary F1 0.516. Well short of the roughly 0.62 to 0.65 published full fine-tunes reach, which is what one epoch on 9% of the data buys.

**Active learning** (`make active-learn`). Over the 1,368-row pool the improved model cleared the 0.92 auto-label threshold on 0 rows: 58 went to review, 1,310 to the uncertain queue. At this training scope the loop saves no labeling effort at all. The routing works; it is worth something only once the model is confident enough to auto-label, which this run is not.

## Quickstart

Python 3.11 or 3.12, GNU Make, and [uv](https://docs.astral.sh/uv/).

```bash
make install     # uv venv + editable install
make download    # DiverseVul + CodeXGLUE Devign, about 200 MB
make splits      # train / val / test / holdout / pool parquets
make test lint
```

Then train, export, and serve:

```bash
MAX_TRAIN_ROWS=4000 EPOCHS=2 make train-improved   # drop the overrides for a full run
make export-onnx && make quantize && make serve    # FastAPI on :8000
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"code":"void f(char *b){ strcpy(b, getenv(\"X\")); }"}'
```

`POST /predict` returns `label`, `label_id`, `confidence`, `cached`, and `inference_ms`. `POST /predict/batch` takes up to 64 codes and bypasses the batch queue. `GET /stats` reports the cache hit rate, `GET /healthz` is a liveness probe.

Any training or benchmark run can be scoped down with `MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_TRAIN_ROWS`, `MAX_EVAL_ROWS`, and `MAX_SEQ_LEN`. Whatever you set is written into the report, so a number always carries the run that produced it.

## Limitations

- Every accuracy figure comes from a short run on a subset. No model here is trained to convergence and the absolute scores are low.
- DiverseVul attaches a fixing commit's CWE to both the vulnerable and the patched function, so `build_splits.py` keeps only `target == 1` rows, leaving 16,101 of 330,492. Those still carry the attribution noise of the original mining, and the quarter of rows citing several CWEs keep only the first.
- Only the top-10 CWEs are distinct classes. `__OTHER__` absorbs the long tail and is the largest class in the split.
- Latency is single-host, batch size 1, warm process, no network. The load test runs one uvicorn worker on the same machine as the client, and the per-process LRU cache means `/stats` reports one worker's view.
- No authentication, rate limiting, or drift monitoring on any endpoint. Full list in `docs/limitations.md`.

## License

MIT. See [LICENSE](LICENSE).
