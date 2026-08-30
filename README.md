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
               +--> arch_bench.py        FP32 vs INT8 on this CPU and, via
               |                         .github/workflows/arch-bench.yml,
               |                         on x86-64 CI runners
               +--> api/main.py          ORT session + LRU cache + 8ms
               |         |               micro-batch window
               |         v
               |     bench/locustfile.py    load test
               v
        active_learning_loop.py   pool -> auto / review / uncertain

CodeXGLUE Devign --> train_devign.py   independent binary benchmark
```

## Measured results

All numbers below were produced on an **Apple M4 Pro, 14 cores, 48 GB RAM, macOS arm64, no CUDA GPU**, at the reduced scope printed beside each. Every run writes a JSON report to `bench/reports/` carrying its own `scope` block, and those reports are committed. Training used MPS; all inference measurements are CPU-only, because ONNX Runtime has no MPS execution provider and timing an MPS PyTorch model against a CPU ONNX graph would measure the accelerator instead of the quantization.

**Classifier quality.** Top-10 CWE classes plus `__OTHER__`, 4,000 sampled training rows, 2 epochs, 256 tokens, evaluated on the full 2,415-row holdout. This is a small-scope run, not a converged one.

| Config | Macro F1 | Accuracy | Macro recall | Train wall clock |
|---|---|---|---|---|
| `codebert-base`, plain cross-entropy | 0.160 | 0.438 | 0.176 | 391 s |
| `unixcoder-base`, focal + class weights + augmentation | 0.239 | 0.260 | 0.283 | 357 s |

The second config trades accuracy for macro F1: it stops the model from collapsing onto `__OTHER__`, the largest class, and raises macro recall from 0.176 to 0.283. Reporting only macro F1 would hide that the accuracy went down.

The two rows differ in more than the loss function, and the difference cannot be attributed to class weighting alone. The `scope` block in each report records what actually changed: base model (`codebert-base` against `unixcoder-base`), learning rate (5e-5 against 2e-5), gradient accumulation (1 step against 2), and training rows after minority-class augmentation (4,000 against 4,284). Isolating one factor needs an ablation this repository does not contain.

```
MAX_TRAIN_ROWS=4000 EPOCHS=2 make train-baseline train-improved
```

**Quantization and inference latency.** Batch size 1, CPU, 100 timed inputs per variant, macro F1 over 1,000 holdout rows, on this arm64 host at ONNX Runtime's default thread count. Two runs of the identical command, because one run of a percentile is not a measurement.

| Variant | P50 (run 1 / run 2) | P95 (run 1 / run 2) | Macro F1 | Model size |
|---|---|---|---|---|
| PyTorch FP32 | 30.7 / 32.8 ms | 37.1 / 38.4 ms | 0.258 | n/a |
| ONNX FP32 | 26.2 / 31.2 ms | 32.3 / 43.8 ms | 0.258 | 504 MB |
| ONNX INT8 dynamic | 27.2 / 28.5 ms | 33.2 / 38.7 ms | 0.255 | 127 MB |

INT8 shrinks the model 3.96x and costs 1.3% of macro F1. Its latency effect on this host is **not measurable at this sample size**: INT8 comes out at 0.96x of ONNX FP32 at P50 in the first run and 1.10x in the second, so the sign of the effect flips between two runs of the same command on the same machine. Macro F1 is identical to four decimals across both runs, so the accuracy cost of quantization is a real number and the latency difference here is not one. Both reports are committed as `inference_bench_run1.json` and `inference_bench.json`.

Separating the two needs a benchmark that pins the thread count and takes medians over repeats, and it needs more than one CPU, because whether dynamic INT8 is faster is a property of the instruction set rather than of the model.

```
make export-onnx && make quantize && make bench-inference
```

**The same question on six CI runners.** `arch_bench.py` exports and quantizes the same architecture from the public base checkpoint, so it needs neither the fine-tuned weights nor the dataset and runs unchanged on a GitHub Actions runner. `.github/workflows/arch-bench.yml` runs it across three runner images; which CPU a job lands on is not the job's choice, so each report records the CPU model and its instruction flags. Six jobs reached four distinct CPU models. Intra-op threads are pinned to 4 on every host, including this one, so the rows differ in instruction set rather than in core count, and each row is the median of its runs.

| CPU | INT8 instructions present | Preset | Runs | FP32 P50 | INT8 P50 | INT8 speedup P50 | INT8 speedup P95 |
|---|---|---|---|---|---|---|---|
| Apple M4 Pro | FEAT_DotProd, FEAT_I8MM, FEAT_BF16 | `arm64` | 2 | 36.98 ms | 35.85 ms | 1.03x | 1.13x |
| Apple M4 Pro (fine-tuned weights) | FEAT_DotProd, FEAT_I8MM, FEAT_BF16 | `arm64` | 2 | 38.01 ms | 34.73 ms | 1.09x | 1.27x |
| AMD EPYC 7763 64-Core Processor | avx2 | `avx2` | 3 | 272.21 ms | 191.56 ms | 1.42x | 1.4x |
| AMD EPYC 9V74 80-Core Processor | avx2 | `avx2` | 1 | 292.25 ms | 206.02 ms | 1.42x | 1.42x |
| Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | avx512_vnni, avx512f, avx2 | `avx512_vnni` | 1 | 208.16 ms | 99.73 ms | 2.09x | 2.02x |
| Intel(R) Xeon(R) 6973P-C | amx_int8, avx512_vnni, avx_vnni, avx512f, avx2 | `avx512_vnni` | 1 | 146.42 ms | 51.56 ms | 2.84x | 2.56x |

INT8 is faster on every x86-64 part measured, **including the AMD ones that have neither AVX-512 nor VNNI**, so VNNI is not what makes dynamic INT8 worth doing. It roughly doubles a win that AVX2 already delivers, and AMX-INT8 doubles it again. The speedup tracks the CPU's integer dot-product support in order: 1.4x, 2.1x, 2.8x. On this arm64 host it is 1.03x to 1.09x at P50, a spread no wider than the noise between two runs of the same configuration, and that is despite the CPU reporting both DotProd and I8MM. Whether the AArch64 kernels ONNX Runtime ships or the `arm64` quantization preset is responsible is not something these measurements separate.

That holds at both thread counts measured: pinned to 4 threads INT8 sits at 1.03x to 1.09x, and at ONNX Runtime's default of all 14 cores it is indistinguishable from FP32. Neither reading turns it into a win.

Read the ratio within a row, not down a column. A dedicated laptop performance core and a share of a virtualised server are not comparable in absolute latency, which is why the M4 Pro's FP32 P50 is a quarter of the fastest Xeon's.

The practical reading: take the 4x size reduction everywhere, and expect the latency reduction only on x86. The `(fine-tuned weights)` row is what makes the other rows usable. Latency here does not depend on weight values: every input is padded to 256 tokens, so each forward pass runs the same operators over the same shapes whatever the weights hold, and integer GEMM does not run faster on one bit pattern than another. Running the identical probe against the fine-tuned checkpoint checks that rather than asserting it, and the two arm64 rows agree to within 3% on median latency for both variants.

```
.venv/bin/python -m ml_inference.arch_bench                               # base checkpoint
ARCH_BENCH_CHECKPOINT=models/improved/final \
  .venv/bin/python -m ml_inference.arch_bench                             # fine-tuned weights
.venv/bin/python scripts/arch_table.py --by-cpu                           # renders the table above
```

Each writes `bench/reports/arch_latency_<cpu capability>_<weights>.json`. The committed arm64 reports carry a `-run1` / `-run2` suffix set through `ARCH_BENCH_TAG`, because a row is a median over repeats and one run of a percentile is not a measurement. The x86-64 reports come from the workflow, unmodified, under `bench/reports/ci/`.

**Serving throughput.** 50 concurrent locust users, 60 s, single uvicorn worker, 30% payload repeat rate, per-user request streams seeded so all five configurations see the same sequence of decisions. Five configurations rather than an optimized-against-baseline pair, because a single blended number cannot say which part earned the gain.

| Service configuration | Requests/s | P50 | P95 | P99 | Cache hit rate | Failures |
|---|---|---|---|---|---|---|
| FP32, no cache, no batching | 37.7 | 980 ms | 1300 ms | 2700 ms | n/a | 0 / 2247 |
| INT8, no cache, no batching | 41.7 | 860 ms | 1100 ms | 1300 ms | n/a | 0 / 2472 |
| INT8 + cache | 100.3 | 120 ms | 510 ms | 650 ms | 64.6% | 0 / 5993 |
| INT8 + micro-batching | 35.5 | 1100 ms | 1700 ms | 1900 ms | n/a | 0 / 2117 |
| INT8 + cache + micro-batching | 85.7 | 170 ms | 880 ms | 1100 ms | 61.1% | 0 / 5118 |

The cache is the entire gain, and micro-batching costs throughput. Against the INT8 no-cache no-batching reference: the cache alone is **2.4x**, batching alone is **0.85x**, and the two together are 2.1x, which is worse than the cache on its own. Quantization contributes 1.11x in the serving path, in the same neighbourhood as the 1.03x it gets at batch size 1 on this architecture.

Micro-batching losing is the interesting row. The most likely reason, which these runs support but do not isolate, is that both of its premises fail here. A batcher pays off when the compute unit is idle at batch size 1, and ONNX Runtime at batch size 1 already spreads a 256-token forward pass across every core on this host, so a batch of 16 does 16 times the work rather than filling idle capacity. The forward pass also runs on the event loop thread, so while a batch executes nothing else is served, including the cache hits that would otherwise return in microseconds. That is why adding the batcher to the cache costs 15% of the cache's throughput. The batcher reached a mean batch size of 14.55 out of a ceiling of 16, so it was fully engaged and still lost.

The measured best configuration on this host is therefore INT8 with the cache and **without** the batcher: `MODEL_VARIANT=quantized NO_BATCHING=1`. The batcher is kept because the argument for it is hardware-dependent in the same way quantization is, and because `/predict/batch` uses the same batched forward path for callers that genuinely have 64 functions at once.

The 64.6% hit rate at a 30% repeat rate is a property of the synthetic traffic, not a claim about real scanners. Every configuration ran 0 failures; the request-size cap matters here, because at the original 20,000-character limit 2.5% of the CWE-labeled DiverseVul functions in these splits were rejected with a 422 (`scripts/dataset_stats.py`). The five runs are sequential against separate processes, so a comparison between two rows carries the run-to-run variance of both.

```
scripts/run_load_test.sh all 50 60s      # all five, ~6 minutes
scripts/run_load_test.sh int8-cache 50 60s
```

Each run writes `bench/reports/api_load_summary_<config>.json` from locust and `bench/reports/api_stats_<config>.json` from the service itself, so the configuration a number was measured under is recorded by the process that served it rather than by the script that drove it.

**CodeXGLUE Devign** (`MAX_TRAIN_ROWS=2000 MAX_EVAL_ROWS=1000 EPOCHS=1 make train-devign`). Binary vulnerable/benign, 2,000 of 21,854 training rows, 1 epoch, 1,000 test rows: accuracy 0.533, binary F1 0.516. Well short of the roughly 0.62 to 0.65 published full fine-tunes reach, which is what one epoch on 9% of the data buys.

**Dataset** (`.venv/bin/python scripts/dataset_stats.py`, written to `bench/reports/dataset_stats.json`). The mirror holds 330,492 rows, of which 18,945 are `target == 1` and 16,109 of those carry a CWE; 26.2% of the labeled rows cite more than one. The splits keep 16,101 rows. **Active learning** (`make active-learn`). Over the 1,368-row pool the improved model cleared the 0.92 auto-label threshold on 0 rows: 58 went to review, 1,310 to the uncertain queue. At this training scope the loop saves no labeling effort at all. The routing works; it is worth something only once the model is confident enough to auto-label, which this run is not.

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

`POST /predict` returns `label`, `label_id`, `confidence`, `cached`, and `inference_ms`. `POST /predict/batch` takes up to 64 codes and bypasses the batch queue. `GET /healthz` is a liveness probe.

`GET /stats` reports the configuration the process is running under and the counters it has run up, so the settings behind a published number can be read off the service rather than inferred from the source:

```json
{"model_variant": "quantized", "max_seq_len": 256,
 "cache":    {"enabled": true, "capacity": 8192, "hits": 2867, "misses": 1827, "hit_rate": 0.6108},
 "batching": {"enabled": true, "window_ms": 8.0, "max_batch": 16,
              "batches_run": 274, "requests_batched": 1827, "mean_batch_size": 6.67}}
```

That body is `bench/reports/api_stats_int8-cache-batch.json`, written by the service at the end of its own load-test run.

The served defaults are an 8 ms micro-batch window, a batch ceiling of 16, and an 8192-entry cache, and those are the values every serving number above was measured at. The window is short because an uncached forward pass costs about 30 ms, so a window long enough to fill a batch of 16 under light traffic would add more queueing delay than the batch saves. The cache holds roughly half the 16,101-row split, so nothing evicts during a load test and the reported hit rate is the repeat rate of the traffic rather than an artifact of capacity. Override any of them with `BATCH_WINDOW_MS`, `BATCH_MAX`, `CACHE_CAPACITY`, `NO_CACHE=1`, and `NO_BATCHING=1`.

Any training or benchmark run can be scoped down with `MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_TRAIN_ROWS`, `MAX_EVAL_ROWS`, and `MAX_SEQ_LEN`. Whatever you set is written into the report, so a number always carries the run that produced it.

## Limitations

- Every accuracy figure comes from a short run on a subset. No model here is trained to convergence and the absolute scores are low.
- DiverseVul attaches a fixing commit's CWE to both the vulnerable and the patched function, so `build_splits.py` keeps only `target == 1` rows, leaving 16,101 of 330,492. Those still carry the attribution noise of the original mining, and the 26.2% of labeled rows citing several CWEs keep only the first.
- Only the top-10 CWEs are distinct classes. `__OTHER__` absorbs the long tail and is the largest class in the split.
- Latency is single-host, batch size 1, warm process, no network. The load test runs one uvicorn worker on the same machine as the client, and the per-process LRU cache means `/stats` reports one worker's view.
- No authentication, rate limiting, or drift monitoring on any endpoint. Full list in `docs/limitations.md`; `docs/architecture.md` covers the components.

## License

MIT. See [LICENSE](LICENSE).
