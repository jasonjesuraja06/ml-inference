# ml-inference

[![CI](https://github.com/jasonjesuraja06/ml-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonjesuraja06/ml-inference/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jasonjesuraja06/ml-inference/blob/main/notebooks/train_gpu.ipynb)

Classifies a C/C++ function by CWE class, then serves the model over HTTP with ONNX Runtime, an LRU cache, and dynamic micro-batching. Every number here was produced by the command printed beside it on an Apple M4 Pro (14 cores, 48 GB, macOS arm64, no CUDA GPU), and its raw output is committed under `results/` or `bench/reports/`. `make results` re-renders the tables from those files; `tests/test_readme_matches_results.py` fails the build when a number here stops matching the run it came from.

## Finding: 94% of the CWE-labeled rows in DiverseVul are not vulnerable

DiverseVul ships both the vulnerable and the patched version of every function a fixing commit touched, and attaches the commit's CWE to both. The CWE is a property of the commit, not of the function. 281,734 of the mirror's 330,492 rows carry a non-empty `cwe` field, but only 16,109 of those are `target == 1`. Keying training data off the `cwe` field alone feeds the model 265,625 rows of patched, benign code wearing a vulnerability label against 16,109 genuine ones, and the resulting classifier is largely learning which functions a security commit touched.

A type error hides the first half. The `cwe` field deserializes to a `numpy.ndarray`, not a string, so `str(row["cwe"])` renders as `"['CWE-119' 'CWE-787']"`. 26.2% of labeled rows cite more than one CWE, so naive stringification invents composite classes and turns 142 real CWEs into 625 label strings, most holding a handful of rows. Neither failure raises: both produce a clean dataframe and a training run that converges to a meaningless target. `scripts/build_splits.py` filters on `target == 1` and normalizes the array through `_first_cwe`, which takes the first entry and drops the rest. That leaves 16,101 usable rows, the split everything below is trained on, and it is the reason the class counts are in the thousands rather than the hundreds of thousands. Multi-CWE rows keep only their first label and carry an irreducible error floor under a single-label objective.

```
.venv/bin/python scripts/dataset_stats.py      # bench/reports/dataset_stats.json
make cwe-support                               # results/dataset/cwe_support.json
```

## Task A: top-10 CWE multiclass

DiverseVul, the 10 most frequent CWEs plus `__OTHER__`, all 9,854 train rows, 3 epochs, 256 tokens, full 2,415-row holdout. `TOP_K_CWES = 10` was fixed in `config.py` before any model was trained; there is no elbow there (rank 10 holds 462 rows, rank 11 holds 398), so `__OTHER__` absorbs 132 CWEs and 36.3% of the data. That is why the no-training floor is on the table.

| Config | Train rows | Macro F1 | Weighted F1 | Accuracy | Macro recall | Wall clock |
|---|---|---|---|---|---|---|
| majority class, no training | 0 | 0.0497 | 0.2059 | 0.3764 | 0.0909 | 0 s |
| `codebert-base`, plain cross-entropy | 9,854 | 0.4093 | 0.5196 | 0.5350 | 0.3973 | 23 min 31 s |
| `unixcoder-base`, focal + class weights + augmentation | 10,552 | 0.4048 | 0.4285 | 0.4186 | 0.4581 | 26 min 46 s |

**The two configurations are a wash on macro F1: 0.4093 against 0.4048 on a 2,415-row holdout.** Neither wins. They reach it differently, and that is the result. The reweighted arm lifts the rare classes (CWE-399 0.0882 to 0.2888, CWE-20 0.2965 to 0.3676) and pays on the common ones (`__OTHER__` 0.6833 to 0.4453, CWE-125 0.5866 to 0.4969); macro recall rises 0.3973 to 0.4581 while accuracy falls 0.5350 to 0.4186. At an earlier 4,000-row scope these same two configurations scored 0.1597 and 0.2392 and reweighting looked like a clear win. On the whole split that margin is gone, which reads as the imbalance handling having compensated for data scarcity rather than for imbalance. Per-class F1 for all 11 classes is in [docs/benchmarks.md](docs/benchmarks.md). The two rows differ in backbone, learning rate (5e-5 against 2e-5), gradient accumulation, and augmented row count, so no single factor can be credited; separating them needs an ablation this repository does not contain. Neither run is converged: validation macro F1 went 0.2094, 0.3711, 0.4288 for the baseline and 0.3036, 0.3809, 0.4019 for the improved arm, still rising when the epoch budget ended. Epoch counts come from the 2-to-4 range in Devlin et al. Section A.3 and a wall-clock budget, were fixed before the runs, and were held identical across both arms. There is no hyperparameter search anywhere in this repository.

```
make majority-baseline && EPOCHS=3 make train-baseline && EPOCHS=3 NO_AUTO_LABELS=1 make train-improved
```

## Task B: binary detection on CodeXGLUE Defect Detection

All 21,854 Devign train rows, 2 epochs, 256 tokens, full 2,732-row test split, same `microsoft/codebert-base` checkpoint as the leaderboard row, so these sit on one scale.

| System | Accuracy | Binary F1 | Source |
|---|---|---|---|
| BiLSTM, CodeXGLUE leaderboard | 0.5937 | not reported | Lu et al., NeurIPS 2021 D&B |
| TextCNN, CodeXGLUE leaderboard | 0.6069 | not reported | Lu et al., NeurIPS 2021 D&B |
| RoBERTa, CodeXGLUE leaderboard | 0.6105 | not reported | Lu et al., NeurIPS 2021 D&B |
| CodeBERT, CodeXGLUE leaderboard | 0.6208 | not reported | Lu et al., NeurIPS 2021 D&B |
| **this repository**, `codebert-base`, 2 epochs, 256 tokens | **0.6332** | **0.5197** | `results/devign_codebert/` |

This **ties** the published CodeBERT row. The 0.6332 against 0.6208 gap is 34 test examples, 1.3 times the 0.0092 binomial standard error at this sample size, from one seed with no variance estimate. It is not a win and should not be read as one. What it cost is the interesting part: 2 epochs against the reference recipe's 5, 256 tokens against 400, batch 16 against 32, no hyperparameter search, on a CPU host with no CUDA GPU. Accuracy is the only metric the leaderboard reports and the only column that compares. The rest does not flatter: binary F1 is 0.5197 from precision 0.6522 and recall 0.4319, so the model misses 57% of vulnerable functions. On a near-balanced benchmark accuracy hides that; `results/devign_codebert/confusion_matrix.txt` does not. Citations, tables, and a comparability matrix for every published number are in `results/published_baselines.json` and [docs/benchmarks.md](docs/benchmarks.md); the Devign and DiverseVul papers are **not** comparable to this and the matrix says why.

```
EPOCHS=2 make train-devign
```

## Quantization is an architecture-dependent win

INT8 dynamic quantization shrinks the model 3.96x, from 504 MB to 127 MB, and costs 3.49% of macro F1 (0.4123 to 0.3979 over 1,000 holdout rows). Both figures are stable to four decimals across two runs. The latency effect is not: on this arm64 host INT8 comes out at 1.00x and 0.96x of ONNX FP32 at P50 across two runs of the identical command, so the sign flips. At P95 it is 1.12x and 1.09x, a small consistent edge.

Whether dynamic INT8 is faster is a property of the instruction set, so `arch_bench.py` rebuilds the same graph from the public base checkpoint (no fine-tuned weights, no dataset) and `.github/workflows/arch-bench.yml` runs it on x86-64 CI. Six jobs reached four CPU models. Intra-op threads are pinned to 4 everywhere, so rows differ in instruction set rather than core count, and each row is a median over its runs.

| CPU | INT8 instructions | Preset | Runs | FP32 P50 | INT8 P50 | Speedup P50 | Speedup P95 |
|---|---|---|---|---|---|---|---|
| Apple M4 Pro | FEAT_DotProd, FEAT_I8MM, FEAT_BF16 | `arm64` | 2 | 36.98 ms | 35.85 ms | 1.03x | 1.13x |
| Apple M4 Pro (fine-tuned weights) | FEAT_DotProd, FEAT_I8MM, FEAT_BF16 | `arm64` | 2 | 38.01 ms | 34.73 ms | 1.09x | 1.27x |
| AMD EPYC 7763 | avx2 | `avx2` | 3 | 272.21 ms | 191.56 ms | 1.42x | 1.4x |
| AMD EPYC 9V74 | avx2 | `avx2` | 1 | 292.25 ms | 206.02 ms | 1.42x | 1.42x |
| Intel Xeon Platinum 8370C | avx512_vnni, avx512f, avx2 | `avx512_vnni` | 1 | 208.16 ms | 99.73 ms | 2.09x | 2.02x |
| Intel Xeon 6973P-C | amx_int8, avx512_vnni, avx_vnni, avx512f, avx2 | `avx512_vnni` | 1 | 146.42 ms | 51.56 ms | 2.84x | 2.56x |

INT8 wins on every x86-64 part measured, **including the AMD ones that have neither AVX-512 nor VNNI**, so VNNI is not what makes dynamic INT8 worth doing. It roughly doubles a win AVX2 already delivers, and AMX-INT8 doubles it again: 1.4x, 2.1x, 2.8x in order of integer dot-product support. On arm64 it is 1.03x to 1.09x at P50 despite the CPU reporting both DotProd and I8MM, and indistinguishable from FP32 at the default thread count. These runs cannot separate ONNX Runtime's AArch64 kernels from the `arm64` preset as the cause. Read ratios within a row, never down a column: a laptop performance core and a share of a virtualised server are not comparable in absolute latency. The practical reading: take the 4x size reduction everywhere, expect the latency reduction only on x86. The fine-tuned row exists because latency here should not depend on weight values (every input is padded to 256 tokens, so each pass runs the same operators over the same shapes), and running the identical probe against the fine-tuned checkpoint checks that instead of asserting it. The two arm64 rows agree within 3%.

```
make export-onnx && make quantize && make bench-inference
.venv/bin/python -m ml_inference.arch_bench && .venv/bin/python scripts/arch_table.py --by-cpu
```

## Serving: the cache is the whole gain, and batching costs throughput

50 concurrent locust users, 60 s, one uvicorn worker, 30% payload repeat rate, per-user streams seeded so all five configurations see the same decisions. Five configurations rather than an optimized-against-baseline pair, because a blended number cannot say which part earned the gain.

| Service configuration | Requests/s | P50 | P95 | P99 | Cache hit rate | Failures |
|---|---|---|---|---|---|---|
| FP32, no cache, no batching | 37.7 | 980 ms | 1300 ms | 2700 ms | n/a | 0 / 2247 |
| INT8, no cache, no batching | 41.7 | 860 ms | 1100 ms | 1300 ms | n/a | 0 / 2472 |
| INT8 + cache | 100.3 | 120 ms | 510 ms | 650 ms | 64.6% | 0 / 5993 |
| INT8 + micro-batching | 35.5 | 1100 ms | 1700 ms | 1900 ms | n/a | 0 / 2117 |
| INT8 + cache + micro-batching | 85.7 | 170 ms | 880 ms | 1100 ms | 61.1% | 0 / 5118 |

Against the INT8 no-cache no-batching row: the cache alone is **2.4x**, batching alone is **0.85x**, and both together are 2.1x, worse than the cache by itself. Quantization contributes 1.11x in the serving path, near the 1.03x it gets at batch size 1 on this architecture. Micro-batching losing is the useful negative result, and the previous two-row comparison could not have found it because it moved the model, the cache, and the batcher at once. Both premises fail here. A batcher pays off when the compute unit idles at batch size 1, but ONNX Runtime already spreads a 256-token pass across every core, so a batch of 16 does 16x the work rather than filling idle capacity. The forward pass also runs on the event loop thread, so while a batch executes nothing else is served, including cache hits that would return in microseconds. That is why adding the batcher to the cache costs 15% of its throughput. The batcher reached a mean batch size of 14.55 against a ceiling of 16, so it was fully engaged and still lost. The measured best configuration on this host is `MODEL_VARIANT=quantized NO_BATCHING=1`. The batcher stays because the argument for it is hardware-dependent in the same way quantization is, and `/predict/batch` uses the same path for callers that genuinely have 64 functions at once. The 64.6% hit rate at a 30% repeat rate is a property of synthetic traffic, not a claim about real scanners.

```
scripts/run_load_test.sh all 50 60s     # all five, about 6 minutes
```

## Quickstart

Python 3.11 or 3.12, GNU Make, and [uv](https://docs.astral.sh/uv/).

```bash
make install                                       # uv venv + editable install
make download && make splits                       # about 200 MB
make test lint
EPOCHS=3 make train-improved                       # add MAX_TRAIN_ROWS=N to scope down
make export-onnx && make quantize && make serve    # FastAPI on :8000
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"code":"void f(char *b){ strcpy(b, getenv(\"X\")); }"}'
```

`POST /predict` returns `label`, `label_id`, `confidence`, `cached`, and `inference_ms`. `POST /predict/batch` takes up to 64 codes and bypasses the batch queue. `GET /healthz` is a liveness probe. `GET /stats` reports the configuration the process is actually running under alongside its counters, so the settings behind a published number can be read off the service rather than inferred from source; `bench/reports/api_stats_*.json` are those bodies, written by the service at the end of each load run. Defaults are an 8 ms batch window, a batch ceiling of 16, and an 8192-entry cache, and every serving number above was measured at them. Override with `BATCH_WINDOW_MS`, `BATCH_MAX`, `CACHE_CAPACITY`, `NO_CACHE=1`, `NO_BATCHING=1`. Training and benchmark scope is set by `MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_TRAIN_ROWS`, `MAX_EVAL_ROWS`, `MAX_SEQ_LEN`; whatever is set gets written into the report. [`notebooks/train_gpu.ipynb`](notebooks/train_gpu.ipynb) runs this same path at full scope on a free Colab T4 by calling these scripts unchanged, and ships with every cell output cleared: its numbers exist only once it is run.

## Limitations

- Short schedules, no hyperparameter search, nothing trained to convergence. Absolute scores are low and the Task A macro F1 has no published comparison point, which is why Task B is here.
- The CWE label is commit-level even after the `target == 1` filter, so the retained rows still carry the attribution noise of the original mining. Only the top-10 CWEs are distinct classes and `__OTHER__` is the largest one.
- Latency is single-host, batch size 1, warm process, no network. Two runs of the same command disagree on the sign of the arm64 INT8 P50 effect, so no single-run difference under about 15% here is an effect.
- The cross-architecture table measures latency only and never accuracy: CI exports an untrained head, which is sound for timing and worthless for scoring. `ubuntu-latest` is not a fixed machine, so a rerun lands on a different CPU mix.
- The load test runs one uvicorn worker on the client's machine, and the per-process LRU cache means `/stats` shows one worker's view.
- No authentication, rate limiting, calibration, or drift monitoring. Full list in [docs/limitations.md](docs/limitations.md); [docs/architecture.md](docs/architecture.md) covers the components.

## License

MIT. See [LICENSE](LICENSE).
