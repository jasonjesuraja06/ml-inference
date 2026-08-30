# Architecture

```
   DiverseVul (HF mirror)                CodeXGLUE Devign (HF)
            |                                     |
            v                                     v
   scripts/build_splits.py                 train_devign.py
   train / val / test / holdout / pool      (binary benchmark)
            |
            v
   train_baseline.py            train_improved.py
   (CodeBERT, plain CE)         (UniXcoder, focal + class weights
            |                    + augmentation + early stopping)
            |                              |
            |                              v
            |                    export_onnx.py -> quantize_onnx.py
            |                    (FP32 ONNX)      (INT8 dynamic)
            |                              |
            v                              v
   holdout macro F1            bench_inference.py
                               (latency + macro F1 per variant)
                                           |
                                           v
                               api/main.py (FastAPI)
                               Engine = ORT session + LRU cache + micro-batching
                                           |
                                           v
                               bench/locustfile.py (load test)

   active_learning_loop.py
        scores the pool -> auto_labeled / review / uncertain queues
        auto_labeled.parquet feeds the next train_improved.py run
```

## Datasets

**DiverseVul** (Chen et al., 2023) is the primary source for the multi-class
CWE task. Two properties of the mirror drive `build_splits.py`:

- It contains 330,492 functions, of which 18,945 carry `target == 1`. The
  dataset ships both the vulnerable and the patched version of each function
  touched by a fixing commit, and attaches the commit's CWE to both. The CWE is
  therefore a property of the commit, not of the function, and it is only a
  meaningful label on the `target == 1` rows. The split builder keeps those.
- The `cwe` field is an array, because one commit can cite several CWEs. About
  a quarter of rows carry more than one. The builder takes the first and drops
  the rest, so a multi-CWE function trains as a single-label example. This is a
  real simplification and a source of irreducible error on those rows.

After filtering to vulnerable rows with a parseable CWE and a function body
longer than 20 characters, 16,101 rows remain. They are split into the top-10
most frequent CWEs plus one `__OTHER__` bucket, which holds the long tail and
is the largest single class.

**CodeXGLUE Defect Detection (Devign)** (Zhou et al., 2019) is a binary
vulnerable/benign benchmark over 27,318 C functions from FFmpeg and QEMU. It is
used as an independently published second task.

`scripts/build_splits.py` writes `train`, `val`, `test`, `holdout`,
`unlabeled_pool` parquets, a `label_map.json`, and a `manifest.json` recording
the row counts and the training label distribution.

## Training

Both trainers share the tokenizer, dataset class, and class-weight helper in
`data.py`, and both accept the same environment overrides (`MODEL_NAME`,
`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_TRAIN_ROWS`, `MAX_EVAL_ROWS`,
`MAX_SEQ_LEN`). Every report they write embeds a `scope` block with the values
actually used, the device, and the wall clock, so a score is never separable
from the run that produced it.

Those reports land in `bench/reports/`, one file per script, which the next run
at a different scope overwrites. `scripts/collect_result.py` freezes a run into
`results/<name>/` before that happens, with the metrics JSON, the command, a
per-class table, a confusion matrix, and the stdout. `scripts/results_table.py`
renders what is in `results/` next to the published numbers transcribed in
`results/published_baselines.json` and computes nothing itself, so the README,
`docs/benchmarks.md`, and the committed artifacts cannot disagree.
`notebooks/train_gpu.ipynb` runs the same scripts unchanged on a Colab T4 for
anyone without a local GPU.

### Baseline (`train_baseline.py`)

`microsoft/codebert-base`, plain cross-entropy, no class weights, no warmup, no
weight decay, no augmentation, no early stopping. It is the control arm: it
shows what the task looks like when the class imbalance is left untreated.

### Improved (`train_improved.py`)

`microsoft/unixcoder-base` with:

- Class-weighted focal loss, gamma 2.0, with inverse-frequency weights taken
  from the training distribution
- Label smoothing 0.1
- Minority-class augmentation, applied to classes whose training count is below
  70 percent of the median per-class count
- LR warmup 0.06, weight decay 0.01, effective batch 32 via gradient
  accumulation
- Early stopping on validation macro F1, patience 2
- Auto-labeled rows from a prior active-learning run, when present

The two differ in backbone as well as in loss, so the gap between them is the
combined effect of every change, not an ablation of any single one.

## Quantization

`export_onnx.py` produces an FP32 ONNX graph through Optimum's
`ORTModelForSequenceClassification(export=True)`. `quantize_onnx.py` applies
dynamic INT8 quantization with `ORTQuantizer`.

The quantization preset is chosen from the CPU's instruction-set flags, not
from its architecture name, because Optimum's presets are feature-specific and
"x86-64" is not one instruction set. `avx512_vnni` needs the VNNI extension,
which an Intel Cascade Lake part has and an AMD Zen 3 part does not; `avx512`
and `avx2` are the fallbacks below it, and `arm64` targets AArch64.
`hostinfo.py` reads the flags the OS reports, and `quant_arch()` picks the best
preset those flags support. When no flags are readable it falls back to `avx2`
rather than up to `avx512_vnni`: guessing upward produces a graph quantized for
a machine that is not this one, and does it silently. `QUANT_ARCH` overrides
the detection when quantizing for a different target. The preset actually used
and the resulting file sizes are written to
`models/onnx/improved-int8/quantization.json`.

### Measuring quantization on more than one architecture

Whether dynamic INT8 is faster than FP32 is not a property of the model. It is
a property of the CPU running it and of the kernels ONNX Runtime ships for that
CPU, and this project's development host answers it one way while an x86-64
machine answers it another. One host cannot settle the question.

`bench_inference.py` cannot be moved to a second host: it needs the fine-tuned
checkpoint and the holdout split, and neither is in the repository.
`arch_bench.py` is the part that can. It exports and quantizes the same
architecture from the public base checkpoint, times FP32 against INT8 on
synthetic inputs, and needs no dataset, so the identical module runs on a
GitHub Actions runner. `.github/workflows/arch-bench.yml` runs it there and
uploads the report.

Two design choices make the two hosts comparable:

- **Intra-op threads are pinned** (`ARCH_BENCH_THREADS`, default 4). A 14-core
  laptop against a 4-vCPU runner would otherwise mix the effect of the
  instruction set with the effect of having three times the cores. Even pinned,
  absolute latency across the two hosts is not comparable: one is a dedicated
  performance core and the other a share of a virtualised server. The
  comparable quantity is the INT8-against-FP32 ratio within each host.
- **Every input is padded to `MAX_SEQ_LEN`**, so each forward pass runs the
  same operators over the same shapes whatever the weights hold. That is why an
  untrained classification head is sound for a latency measurement and useless
  for an accuracy one, and `arch_bench.py` reports latency only. The claim is
  checked rather than asserted: the same probe is run locally against the
  fine-tuned checkpoint as well, and both arm64 reports are committed.

The report is named after what the CPU can do rather than what it is called
(`arch_latency_x86-avx2_untrained.json`, `arch_latency_arm64-i8mm_trained.json`),
because a run that lands on an AMD part and a run that lands on an Intel part
are two measurements, not one.

## Serving

`api/inference.py` builds one `Engine` at application startup holding the ONNX
Runtime session, the tokenizer, the inverse label map, an `LRUEmbeddingCache`,
and an asyncio queue with a background batching task.

`/predict` flow:

1. Cache lookup, keyed by xxhash of the code string. A hit returns immediately
   with `cached: true`.
2. On a miss the request is parked on the batch queue. The batch loop collects
   requests for up to `BATCH_WINDOW_MS` (default 8) or until `BATCH_MAX`
   (default 16) accumulate, then runs one ORT forward pass.
3. Each waiting future receives its row of the logits, and the result is cached.

Caching logits is sound because the model is a deterministic function from code
to logits. The hit rate is only useful to the extent that real traffic repeats
snippets; the load test models this with a configurable repeat rate rather than
assuming it.

`NO_CACHE=1` and `NO_BATCHING=1` disable each path independently, which is what
makes `serve-baseline` a like-for-like comparison target.

`/predict/batch` bypasses the queue for up to 64 inputs in one call.

## Active learning

`active_learning_loop.py` scores the pool with the improved model and splits it
at two confidence thresholds into auto-labeled, human-review, and uncertain
queues. Because `build_splits.py` retains the pool's real labels in a
`true_label` column the loop never reads before predicting, the auto-labels can
be scored against ground truth afterwards. That check does not exist on a real
unlabeled pool, where the equivalent guardrail is a spot-check sample.

## Trade-offs

- **Dynamic INT8 rather than QAT.** Post-training dynamic quantization is cheap
  and needs no calibration data. Quantization-aware training would recover more
  accuracy at a large training cost.
- **CPU for all inference measurements.** ONNX Runtime has no MPS execution
  provider, so an ONNX graph cannot use the Apple GPU. Benchmarking the PyTorch
  model on MPS against ONNX on CPU would measure the accelerator rather than
  the quantization, so `bench_inference.py` puts every variant on CPU.
- **Backbone and technique changed together.** The improved config differs from
  the baseline in both, so the difference between them cannot be attributed to
  either alone.
- **Label noise.** DiverseVul labels come from CVE-fixing commits and carry
  attribution noise. Label smoothing and focal loss compensate partially.
