# Architecture

```
   datasets (DiverseVul, CodeXGLUE Devign)
                 |
                 v
   build_splits.py  ->  train / val / test / 10K holdout / unlabeled pool
                 |
                 v
   train_baseline.py        train_improved.py        train_devign.py
                 |
                 v
   export_onnx.py  ->  quantize_onnx.py  (INT8 dynamic, AVX-512 VNNI)
                 |
                 v
   bench_inference.py  (FP32 vs INT8 latency + accuracy)
                 |
                 v
   api/main.py (FastAPI)
        Engine = ORT session + LRUEmbeddingCache + dynamic micro-batching
                 |
                 v
   bench/locustfile.py  (sustained load, configurable cache-repeat rate)

   active_learning_loop.py
        scores unlabeled pool -> auto_labeled / review / uncertain queues
```

## Datasets

- **DiverseVul** (Wagner et al., 2023) — ~18K vulnerable C/C++ functions across 295 CWE classes. The training pipeline restricts to the top-10 most-frequent CWEs plus an "other" bucket, which makes the per-class sample counts large enough to learn meaningful per-class boundaries.
- **CodeXGLUE Defect Detection (Devign)** (Zhou et al., 2019) — ~27K binary-labeled (vulnerable/benign) functions from FFmpeg and Qemu. Used here as a recognized comparison benchmark.

`scripts/build_splits.py` produces:
- `train.parquet`, `val.parquet`, `test.parquet` — 80/10/10 split of the labeled remainder after the holdout and pool are removed.
- `holdout_10k.parquet` — 10,000-row evaluation holdout, not touched by training.
- `unlabeled_pool.parquet` — ~10% of the remainder, with `label = -1`; the true labels are retained in a separate `true_label` column for active-learning evaluation.
- `label_map.json` — string-CWE-id ↔ integer-label-id mapping; included in the API container.

## Training

Two training scripts share a tokenizer, dataset class, and class-weight helper.

### Baseline (`train_baseline.py`)

- Backbone: `microsoft/codebert-base`
- 2 epochs, batch size 16, LR 5e-5, no warmup, no weight decay
- Standard cross-entropy loss, no class weights
- No early stopping; the final checkpoint is the last epoch
- Final evaluation on `holdout_10k.parquet`

### Improved (`train_improved.py`)

- Backbone: `microsoft/unixcoder-base`
- 6 epochs (with early stopping), effective batch 32 via grad accumulation
- Class-weighted focal loss (gamma 2.0) — inverse-frequency weights from the training distribution
- Label smoothing 0.1
- Minority-class augmentation via identifier renaming and benign line duplication (see `augment.py`); only applied to classes whose training count is below 70% of the median per-class count
- LR warmup ratio 0.06, weight decay 0.01
- Early stopping on validation macro F1 with patience 2
- Final evaluation on `holdout_10k.parquet`

### CodeXGLUE Defect Detection (`train_devign.py`)

CodeBERT fine-tuned on the Devign binary-classification task. 3 epochs, LR 2e-5, warmup 0.06, weight decay 0.01. Reports accuracy and binary F1 on the official test split.

## Quantization

`export_onnx.py` uses Optimum's `ORTModelForSequenceClassification.from_pretrained(export=True)` to produce a FP32 ONNX graph. `quantize_onnx.py` then runs Optimum's `ORTQuantizer` with an AVX-512 VNNI dynamic-quantization config (`is_static=False, per_channel=True`). The output `model_quantized.onnx` is what the FastAPI service loads in production mode.

## Serving

`api/inference.py` constructs a single `Engine` on application startup, holding:

- The ONNX Runtime session (`ORTModelForSequenceClassification`).
- The tokenizer.
- A `label_map.json`-derived inverse map for humanizing predictions.
- An `LRUEmbeddingCache` (xxhash-keyed; configurable capacity).
- An asyncio queue and background task implementing dynamic micro-batching.

`/predict` flow:
1. Cache lookup (xxhash of code → cached logits). Cache hit returns immediately.
2. On miss, the request waits in the batch queue. The batch loop collects requests for up to 8ms (configurable) or until the batch reaches `BATCH_MAX` (default 16), then runs a single ORT forward pass.
3. The resulting logits are returned to each waiting future and inserted into the cache.

`/predict/batch` bypasses the queue and runs a direct batched forward pass on up to 64 inputs.

## Active learning

`active_learning_loop.py` loads the improved model, scores every row in `unlabeled_pool.parquet`, and partitions:
- `confidence >= 0.92` → auto-labeled, written to `data/splits/auto_labeled.parquet` for inclusion in the next `train-improved` run.
- `0.45 <= confidence < 0.92` → human-review queue.
- `confidence < 0.45` → uncertain queue (high information gain; these are prioritized for manual labeling).

Because the pool retains true labels, the script also reports the agreement rate of auto-labels against ground truth — a guardrail against silent quality regressions.

## Trade-offs

- **Post-training dynamic INT8 vs QAT.** Dynamic quantization is cheap and recovers most of the speedup with a small accuracy hit. Quantization-aware training (QAT) would close the small accuracy gap further but at significant training cost; out of scope here.
- **CPU inference assumption.** All latency numbers in the README assume CPU inference. GPU inference would change the relative ranking of FP32 vs INT8 (INT8 wins less on GPU because of overhead).
- **One backbone per variant.** A more thorough study would ablate (backbone × loss × augmentation) and isolate which change contributes which fraction of the macro-F1 lift. For a reference implementation we batch them together and report the combined effect.
- **DiverseVul label noise.** The dataset is mined from real-world CVE fixes and has some label noise. The `label_smoothing` and focal-loss settings partially compensate; a production deployment would invest in re-labeling the noisiest CWE classes.
