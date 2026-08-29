# Limitations

## What the measured numbers do and do not cover

Every number in the README comes from a run on one machine, an Apple M4 Pro
with 14 cores and 48 GB of RAM running macOS on arm64. Each was produced at a
reduced scope chosen to fit a fixed time budget, and each report in
`bench/reports/` carries a `scope` block recording exactly what that run used.
The reduced scope is the main caveat on the accuracy numbers: they say what
these configurations reach after a short run on a subset, not what they would
reach if trained to convergence on the full split.

The latency numbers are less scope-sensitive, but they are single-host, batch
size 1, CPU-only, and measured with a warm process. They do not include network
time, and they are not a claim about throughput under concurrency; the load
test covers that separately.

## Task limitations

- **The CWE label is commit-level.** DiverseVul attaches a fixing commit's CWE
  to every function the commit touched. Only the `target == 1` rows are used
  here, but even those carry the attribution noise of the original mining.
- **Multi-CWE functions are truncated to one label.** About a quarter of rows
  cite more than one CWE; the split builder keeps the first. Those rows have an
  irreducible error floor under a single-label objective.
- **The long tail collapses.** Only the top-10 CWEs are distinct classes;
  everything else becomes `__OTHER__`, which is the largest class in the split.
- **C and C++ only.** DiverseVul covers no other language.
- **Whole-function classification.** The model does not localize the defect to
  a line or an expression.
- **Truncation.** Functions are cut at `MAX_SEQ_LEN` tokens, so evidence past
  that point is invisible to the model.

## Modeling limitations

- The improved configuration changes the backbone and the training technique at
  once, so the difference between the two configurations cannot be attributed
  to either alone.
- Only post-training dynamic INT8 quantization is implemented. No QAT, no
  distillation, no confidence calibration.
- Confidence scores are raw softmax outputs and are not calibrated. The
  active-learning thresholds are therefore chosen, not derived.

## Serving limitations

- No authentication, authorization, rate limiting, or quota on any endpoint.
- Single global model; no multi-tenancy or per-tenant fine-tunes.
- Synchronous HTTP only; no batch ingest path.
- No drift monitoring on input distribution or per-class confidence.
- The LRU cache is per process. Running uvicorn with multiple workers gives
  each worker its own cache, so the hit rate reported by `/stats` is that
  worker's, not the service's.

## Labeling limitations

- The active-learning loop writes parquet queues. Connecting them to a labeling
  tool is integration work that is not done here.
- This project has not measured human labeling time, so it reports triage rates
  rather than hours saved. See `labeling_runbook.md`.
- No inter-annotator agreement tracking.
