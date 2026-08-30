# Limitations

## What the measured numbers do and do not cover

Every number in the README comes from a run on one machine, an Apple M4 Pro
with 14 cores and 48 GB of RAM running macOS on arm64. Each was produced at a
reduced scope chosen to fit a fixed time budget, and each report in
`bench/reports/` carries a `scope` block recording exactly what that run used.
The reduced scope is the main caveat on the accuracy numbers: they say what
these configurations reach after a short run on a subset, not what they would
reach if trained to convergence on the full split.

Latency percentiles on this host carry more run-to-run variance than the
effects being measured: two runs of `make bench-inference` disagree on
whether INT8 is faster than FP32 at all. That is why the cross-architecture
table pins the thread count and reports medians over repeats, and why no
single-run latency difference under about 15% should be read as an effect.

The latency numbers are less scope-sensitive, but they are batch size 1,
CPU-only, and measured with a warm process. They do not include network time,
and they are not a claim about throughput under concurrency; the load test
covers that separately.

The cross-architecture INT8 table is the one set of numbers not from this host,
and it carries its own limits:

- **It measures latency, never accuracy.** The CI runs export an untrained
  classification head, which is sound for timing and worthless for scoring.
  Nothing in that table says anything about macro F1.
- **Absolute latency is not comparable across its rows.** A dedicated laptop
  performance core and a share of a virtualised CI server differ in clock,
  memory bandwidth, and what else is resident. Only the INT8-against-FP32 ratio
  within a row is a like-for-like comparison.
- **One allocation of a CPU model is one sample.** The EPYC 7763 and both
  Apple M4 Pro rows are medians of more than one run; the others are not. The Intel rows are single runs, so their third
  digit is not meaningful; the ordering between CPU families is much larger
  than the spread within the repeated one.
- **It does not explain the arm64 result, only records it.** These runs cannot
  separate ONNX Runtime's AArch64 kernels from the `arm64` quantization preset
  as the reason INT8 is not faster there. Doing that needs a kernel-level
  profile this repository does not contain.
- **`ubuntu-latest` is not a fixed machine.** Re-running the workflow will land
  on a different mix of CPUs and produce a different set of rows.

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
- The load-test configurations run sequentially against separate processes on
  the same machine as the client. Per-user request streams are seeded, so the
  five configurations see the same sequence of payload decisions, but a slower
  configuration gets through fewer of them and concurrency means two runs never
  interleave identically. A difference between two rows carries the run-to-run
  variance of both, and the attribution between cache and batching is only as
  stable as that.
- The cache hit rate is a property of the synthetic repeat rate, which is set
  to 0.30. It is a knob, not a measurement of how often real scanner traffic
  repeats a function, and every throughput number that depends on the cache
  inherits that assumption.

## Labeling limitations

- The active-learning loop writes parquet queues. Connecting them to a labeling
  tool is integration work that is not done here.
- This project has not measured human labeling time, so it reports triage rates
  rather than hours saved. See `labeling_runbook.md`.
- No inter-annotator agreement tracking.
