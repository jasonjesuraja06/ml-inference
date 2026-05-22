# Scope and Roadmap

## Current scope

A reproducible reference implementation: dataset preparation, two training variants on the top-10 CWE multi-class task, a CodeXGLUE Devign binary benchmark, ONNX export with INT8 dynamic quantization, a FastAPI inference service with caching and dynamic batching, a load test, and an active-learning loop.

All benchmarks run on commodity hardware (Apple Silicon, mid-range x86 with AVX-512 VNNI for the best INT8 results, GPU optional for training).

## Not yet in scope

### Datasets and tasks
- **Multi-language support.** DiverseVul is C/C++; this project does not handle Python, JavaScript, Java, or other languages.
- **Fine-grained CWE classification.** Only the top-10 most-frequent CWE classes are treated as distinct; the long tail collapses to "other."
- **Vulnerability *localization*.** The model classifies whole functions; it does not point at the offending line.

### Modeling
- **Quantization-aware training (QAT).** Only post-training dynamic INT8 is included. QAT would narrow the small accuracy gap further at higher training cost.
- **Distilled student models.** Knowledge distillation to a smaller backbone could trade more accuracy for additional speedup.
- **Ensembles and calibration.** Per-class calibration of confidence scores would tighten the active-learning thresholds.

### Production serving
- **Authentication / authorization.** No auth on any endpoint.
- **Rate limiting / quota.** Open service.
- **Multi-tenancy.** Single global model; no per-tenant fine-tunes.
- **Streaming / batched ingest.** Synchronous HTTP only. A real deployment would also accept S3/SQS-style batch jobs.
- **Drift detection.** No monitoring of input distribution drift or per-class confidence drift.

### Labeling
- **Labeling UI.** The active-learning loop writes parquet queues; pairing them with a Label Studio or doccano workflow is left as integration work.
- **Inter-annotator agreement tracking.** Out of scope for the reference; an integration would need a labeler-rotation plan and a Krippendorff-alpha or Cohen-kappa report.

## Known sharp edges

- **AVX-512 VNNI dependency for INT8 wins.** On hardware without VNNI, dynamic INT8 still works but the speedup is smaller. Apple Silicon's NEON path through ONNX Runtime gives similar relative wins.
- **DiverseVul label noise.** Labels are mined from CVE-fixing commits; some functions are mis-attributed. Label smoothing and focal loss compensate partially but not fully.
- **Active learning seed-dependence.** The initial improved-model quality determines the auto-label bucket quality. Bootstrap with a high-quality improved model before trusting auto-labels.
- **Memory pressure during quantization.** Optimum's quantization pass loads the FP32 model fully; large models may need a host with sufficient RAM.

## Contributing

Issues and pull requests welcome. Changes that alter the benchmark numbers in the README should include a reproduction recipe and updated reference report.
