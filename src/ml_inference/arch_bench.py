"""
Cross-architecture INT8 latency probe: FP32 ONNX against INT8 ONNX, batch size
1, on whatever CPU it is run on.

Why this exists separately from bench_inference.py. Dynamic INT8 quantization
is an instruction-set bet: the win comes from integer dot-product instructions
(AVX-512 VNNI on x86-64, SDOT/SMMLA on AArch64) that a given CPU may or may not
have, and that a given ONNX Runtime build may or may not emit kernels for. On
this project's development host, an Apple M4 Pro, INT8 is slower than FP32. The
question that answers is "is dynamic INT8 worth it here", not "is dynamic INT8
worth it", and the second question needs a second architecture.

bench_inference.py cannot run on a second architecture, because it needs the
fine-tuned checkpoint and the holdout split, neither of which is in the
repository. This module needs neither: it builds the graph from a public base
checkpoint, times it against synthetic inputs, and so runs unchanged on a
GitHub Actions x86-64 runner. See .github/workflows/arch-bench.yml.

What that costs, stated plainly: with no fine-tuned weights it measures latency
only, never accuracy. That is sound because latency here does not depend on
weight values. Every input is padded to MAX_SEQ_LEN, so every forward pass
executes the same operators over the same tensor shapes whatever the weights
hold, and integer GEMM does not run faster on one bit pattern than another.
`ARCH_BENCH_CHECKPOINT=models/improved/final` runs the same probe against the
fine-tuned weights, which is how that claim is checked rather than asserted;
bench/reports/ carries both arm64 runs.

Environment:
  ARCH_BENCH_CHECKPOINT   checkpoint to export      (default: the base model)
  ARCH_BENCH_THREADS      ORT intra-op threads      (default 4)
  ARCH_BENCH_SAMPLES      timed inputs per variant  (default 100)
  ARCH_BENCH_TAG          report filename stem      (default: host ISA + weights)
  QUANT_ARCH              quantization preset       (default from host ISA)
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_inference.bench_inference import percentile
from ml_inference.config import MAX_SEQ_LEN, REPORTS_DIR, TOP_K_CWES, env_int, env_str
from ml_inference.export_onnx import export_fp32
from ml_inference.hostinfo import cpu_details, isa_tag
from ml_inference.quantize_onnx import quant_arch, quantize_dir

# The base checkpoint of the improved config. Exported with a randomly
# initialised classification head when no fine-tuned checkpoint is supplied:
# same operators, same shapes, same kernels, different numbers.
BASE_CHECKPOINT = "microsoft/unixcoder-base"

# One C function per shape of code the service sees. Content is irrelevant to
# latency because every input is padded to MAX_SEQ_LEN; these exist so the
# tokenizer is exercised on plausible source rather than on filler.
SYNTHETIC_SNIPPETS = (
    "int copy_name(char *dst, const char *src) {\n"
    "  size_t n = strlen(src);\n"
    "  memcpy(dst, src, n);\n"
    "  dst[n] = '\\0';\n"
    "  return (int)n;\n}",
    "static int parse_header(struct buf *b, uint32_t *out) {\n"
    "  if (b->len < 4) return -1;\n"
    "  *out = (b->data[0] << 24) | (b->data[1] << 16) | (b->data[2] << 8) | b->data[3];\n"
    "  b->len -= 4;\n"
    "  return 0;\n}",
    "void release(struct node *n) {\n"
    "  if (!n) return;\n"
    "  free(n->payload);\n"
    "  free(n);\n"
    "  n->payload = NULL;\n}",
    "int scale(int count, int width) {\n"
    "  int total = count * width;\n"
    "  if (total < 0) return -1;\n"
    "  return total;\n}",
)


def session_options(threads: int) -> ort.SessionOptions:
    """Pin intra-op parallelism so two hosts differ in ISA, not in core count.

    The default is the host's core count, which on a 14-core laptop against a
    4-vCPU CI runner would mix the effect of the instruction set with the
    effect of having three times the cores.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return so


def build_checkpoint(spec: str, workdir: Path) -> tuple[str, str]:
    """Resolve the checkpoint to export. Returns (path, weights_description)."""
    local = Path(spec)
    if local.exists():
        return str(local), "fine-tuned"
    n_labels = TOP_K_CWES + 1
    dest = workdir / "base-checkpoint"
    print(f"materialising {spec} with a {n_labels}-label head (untrained) -> {dest}")
    model = AutoModelForSequenceClassification.from_pretrained(spec, num_labels=n_labels)
    model.save_pretrained(str(dest))
    AutoTokenizer.from_pretrained(spec).save_pretrained(str(dest))
    return str(dest), "untrained"


def time_variant(model, tok, samples: list[str], warmup: int = 10) -> dict:
    """Single-input latency at batch size 1, inputs padded to MAX_SEQ_LEN."""
    def call(code: str) -> None:
        enc = tok(code, truncation=True, padding="max_length",
                  max_length=MAX_SEQ_LEN, return_tensors="pt")
        model(**dict(enc))

    for i in range(warmup):
        call(samples[i % len(samples)])
    lats = []
    for i in range(len(samples)):
        t0 = time.perf_counter()
        call(samples[i])
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


def main() -> None:
    threads = env_int("ARCH_BENCH_THREADS", 4)
    n_samples = env_int("ARCH_BENCH_SAMPLES", 100)
    spec = env_str("ARCH_BENCH_CHECKPOINT", BASE_CHECKPOINT)
    arch = quant_arch()
    cpu = cpu_details()

    with tempfile.TemporaryDirectory(prefix="arch-bench-") as tmp:
        workdir = Path(tmp)
        checkpoint, weights = build_checkpoint(spec, workdir)
        # Named after what the CPU can do, not what it is called, so an Intel
        # runner and an AMD runner cannot overwrite each other's report.
        tag = env_str("ARCH_BENCH_TAG", f"{isa_tag()}_{weights}")

        fp32_dir = workdir / "fp32"
        int8_dir = workdir / "int8"
        print(f"exporting {checkpoint} -> FP32 ONNX")
        export_fp32(checkpoint, fp32_dir)
        print(f"quantizing -> INT8 dynamic, preset={arch}")
        quant_meta = quantize_dir(fp32_dir, int8_dir, arch)

        so = session_options(threads)
        tok = AutoTokenizer.from_pretrained(str(fp32_dir))
        fp32 = ORTModelForSequenceClassification.from_pretrained(
            str(fp32_dir), session_options=so, provider="CPUExecutionProvider"
        )
        int8 = ORTModelForSequenceClassification.from_pretrained(
            str(int8_dir), file_name="model_quantized.onnx",
            session_options=so, provider="CPUExecutionProvider",
        )

        samples = [SYNTHETIC_SNIPPETS[i % len(SYNTHETIC_SNIPPETS)] for i in range(n_samples)]
        print(f"timing {n_samples} single inputs per variant, {threads} intra-op threads")
        fp32_lat = time_variant(fp32, tok, samples)
        int8_lat = time_variant(int8, tok, samples)

    report = {
        "scope": {
            "checkpoint": spec,
            "weights": weights,
            "latency_samples_per_variant": n_samples,
            "batch_size": 1,
            "max_seq_len": MAX_SEQ_LEN,
            "padding": "max_length",
            "intra_op_threads": threads,
            "execution_provider": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
            "python_version": sys.version.split()[0],
            "quant_preset": arch,
        },
        "cpu": cpu,
        "model_bytes": {
            "fp32": quant_meta["fp32_bytes"],
            "int8": quant_meta["int8_bytes"],
            "size_ratio": quant_meta["size_ratio"],
        },
        "onnx_fp32": fp32_lat,
        "onnx_int8": int8_lat,
        "int8_speedup_vs_fp32": {
            "p50": round(fp32_lat["p50_ms"] / int8_lat["p50_ms"], 3),
            "p95": round(fp32_lat["p95_ms"] / int8_lat["p95_ms"], 3),
            "p99": round(fp32_lat["p99_ms"] / int8_lat["p99_ms"], 3),
        },
    }
    dest = REPORTS_DIR / f"arch_latency_{tag}.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
