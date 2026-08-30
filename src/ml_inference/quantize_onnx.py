"""
INT8 dynamic quantization of the FP32 ONNX export.

The quantization config is chosen from the host's instruction-set flags, not
from its architecture name. Optimum's presets are feature-specific, and
"x86-64" is not one instruction set: `avx512_vnni` needs VNNI, which an Intel
Cascade Lake part has and an AMD Zen 3 part does not, while `arm64` targets
AArch64. Picking a preset for instructions the host lacks produces a graph
quantized for a machine that is not this one. Reading the flags rather than
`platform.machine()` is what keeps that from happening silently.

Override with QUANT_ARCH=<preset> to quantize for a different target; the
preset actually used is written into quantization.json.

Output: models/onnx/improved-int8/model_quantized.onnx + tokenizer files.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from ml_inference.config import ONNX_DIR
from ml_inference.hostinfo import cpu_flags


def quant_arch() -> str:
    """The Optimum preset matching this host, unless QUANT_ARCH overrides it.

    On x86-64 the preset follows the flags the OS reports, best first. The
    fallback when no flags are readable is `avx2`, which every x86-64 part
    ONNX Runtime ships wheels for has had since 2013; guessing `avx512_vnni`
    instead would be the optimistic direction, and the optimistic direction is
    the one that produces a graph the host cannot run well.
    """
    override = os.environ.get("QUANT_ARCH")
    if override:
        return override
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return "arm64"
    flags = cpu_flags()
    if "avx512_vnni" in flags:
        return "avx512_vnni"
    if "avx512f" in flags:
        return "avx512"
    return "avx2"


def quantize_dir(src: str | Path, out: str | Path, arch: str | None = None) -> dict:
    """Dynamically quantize an FP32 ONNX export to INT8 and return its metadata.

    Shared by the `make quantize` path and by arch_bench.py so the INT8 graph
    timed on a CI runner comes out of the same code as the one timed here.
    """
    src, out = Path(src), Path(out)
    arch = arch or quant_arch()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    q = ORTQuantizer.from_pretrained(str(src))
    qconfig = getattr(AutoQuantizationConfig, arch)(is_static=False, per_channel=True)
    q.quantize(save_dir=str(out), quantization_config=qconfig)
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))

    fp32_bytes = (src / "model.onnx").stat().st_size
    int8_bytes = (out / "model_quantized.onnx").stat().st_size
    meta = {
        "quant_preset": arch,
        "host_machine": platform.machine(),
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "size_ratio": round(fp32_bytes / int8_bytes, 2),
    }
    (out / "quantization.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    src = ONNX_DIR / "improved-fp32"
    if not src.exists():
        raise SystemExit(f"missing FP32 export at {src}; run `make export-onnx` first")
    out = ONNX_DIR / "improved-int8"
    arch = quant_arch()
    print(f"quantizing {src} -> {out} (INT8 dynamic, preset={arch}, host={platform.machine()})")
    meta = quantize_dir(src, out, arch)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
