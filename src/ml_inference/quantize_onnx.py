"""
INT8 dynamic quantization of the FP32 ONNX export.

The quantization config is chosen from the host architecture, because Optimum's
presets are instruction-set specific: `avx512_vnni` targets x86-64 with VNNI,
and `arm64` targets AArch64. Picking the wrong one silently produces a graph
tuned for instructions the host does not have. Override with QUANT_ARCH=arm64
or QUANT_ARCH=avx512_vnni to quantize for a machine other than this one.

Output: models/onnx/improved-int8/model_quantized.onnx + tokenizer files.
"""
from __future__ import annotations

import json
import os
import platform
import shutil

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from ml_inference.config import ONNX_DIR


def quant_arch() -> str:
    """The Optimum preset matching this host, unless QUANT_ARCH overrides it."""
    override = os.environ.get("QUANT_ARCH")
    if override:
        return override
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "avx512_vnni"


def main() -> None:
    src = ONNX_DIR / "improved-fp32"
    if not src.exists():
        raise SystemExit(f"missing FP32 export at {src}; run `make export-onnx` first")
    out = ONNX_DIR / "improved-int8"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    arch = quant_arch()
    print(f"quantizing {src} -> {out} (INT8 dynamic, preset={arch}, host={platform.machine()})")
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
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
