"""
INT8 dynamic quantization of the FP32 ONNX export.

Output: models/onnx/improved-int8/model_quantized.onnx + tokenizer files.
The quantized model is the artifact used by the FastAPI service and is what
the inference benchmark (`bench_inference.py`) measures for edge/mobile-style
single-input latency.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from ml_inference.config import ONNX_DIR


def main() -> None:
    src = ONNX_DIR / "improved-fp32"
    if not src.exists():
        raise SystemExit(f"missing FP32 export at {src}; run `make export-onnx` first")
    out = ONNX_DIR / "improved-int8"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"quantizing {src} -> {out} (INT8 dynamic)")
    q = ORTQuantizer.from_pretrained(str(src))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    q.quantize(save_dir=str(out), quantization_config=qconfig)
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    print("done")


if __name__ == "__main__":
    main()
