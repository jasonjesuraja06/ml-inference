"""
Export the IMPROVED model to ONNX (FP32). This is the pre-quantization artifact.

Output: models/onnx/improved-fp32/model.onnx + tokenizer files
"""
from __future__ import annotations

import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from ml_inference.config import IMPROVED, ONNX_DIR


def main() -> None:
    src = IMPROVED.output_dir / "final"
    if not src.exists():
        raise SystemExit(f"missing improved model at {src}; run `make train-improved` first")
    out = ONNX_DIR / "improved-fp32"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"exporting {src} -> {out} (FP32 ONNX)")
    model = ORTModelForSequenceClassification.from_pretrained(str(src), export=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    print("done")


if __name__ == "__main__":
    main()
