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


def export_fp32(src: str | Path, out: str | Path) -> Path:
    """Export a sequence-classification checkpoint to an FP32 ONNX graph.

    `src` may be a local directory or a Hub id. Shared by the `make export-onnx`
    path and by arch_bench.py, so the graph benchmarked on a CI runner is
    produced by the same code as the one benchmarked here.
    """
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(str(src), export=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    return out


def main() -> None:
    src = IMPROVED.output_dir / "final"
    if not src.exists():
        raise SystemExit(f"missing improved model at {src}; run `make train-improved` first")
    out = ONNX_DIR / "improved-fp32"
    print(f"exporting {src} -> {out} (FP32 ONNX)")
    export_fp32(src, out)
    print("done")


if __name__ == "__main__":
    main()
