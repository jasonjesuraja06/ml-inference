#!/usr/bin/env python3
"""
Download DiverseVul + CodeXGLUE Defect Detection.

DiverseVul: 18K+ vulnerable functions, 295 CWEs, 2023.
  - HF Hub: claudios/DiverseVul (mirror) OR original release
  - Fallback URL: https://drive.google.com/.../diversevul.json (manual)

CodeXGLUE Defect Detection (Devign): 27K binary-labeled functions.
  - HF Hub: code_x_glue_cc_defect_detection
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ml_inference.config import DATA_RAW  # noqa: E402


def download_diversevul() -> pathlib.Path:
    out = DATA_RAW / "diversevul"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(f"datasets not installed: {e}")

    candidates = [
        "claudios/DiverseVul",
        "wagner-group/DiverseVul",
    ]
    last_err: Exception | None = None
    for repo in candidates:
        try:
            print(f"[diversevul] trying HF repo: {repo}")
            ds = load_dataset(repo)
            saved = out / "hf_dataset"
            ds.save_to_disk(str(saved))
            print(f"[diversevul] saved to {saved}")
            return saved
        except Exception as e:  # noqa: BLE001
            print(f"[diversevul] failed on {repo}: {e}")
            last_err = e
    raise SystemExit(
        f"could not download DiverseVul from any mirror; last error: {last_err}\n"
        "Manual fallback: download diversevul.json from the paper's release page and place at "
        f"{out / 'diversevul.json'}"
    )


def download_codexglue_devign() -> pathlib.Path:
    out = DATA_RAW / "codexglue_devign"
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    # The HF mirror of CodeXGLUE defect detection (Devign).
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    saved = out / "hf_dataset"
    ds.save_to_disk(str(saved))
    print(f"[codexglue-devign] saved to {saved}")
    return saved


def main() -> None:
    print("=== DiverseVul ===")
    download_diversevul()
    print("=== CodeXGLUE Defect Detection (Devign) ===")
    download_codexglue_devign()
    print("done")


if __name__ == "__main__":
    main()
