#!/usr/bin/env python3
"""
Download DiverseVul and CodeXGLUE Defect Detection from the Hugging Face Hub.

DiverseVul is fetched from the community mirror `claudios/DiverseVul`, which
holds 330,492 rows of which 18,945 are vulnerable functions. The original
release is distributed as a request-gated download rather than from the Hub;
if you have that file, place it at data/raw/diversevul/diversevul.json as
newline-delimited JSON and build_splits.py will read it instead.

CodeXGLUE Defect Detection (Devign) is fetched from
`google/code_x_glue_cc_defect_detection`: 21,854 train, 2,732 validation, and
2,732 test rows of binary-labeled C functions.

Total download is roughly 200 MB and expands to about 700 MB on disk.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ml_inference.config import DATA_RAW  # noqa: E402


def download_diversevul() -> pathlib.Path:
    out = DATA_RAW / "diversevul"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(f"datasets not installed: {e}") from e

    candidates = [c for c in os.environ.get("DIVERSEVUL_REPOS", "claudios/DiverseVul").split(",") if c]
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
        f"could not download DiverseVul from any of {candidates}; last error: {last_err}\n"
        "Manual fallback: place newline-delimited JSON at "
        f"{out / 'diversevul.json'} and run `make splits`."
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
