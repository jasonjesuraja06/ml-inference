"""Central config: filesystem paths, task constants, and training configurations.

Two training configurations are defined. BASELINE is a plain cross-entropy
fine-tune with no imbalance handling. IMPROVED adds class-weighted focal loss,
label smoothing, minority-class augmentation, warmup, and early stopping. They
exist so the effect of those techniques can be measured against a fixed split;
neither is tuned toward a particular score.

Every field below can be overridden from the environment so a run can be scoped
down to fit a given machine or time budget. See `scoped_config`.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field, replace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_SPLITS = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
REPORTS_DIR = REPO_ROOT / "bench" / "reports"

for d in (DATA_RAW, DATA_PROCESSED, DATA_SPLITS, ONNX_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TOP_K_CWES = 10  # multi-class task: top-10 most-frequent CWEs + "other"
HOLDOUT_FRACTION = 0.15  # share of CWE-labeled rows reserved as the evaluation holdout
SEED = 1729


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Token budget per function. Lower it to trade truncation for speed.
MAX_SEQ_LEN = env_int("MAX_SEQ_LEN", 256)


@dataclass
class TrainConfig:
    name: str
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    use_class_weights: bool
    use_focal_loss: bool
    use_augmentation: bool
    label_smoothing: float
    early_stopping_patience: int
    output_dir: pathlib.Path = field(default=MODELS_DIR)
    max_train_rows: int = 0  # 0 means "use the whole train split"
    max_eval_rows: int = 0  # 0 means "use the whole holdout"


# Plain cross-entropy fine-tune. No class weights, no warmup, no weight decay,
# no augmentation, no early stopping. Class imbalance is left untreated.
BASELINE = TrainConfig(
    name="codebert-baseline",
    model_name="microsoft/codebert-base",
    epochs=2,
    batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.0,
    warmup_ratio=0.0,
    grad_accum_steps=1,
    use_class_weights=False,
    use_focal_loss=False,
    use_augmentation=False,
    label_smoothing=0.0,
    early_stopping_patience=0,
    output_dir=MODELS_DIR / "baseline",
)

# Class-weighted focal loss, label smoothing, minority-class augmentation,
# LR warmup, weight decay, and early stopping on validation macro F1.
IMPROVED = TrainConfig(
    name="unixcoder-improved",
    model_name="microsoft/unixcoder-base",
    epochs=6,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    grad_accum_steps=2,
    use_class_weights=True,
    use_focal_loss=True,
    use_augmentation=True,
    label_smoothing=0.1,
    early_stopping_patience=2,
    output_dir=MODELS_DIR / "improved",
)


def scoped_config(cfg: TrainConfig) -> TrainConfig:
    """Apply environment overrides to a training configuration.

    This is how a run is reduced to fit a time budget without editing source.
    Recognised variables: MODEL_NAME, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MAX_TRAIN_ROWS, MAX_EVAL_ROWS. The values actually used are written into
    every metrics report so a reported score always carries its own scope.
    """
    return replace(
        cfg,
        model_name=env_str("MODEL_NAME", cfg.model_name),
        epochs=env_int("EPOCHS", cfg.epochs),
        batch_size=env_int("BATCH_SIZE", cfg.batch_size),
        learning_rate=env_float("LEARNING_RATE", cfg.learning_rate),
        max_train_rows=env_int("MAX_TRAIN_ROWS", cfg.max_train_rows),
        max_eval_rows=env_int("MAX_EVAL_ROWS", cfg.max_eval_rows),
    )


def run_scope(cfg: TrainConfig, n_train: int, n_eval: int) -> dict:
    """The provenance block attached to every metrics report."""
    import platform

    return {
        "config_name": cfg.name,
        "model_name": cfg.model_name,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "learning_rate": cfg.learning_rate,
        "max_seq_len": MAX_SEQ_LEN,
        "train_rows": int(n_train),
        "eval_rows": int(n_eval),
        "device": device(),
        "host": f"{platform.system()} {platform.machine()}",
        "seed": SEED,
    }


def device() -> str:
    """Prefer MPS on Apple Silicon, else CUDA, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
