"""Minimal smoke tests that don't require trained models."""
from __future__ import annotations

import json
import pathlib
import random

import pytest


def test_import_root():
    import ml_inference  # noqa: F401


def test_config_paths_exist():
    from ml_inference.config import DATA_SPLITS, MODELS_DIR, ONNX_DIR, REPORTS_DIR
    for d in (DATA_SPLITS, MODELS_DIR, ONNX_DIR, REPORTS_DIR):
        assert pathlib.Path(d).exists()


def test_focal_loss_smoke():
    import torch
    from ml_inference.loss import FocalLoss

    logits = torch.randn(8, 11)
    targets = torch.randint(0, 11, (8,))
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.item() > 0


def test_augment_does_not_explode():
    from ml_inference.augment import augment
    rng = random.Random(0)
    code = "int main() {\n  int x = 0;\n  return x;\n}"
    out = augment(code, rng)
    assert isinstance(out, str)
    assert "main" in out or "main_v" in out


def test_lru_cache():
    import numpy as np
    from api.inference import LRUEmbeddingCache
    c = LRUEmbeddingCache(capacity=2)
    k = LRUEmbeddingCache.key("hello")
    assert c.get(k) is None
    c.put(k, np.zeros(3))
    assert c.get(k) is not None
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1


@pytest.mark.skipif(
    not (pathlib.Path("data/splits/label_map.json").exists()),
    reason="splits not built yet; run `make download && make splits`",
)
def test_label_map_loadable():
    from ml_inference.data import load_label_map
    m = load_label_map()
    assert isinstance(m, dict) and m
