"""Tests for the pieces that run without a trained model or a downloaded dataset."""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch


def test_focal_loss_downweights_easy_examples():
    """Focal loss must penalise a confident-correct prediction less than plain CE."""
    from ml_inference.loss import FocalLoss

    targets = torch.tensor([0])
    confident = torch.tensor([[10.0, 0.0, 0.0]])
    focal = FocalLoss(gamma=2.0)(confident, targets)
    ce = torch.nn.functional.cross_entropy(confident, targets)
    assert focal.item() < ce.item()

    # On a confidently wrong prediction the focal term must not shrink the loss
    # below cross-entropy; that is the whole point of the (1 - pt) ** gamma factor.
    wrong = torch.tensor([[0.0, 10.0, 0.0]])
    assert FocalLoss(gamma=2.0)(wrong, targets).item() == pytest.approx(
        torch.nn.functional.cross_entropy(wrong, targets).item(), rel=0.05
    )


def test_focal_loss_alpha_weights_classes():
    """A larger alpha for the true class must produce a larger loss."""
    from ml_inference.loss import FocalLoss

    logits = torch.tensor([[0.5, 0.5, 0.5]])
    targets = torch.tensor([1])
    low = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0]), gamma=2.0)(logits, targets)
    high = FocalLoss(alpha=torch.tensor([1.0, 5.0, 1.0]), gamma=2.0)(logits, targets)
    assert high.item() == pytest.approx(5.0 * low.item(), rel=1e-4)


def test_class_weights_are_inverse_frequency():
    from ml_inference.data import class_weights

    df = pd.DataFrame({"label": [0] * 90 + [1] * 10})
    w = class_weights(df, 2)
    # Rare class gets the larger weight, in inverse proportion to its frequency.
    assert w[1] > w[0]
    assert (w[1] / w[0]).item() == pytest.approx(9.0, rel=1e-4)
    # Weighting each class by its frequency recovers an average weight of 1.
    assert (0.90 * w[0] + 0.10 * w[1]).item() == pytest.approx(1.0, rel=1e-4)


def test_class_weights_handle_absent_class():
    """A class with zero rows must not produce inf or nan weights."""
    from ml_inference.data import class_weights

    w = class_weights(pd.DataFrame({"label": [0, 0, 1]}), 4)
    assert torch.isfinite(w).all()


def test_augment_preserves_code_structure():
    """Augmentation may rename identifiers but must not drop or reorder braces."""
    from ml_inference.augment import augment

    code = "int main(int argc) {\n  int total = argc + 1;\n  return total;\n}"
    for seed in range(25):
        out = augment(code, random.Random(seed))
        assert out.count("{") == code.count("{")
        assert out.count("}") == code.count("}")
        # Keywords are never renamed.
        assert "int " in out and "return" in out


def test_augment_actually_changes_something():
    """Over many seeds the augmenter must produce at least one variant."""
    from ml_inference.augment import augment

    code = "int handler(char *buffer) {\n  int length = 0;\n  length = strlen(buffer);\n  return length;\n}"
    variants = {augment(code, random.Random(s)) for s in range(50)}
    assert len(variants) > 1


def test_augment_never_renames_keywords():
    from ml_inference.augment import rename_locals

    code = "for (int i = 0; i < n; i++) { if (i) continue; }"
    out = rename_locals(code, random.Random(3))
    for kw in ("for", "int", "if", "continue"):
        assert f"{kw}_v" not in out


def test_lru_cache_evicts_least_recently_used():
    from api.inference import LRUEmbeddingCache

    c = LRUEmbeddingCache(capacity=2)
    ka, kb, kc = (LRUEmbeddingCache.key(x) for x in ("a", "b", "c"))
    c.put(ka, np.zeros(3))
    c.put(kb, np.ones(3))
    c.get(ka)  # touch "a" so "b" becomes least-recently-used
    c.put(kc, np.full(3, 2.0))
    assert c.get(ka) is not None
    assert c.get(kb) is None, "least-recently-used entry should have been evicted"
    assert c.get(kc) is not None


def test_lru_cache_returns_stored_value_and_counts_stats():
    from api.inference import LRUEmbeddingCache

    c = LRUEmbeddingCache(capacity=4)
    k = LRUEmbeddingCache.key("void f(){}")
    assert c.get(k) is None
    c.put(k, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(c.get(k), np.array([1.0, 2.0, 3.0]))
    stats = c.stats()
    assert stats == {"hits": 1, "misses": 1, "hit_rate": 0.5}


def test_lru_cache_key_is_content_addressed():
    from api.inference import LRUEmbeddingCache

    assert LRUEmbeddingCache.key("int x;") == LRUEmbeddingCache.key("int x;")
    assert LRUEmbeddingCache.key("int x;") != LRUEmbeddingCache.key("int y;")


def test_subsample_is_deterministic_and_bounded():
    from ml_inference.data import subsample

    df = pd.DataFrame({"code": [str(i) for i in range(100)], "label": list(range(100))})
    a = subsample(df, 10)
    b = subsample(df, 10)
    assert len(a) == 10
    pd.testing.assert_frame_equal(a, b)
    # 0 and oversized limits both mean "everything".
    assert len(subsample(df, 0)) == 100
    assert len(subsample(df, 500)) == 100


def test_metrics_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    from ml_inference.metrics import compute_classification_metrics

    y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    y_pred = np.array([0, 2, 2, 2, 1, 0, 1, 1])
    m = compute_classification_metrics(y_true, y_pred)
    assert m["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert np.array(m["confusion_matrix"]).sum() == len(y_true)


def test_quant_arch_follows_host_and_override(monkeypatch):
    """The quantization preset must match the host ISA, not a hard-coded x86 one."""
    from ml_inference import quantize_onnx

    monkeypatch.delenv("QUANT_ARCH", raising=False)
    monkeypatch.setattr(quantize_onnx.platform, "machine", lambda: "arm64")
    assert quantize_onnx.quant_arch() == "arm64"
    monkeypatch.setattr(quantize_onnx.platform, "machine", lambda: "x86_64")
    assert quantize_onnx.quant_arch() == "avx512_vnni"
    monkeypatch.setenv("QUANT_ARCH", "arm64")
    assert quantize_onnx.quant_arch() == "arm64"


def test_scoped_config_reads_environment(monkeypatch):
    from ml_inference.config import BASELINE, scoped_config

    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("MAX_TRAIN_ROWS", "512")
    monkeypatch.setenv("MODEL_NAME", "some/other-checkpoint")
    cfg = scoped_config(BASELINE)
    assert (cfg.epochs, cfg.max_train_rows, cfg.model_name) == (1, 512, "some/other-checkpoint")
    # The module-level config object must not be mutated.
    assert BASELINE.epochs == 2 and BASELINE.max_train_rows == 0


def test_first_cwe_normalisation():
    """The CWE field is an array; benign rows carry an empty one and must be dropped."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from build_splits import _first_cwe

    assert _first_cwe(np.array(["CWE-125", "CWE-787"])) == "CWE-125"
    assert _first_cwe(np.array([], dtype=object)) is None
    assert _first_cwe([]) is None
    assert _first_cwe(None) is None
    assert _first_cwe("787") == "CWE-787"
    assert _first_cwe("cwe-125") == "CWE-125"


def test_auto_labeled_round_trip(tmp_path, monkeypatch):
    """Auto-labeled rows written by the active-learning loop must load back for training."""
    from ml_inference import data as data_mod

    monkeypatch.setattr(data_mod, "DATA_SPLITS", tmp_path)
    assert data_mod.load_auto_labeled() is None

    pd.DataFrame({"code": ["int a;", "int b;"], "label": [1, 2], "confidence": [0.99, 0.95]}).to_parquet(
        tmp_path / "auto_labeled.parquet", index=False
    )
    loaded = data_mod.load_auto_labeled()
    assert list(loaded.columns) == ["code", "label"]
    assert len(loaded) == 2


def test_predict_request_accepts_a_long_real_function():
    """Real DiverseVul functions run past 200k characters; the cap must clear them."""
    from api.schemas import MAX_CODE_CHARS, PredictRequest

    assert MAX_CODE_CHARS > 226_800
    PredictRequest(code="int x;\n" * 30_000)
    with pytest.raises(__import__("pydantic").ValidationError):
        PredictRequest(code="x" * (MAX_CODE_CHARS + 1))


def test_predict_request_rejects_empty_code():
    from pydantic import ValidationError

    from api.schemas import BatchPredictRequest, PredictRequest

    PredictRequest(code="int x;")
    with pytest.raises(ValidationError):
        PredictRequest(code="")
    with pytest.raises(ValidationError):
        BatchPredictRequest(codes=[])
    with pytest.raises(ValidationError):
        BatchPredictRequest(codes=["x"] * 65)


def test_percentile_is_nearest_rank():
    from ml_inference.bench_inference import percentile

    lats = [float(i) for i in range(1, 101)]
    assert percentile(lats, 0.50) == 50.0
    assert percentile(lats, 0.95) == 95.0
    assert percentile(lats, 0.99) == 99.0
    # Must not index past the end of a short sample.
    assert percentile([1.0, 2.0], 0.99) == 2.0


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[1] / "data/splits/label_map.json").exists(),
    reason="splits not built; run `make download && make splits`",
)
def test_label_map_matches_configured_class_count():
    from ml_inference.config import DATA_SPLITS, TOP_K_CWES
    from ml_inference.data import load_label_map, n_labels

    m = load_label_map()
    assert len(m) == TOP_K_CWES + 1, "top-K CWEs plus one 'other' bucket"
    assert "__OTHER__" in m
    assert sorted(m.values()) == list(range(len(m))), "label ids must be dense and 0-based"
    assert n_labels() == len(m)
    manifest = json.loads((DATA_SPLITS / "manifest.json").read_text())
    assert manifest["train"] > 0 and manifest["holdout"] > 0
