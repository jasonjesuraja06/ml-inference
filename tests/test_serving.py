"""Tests for the serving surface: the /stats contract and the host probe.

These run without a trained model or a downloaded dataset, which is the point:
the guarantee that every knob a published measurement depended on is readable
back off the running service should be checkable in CI, where no 500 MB ONNX
graph exists.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _engine(**over):
    from api.inference import (
        DEFAULT_BATCH_MAX,
        DEFAULT_BATCH_WINDOW_MS,
        DEFAULT_CACHE_CAPACITY,
        LRUEmbeddingCache,
    )

    defaults = dict(
        model_variant="quantized",
        cache_enabled=True,
        cache_capacity=DEFAULT_CACHE_CAPACITY,
        cache=LRUEmbeddingCache(DEFAULT_CACHE_CAPACITY),
        batching_enabled=True,
        batch_window_ms=DEFAULT_BATCH_WINDOW_MS,
        batch_max=DEFAULT_BATCH_MAX,
        batches_run=0,
        batched_requests=0,
    )
    defaults.update(over)
    return SimpleNamespace(**defaults)


def test_stats_reports_every_knob_a_measurement_depended_on():
    """README documents the window and the capacity; /stats must show them."""
    from api.inference import stats_payload

    s = stats_payload(_engine())
    assert s["model_variant"] == "quantized"
    assert s["cache"]["enabled"] is True
    assert s["cache"]["capacity"] == 8192
    assert s["cache"]["hit_rate"] == 0.0
    assert s["batching"]["enabled"] is True
    assert s["batching"]["window_ms"] == 8.0
    assert s["batching"]["max_batch"] == 16


def test_stats_reports_cache_hit_rate_and_batch_occupancy():
    from api.inference import LRUEmbeddingCache, stats_payload

    cache = LRUEmbeddingCache(4)
    k = LRUEmbeddingCache.key("int x;")
    cache.get(k)
    cache.put(k, np.zeros(3))
    cache.get(k)
    s = stats_payload(_engine(cache=cache, batches_run=4, batched_requests=26))
    assert s["cache"]["hits"] == 1 and s["cache"]["misses"] == 1
    assert s["cache"]["hit_rate"] == 0.5
    assert s["batching"]["mean_batch_size"] == 6.5


def test_stats_marks_disabled_paths_without_inventing_counters():
    """The baseline configurations must not report a hit rate they never had."""
    from api.inference import stats_payload

    s = stats_payload(_engine(cache_enabled=False, cache=None, batching_enabled=False))
    assert s["cache"] == {"enabled": False}
    assert s["batching"] == {"enabled": False}


def test_isa_tag_names_the_int8_capability_not_the_architecture(monkeypatch):
    from ml_inference import hostinfo

    monkeypatch.setattr(hostinfo.platform, "machine", lambda: "x86_64")
    for flags, expected in [
        ({"avx2", "avx512f", "avx512_vnni"}, "x86-avx512_vnni"),
        ({"avx2", "avx512f"}, "x86-avx512"),
        ({"avx2"}, "x86-avx2"),
        (set(), "x86_64-unknown"),
    ]:
        monkeypatch.setattr(hostinfo, "cpu_flags", lambda f=flags: f)
        assert hostinfo.isa_tag() == expected


def test_cpu_details_always_carries_the_fields_a_report_quotes():
    from ml_inference.hostinfo import cpu_details

    d = cpu_details()
    assert set(d) >= {"machine", "system", "logical_cpus", "model", "int8_features"}
    assert isinstance(d["int8_features"], dict)
