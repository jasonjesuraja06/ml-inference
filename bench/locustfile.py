"""
Locust load test for the FastAPI inference service.

Concurrent users hit /predict with payloads drawn from the holdout parquet. A
configurable fraction of requests replay a payload the same user has already
sent, which is what exercises the LRU logit cache.

Each user draws from its own `random.Random(LOAD_SEED + user_index)`, so user k
issues the same sequence of decisions in every run. Concurrency still means two
runs do not interleave identically and a slower configuration gets through
fewer requests, so a run is a prefix of the same per-user sequence rather than
a byte-for-byte replay. That is what makes the five configurations comparable
without claiming they saw identical traffic.

Usage:
  scripts/run_load_test.sh int8-cache-batch 50 60s
  CACHE_REPEAT_RATE=0.4 scripts/run_load_test.sh int8-cache 50 60s

CACHE_REPEAT_RATE is a knob, not a measurement of real scanner traffic. The
cache hit rate the service reports is only as representative as the rate set
here, and the report records both the requested rate and the realised one.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import random

import pandas as pd
from locust import HttpUser, between, events, task

from ml_inference.config import DATA_SPLITS

CACHE_REPEAT_RATE = float(os.environ.get("CACHE_REPEAT_RATE", "0.30"))
LOAD_SEED = int(os.environ.get("LOAD_SEED", "1729"))
LOAD_CONFIG = os.environ.get("LOAD_CONFIG", "unnamed")

# Payloads a user remembers and may replay. Bounded so the replay set stays
# small enough to actually repeat within a 60 s run.
RECENT_WINDOW = 256

_codes: list[str] = []
_user_counter = itertools.count()
_replayed = 0
_sent = 0


@events.test_start.add_listener
def _load_payloads(environment, **kwargs):
    global _codes
    holdout_path = DATA_SPLITS / "holdout.parquet"
    if not holdout_path.exists():
        holdout_path = DATA_SPLITS / "test.parquet"
    df = pd.read_parquet(holdout_path)
    _codes = df["code"].dropna().astype(str).tolist()
    if not _codes:
        raise SystemExit(f"no payloads available at {holdout_path}; run `make splits` first")


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = os.environ.get("API_URL", "http://localhost:8000")

    def on_start(self) -> None:
        self._recent: list[str] = []
        # Per-user stream: user k makes the same choices in every configuration.
        self._rng = random.Random(LOAD_SEED + next(_user_counter))

    @task(10)
    def predict(self) -> None:
        global _replayed, _sent
        if not _codes:
            return
        if self._recent and self._rng.random() < CACHE_REPEAT_RATE:
            code = self._rng.choice(self._recent)
            _replayed += 1
        else:
            code = self._rng.choice(_codes)
            self._recent.append(code)
            if len(self._recent) > RECENT_WINDOW:
                self._recent.pop(0)
        _sent += 1
        self.client.post(
            "/predict",
            json={"code": code, "return_probs": False},
            name="POST /predict",
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/healthz", name="GET /healthz")


@events.quitting.add_listener
def _on_quit(environment, **kwargs):
    s = environment.stats.total
    reports_dir = pathlib.Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    predict = environment.stats.get("POST /predict", "POST")
    out = {
        "scope": {
            "config": LOAD_CONFIG,
            "users": getattr(environment.parsed_options, "num_users", None),
            "cache_repeat_rate_requested": CACHE_REPEAT_RATE,
            "cache_repeat_rate_realised": round(_replayed / _sent, 4) if _sent else None,
            "load_seed": LOAD_SEED,
            "duration_seconds": round(s.last_request_timestamp - s.start_time, 1)
            if s.last_request_timestamp else None,
            "host": getattr(environment.parsed_options, "host", None) or InferenceUser.host,
            "model_variant": os.environ.get("MODEL_VARIANT", "quantized"),
            "cache_enabled": os.environ.get("NO_CACHE") != "1",
            "batching_enabled": os.environ.get("NO_BATCHING") != "1",
        },
        "total_requests": s.num_requests,
        "failures": s.num_failures,
        "rps": round(s.total_rps, 2),
        "all_endpoints": {
            "p50_ms": s.get_response_time_percentile(0.5),
            "p95_ms": s.get_response_time_percentile(0.95),
            "p99_ms": s.get_response_time_percentile(0.99),
        },
        "predict_endpoint": {
            "requests": predict.num_requests,
            "failures": predict.num_failures,
            "p50_ms": predict.get_response_time_percentile(0.5),
            "p95_ms": predict.get_response_time_percentile(0.95),
            "p99_ms": predict.get_response_time_percentile(0.99),
            "rps": round(predict.total_rps, 2),
        },
    }
    # Named after the service configuration, so one run cannot overwrite another.
    dest = reports_dir / f"api_load_summary_{LOAD_CONFIG}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")
