"""
Locust load test for the FastAPI inference service.

Runs concurrent users hitting /predict with payloads drawn from the holdout
parquet. A configurable fraction of requests re-use a recent payload to
exercise the LRU embedding cache (mirrors real scanner traffic where the same
snippet often appears repeatedly in a scan).

Usage:
  scripts/run_load_test.sh 50 60s optimized
  CACHE_REPEAT_RATE=0.4 scripts/run_load_test.sh 50 60s optimized

CACHE_REPEAT_RATE is the fraction of requests that replay a payload this user
has already sent. It is a knob, not a measurement of real scanner traffic; the
cache hit rate the service reports is only as representative as the rate set
here.
"""
from __future__ import annotations

import os
import pathlib
import random

import pandas as pd
from locust import HttpUser, between, events, task

from ml_inference.config import DATA_SPLITS

CACHE_REPEAT_RATE = float(os.environ.get("CACHE_REPEAT_RATE", "0.30"))


_codes: list[str] = []


@events.test_start.add_listener
def _load_payloads(environment, **kwargs):
    global _codes
    holdout_path = DATA_SPLITS / "holdout.parquet"
    if not holdout_path.exists():
        # Fallback: any test split
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

    @task(10)
    def predict(self) -> None:
        if not _codes:
            return
        if self._recent and random.random() < CACHE_REPEAT_RATE:
            code = random.choice(self._recent)
        else:
            code = random.choice(_codes)
            self._recent.append(code)
            if len(self._recent) > 256:
                self._recent.pop(0)
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
            "users": getattr(environment.parsed_options, "num_users", None),
            "cache_repeat_rate": CACHE_REPEAT_RATE,
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
    import json
    # Name the report after the service configuration so a baseline run cannot
    # overwrite the optimized run's numbers.
    variant = "baseline" if os.environ.get("NO_CACHE") == "1" else "optimized"
    dest = reports_dir / f"api_load_summary_{variant}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")
