"""
Locust load test for the FastAPI inference service.

Runs concurrent users hitting /predict with payloads drawn from the holdout
parquet. A configurable fraction of requests re-use a recent payload to
exercise the LRU embedding cache (mirrors real scanner traffic where the same
snippet often appears repeatedly in a scan).

Usage:
  make bench-api          # 50 users, 5/s ramp, 1h sustained, HTML report
  CACHE_REPEAT_RATE=0.4 locust -f bench/locustfile.py ...
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
    holdout_path = DATA_SPLITS / "holdout_10k.parquet"
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
    out = {
        "total_requests": s.num_requests,
        "failures": s.num_failures,
        "p50_ms": s.get_response_time_percentile(0.5),
        "p95_ms": s.get_response_time_percentile(0.95),
        "p99_ms": s.get_response_time_percentile(0.99),
        "rps": s.total_rps,
        "extrapolated_daily_capacity": int((s.total_rps or 0) * 86400),
    }
    import json
    (reports_dir / "api_load_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
