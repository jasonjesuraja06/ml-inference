"""FastAPI service for code vulnerability classification."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.inference import Engine, latency_ms
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = Engine()
    await _engine.start()
    yield
    await _engine.stop()


app = FastAPI(title="ml-inference", version="0.1.0", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    return HealthResponse(
        status="ok",
        model_variant=_engine.model_variant,
        cache_enabled=_engine.cache_enabled,
        batching_enabled=_engine.batching_enabled,
    )


@app.get("/stats")
async def stats() -> dict:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    return {
        "model_variant": _engine.model_variant,
        "cache": _engine.cache.stats() if _engine.cache_enabled else {"enabled": False},
        "batching_enabled": _engine.batching_enabled,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    t0 = time.perf_counter()
    logits, cached = await _engine.predict(req.code)
    out = _engine.humanize(logits, req.return_probs)
    return PredictResponse(
        **out,
        cached=cached,
        inference_ms=latency_ms(t0),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    t0 = time.perf_counter()
    logits = _engine.predict_batch_sync(req.codes)
    items = []
    for i in range(len(req.codes)):
        d = _engine.humanize(logits[i], req.return_probs)
        items.append(PredictResponse(**d, cached=False, inference_ms=0.0))
    return BatchPredictResponse(predictions=items, batch_ms=latency_ms(t0))
