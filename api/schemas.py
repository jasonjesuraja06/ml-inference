"""Request/response models for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20_000)
    return_probs: bool = False


class BatchPredictRequest(BaseModel):
    codes: list[str] = Field(..., min_length=1, max_length=64)
    return_probs: bool = False


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float] | None = None
    cached: bool = False
    # None on the batch endpoint, where only the whole-batch time is meaningful.
    inference_ms: float | None = None


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    batch_ms: float


class HealthResponse(BaseModel):
    status: str
    model_variant: str
    cache_enabled: bool
    batching_enabled: bool
