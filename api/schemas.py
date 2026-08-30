"""Request/response models for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field

# A request-size guard, not a modeling constraint: the tokenizer truncates to
# MAX_SEQ_LEN tokens regardless. The previous 20,000-character cap rejected
# 2.4 percent of real DiverseVul functions with a 422, which showed up as a
# 2.3 percent failure rate under load. Longest function in the holdout split
# is 226,800 characters.
MAX_CODE_CHARS = 262_144


class PredictRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=MAX_CODE_CHARS)
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
