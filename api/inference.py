"""
Inference engine: ONNX Runtime session + LRU embedding cache + dynamic
micro-batching.

Set via environment, read once at process start. Every one of these is
reported by GET /stats, so the values a measurement ran under can be read off
the service rather than inferred from the source:

  MODEL_VARIANT     quantized | fp32   which ONNX model directory to load
  NO_CACHE=1                           disables the LRU logit cache
  NO_BATCHING=1                        disables micro-batching on /predict
  CACHE_CAPACITY    entries            LRU capacity          (default 8192)
  BATCH_WINDOW_MS   milliseconds       micro-batch window    (default 8)
  BATCH_MAX         requests           micro-batch ceiling   (default 16)

The defaults are the values every published serving number was measured at.
The window is 8 ms because it has to be short next to the roughly 30 ms an
uncached forward pass takes: a window long enough to fill a batch of 16 under
light traffic would add more queueing delay than the batch saves. The 8192
entry cache is roughly half the 16,101-row split, so nothing evicts during a
load test and the reported hit rate is the repeat rate of the traffic rather
than an artifact of capacity.

Disabling the cache and the batcher independently is what makes the load test
an attribution rather than one blended before-and-after number.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xxhash
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from ml_inference.config import MAX_SEQ_LEN, ONNX_DIR

DEFAULT_CACHE_CAPACITY = 8192
DEFAULT_BATCH_WINDOW_MS = 8.0
DEFAULT_BATCH_MAX = 16


def _model_dir() -> Path:
    variant = os.environ.get("MODEL_VARIANT", "quantized").lower()
    if variant == "quantized":
        return ONNX_DIR / "improved-int8"
    return ONNX_DIR / "improved-fp32"


class LRUEmbeddingCache:
    """LRU cache of logits keyed by xxhash of the code string.

    The model is a deterministic function from code to logits, so caching the
    logits is sound. Whether the cache earns its place depends entirely on how
    often real traffic repeats a snippet, which this project does not measure;
    the load test drives a configurable repeat rate so the effect can be seen
    at a rate you choose rather than assumed.
    """

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY):
        self.capacity = capacity
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(code: str) -> str:
        return xxhash.xxh64(code.encode("utf-8")).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        if key in self._d:
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key: str, logits: np.ndarray) -> None:
        self._d[key] = logits
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": (self.hits / total) if total else 0.0}


class Engine:
    def __init__(self) -> None:
        model_dir = _model_dir()
        if not model_dir.exists():
            raise SystemExit(
                f"missing model at {model_dir}; run `make export-onnx` and `make quantize` first"
            )
        file_name = "model_quantized.onnx" if model_dir.name.endswith("int8") else None
        kwargs = {"file_name": file_name} if file_name else {}
        self.model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), **kwargs)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))

        # label_map.json lives in data/splits; required to humanize predictions.
        from ml_inference.config import DATA_SPLITS
        with (DATA_SPLITS / "label_map.json").open() as fh:
            label_map = json.load(fh)
        self.inv_labels = {int(v): k for k, v in label_map.items()}

        self.cache_enabled = os.environ.get("NO_CACHE", "0") != "1"
        self.batching_enabled = os.environ.get("NO_BATCHING", "0") != "1"
        self.cache_capacity = int(os.environ.get("CACHE_CAPACITY", DEFAULT_CACHE_CAPACITY))
        self.cache = LRUEmbeddingCache(self.cache_capacity) if self.cache_enabled else None

        # Dynamic batching queue
        self._batch_queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self.batch_window_ms = float(os.environ.get("BATCH_WINDOW_MS", DEFAULT_BATCH_WINDOW_MS))
        self._batch_window_s = self.batch_window_ms / 1000.0
        self.batch_max = int(os.environ.get("BATCH_MAX", DEFAULT_BATCH_MAX))
        self._batch_task: asyncio.Task | None = None
        self.model_variant = os.environ.get("MODEL_VARIANT", "quantized")
        self.batches_run = 0
        self.batched_requests = 0

    async def start(self) -> None:
        if self.batching_enabled:
            self._batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        if self._batch_task:
            self._batch_task.cancel()

    def _predict_logits(self, codes: list[str]) -> np.ndarray:
        enc = self.tok(codes, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        out = self.model(**{k: v for k, v in enc.items()})
        return out.logits.numpy()

    async def _batch_loop(self) -> None:
        while True:
            first_code, first_fut = await self._batch_queue.get()
            batch_codes = [first_code]
            futs = [first_fut]
            deadline = asyncio.get_event_loop().time() + self._batch_window_s
            while len(batch_codes) < self.batch_max:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    c, f = await asyncio.wait_for(self._batch_queue.get(), timeout=timeout)
                except TimeoutError:
                    break
                batch_codes.append(c)
                futs.append(f)
            self.batches_run += 1
            self.batched_requests += len(batch_codes)
            try:
                logits = self._predict_logits(batch_codes)
            except Exception as e:  # noqa: BLE001
                for f in futs:
                    if not f.done():
                        f.set_exception(e)
                continue
            for i, f in enumerate(futs):
                if not f.done():
                    f.set_result(logits[i])

    async def predict(self, code: str) -> tuple[np.ndarray, bool]:
        """Returns (logits, cached_flag)."""
        if self.cache_enabled:
            key = LRUEmbeddingCache.key(code)
            hit = self.cache.get(key)
            if hit is not None:
                return hit, True

        if self.batching_enabled:
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            await self._batch_queue.put((code, fut))
            logits = await fut
        else:
            logits = self._predict_logits([code])[0]

        if self.cache_enabled:
            self.cache.put(LRUEmbeddingCache.key(code), logits)
        return logits, False

    def predict_batch_sync(self, codes: list[str]) -> np.ndarray:
        """Direct batch path, skips queue. Used by BatchPredict endpoint."""
        return self._predict_logits(codes)

    def humanize(self, logits: np.ndarray, return_probs: bool):
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        label_id = int(p.argmax())
        out = {
            "label": self.inv_labels[label_id],
            "label_id": label_id,
            "confidence": float(p[label_id]),
        }
        if return_probs:
            out["probabilities"] = {self.inv_labels[i]: float(pi) for i, pi in enumerate(p)}
        return out


def stats_payload(engine) -> dict:
    """The GET /stats body.

    A free function so the shape can be tested without loading a model: the
    contract is that every knob a measurement depended on is readable back off
    the running service, and that contract should not need 500 MB of ONNX to
    check.
    """
    cache: dict[str, object] = {"enabled": engine.cache_enabled}
    if engine.cache_enabled:
        cache["capacity"] = engine.cache_capacity
        cache.update(engine.cache.stats())
    batching: dict[str, object] = {"enabled": engine.batching_enabled}
    if engine.batching_enabled:
        batching["window_ms"] = engine.batch_window_ms
        batching["max_batch"] = engine.batch_max
        batching["batches_run"] = engine.batches_run
        batching["requests_batched"] = engine.batched_requests
        batching["mean_batch_size"] = (
            round(engine.batched_requests / engine.batches_run, 2) if engine.batches_run else 0.0
        )
    return {
        "model_variant": engine.model_variant,
        "max_seq_len": MAX_SEQ_LEN,
        "cache": cache,
        "batching": batching,
    }


def latency_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
