PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYV := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn

.PHONY: help venv install download splits \
	train-baseline train-improved train-devign \
	export-onnx quantize bench-inference bench-api \
	serve serve-baseline active-learn test lint clean

help:
	@echo "Targets (run in order for a full reproduction):"
	@echo "  venv             - create local virtualenv"
	@echo "  install          - install deps into venv"
	@echo "  download         - download DiverseVul + CodeXGLUE Defect Detection"
	@echo "  splits           - build train/val/test/holdout splits (top-10 CWE)"
	@echo "  train-baseline   - CodeBERT, weak config -> F1 ~0.72"
	@echo "  train-improved   - UniXcoder + class weights + active learning -> F1 ~0.84"
	@echo "  train-devign     - benchmark on CodeXGLUE Defect Detection (binary vuln)"
	@echo "  export-onnx      - export improved model to ONNX (FP32)"
	@echo "  quantize         - quantize ONNX to INT8 dynamic"
	@echo "  bench-inference  - latency before/after quantization + accuracy delta on 10K holdout"
	@echo "  serve            - run FastAPI with quantized model + embedding cache"
	@echo "  serve-baseline   - run FastAPI with FP32 model (no cache, no batching) for comparison"
	@echo "  bench-api        - locust 10K/day sustained load, P95 target <150ms"
	@echo "  active-learn     - run one active-learning iteration on the unlabeled pool"
	@echo "  test             - pytest"
	@echo "  lint             - ruff"
	@echo "  clean            - remove venv, models, reports"

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

download:
	$(PYV) scripts/download_data.py

splits:
	$(PYV) scripts/build_splits.py

train-baseline:
	$(PYV) -m ml_inference.train_baseline

train-improved:
	$(PYV) -m ml_inference.train_improved

train-devign:
	$(PYV) -m ml_inference.train_devign

export-onnx:
	$(PYV) -m ml_inference.export_onnx

quantize:
	$(PYV) -m ml_inference.quantize_onnx

bench-inference:
	$(PYV) -m ml_inference.bench_inference

serve:
	MODEL_VARIANT=quantized $(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 2

serve-baseline:
	MODEL_VARIANT=fp32 NO_BATCHING=1 NO_CACHE=1 $(UVICORN) api.main:app --host 0.0.0.0 --port 8001 --workers 2

bench-api:
	$(VENV)/bin/locust -f bench/locustfile.py --headless -u 50 -r 5 -t 1h --html bench/reports/api-load.html

active-learn:
	$(PYV) -m ml_inference.active_learning_loop

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src api tests scripts

clean:
	rm -rf $(VENV) models/baseline/* models/improved/* models/onnx/* bench/reports/*
