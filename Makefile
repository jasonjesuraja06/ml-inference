UV := uv
VENV := .venv
PYV := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
export PYTHONPATH := src:.

.PHONY: help install download splits \
	train-baseline train-improved train-devign \
	export-onnx quantize bench-inference bench-api \
	serve serve-baseline active-learn test lint clean

help:
	@echo "Targets, in the order a full reproduction runs them:"
	@echo "  install          - create .venv and install dependencies with uv"
	@echo "  download         - fetch DiverseVul + CodeXGLUE Defect Detection from the HF Hub"
	@echo "  splits           - build train/val/test/holdout/pool parquets (top-10 CWE)"
	@echo "  train-baseline   - CodeBERT, plain cross-entropy, no imbalance handling"
	@echo "  train-improved   - UniXcoder, class-weighted focal loss, augmentation, early stopping"
	@echo "  train-devign     - CodeBERT on CodeXGLUE Defect Detection (binary)"
	@echo "  export-onnx      - export the improved model to FP32 ONNX"
	@echo "  quantize         - INT8 dynamic quantization, preset chosen from host ISA"
	@echo "  bench-inference  - FP32 vs INT8 single-input latency and holdout macro F1"
	@echo "  serve            - FastAPI on :8000 with the INT8 model, cache, and batching"
	@echo "  serve-baseline   - FastAPI on :8001 with FP32, no cache, no batching"
	@echo "  bench-api        - locust load test against a running service"
	@echo "  active-learn     - one active-learning iteration over the unlabeled pool"
	@echo "  test             - pytest"
	@echo "  lint             - ruff"
	@echo "  clean            - remove venv, models, and reports"
	@echo ""
	@echo "Scope any training run down with e.g. MAX_TRAIN_ROWS=4000 EPOCHS=2 make train-baseline"

install:
	$(UV) venv --python 3.12 $(VENV)
	$(UV) pip install --python $(PYV) -e ".[dev]"

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
	MODEL_VARIANT=quantized $(UVICORN) api.main:app --host 127.0.0.1 --port 8000

serve-baseline:
	MODEL_VARIANT=fp32 NO_BATCHING=1 NO_CACHE=1 $(UVICORN) api.main:app --host 127.0.0.1 --port 8001

# Override users/run time from the environment, e.g. LOAD_USERS=100 LOAD_TIME=5m make bench-api
LOAD_USERS ?= 50
LOAD_TIME ?= 60s
bench-api:
	$(VENV)/bin/locust -f bench/locustfile.py --headless \
		-u $(LOAD_USERS) -r 10 -t $(LOAD_TIME) \
		--html bench/reports/api-load.html

active-learn:
	$(PYV) -m ml_inference.active_learning_loop

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check src api tests scripts bench

clean:
	rm -rf $(VENV) models/baseline models/improved models/devign models/onnx bench/reports/*.json bench/reports/*.html
