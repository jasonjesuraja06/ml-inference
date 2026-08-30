#!/bin/bash
cd /private/tmp/claude-501/-Users-jason-jesuraja-HeartScreen/fea08555-080d-4ab3-a746-1a45b32d76e5/scratchpad/mlx
export PYTHONPATH=src:.
export TOKENIZERS_PARALLELISM=false
EPOCHS=3 .venv/bin/python -m ml_inference.train_baseline
