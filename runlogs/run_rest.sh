#!/bin/bash
cd /private/tmp/claude-501/-Users-jason-jesuraja-HeartScreen/fea08555-080d-4ab3-a746-1a45b32d76e5/scratchpad/mlx
export PYTHONPATH=src:.
export TOKENIZERS_PARALLELISM=false

# wait for the baseline run to exit
while pgrep -f "ml_inference.train_baseline" > /dev/null; do sleep 10; done
echo "=== baseline finished, starting improved at $(date -u +%FT%TZ) ==="

EPOCHS=3 NO_AUTO_LABELS=1 .venv/bin/python -m ml_inference.train_improved > runlogs/improved.log 2>&1
echo "=== improved exit=$? at $(date -u +%FT%TZ) ==="

EPOCHS=2 .venv/bin/python -m ml_inference.train_devign > runlogs/devign.log 2>&1
echo "=== devign exit=$? at $(date -u +%FT%TZ) ==="
