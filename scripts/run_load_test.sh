#!/usr/bin/env bash
# Start the FastAPI service in one configuration, drive it with locust, record
# the result, then shut it down.
#
# Usage:
#   scripts/run_load_test.sh <config> [users] [duration]
#   scripts/run_load_test.sh all      [users] [duration]
#
# Configurations. The point of having five rather than two is attribution: a
# single optimized-against-baseline number cannot say how much of the gain is
# the cache and how much is the batcher, so each is enabled on its own against
# a common reference.
#
#   fp32-plain         FP32,  no cache, no batching   model-choice reference
#   int8-plain         INT8,  no cache, no batching   serving reference
#   int8-cache         INT8,  cache,    no batching   cache alone
#   int8-batch         INT8,  no cache, batching      batching alone
#   int8-cache-batch   INT8,  cache,    batching      both, the served default
#
# Every run uses the same payload pool, the same seeded per-user request
# sequence (LOAD_SEED), the same concurrency, and the same duration. The runs
# are sequential against separate processes, so a comparison between two rows
# carries the run-to-run variance of both.
#
# Writes bench/reports/api_load_summary_<config>.json (locust side) and
# bench/reports/api_stats_<config>.json (the served process's own counters).
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-int8-cache-batch}"
USERS="${2:-50}"
DURATION="${3:-60s}"
PORT="${PORT:-8000}"
export PYTHONPATH="src:."
export CACHE_REPEAT_RATE="${CACHE_REPEAT_RATE:-0.30}"
export LOAD_SEED="${LOAD_SEED:-1729}"

ALL_CONFIGS="fp32-plain int8-plain int8-cache int8-batch int8-cache-batch"

if [ "$CONFIG" = "all" ]; then
  for c in $ALL_CONFIGS; do
    "$0" "$c" "$USERS" "$DURATION"
  done
  exit 0
fi

case "$CONFIG" in
  fp32-plain)       MODEL_VARIANT=fp32      NO_CACHE=1 NO_BATCHING=1 ;;
  int8-plain)       MODEL_VARIANT=quantized NO_CACHE=1 NO_BATCHING=1 ;;
  int8-cache)       MODEL_VARIANT=quantized NO_CACHE=0 NO_BATCHING=1 ;;
  int8-batch)       MODEL_VARIANT=quantized NO_CACHE=1 NO_BATCHING=0 ;;
  int8-cache-batch) MODEL_VARIANT=quantized NO_CACHE=0 NO_BATCHING=0 ;;
  *) echo "unknown config '$CONFIG'; one of: $ALL_CONFIGS all" >&2; exit 2 ;;
esac
export MODEL_VARIANT NO_CACHE NO_BATCHING
export LOAD_CONFIG="$CONFIG"

mkdir -p bench/reports
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for the service to come up on :$PORT"
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/healthz" || { echo "service never became healthy"; exit 1; }
echo

echo "running locust: config=$CONFIG, $USERS users, $DURATION, repeat rate $CACHE_REPEAT_RATE"
API_URL="http://127.0.0.1:$PORT" .venv/bin/locust -f bench/locustfile.py --headless \
  -u "$USERS" -r 10 -t "$DURATION" \
  --html "bench/reports/api-load-$CONFIG.html"

curl -s "http://127.0.0.1:$PORT/stats" > "bench/reports/api_stats_$CONFIG.json"
cat "bench/reports/api_stats_$CONFIG.json"
echo
