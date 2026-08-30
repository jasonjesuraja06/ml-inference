#!/usr/bin/env bash
# Start the FastAPI service, drive it with locust, then shut it down.
#
# Usage:
#   scripts/run_load_test.sh [users] [duration] [variant]
#
#   users     concurrent locust users            (default 50)
#   duration  locust -t value                    (default 60s)
#   variant   optimized | baseline               (default optimized)
#
# "optimized" serves the INT8 model with the LRU cache and dynamic batching on.
# "baseline" serves FP32 with both disabled, which is the comparison target.
# Results go to bench/reports/api_load_summary_<variant>.json and the served
# cache counters to bench/reports/api_stats_<variant>.json.
set -euo pipefail

cd "$(dirname "$0")/.."

USERS="${1:-50}"
DURATION="${2:-60s}"
VARIANT="${3:-optimized}"
PORT="${PORT:-8000}"
export PYTHONPATH="src:."

if [ "$VARIANT" = "baseline" ]; then
  export MODEL_VARIANT=fp32 NO_CACHE=1 NO_BATCHING=1
else
  export MODEL_VARIANT=quantized
fi

mkdir -p bench/reports
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for the service to come up on :$PORT"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/healthz" || { echo "service never became healthy"; exit 1; }
echo

echo "running locust: $USERS users, $DURATION, variant=$VARIANT"
API_URL="http://127.0.0.1:$PORT" .venv/bin/locust -f bench/locustfile.py --headless \
  -u "$USERS" -r 10 -t "$DURATION" \
  --html "bench/reports/api-load-$VARIANT.html"

curl -s "http://127.0.0.1:$PORT/stats" | tee "bench/reports/api_stats_$VARIANT.json"
echo
