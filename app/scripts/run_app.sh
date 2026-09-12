#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 {build|model|api|dev-ui|all}" >&2
}

wait_for_model() {
  local url="${VLLM_HEALTH_URL:-http://127.0.0.1:8001/v1/models}"
  local timeout="${VLLM_WAIT_SECONDS:-180}"
  local started=$SECONDS
  until curl --noproxy '*' --silent --show-error --fail "$url" 2>/dev/null \
    | grep -q 'verisql-v2'; do
    if (( SECONDS - started >= timeout )); then
      echo "Timed out waiting for vLLM at $url" >&2
      return 1
    fi
    sleep 2
  done
}

case "${1:-}" in
  build) exec bash app/scripts/build_frontend.sh ;;
  model) exec bash app/scripts/serve_model.sh ;;
  api) exec bash app/scripts/serve_api.sh ;;
  dev-ui)
    cd app/frontend
    exec npm run dev
    ;;
  all)
    bash app/scripts/build_frontend.sh
    mkdir -p app/runtime
    bash app/scripts/serve_model.sh &
    MODEL_PID=$!
    cleanup() {
      if kill -0 "$MODEL_PID" 2>/dev/null; then
        kill "$MODEL_PID" 2>/dev/null || true
        wait "$MODEL_PID" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT INT TERM
    wait_for_model
    bash app/scripts/serve_api.sh
    ;;
  *) usage; exit 2 ;;
esac
