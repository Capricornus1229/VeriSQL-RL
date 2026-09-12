#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

readarray -t API_SETTINGS < <("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("app/configs/serve.yaml").read_text()) or {}
app = cfg.get("app", {})
print(app.get("host", "127.0.0.1"))
print(app.get("port", 8000))
PY
)
API_HOST="${API_HOST:-${API_SETTINGS[0]:-127.0.0.1}}"
API_PORT="${API_PORT:-${API_SETTINGS[1]:-8000}}"
API_RELOAD="${API_RELOAD:-0}"

ARGS=(app.backend.api:app --host "$API_HOST" --port "$API_PORT")
if [[ "$API_RELOAD" == "1" || "$API_RELOAD" == "true" ]]; then
  ARGS+=(--reload)
fi
exec "$PYTHON_BIN" -m uvicorn "${ARGS[@]}"
