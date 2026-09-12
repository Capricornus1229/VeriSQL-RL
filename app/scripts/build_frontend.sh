#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT/app/frontend"

if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build
