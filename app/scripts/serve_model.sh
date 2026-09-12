#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p app/runtime

export CUDA_VISIBLE_DEVICES=0
VLLM_BIN="${VLLM_BIN:-vllm}"
exec "$VLLM_BIN" serve model/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules verisql-v2=pipelines/v2/runs/grpo/best_adapter \
  --generation-config vllm \
  --enable-prefix-caching \
  >> app/runtime/vllm.log 2>&1
