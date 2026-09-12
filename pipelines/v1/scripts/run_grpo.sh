#!/usr/bin/env bash

set -euo pipefail

V1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$V1_ROOT/../.." && pwd)"
CONFIG_DIR="$V1_ROOT/configs"
CONFIG_PATH="$CONFIG_DIR/grpo.yaml"
RUN_NAME="qwen3_8b_bird_grpo"
MODEL_PATH="$PROJECT_ROOT/model/Qwen3-8B"
ADAPTER_PATH="$V1_ROOT/runs/grpo/model/final_adapter"
DEV_OUTPUT_DIR="$V1_ROOT/runs/grpo/dev"
PREDICTIONS_PATH="$DEV_OUTPUT_DIR/predictions.jsonl"
SCORED_RESULTS_PATH="$DEV_OUTPUT_DIR/scored_results.jsonl"
METRICS_PATH="$DEV_OUTPUT_DIR/metrics.json"

usage() {
  echo "Usage: bash pipelines/v1/scripts/run_grpo.sh {train|dev} [--resume]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

STAGE="$1"
shift

RESUME=false
if [[ $# -eq 1 ]]; then
  if [[ "$1" != "--resume" ]]; then
    usage
    exit 2
  fi
  RESUME=true
fi

case "$STAGE" in
  train|dev)
    ;;
  *)
    usage
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1

if [[ "$STAGE" == "train" ]]; then
  TRAIN_ARGS=(--config "$CONFIG_PATH")
  if [[ "$RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume)
  fi

  accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m pipelines.v1.src.train_grpo \
    "${TRAIN_ARGS[@]}"
  exit 0
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "GRPO final Adapter does not exist: $ADAPTER_PATH" >&2
  echo "Run 'bash pipelines/v1/scripts/run_grpo.sh train' first." >&2
  exit 1
fi

if [[ "$RESUME" == true && -f "$PREDICTIONS_PATH" ]]; then
  echo "Dev predictions are already complete: $PREDICTIONS_PATH"
else
  GENERATION_ARGS=(
    --model-path "$MODEL_PATH"
    --adapter-path "$ADAPTER_PATH"
    --tokenizer-path "$MODEL_PATH"
    --run-name "$RUN_NAME"
    --output-dir "$DEV_OUTPUT_DIR"
    --batch-size 1
    --max-new-tokens 2048
    --devices cuda:0 cuda:1
  )
  if [[ "$RESUME" == true ]]; then
    GENERATION_ARGS+=(--resume)
  fi
  python -m pipelines.v1.src.generate_dev "${GENERATION_ARGS[@]}"
fi

if [[ "$RESUME" == true && -f "$SCORED_RESULTS_PATH" && -f "$METRICS_PATH" ]]; then
  echo "Dev evaluation is already complete: $METRICS_PATH"
else
  python -m src.evaluation.bird_evaluation \
    --predictions-path "$PREDICTIONS_PATH" \
    --run-name "$RUN_NAME" \
    --output-dir "$DEV_OUTPUT_DIR" \
    --timeout-seconds 60
fi

python -m pipelines.v1.src.train_grpo \
  --config "$CONFIG_PATH" \
  --finalize-dev
