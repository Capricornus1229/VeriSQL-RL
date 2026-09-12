#!/usr/bin/env bash

set -euo pipefail

V1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$V1_ROOT/../.." && pwd)"
CONFIG_DIR="$V1_ROOT/configs"
CONFIG_PATH="$CONFIG_DIR/sft.yaml"

usage() {
  echo "Usage: bash pipelines/v1/scripts/run_sft.sh {train|resume <checkpoint_path>}" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
shift

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1

launch_sft_phase() {
  local phase="$1"
  shift
  accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m pipelines.v1.src.train_sft \
    --config "$CONFIG_PATH" \
    --mode "$phase" \
    "$@"
}

case "$MODE" in
  train)
    if [[ $# -ne 0 ]]; then
      usage
      exit 2
    fi
    echo "Running automatic SFT preflight..."
    launch_sft_phase preflight
    echo "Running automatic SFT smoke test..."
    launch_sft_phase smoke
    echo "Starting formal SFT training from the base model..."
    launch_sft_phase train
    ;;
  resume)
    if [[ $# -ne 1 ]]; then
      usage
      exit 2
    fi
    launch_sft_phase train --resume-from "$1"
    ;;
  *)
    usage
    exit 2
    ;;
esac
