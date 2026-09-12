#!/usr/bin/env bash

set -euo pipefail

V1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$V1_ROOT/../.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/model/Qwen3-8B"

ZERO_SHOT_RUN_NAME="qwen3_8b_zero_shot"
ZERO_SHOT_OUTPUT_DIR="$V1_ROOT/runs/zero_shot"
SFT_RUN_NAME="qwen3_8b_bird_sft"
SFT_ADAPTER_PATH="$V1_ROOT/runs/sft/model/best_adapter"
SFT_OUTPUT_DIR="$V1_ROOT/runs/sft/dev"

usage() {
  cat <<'EOF'
Usage:
  bash pipelines/v1/scripts/run_pipeline.sh prepare
  bash pipelines/v1/scripts/run_pipeline.sh zero-shot [--resume]
  bash pipelines/v1/scripts/run_pipeline.sh sft [--resume-from CHECKPOINT]
  bash pipelines/v1/scripts/run_pipeline.sh grpo [--resume]
  bash pipelines/v1/scripts/run_pipeline.sh dev [--resume]
EOF
}

require_no_arguments() {
  if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
  fi
}

parse_optional_resume() {
  if [[ $# -eq 0 ]]; then
    printf '%s\n' false
  elif [[ $# -eq 1 && "$1" == "--resume" ]]; then
    printf '%s\n' true
  else
    usage >&2
    exit 2
  fi
}

generate_and_evaluate() {
  local run_name="$1"
  local output_dir="$2"
  local adapter_path="$3"
  local resume="$4"
  local predictions_path="$output_dir/predictions.jsonl"
  local scored_results_path="$output_dir/scored_results.jsonl"
  local metrics_path="$output_dir/metrics.json"
  local generation_args=(
    --model-path "$MODEL_PATH"
    --tokenizer-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-dir "$output_dir"
    --batch-size 1
    --max-new-tokens 2048
    --devices cuda:0 cuda:1
  )

  if [[ -n "$adapter_path" ]]; then
    if [[ ! -d "$adapter_path" ]]; then
      echo "Adapter does not exist: $adapter_path" >&2
      exit 1
    fi
    generation_args+=(--adapter-path "$adapter_path")
  fi
  if [[ "$resume" == true ]]; then
    generation_args+=(--resume)
  fi

  if [[ "$resume" == true && -f "$predictions_path" ]]; then
    echo "Dev predictions are already complete: $predictions_path"
  else
    python -m pipelines.v1.src.generate_dev "${generation_args[@]}"
  fi

  if [[ "$resume" == true && -f "$scored_results_path" && -f "$metrics_path" ]]; then
    echo "Dev evaluation is already complete: $metrics_path"
  else
    python -m src.evaluation.bird_evaluation \
      --predictions-path "$predictions_path" \
      --run-name "$run_name" \
      --output-dir "$output_dir" \
      --timeout-seconds 60
  fi
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
  fi
  usage
  exit 0
fi

STAGE="$1"
shift

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1

case "$STAGE" in
  prepare)
    require_no_arguments "$@"
    python -m src.data.build_schema_catalog
    python -m src.execution.sql_executor --timeout-seconds 30
    python -m pipelines.v1.src.build_sft_dataset
    python -m pipelines.v1.src.audit_sft_tokens \
      --tokenizer-path "$MODEL_PATH"
    ;;
  zero-shot)
    RESUME="$(parse_optional_resume "$@")"
    generate_and_evaluate \
      "$ZERO_SHOT_RUN_NAME" \
      "$ZERO_SHOT_OUTPUT_DIR" \
      "" \
      "$RESUME"
    ;;
  sft)
    if [[ $# -eq 0 ]]; then
      bash "$V1_ROOT/scripts/run_sft.sh" train
    elif [[ $# -eq 2 && "$1" == "--resume-from" ]]; then
      bash "$V1_ROOT/scripts/run_sft.sh" resume "$2"
    else
      usage >&2
      exit 2
    fi
    generate_and_evaluate \
      "$SFT_RUN_NAME" \
      "$SFT_OUTPUT_DIR" \
      "$SFT_ADAPTER_PATH" \
      false
    ;;
  grpo)
    RESUME="$(parse_optional_resume "$@")"
    if [[ "$RESUME" == true ]]; then
      bash "$V1_ROOT/scripts/run_grpo.sh" train --resume
    else
      bash "$V1_ROOT/scripts/run_grpo.sh" train
    fi
    ;;
  dev)
    RESUME="$(parse_optional_resume "$@")"
    if [[ "$RESUME" == true ]]; then
      bash "$V1_ROOT/scripts/run_grpo.sh" dev --resume
    else
      bash "$V1_ROOT/scripts/run_grpo.sh" dev
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

python -m pipelines.v1.src.publish_results --stage "$STAGE"
