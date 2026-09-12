#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PIPELINE_ROOT/../.." && pwd)"
CONFIG_PATH="${V2_CONFIG:-$PIPELINE_ROOT/configs/pipeline.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

usage() {
  cat <<'EOF'
Usage:
  bash pipelines/v2/scripts/run_pipeline.sh prepare
  bash pipelines/v2/scripts/run_pipeline.sh teacher [--resume]
  bash pipelines/v2/scripts/run_pipeline.sh sft [--resume-from CHECKPOINT]
  bash pipelines/v2/scripts/run_pipeline.sh screen [--resume]
  bash pipelines/v2/scripts/run_pipeline.sh grpo [--resume]
  bash pipelines/v2/scripts/run_pipeline.sh dev [--resume]
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

STAGE="$1"
shift
if [[ "$STAGE" == "-h" || "$STAGE" == "--help" ]]; then
  usage
  exit 0
fi

case "$STAGE" in
  prepare|teacher|sft|screen|grpo|dev) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

RESUMING=false
if [[ "${1:-}" == "--resume" || "${1:-}" == "--resume-from" ]]; then
  RESUMING=true
fi

mkdir -p "$PIPELINE_ROOT/logs"
LOG_PATH="$PIPELINE_ROOT/logs/$STAGE.log"
if [[ "$RESUMING" == true ]]; then
  exec > >(tee -a "$LOG_PATH") 2>&1
else
  exec > >(tee "$LOG_PATH") 2>&1
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

run_python() {
  "$PYTHON_BIN" -B -m "$@"
}

run_accelerate() {
  "$ACCELERATE_BIN" launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m "$@"
}

selected_sft_adapter() {
  "$PYTHON_BIN" -B -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["adapter_path"])' \
    "$PIPELINE_ROOT/runs/sft/selection.json"
}

printf '[%s] started at %s\n' "$STAGE" "$(date '+%F %T %z')"

case "$STAGE" in
  prepare)
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    run_python pipelines.v2.src.prepare_data --config "$CONFIG_PATH"
    run_python pipelines.v2.src.publish_results --stage prepare
    printf 'Prepared data: pipelines/v2/artifacts/data\n'
    ;;

  teacher)
    if [[ "$RESUMING" == true ]]; then
      [[ $# -eq 1 && "$1" == "--resume" ]] || { usage >&2; exit 2; }
      if [[ ! -f "$PIPELINE_ROOT/artifacts/teacher/rationales.train.jsonl" ]]; then
        run_python pipelines.v2.src.generate_teacher generate-train \
          --config "$CONFIG_PATH" --resume
      fi
      if [[ ! -f "$PIPELINE_ROOT/artifacts/teacher/rationales.validation.jsonl" ]]; then
        run_python pipelines.v2.src.generate_teacher generate-validation \
          --config "$CONFIG_PATH" --resume
      fi
    else
      [[ $# -eq 0 ]] || { usage >&2; exit 2; }
      rm -rf "$PIPELINE_ROOT/.work/teacher"
      run_python pipelines.v2.src.generate_teacher generate-train \
        --config "$CONFIG_PATH" --smoke
      rm -rf "$PIPELINE_ROOT/.work/teacher"
      run_python pipelines.v2.src.generate_teacher generate-train \
        --config "$CONFIG_PATH"
      run_python pipelines.v2.src.generate_teacher generate-validation \
        --config "$CONFIG_PATH"
    fi
    run_python pipelines.v2.src.generate_teacher build-sft --config "$CONFIG_PATH"
    rm -rf "$PIPELINE_ROOT/.work/teacher"
    printf 'Teacher and SFT data: pipelines/v2/artifacts/teacher\n'
    ;;

  sft)
    if [[ "$RESUMING" == true ]]; then
      [[ $# -eq 2 && "$1" == "--resume-from" ]] || { usage >&2; exit 2; }
      RESUME_CHECKPOINT="$2"
      run_accelerate pipelines.v2.src.train_sft \
        --config "$CONFIG_PATH" --mode train --resume-from "$RESUME_CHECKPOINT"
      run_python pipelines.v2.src.select_model \
        --kind sft --config "$CONFIG_PATH" --resume --devices cuda:0 cuda:1
    else
      [[ $# -eq 0 ]] || { usage >&2; exit 2; }
      rm -rf "$PIPELINE_ROOT/.work/sft"
      run_accelerate pipelines.v2.src.train_sft \
        --config "$CONFIG_PATH" --mode preflight \
        --run-root "$PIPELINE_ROOT/.work/sft/preflight"
      run_accelerate pipelines.v2.src.train_sft \
        --config "$CONFIG_PATH" --mode smoke \
        --run-root "$PIPELINE_ROOT/.work/sft/smoke"
      rm -rf "$PIPELINE_ROOT/.work/sft"
      run_accelerate pipelines.v2.src.train_sft \
        --config "$CONFIG_PATH" --mode train
      run_python pipelines.v2.src.select_model \
        --kind sft --config "$CONFIG_PATH" --devices cuda:0 cuda:1
    fi
    rm -rf "$PIPELINE_ROOT/.work/sft"
    run_python pipelines.v2.src.publish_results --stage sft
    printf 'Selected SFT Adapter: pipelines/v2/runs/sft/best_adapter\n'
    ;;

  screen)
    if [[ "$RESUMING" == true ]]; then
      [[ $# -eq 1 && "$1" == "--resume" ]] || { usage >&2; exit 2; }
    else
      [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    fi
    ADAPTER_PATH="$(selected_sft_adapter)"
    SCREEN_ARGS=()
    [[ "$RESUMING" == true ]] && SCREEN_ARGS+=(--resume)
    run_python pipelines.v2.src.screen_grpo \
      --config "$CONFIG_PATH" \
      --adapter-path "$ADAPTER_PATH" \
      --devices cuda:0 cuda:1 \
      "${SCREEN_ARGS[@]}"
    run_python pipelines.v2.src.publish_results --stage screen
    printf 'Selected GRPO prompts: pipelines/v2/runs/screen/selected_prompts.jsonl\n'
    ;;

  grpo)
    if [[ "$RESUMING" == true ]]; then
      [[ $# -eq 1 && "$1" == "--resume" ]] || { usage >&2; exit 2; }
      if [[ -f "$PIPELINE_ROOT/runs/grpo/training_summary.json" && \
            -f "$PIPELINE_ROOT/runs/grpo/final_checkpoint/training_state.json" ]]; then
        echo "GRPO training is complete; resuming checkpoint selection."
      else
        run_accelerate pipelines.v2.src.train_grpo \
          --config "$CONFIG_PATH" --mode train --resume
      fi
      run_python pipelines.v2.src.select_model \
        --kind grpo --config "$CONFIG_PATH" --resume --devices cuda:0 cuda:1
    else
      [[ $# -eq 0 ]] || { usage >&2; exit 2; }
      rm -rf "$PIPELINE_ROOT/.work/grpo"
      run_accelerate pipelines.v2.src.train_grpo \
        --config "$CONFIG_PATH" --mode smoke
      rm -rf "$PIPELINE_ROOT/.work/grpo/smoke"
      run_accelerate pipelines.v2.src.train_grpo \
        --config "$CONFIG_PATH" --mode train
      run_python pipelines.v2.src.select_model \
        --kind grpo --config "$CONFIG_PATH" --devices cuda:0 cuda:1
    fi
    rm -rf "$PIPELINE_ROOT/.work/grpo"
    run_python pipelines.v2.src.publish_results --stage grpo
    printf 'Selected Exact-GRPO Adapter: pipelines/v2/runs/grpo/best_adapter\n'
    ;;

  dev)
    if [[ "$RESUMING" == true ]]; then
      [[ $# -eq 1 && "$1" == "--resume" ]] || { usage >&2; exit 2; }
      DEV_ARGS=(--resume)
    else
      [[ $# -eq 0 ]] || { usage >&2; exit 2; }
      DEV_ARGS=()
    fi
    run_python pipelines.v2.src.generate_and_evaluate dev \
      --config "$CONFIG_PATH" --devices cuda:0 cuda:1 "${DEV_ARGS[@]}"
    run_python pipelines.v2.src.select_model --kind summary --config "$CONFIG_PATH"
    rm -rf "$PIPELINE_ROOT/.work/dev"
    run_python pipelines.v2.src.publish_results --stage dev
    printf 'Final Dev summary: pipelines/v2/runs/summary.json\n'
    ;;
esac

printf '[%s] completed at %s\n' "$STAGE" "$(date '+%F %T %z')"
