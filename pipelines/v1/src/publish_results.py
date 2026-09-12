"""Publish lightweight V1 metrics from the reproducible run directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PIPELINE_ROOT / "runs"
ARTIFACTS_ROOT = PIPELINE_ROOT / "artifacts"
RESULTS_ROOT = PIPELINE_ROOT / "results"


STAGE_FILES = {
    "prepare": (
        (
            ARTIFACTS_ROOT / "data" / "sft_token_stats.json",
            RESULTS_ROOT / "data" / "sft_token_stats.json",
        ),
    ),
    "zero-shot": (
        (
            RUNS_ROOT / "zero_shot" / "metrics.json",
            RESULTS_ROOT / "zero_shot" / "metrics.json",
        ),
    ),
    "sft": (
        (
            RUNS_ROOT / "sft" / "train_metrics.jsonl",
            RESULTS_ROOT / "sft" / "train_metrics.jsonl",
        ),
        (
            RUNS_ROOT / "sft" / "eval_metrics.jsonl",
            RESULTS_ROOT / "sft" / "eval_metrics.jsonl",
        ),
        (
            RUNS_ROOT / "sft" / "summary.json",
            RESULTS_ROOT / "sft" / "training_summary.json",
        ),
        (
            RUNS_ROOT / "sft" / "dev" / "metrics.json",
            RESULTS_ROOT / "sft" / "metrics.json",
        ),
    ),
    "grpo": (
        (
            RUNS_ROOT / "grpo" / "train_metrics.jsonl",
            RESULTS_ROOT / "grpo" / "train_metrics.jsonl",
        ),
        (
            RUNS_ROOT / "grpo" / "summary.json",
            RESULTS_ROOT / "grpo" / "training_summary.json",
        ),
    ),
    "dev": (
        (
            RUNS_ROOT / "grpo" / "dev" / "metrics.json",
            RESULTS_ROOT / "grpo" / "metrics.json",
        ),
        (
            RUNS_ROOT / "grpo" / "summary.json",
            RESULTS_ROOT / "grpo" / "training_summary.json",
        ),
    ),
}


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Cannot publish missing V1 result: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return metrics


def _write_summary() -> None:
    zero_shot = _load_metrics(RESULTS_ROOT / "zero_shot" / "metrics.json")
    sft = _load_metrics(RESULTS_ROOT / "sft" / "metrics.json")
    grpo = _load_metrics(RESULTS_ROOT / "grpo" / "metrics.json")
    zero_shot_ex = float(zero_shot["overall_ex"])
    sft_ex = float(sft["overall_ex"])
    grpo_ex = float(grpo["overall_ex"])
    summary = {
        "zero_shot": {"overall_ex": zero_shot_ex},
        "sft": {
            "overall_ex": sft_ex,
            "gain_over_zero_shot": round(sft_ex - zero_shot_ex, 2),
        },
        "grpo": {
            "overall_ex": grpo_ex,
            "gain_over_sft": round(grpo_ex - sft_ex, 2),
        },
    }
    destination = RESULTS_ROOT / "summary.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_results(stage: str) -> None:
    """Refresh the lightweight result files produced by one completed stage."""
    for source, destination in STAGE_FILES[stage]:
        _atomic_copy(source, destination)
    if stage == "dev":
        _write_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish lightweight V1 results.")
    parser.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    arguments = parser.parse_args()
    publish_results(arguments.stage)


if __name__ == "__main__":
    main()
