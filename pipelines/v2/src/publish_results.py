"""Publish lightweight V2 metrics from completed Pipeline stages."""

from __future__ import annotations

import argparse
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
            ARTIFACTS_ROOT / "data" / "grounding_audit.json",
            RESULTS_ROOT / "data" / "grounding_audit.json",
        ),
    ),
    "sft": (
        (
            RUNS_ROOT / "sft" / "selection.json",
            RESULTS_ROOT / "sft" / "selection.json",
        ),
        (
            RUNS_ROOT / "sft" / "summary.json",
            RESULTS_ROOT / "sft" / "summary.json",
        ),
    ),
    "screen": (
        (
            RUNS_ROOT / "screen" / "summary.json",
            RESULTS_ROOT / "screen" / "summary.json",
        ),
    ),
    "grpo": (
        (
            RUNS_ROOT / "grpo" / "selection.json",
            RESULTS_ROOT / "grpo" / "selection.json",
        ),
        (
            RUNS_ROOT / "grpo" / "summary.json",
            RESULTS_ROOT / "grpo" / "summary.json",
        ),
        (
            RUNS_ROOT / "grpo" / "train_metrics.jsonl",
            RESULTS_ROOT / "grpo" / "train_metrics.jsonl",
        ),
        (
            RUNS_ROOT / "grpo" / "training_summary.json",
            RESULTS_ROOT / "grpo" / "training_summary.json",
        ),
    ),
    "dev": (
        (
            RUNS_ROOT / "dev" / "sft_greedy" / "metrics.json",
            RESULTS_ROOT / "dev" / "sft_greedy_metrics.json",
        ),
        (
            RUNS_ROOT / "dev" / "greedy" / "metrics.json",
            RESULTS_ROOT / "dev" / "greedy_metrics.json",
        ),
        (
            RUNS_ROOT / "dev" / "vote" / "metrics.json",
            RESULTS_ROOT / "dev" / "vote_metrics.json",
        ),
        (
            RUNS_ROOT / "dev" / "summary.json",
            RESULTS_ROOT / "dev" / "summary.json",
        ),
        (
            RUNS_ROOT / "summary.json",
            RESULTS_ROOT / "summary.json",
        ),
    ),
}


def _copy_result(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Cannot publish missing V2 result: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_results(stage: str) -> None:
    """Refresh the lightweight result files for one completed stage."""
    for source, destination in STAGE_FILES[stage]:
        _copy_result(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    arguments = parser.parse_args()
    publish_results(arguments.stage)


if __name__ == "__main__":
    main()
