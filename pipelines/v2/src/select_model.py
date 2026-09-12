"""Evaluate and select the SFT and exact-GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from pipelines.v2.src.generate_and_evaluate import (
    evaluate_adapter_greedy,
    evaluate_adapter_pass8,
)
from pipelines.v2.src.pipeline_io import (
    atomic_write_json,
    load_config,
    load_json,
    project_relative_path,
    resolve_project_path,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"
VALIDATION_DATA_PATH = PIPELINE_ROOT / "artifacts" / "data" / "validation.jsonl"
SFT_RUN_ROOT = PIPELINE_ROOT / "runs" / "sft"
GRPO_RUN_ROOT = PIPELINE_ROOT / "runs" / "grpo"
DEV_RUN_ROOT = PIPELINE_ROOT / "runs" / "dev"


def _correct_count(metrics: dict[str, Any]) -> int:
    if "correct_count" in metrics:
        return int(metrics["correct_count"])
    return int(round(float(metrics["official_ex"]) * int(metrics["sample_count"]) / 100.0))


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(metrics["sample_count"]),
        "correct_count": _correct_count(metrics),
        "official_ex": round(float(metrics["official_ex"]), 2),
        "executable_rate": round(float(metrics["executable_rate"]), 2),
    }


def _copy_adapter(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)


def _sft_checkpoints() -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in sorted((SFT_RUN_ROOT / "checkpoints").glob("epoch_*")):
        try:
            epoch = int(checkpoint.name.removeprefix("epoch_"))
        except ValueError:
            continue
        if (checkpoint / "adapter").is_dir():
            checkpoints.append((epoch, checkpoint))
    if not checkpoints:
        raise FileNotFoundError("No completed SFT epoch checkpoint was found.")
    return checkpoints


def _checkpoint_step(checkpoint: Path) -> int:
    state = load_json(checkpoint / "training_state.json")
    return int(state["optimizer_step"])


def _grpo_checkpoints() -> list[tuple[int, Path]]:
    by_step: dict[int, Path] = {}
    for checkpoint in sorted((GRPO_RUN_ROOT / "checkpoints").glob("step_*")):
        if (checkpoint / "adapter").is_dir() and (checkpoint / "training_state.json").is_file():
            by_step.setdefault(_checkpoint_step(checkpoint), checkpoint)

    final_checkpoint = GRPO_RUN_ROOT / "final_checkpoint"
    if (final_checkpoint / "adapter").is_dir() and (final_checkpoint / "training_state.json").is_file():
        by_step.setdefault(_checkpoint_step(final_checkpoint), final_checkpoint)

    if not by_step:
        raise FileNotFoundError("No completed GRPO checkpoint was found.")
    return sorted(by_step.items())


def select_sft(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    resume: bool = False,
    devices: Sequence[str] = ("cuda:0", "cuda:1"),
) -> dict[str, Any]:
    """Select the SFT epoch with the best full greedy Validation EX."""
    config_path = Path(config_path).resolve()
    candidates: list[dict[str, Any]] = []
    for epoch, checkpoint in _sft_checkpoints():
        metrics = evaluate_adapter_greedy(
            config_path=config_path,
            adapter_path=checkpoint / "adapter",
            data_path=VALIDATION_DATA_PATH,
            output_dir=checkpoint / "validation",
            work_dir=(
                PIPELINE_ROOT / ".work" / "sft" / "validation" / checkpoint.name
            ),
            resume=resume,
            devices=tuple(devices),
        )
        candidates.append({
            "epoch": epoch,
            "checkpoint_path": project_relative_path(checkpoint),
            "adapter_path": project_relative_path(checkpoint / "adapter"),
            "metrics": _compact_metrics(metrics),
        })

    best = max(candidates, key=lambda item: (
        item["metrics"]["correct_count"],
        item["metrics"]["executable_rate"],
        -item["epoch"],
    ))
    best_adapter = SFT_RUN_ROOT / "best_adapter"
    _copy_adapter(resolve_project_path(best["adapter_path"]), best_adapter)
    selection = {
        "selected_epoch": best["epoch"],
        "adapter_path": project_relative_path(best_adapter),
        "metrics": best["metrics"],
        "candidates": candidates,
    }
    atomic_write_json(SFT_RUN_ROOT / "selection.json", selection)
    summary_path = SFT_RUN_ROOT / "summary.json"
    training_summary = load_json(summary_path) if summary_path.is_file() else {}
    if "training" in training_summary:
        training_summary = training_summary["training"]
    atomic_write_json(summary_path, {"training": training_summary, "selection": selection})
    return selection


def select_grpo(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    resume: bool = False,
    devices: Sequence[str] = ("cuda:0", "cuda:1"),
) -> dict[str, Any]:
    """Select only among exact-GRPO checkpoints using full pass@8 Vote."""
    config_path = Path(config_path).resolve()
    sft_selection = load_json(SFT_RUN_ROOT / "selection.json")
    sft_baseline = evaluate_adapter_pass8(
        config_path=config_path,
        adapter_path=resolve_project_path(sft_selection["adapter_path"]),
        data_path=VALIDATION_DATA_PATH,
        output_dir=GRPO_RUN_ROOT / "sft_baseline_validation",
        work_dir=(
            PIPELINE_ROOT / ".work" / "grpo" / "selection" / "sft_baseline"
        ),
        resume=resume,
        devices=tuple(devices),
    )

    candidates: list[dict[str, Any]] = []
    for optimizer_step, checkpoint in _grpo_checkpoints():
        metrics = evaluate_adapter_pass8(
            config_path=config_path,
            adapter_path=checkpoint / "adapter",
            data_path=VALIDATION_DATA_PATH,
            output_dir=checkpoint / "validation",
            work_dir=(
                PIPELINE_ROOT / ".work" / "grpo" / "selection" / checkpoint.name
            ),
            resume=resume,
            devices=tuple(devices),
        )
        candidates.append({
            "optimizer_step": optimizer_step,
            "checkpoint_path": project_relative_path(checkpoint),
            "adapter_path": project_relative_path(checkpoint / "adapter"),
            "greedy": _compact_metrics(metrics["greedy"]),
            "vote": _compact_metrics(metrics["vote"]),
        })

    best = max(candidates, key=lambda item: (
        item["vote"]["correct_count"],
        item["vote"]["executable_rate"],
        item["greedy"]["correct_count"],
        -item["optimizer_step"],
    ))
    best_adapter = GRPO_RUN_ROOT / "best_adapter"
    _copy_adapter(resolve_project_path(best["adapter_path"]), best_adapter)
    selection = {
        "selected_optimizer_step": best["optimizer_step"],
        "best_adapter_path": project_relative_path(best_adapter),
        "selection_metric": "pass_at_8_execution_vote",
        "greedy": best["greedy"],
        "vote": best["vote"],
        "sft_baseline": {
            "adapter_path": sft_selection["adapter_path"],
            "greedy": _compact_metrics(sft_baseline["greedy"]),
            "vote": _compact_metrics(sft_baseline["vote"]),
        },
        "candidates": candidates,
    }
    atomic_write_json(GRPO_RUN_ROOT / "selection.json", selection)
    training_summary_path = GRPO_RUN_ROOT / "training_summary.json"
    training_summary = load_json(training_summary_path) if training_summary_path.is_file() else {}
    atomic_write_json(GRPO_RUN_ROOT / "summary.json", {
        "training": training_summary,
        "selection": selection,
    })
    return selection


def summarize(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    sft_selection = load_json(SFT_RUN_ROOT / "selection.json")
    grpo_selection = load_json(GRPO_RUN_ROOT / "selection.json")
    dev_summary = load_json(DEV_RUN_ROOT / "summary.json")
    summary = {
        "method": [
            "grounding_and_value_retrieval",
            "teacher_reasoning_sft",
            "hybrid_difficulty_screening",
            "exact_only_grpo",
            "pass_at_8_execution_vote",
        ],
        "v1_baselines": config["baselines"],
        "internal_validation": {
            "sft": sft_selection["metrics"],
            "sft_pass_at_8": grpo_selection["sft_baseline"],
            "selected_grpo_greedy": grpo_selection["greedy"],
            "selected_grpo_vote": grpo_selection["vote"],
        },
        "dev": dev_summary,
    }
    atomic_write_json(PIPELINE_ROOT / "runs" / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("sft", "grpo", "summary"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.kind == "sft":
        result = select_sft(args.config, resume=args.resume, devices=args.devices)
    elif args.kind == "grpo":
        result = select_grpo(args.config, resume=args.resume, devices=args.devices)
    else:
        result = summarize(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
