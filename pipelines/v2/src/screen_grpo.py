"""Select Train prompts that can provide useful online GRPO signal."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipelines.v2.src.pipeline_io import (
    append_jsonl_flush,
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    load_jsonl,
    resolve_project_path,
)
from pipelines.v2.src.sql_runtime import (
    compute_screening_partial_score_details,
    execute_sql_capped,
    has_screening_score_variance,
    official_execution_match,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"
TRAIN_REWARD_MANIFEST_PATH = (
    PIPELINE_ROOT / "artifacts/data/train_reward_manifest.jsonl"
)
OUTPUT_ROOT = PIPELINE_ROOT / "runs/screen"
SELECTED_PROMPTS_PATH = OUTPUT_ROOT / "selected_prompts.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
WORK_ROOT = PIPELINE_ROOT / ".work/screen"
SCORED_PARTIAL_PATH = WORK_ROOT / "scored.partial.jsonl"
SCREEN_SEED_PHASE = 67


def _score_prompt(
    record: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    timeout_seconds: float,
    max_rows: int,
    score_config: Mapping[str, float],
) -> dict[str, Any]:
    database_path = resolve_project_path(record["sqlite_path"])
    gold = execute_sql_capped(
        database_path,
        record["gold_sql"],
        timeout_seconds=timeout_seconds,
        max_rows=max_rows,
    )
    candidates = generated["candidates"]
    truncated_count = sum(
        candidate.get("truncated") is True for candidate in candidates
    )
    if gold["status"] != "success" or gold["empty_result"]:
        reason = (
            "empty_gold_result"
            if gold["status"] == "success"
            else f"gold_{gold['status']}"
        )
        outcomes = [
            {
                "truncated": candidate.get("truncated") is True,
                "extraction_status": candidate.get("extraction_status", "failed"),
                "format_compliance": candidate.get("format_compliance", False),
                "execution_status": None,
                "error_type": None,
                "official_ex": None,
                "screening_score": None,
            }
            for candidate in candidates
        ]
        return {
            "sample_id": record["sample_id"],
            "source_index": record["source_index"],
            "db_id": record["db_id"],
            "scorable": False,
            "exclusion_reason": reason,
            "valid_candidate_count": len(candidates) - truncated_count,
            "truncated_candidate_count": truncated_count,
            "exact_pass_count": 0,
            "screening_score_min": None,
            "screening_score_max": None,
            "selection_reason": None,
            "candidate_outcomes": outcomes,
        }

    outcomes: list[dict[str, Any]] = []
    exact_scores: list[int] = []
    partial_scores: list[float] = []
    for candidate in candidates:
        if candidate.get("truncated") is True:
            outcomes.append(
                {
                    "truncated": True,
                    "extraction_status": candidate.get(
                        "extraction_status", "failed"
                    ),
                    "format_compliance": candidate.get(
                        "format_compliance", False
                    ),
                    "execution_status": None,
                    "error_type": None,
                    "official_ex": None,
                    "screening_score": None,
                }
            )
            continue

        predicted_sql = candidate.get("predicted_sql", "")
        prediction = execute_sql_capped(
            database_path,
            predicted_sql if isinstance(predicted_sql, str) else "",
            timeout_seconds=timeout_seconds,
            max_rows=max_rows,
        )
        details = compute_screening_partial_score_details(
            prediction,
            gold,
            score_config=score_config,
        )
        exact = official_execution_match(prediction, gold)
        score = float(details["screening_score"])
        exact_scores.append(exact)
        partial_scores.append(score)
        outcomes.append(
            {
                "truncated": False,
                "extraction_status": candidate.get("extraction_status", "failed"),
                "format_compliance": candidate.get("format_compliance", False),
                "execution_status": prediction["status"],
                "error_type": prediction["error_type"],
                "official_ex": exact,
                "screening_score": score,
            }
        )

    valid_count = len(partial_scores)
    exact_count = sum(exact_scores)
    exact_mixed = 0 < exact_count < valid_count
    zero_exact_partial_variance = (
        exact_count == 0
        and len(partial_scores) >= 2
        and has_screening_score_variance(partial_scores)
    )
    if exact_mixed:
        selection_reason = "exact_mixed"
    elif zero_exact_partial_variance:
        selection_reason = "zero_exact_partial_variance"
    else:
        selection_reason = None

    return {
        "sample_id": record["sample_id"],
        "source_index": record["source_index"],
        "db_id": record["db_id"],
        "scorable": True,
        "exclusion_reason": None,
        "valid_candidate_count": valid_count,
        "truncated_candidate_count": truncated_count,
        "exact_pass_count": exact_count,
        "screening_score_min": min(partial_scores) if partial_scores else None,
        "screening_score_max": max(partial_scores) if partial_scores else None,
        "selection_reason": selection_reason,
        "candidate_outcomes": outcomes,
    }


def _summary(
    *,
    manifest_count: int,
    rows: Sequence[Mapping[str, Any]],
    num_candidates: int,
) -> dict[str, Any]:
    selection_reasons = Counter(
        row["selection_reason"] for row in rows if row["selection_reason"]
    )
    exact_buckets = Counter(
        str(row["exact_pass_count"]) for row in rows if row["scorable"]
    )
    valid_count_buckets = Counter(
        str(row["valid_candidate_count"]) for row in rows if row["scorable"]
    )
    exclusion_reasons = Counter(
        row["exclusion_reason"] for row in rows if not row["scorable"]
    )
    extraction_statuses: Counter[str] = Counter()
    execution_statuses: Counter[str] = Counter()
    format_compliant_count = 0
    for row in rows:
        for outcome in row["candidate_outcomes"]:
            extraction_statuses[outcome["extraction_status"]] += 1
            format_compliant_count += outcome["format_compliance"] is True
            if outcome["execution_status"] is not None:
                execution_statuses[outcome["execution_status"]] += 1

    generated_candidate_count = len(rows) * num_candidates
    selected_count = sum(row["selection_reason"] is not None for row in rows)
    return {
        "manifest_prompt_count": manifest_count,
        "eligible_prompt_count": len(rows),
        "manifest_excluded_prompt_count": manifest_count - len(rows),
        "scorable_prompt_count": sum(row["scorable"] for row in rows),
        "unscorable_prompt_count": sum(not row["scorable"] for row in rows),
        "num_candidates": num_candidates,
        "generated_candidate_count": generated_candidate_count,
        "valid_candidate_count": sum(row["valid_candidate_count"] for row in rows),
        "truncated_candidate_count": sum(
            row["truncated_candidate_count"] for row in rows
        ),
        "selected_prompt_count": selected_count,
        "selection_rate": round(100.0 * selected_count / len(rows), 2),
        "exact_mixed_count": selection_reasons["exact_mixed"],
        "zero_exact_partial_variance_count": selection_reasons[
            "zero_exact_partial_variance"
        ],
        "exact_pass_count_buckets": dict(sorted(exact_buckets.items())),
        "valid_candidate_count_buckets": dict(sorted(valid_count_buckets.items())),
        "gold_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "candidate_extraction_statuses": dict(sorted(extraction_statuses.items())),
        "candidate_execution_statuses": dict(sorted(execution_statuses.items())),
        "format_compliance_rate": round(
            100.0 * format_compliant_count / generated_candidate_count, 2
        ),
    }


def screen_grpo_prompts(
    config_path: str | Path,
    adapter_path: str | Path,
    *,
    resume: bool,
    devices: Sequence[str],
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    manifest = load_jsonl(TRAIN_REWARD_MANIFEST_PATH)
    eligible = [record for record in manifest if record.get("reward_eligible") is True]
    num_candidates = int(config["screening"]["num_generations"])
    if num_candidates != 4:
        raise ValueError("GRPO screening requires exactly four candidates per prompt.")

    if SELECTED_PROMPTS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            f"GRPO screening output already exists under {OUTPUT_ROOT}."
        )
    if not resume:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    from pipelines.v2.src.generate_and_evaluate import run_sharded_generation

    generated_rows = run_sharded_generation(
        config_path,
        eligible,
        resolve_project_path(adapter_path),
        WORK_ROOT / "generation",
        num_candidates=num_candidates,
        include_greedy=False,
        seed_phase=SCREEN_SEED_PHASE,
        resume=resume,
        devices=devices,
    )
    generated_by_id = {row["sample_id"]: row for row in generated_rows}

    scored_rows = load_jsonl(SCORED_PARTIAL_PATH) if SCORED_PARTIAL_PATH.exists() else []
    expected_prefix = [record["sample_id"] for record in eligible[: len(scored_rows)]]
    if [row["sample_id"] for row in scored_rows] != expected_prefix:
        raise ValueError("The screening score partial does not match the Train prefix.")

    execution_config = config["execution"]
    for record in eligible[len(scored_rows) :]:
        scored = _score_prompt(
            record,
            generated_by_id[record["sample_id"]],
            timeout_seconds=float(execution_config["timeout_seconds"]),
            max_rows=int(execution_config["max_rows"]),
            score_config=config["screening"],
        )
        append_jsonl_flush(SCORED_PARTIAL_PATH, scored)
        scored_rows.append(scored)

    selected = [
        {
            "sample_id": row["sample_id"],
            "source_index": row["source_index"],
            "db_id": row["db_id"],
            "valid_candidate_count": row["valid_candidate_count"],
            "exact_pass_count": row["exact_pass_count"],
            "screening_score_min": row["screening_score_min"],
            "screening_score_max": row["screening_score_max"],
            "selection_reason": row["selection_reason"],
        }
        for row in scored_rows
        if row["selection_reason"] is not None
    ]
    summary = _summary(
        manifest_count=len(manifest),
        rows=scored_rows,
        num_candidates=num_candidates,
    )
    atomic_write_jsonl(SELECTED_PROMPTS_PATH, selected)
    atomic_write_json(SUMMARY_PATH, summary)
    shutil.rmtree(WORK_ROOT)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = screen_grpo_prompts(
        args.config,
        args.adapter_path,
        resume=args.resume,
        devices=args.devices,
    )
    print(
        "GRPO screening completed: "
        f"eligible={summary['eligible_prompt_count']}, "
        f"selected={summary['selected_prompt_count']}, "
        f"exact_mixed={summary['exact_mixed_count']}, "
        "zero_exact_partial_variance="
        f"{summary['zero_exact_partial_variance_count']}"
    )


if __name__ == "__main__":
    main()
