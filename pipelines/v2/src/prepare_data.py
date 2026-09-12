from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipelines.v2.src.pipeline_io import (
    DEFAULT_CONFIG_PATH,
    PIPELINE_ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    load_json,
    load_jsonl,
    resolve_project_path,
)
from pipelines.v2.src.prompt import (
    build_prompt,
    build_value_index,
    render_schema,
    retrieve_grounding,
)


TRAIN_OUTPUT_PATH = PIPELINE_ROOT / "artifacts" / "data" / "train.jsonl"
VALIDATION_OUTPUT_PATH = PIPELINE_ROOT / "artifacts" / "data" / "validation.jsonl"
DEV_OUTPUT_PATH = PIPELINE_ROOT / "artifacts" / "data" / "dev.jsonl"
TRAIN_REWARD_MANIFEST_PATH = (
    PIPELINE_ROOT / "artifacts" / "data" / "train_reward_manifest.jsonl"
)
GROUNDING_AUDIT_PATH = PIPELINE_ROOT / "artifacts" / "data" / "grounding_audit.json"
VALUE_INDEX_PATH = PIPELINE_ROOT / "artifacts" / "grounding" / "value_index.sqlite"

VALIDATION_DB_IDS = {
    "airline",
    "authors",
    "bike_share_1",
    "cs_semester",
    "donor",
    "music_tracker",
    "simpson_episodes",
}
EXPECTED_COUNTS = {"train": 6013, "validation": 588, "dev": 1534}


def _catalog_by_db_id(catalog: object, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(catalog, list):
        raise ValueError(f"{name} schema catalog must be a list.")
    result: dict[str, dict[str, Any]] = {}
    for database in catalog:
        if not isinstance(database, dict) or not isinstance(database.get("db_id"), str):
            raise ValueError(f"{name} schema catalog contains an invalid database.")
        db_id = database["db_id"]
        if db_id in result:
            raise ValueError(f"{name} schema catalog repeats db_id={db_id!r}.")
        result[db_id] = database
    return result


def _gold_reports_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str) and sample_id.startswith("train_"):
            result[sample_id] = record
    return result


def _grounding_for_sample(
    value_index_path: Path,
    grounding_config: dict[str, Any],
    *,
    split: str,
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    return retrieve_grounding(
        value_index_path,
        split=split,
        db_id=sample["db_id"],
        question=sample["question"],
        evidence=sample.get("evidence", ""),
        top_columns=int(grounding_config["top_columns"]),
        values_per_column=int(grounding_config["values_per_column"]),
    )


def _base_train_record(
    sample: dict[str, Any],
    source_index: int,
    schema: dict[str, Any],
    grounding: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_sql = sample["SQL"]
    return {
        "sample_id": f"train_{source_index:06d}",
        "source_index": source_index,
        "db_id": sample["db_id"],
        "prompt": build_prompt(
            render_schema(schema),
            sample["question"],
            sample.get("evidence", ""),
            [hit["line"] for hit in grounding],
        ),
        "completion": [
            {"role": "assistant", "content": f"```sql\n{gold_sql}\n```"}
        ],
        "gold_sql": gold_sql,
        "sqlite_path": schema["sqlite_path"],
        "grounding": grounding,
    }


def _base_dev_record(
    sample: dict[str, Any],
    source_index: int,
    schema: dict[str, Any],
    grounding: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question_id": sample["question_id"],
        "source_index": source_index,
        "db_id": sample["db_id"],
        "difficulty": sample["difficulty"],
        "prompt": build_prompt(
            render_schema(schema),
            sample["question"],
            sample.get("evidence", ""),
            [hit["line"] for hit in grounding],
        ),
        "gold_sql": sample["SQL"],
        "sqlite_path": schema["sqlite_path"],
        "grounding": grounding,
    }


def _reward_exclusion_reason(
    gold_report: dict[str, Any],
    *,
    timeout_seconds: float,
    max_rows: int,
) -> str | None:
    execution = gold_report["gold_execution"]
    if gold_report.get("usable") is not True or execution.get("status") != "success":
        return execution.get("status", "unusable_gold")
    if execution.get("empty_result") is True:
        return "empty_gold_result"
    if float(execution["elapsed_ms"]) > timeout_seconds * 1000:
        return "gold_timeout_limit"
    if int(execution["row_count"]) > max_rows:
        return "gold_row_limit"
    return None


def _build_reward_manifest(
    train_records: list[dict[str, Any]],
    gold_reports: dict[str, dict[str, Any]],
    execution_config: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in train_records:
        report = gold_reports[record["sample_id"]]
        if report["gold_sql"] != record["gold_sql"]:
            raise ValueError(f"Gold SQL mismatch for {record['sample_id']}.")
        if report["db_id"] != record["db_id"]:
            raise ValueError(f"Gold db_id mismatch for {record['sample_id']}.")
        reason = _reward_exclusion_reason(
            report,
            timeout_seconds=float(execution_config["timeout_seconds"]),
            max_rows=int(execution_config["max_rows"]),
        )
        records.append(
            {
                "sample_id": record["sample_id"],
                "source_index": record["source_index"],
                "db_id": record["db_id"],
                "prompt": record["prompt"],
                "gold_sql": record["gold_sql"],
                "sqlite_path": record["sqlite_path"],
                "gold_execution": report["gold_execution"],
                "reward_eligible": reason is None,
                "exclusion_reason": reason,
            }
        )
    return records


def _split_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    hit_counts = [len(record["grounding"]) for record in records]
    matched_value_counts = [
        sum(len(hit["matched_values"]) for hit in record["grounding"])
        for record in records
    ]
    return {
        "sample_count": len(records),
        "database_count": len({record["db_id"] for record in records}),
        "samples_with_grounding": sum(count > 0 for count in hit_counts),
        "samples_without_grounding": sum(count == 0 for count in hit_counts),
        "retrieved_column_count": sum(hit_counts),
        "matched_value_count": sum(matched_value_counts),
        "maximum_columns_per_sample": max(hit_counts, default=0),
        "maximum_values_per_sample": max(matched_value_counts, default=0),
    }


def prepare_data(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Build the grounded datasets, value index, and Train reward manifest."""
    config = load_config(config_path)
    paths = config["paths"]
    grounding_config = config["grounding"]

    train_annotations = load_jsonl(resolve_project_path(paths["train_annotation"]))
    dev_annotations = load_jsonl(resolve_project_path(paths["dev_annotation"]))
    if len(train_annotations) != 6601 or len(dev_annotations) != 1534:
        raise ValueError("Expected 6601 Train annotations and 1534 Dev annotations.")

    train_catalog = load_json(resolve_project_path(paths["train_schema_catalog"]))
    dev_catalog = load_json(resolve_project_path(paths["dev_schema_catalog"]))
    train_schemas = _catalog_by_db_id(train_catalog, "Train")
    dev_schemas = _catalog_by_db_id(dev_catalog, "Dev")
    train_db_ids = {sample["db_id"] for sample in train_annotations}
    dev_db_ids = {sample["db_id"] for sample in dev_annotations}
    if train_db_ids != set(train_schemas) or dev_db_ids != set(dev_schemas):
        raise ValueError("Annotation db_id values do not match their schema catalogs.")
    if train_db_ids & dev_db_ids:
        raise ValueError("Train and Dev databases must remain disjoint.")

    value_index_stats = build_value_index(
        {"train": train_catalog, "dev": dev_catalog},
        {
            "train": resolve_project_path(paths["train_database_root"]),
            "dev": resolve_project_path(paths["dev_database_root"]),
        },
        VALUE_INDEX_PATH,
        scan_rows_per_column=int(grounding_config["scan_rows_per_column"]),
        max_unique_values_per_column=int(
            grounding_config["max_unique_values_per_column"]
        ),
        max_value_chars=int(grounding_config["max_value_chars"]),
    )

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for source_index, sample in enumerate(train_annotations):
        grounding = _grounding_for_sample(
            VALUE_INDEX_PATH,
            grounding_config,
            split="train",
            sample=sample,
        )
        record = _base_train_record(
            sample,
            source_index,
            train_schemas[sample["db_id"]],
            grounding,
        )
        destination = (
            validation_records
            if sample["db_id"] in VALIDATION_DB_IDS
            else train_records
        )
        destination.append(record)

    dev_records: list[dict[str, Any]] = []
    for source_index, sample in enumerate(dev_annotations):
        if sample.get("question_id") != source_index:
            raise ValueError("Dev question_id values must be ordered 0..1533.")
        grounding = _grounding_for_sample(
            VALUE_INDEX_PATH,
            grounding_config,
            split="dev",
            sample=sample,
        )
        dev_records.append(
            _base_dev_record(
                sample,
                source_index,
                dev_schemas[sample["db_id"]],
                grounding,
            )
        )

    actual_counts = {
        "train": len(train_records),
        "validation": len(validation_records),
        "dev": len(dev_records),
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected data counts: {actual_counts}")
    if {record["db_id"] for record in train_records} & {
        record["db_id"] for record in validation_records
    }:
        raise ValueError("Train and Validation databases overlap.")

    gold_reports = _gold_reports_by_sample_id(
        resolve_project_path(paths["gold_execution_report"])
    )
    reward_manifest = _build_reward_manifest(
        train_records,
        gold_reports,
        config["execution"],
    )
    exclusion_counts = Counter(
        record["exclusion_reason"]
        for record in reward_manifest
        if not record["reward_eligible"]
    )
    audit = {
        "value_index": value_index_stats,
        "splits": {
            "train": _split_audit(train_records),
            "validation": _split_audit(validation_records),
            "dev": _split_audit(dev_records),
        },
        "train_validation_database_overlap": 0,
        "sft_dev_database_overlap": 0,
        "reward_manifest": {
            "sample_count": len(reward_manifest),
            "eligible_count": sum(
                record["reward_eligible"] for record in reward_manifest
            ),
            "excluded_count": sum(
                not record["reward_eligible"] for record in reward_manifest
            ),
            "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        },
    }

    atomic_write_jsonl(TRAIN_OUTPUT_PATH, train_records)
    atomic_write_jsonl(VALIDATION_OUTPUT_PATH, validation_records)
    atomic_write_jsonl(DEV_OUTPUT_PATH, dev_records)
    atomic_write_jsonl(TRAIN_REWARD_MANIFEST_PATH, reward_manifest)
    atomic_write_json(GROUNDING_AUDIT_PATH, audit)
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare grounded Train, Validation, and Dev data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = prepare_data(args.config)
    print("Data preparation passed:")
    for split in ("train", "validation", "dev"):
        summary = audit["splits"][split]
        print(
            f"{split}={summary['sample_count']} samples, "
            f"{summary['database_count']} databases, "
            f"grounded={summary['samples_with_grounding']}"
        )
    print(json.dumps(audit["reward_manifest"], ensure_ascii=False))


if __name__ == "__main__":
    main()
