from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.evaluation_utils import (
    load_json_file,
    load_strict_jsonl,
    official_execution_match,
    percentage,
    publish_evaluation_outputs,
    validate_run_name,
    validate_timeout_seconds,
)
from src.execution.sql_executor import execute_sql


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_ANNOTATIONS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "annotations" / "dev" / "bird_sql_dev_20251106.json"
)
DEV_SCHEMA_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "schema_catalog_dev.json"
)
DEV_DATABASE_ROOT = PROJECT_ROOT / "data" / "raw" / "databases" / "dev"
DATASET_NAME = "bird_sql_dev_20251106"
EXPECTED_DEV_SAMPLE_COUNT = 1_534
DIFFICULTIES = ("simple", "moderate", "challenging")
EXECUTION_STATUSES = {
    "success",
    "timeout",
    "execution_error",
    "unsafe_sql",
}
PREDICTION_FIELDS = {
    "question_id",
    "db_id",
    "raw_output",
    "predicted_sql",
    "extraction_status",
    "format_compliance",
}
EXTRACTION_STATUSES = {
    "sql_fence",
    "generic_fence",
    "raw_sql",
    "failed",
}


def _validate_dev_annotations(records: list[dict]) -> list[dict]:
    if len(records) != EXPECTED_DEV_SAMPLE_COUNT:
        raise ValueError(
            f"Dev annotations must contain exactly {EXPECTED_DEV_SAMPLE_COUNT} "
            f"records, got {len(records)} in {DEV_ANNOTATIONS_PATH}."
        )

    records_by_question_id: dict[int, dict] = {}
    for line_number, record in enumerate(records, start=1):
        question_id = record.get("question_id")
        db_id = record.get("db_id")
        gold_sql = record.get("SQL")
        difficulty = record.get("difficulty")

        if type(question_id) is not int:
            raise ValueError(
                "Dev annotation question_id must be an integer and cannot be a "
                f"boolean; line {line_number} has {question_id!r}."
            )
        if question_id in records_by_question_id:
            raise ValueError(
                f"Dev annotation question_id {question_id} is duplicated at "
                f"line {line_number}."
            )
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(
                f"Dev annotation line {line_number}, question_id {question_id}, "
                f"has an invalid db_id: {db_id!r}."
            )
        if not isinstance(gold_sql, str) or not gold_sql.strip():
            raise ValueError(
                f"Dev annotation line {line_number}, question_id {question_id}, "
                f"has an invalid SQL value: {gold_sql!r}."
            )
        if difficulty not in DIFFICULTIES:
            raise ValueError(
                f"Dev annotation line {line_number}, question_id {question_id}, "
                f"has an unsupported difficulty: {difficulty!r}."
            )

        records_by_question_id[question_id] = record

    expected_question_ids = set(range(EXPECTED_DEV_SAMPLE_COUNT))
    actual_question_ids = set(records_by_question_id)
    if actual_question_ids != expected_question_ids:
        raise ValueError(
            "Dev annotation question_id values must be exactly 0 through "
            f"{EXPECTED_DEV_SAMPLE_COUNT - 1}; missing="
            f"{sorted(expected_question_ids - actual_question_ids)!r}, unexpected="
            f"{sorted(actual_question_ids - expected_question_ids)!r}."
        )

    actual_order = [record["question_id"] for record in records]
    expected_order = list(range(EXPECTED_DEV_SAMPLE_COUNT))
    if actual_order != expected_order:
        raise ValueError(
            "Dev annotations must be ordered by question_id from 0 through "
            f"{EXPECTED_DEV_SAMPLE_COUNT - 1}."
        )

    return records


def _validate_predictions(
    records: list[dict],
    annotations: list[dict],
    predictions_path: Path,
) -> dict[int, dict]:
    if len(records) != EXPECTED_DEV_SAMPLE_COUNT:
        raise ValueError(
            f"Predictions must contain exactly {EXPECTED_DEV_SAMPLE_COUNT} "
            f"records, got {len(records)} in {predictions_path}."
        )

    annotations_by_question_id = {
        annotation["question_id"]: annotation for annotation in annotations
    }
    predictions_by_question_id: dict[int, dict] = {}

    for line_number, record in enumerate(records, start=1):
        if set(record) != PREDICTION_FIELDS:
            raise ValueError(
                f"Prediction line {line_number} in {predictions_path} must contain "
                f"exactly fields {sorted(PREDICTION_FIELDS)!r}, got "
                f"{sorted(record)!r}."
            )

        question_id = record["question_id"]
        db_id = record["db_id"]
        raw_output = record["raw_output"]
        predicted_sql = record["predicted_sql"]
        extraction_status = record["extraction_status"]
        format_compliance = record["format_compliance"]

        if type(question_id) is not int:
            raise ValueError(
                "Prediction question_id must be an integer and cannot be a "
                f"boolean; line {line_number} has {question_id!r}."
            )
        if question_id in predictions_by_question_id:
            raise ValueError(
                f"Prediction question_id {question_id} is duplicated at line "
                f"{line_number} in {predictions_path}."
            )
        annotation = annotations_by_question_id.get(question_id)
        if annotation is None:
            raise ValueError(
                f"Prediction line {line_number} has an unexpected question_id: "
                f"{question_id}."
            )
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"an invalid db_id: {db_id!r}."
            )
        if db_id != annotation["db_id"]:
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"db_id {db_id!r}, expected {annotation['db_id']!r}."
            )
        if not isinstance(predicted_sql, str):
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"an invalid predicted_sql: {predicted_sql!r}."
            )
        if not isinstance(raw_output, str):
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"an invalid raw_output: {raw_output!r}."
            )
        if (
            not isinstance(extraction_status, str)
            or extraction_status not in EXTRACTION_STATUSES
        ):
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"an invalid extraction_status: {extraction_status!r}; expected "
                f"one of {sorted(EXTRACTION_STATUSES)!r}."
            )
        if type(format_compliance) is not bool:
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"an invalid format_compliance: {format_compliance!r}; expected "
                "a boolean."
            )
        expected_format_compliance = extraction_status == "sql_fence"
        if format_compliance is not expected_format_compliance:
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, has "
                f"format_compliance={format_compliance!r}, which is inconsistent "
                f"with extraction_status={extraction_status!r}."
            )
        if extraction_status == "failed":
            if predicted_sql != "":
                raise ValueError(
                    f"Prediction line {line_number}, question_id {question_id}, "
                    "must have an empty predicted_sql when extraction_status is "
                    f"'failed', got {predicted_sql!r}."
                )
        elif not predicted_sql.strip():
            raise ValueError(
                f"Prediction line {line_number}, question_id {question_id}, must "
                "have a non-empty predicted_sql when extraction_status is "
                f"{extraction_status!r}."
            )

        predictions_by_question_id[question_id] = record

    expected_question_ids = set(annotations_by_question_id)
    actual_question_ids = set(predictions_by_question_id)
    if actual_question_ids != expected_question_ids:
        raise ValueError(
            "Prediction question_id values do not exactly match Dev annotations; "
            f"missing={sorted(expected_question_ids - actual_question_ids)!r}, "
            f"unexpected={sorted(actual_question_ids - expected_question_ids)!r}."
        )

    return predictions_by_question_id


def _load_dev_database_paths(annotations: list[dict]) -> dict[str, Path]:
    catalog = load_json_file(
        DEV_SCHEMA_CATALOG_PATH,
        description="Dev Schema Catalog",
    )
    if not isinstance(catalog, list):
        raise ValueError(
            "Dev Schema Catalog must have a JSON list at the top level, got "
            f"{type(catalog).__name__}."
        )

    database_root = DEV_DATABASE_ROOT.resolve()
    database_paths: dict[str, Path] = {}
    for record_index, record in enumerate(catalog, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Dev Schema Catalog record {record_index} must be a JSON object, "
                f"got {type(record).__name__}."
            )

        db_id = record.get("db_id")
        sqlite_path_value = record.get("sqlite_path")
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(
                f"Dev Schema Catalog record {record_index} has an invalid db_id: "
                f"{db_id!r}."
            )
        if db_id in database_paths:
            raise ValueError(f"Dev Schema Catalog contains duplicate db_id {db_id!r}.")
        if not isinstance(sqlite_path_value, str) or not sqlite_path_value.strip():
            raise ValueError(
                f"Dev Schema Catalog db_id {db_id!r} has an invalid sqlite_path: "
                f"{sqlite_path_value!r}."
            )

        relative_path = Path(sqlite_path_value)
        if relative_path.is_absolute():
            raise ValueError(
                f"Dev Schema Catalog db_id {db_id!r} sqlite_path must be project-"
                f"relative, got {sqlite_path_value!r}."
            )
        resolved_path = (PROJECT_ROOT / relative_path).resolve()
        try:
            resolved_path.relative_to(database_root)
        except ValueError as error:
            raise ValueError(
                f"Dev Schema Catalog db_id {db_id!r} sqlite_path resolves outside "
                f"the Dev database directory: {sqlite_path_value!r}."
            ) from error

        expected_path = (database_root / db_id / f"{db_id}.sqlite").resolve()
        if resolved_path != expected_path:
            raise ValueError(
                f"Dev Schema Catalog db_id {db_id!r} sqlite_path must resolve to "
                f"{expected_path}, got {sqlite_path_value!r}."
            )
        if not resolved_path.is_file():
            raise ValueError(
                f"Dev Schema Catalog db_id {db_id!r} SQLite file does not exist: "
                f"{resolved_path}."
            )

        database_paths[db_id] = resolved_path

    expected_db_ids = {annotation["db_id"] for annotation in annotations}
    actual_db_ids = set(database_paths)
    if actual_db_ids != expected_db_ids:
        raise ValueError(
            "Dev Schema Catalog db_id values do not exactly match Dev annotations; "
            f"missing={sorted(expected_db_ids - actual_db_ids)!r}, "
            f"unexpected={sorted(actual_db_ids - expected_db_ids)!r}."
        )

    return database_paths


def _validated_execution_result(
    result: Any,
    *,
    question_id: int,
    role: str,
) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError(
            f"execute_sql returned {type(result).__name__} for {role} SQL at "
            f"question_id {question_id}; expected dict."
        )

    required_fields = {
        "status",
        "rows",
        "row_count",
        "column_count",
        "empty_result",
        "elapsed_ms",
        "error_type",
        "error_message",
    }
    missing_fields = sorted(required_fields - set(result))
    if missing_fields:
        raise RuntimeError(
            f"execute_sql omitted required fields {missing_fields!r} for {role} "
            f"SQL at question_id {question_id}."
        )

    status = result.get("status")
    if status not in EXECUTION_STATUSES:
        raise RuntimeError(
            f"execute_sql returned invalid status {status!r} for {role} SQL at "
            f"question_id {question_id}."
        )
    rows = result.get("rows")
    if status == "success":
        if not isinstance(rows, list) or any(not isinstance(row, tuple) for row in rows):
            raise RuntimeError(
                f"execute_sql must return rows as list[tuple] for successful "
                f"{role} SQL at question_id {question_id}, got {rows!r}."
            )
        if result["row_count"] != len(rows):
            raise RuntimeError(
                f"execute_sql returned row_count={result['row_count']!r} but "
                f"{len(rows)} rows for {role} SQL at question_id {question_id}."
            )
        if result["empty_result"] is not (not rows):
            raise RuntimeError(
                f"execute_sql returned inconsistent empty_result="
                f"{result['empty_result']!r} for {role} SQL at question_id "
                f"{question_id}."
            )
        if result["error_type"] is not None or result["error_message"] is not None:
            raise RuntimeError(
                f"execute_sql returned error details for successful {role} SQL "
                f"at question_id {question_id}."
            )
    elif rows is not None:
        raise RuntimeError(
            f"execute_sql must return rows=None for failed {role} SQL at "
            f"question_id {question_id}, got {rows!r}."
        )

    row_count = result["row_count"]
    column_count = result["column_count"]
    empty_result = result["empty_result"]
    if type(row_count) is not int or row_count < 0:
        raise RuntimeError(
            f"execute_sql returned invalid row_count {row_count!r} for {role} "
            f"SQL at question_id {question_id}."
        )
    if type(column_count) is not int or column_count < 0:
        raise RuntimeError(
            f"execute_sql returned invalid column_count {column_count!r} for "
            f"{role} SQL at question_id {question_id}."
        )
    if type(empty_result) is not bool:
        raise RuntimeError(
            f"execute_sql returned invalid empty_result {empty_result!r} for "
            f"{role} SQL at question_id {question_id}."
        )
    if status != "success" and (
        row_count != 0 or column_count != 0 or empty_result
    ):
        raise RuntimeError(
            f"execute_sql returned a non-empty failure shape for {role} SQL at "
            f"question_id {question_id}."
        )

    elapsed_ms = result.get("elapsed_ms")
    if (
        isinstance(elapsed_ms, bool)
        or not isinstance(elapsed_ms, (int, float))
        or not math.isfinite(elapsed_ms)
        or elapsed_ms < 0
    ):
        raise RuntimeError(
            f"execute_sql returned invalid elapsed_ms {elapsed_ms!r} for {role} "
            f"SQL at question_id {question_id}."
        )
    error_type = result.get("error_type")
    error_message = result.get("error_message")
    if error_type is not None and not isinstance(error_type, str):
        raise RuntimeError(
            f"execute_sql returned invalid error_type {error_type!r} for {role} "
            f"SQL at question_id {question_id}."
        )
    if error_message is not None and not isinstance(error_message, str):
        raise RuntimeError(
            f"execute_sql returned invalid error_message {error_message!r} for "
            f"{role} SQL at question_id {question_id}."
        )
    if status != "success" and (
        not isinstance(error_type, str) or not isinstance(error_message, str)
    ):
        raise RuntimeError(
            f"execute_sql must return string error details for failed {role} SQL "
            f"at question_id {question_id}."
        )

    return result


def _score_predictions(
    annotations: list[dict],
    predictions_by_question_id: dict[int, dict],
    database_paths: dict[str, Path],
    timeout_seconds: float,
) -> list[dict]:
    scored_results: list[dict] = []

    for annotation in annotations:
        question_id = annotation["question_id"]
        db_id = annotation["db_id"]
        prediction = predictions_by_question_id[question_id]
        predicted_sql = prediction["predicted_sql"]
        database_path = database_paths[db_id]

        prediction_result = _validated_execution_result(
            execute_sql(
                database_path,
                predicted_sql,
                timeout_seconds=timeout_seconds,
            ),
            question_id=question_id,
            role="prediction",
        )
        gold_result = _validated_execution_result(
            execute_sql(
                database_path,
                annotation["SQL"],
                timeout_seconds=timeout_seconds,
            ),
            question_id=question_id,
            role="gold",
        )

        official_ex = 0
        if (
            prediction_result["status"] == "success"
            and gold_result["status"] == "success"
        ):
            official_ex = official_execution_match(
                prediction_result["rows"],
                gold_result["rows"],
            )

        scored_results.append(
            {
                "question_id": question_id,
                "db_id": db_id,
                "difficulty": annotation["difficulty"],
                "predicted_sql": predicted_sql,
                "extraction_status": prediction["extraction_status"],
                "format_compliance": prediction["format_compliance"],
                "prediction_status": prediction_result["status"],
                "gold_status": gold_result["status"],
                "gold_elapsed_ms": gold_result["elapsed_ms"],
                "gold_error_type": gold_result["error_type"],
                "official_ex": official_ex,
                "elapsed_ms": prediction_result["elapsed_ms"],
                "error_type": prediction_result["error_type"],
                "error_message": prediction_result["error_message"],
            }
        )

    return scored_results


def _build_metrics(run_name: str, scored_results: list[dict]) -> dict:
    total = len(scored_results)
    correct = sum(result["official_ex"] for result in scored_results)
    difficulty_results = {
        difficulty: [
            result
            for result in scored_results
            if result["difficulty"] == difficulty
        ]
        for difficulty in DIFFICULTIES
    }

    return {
        "run_name": run_name,
        "dataset": DATASET_NAME,
        "total": total,
        "overall_ex": percentage(correct, total),
        "simple_ex": percentage(
            sum(result["official_ex"] for result in difficulty_results["simple"]),
            len(difficulty_results["simple"]),
        ),
        "moderate_ex": percentage(
            sum(
                result["official_ex"]
                for result in difficulty_results["moderate"]
            ),
            len(difficulty_results["moderate"]),
        ),
        "challenging_ex": percentage(
            sum(
                result["official_ex"]
                for result in difficulty_results["challenging"]
            ),
            len(difficulty_results["challenging"]),
        ),
        "executable_rate": percentage(
            sum(
                result["prediction_status"] == "success"
                for result in scored_results
            ),
            total,
        ),
        "format_compliance_rate": percentage(
            sum(result["format_compliance"] for result in scored_results),
            total,
        ),
        "timeout_count": sum(
            result["prediction_status"] == "timeout" for result in scored_results
        ),
        "syntax_error_count": sum(
            result["error_type"] == "syntax_error" for result in scored_results
        ),
        "missing_table_count": sum(
            result["error_type"] == "missing_table" for result in scored_results
        ),
        "missing_column_count": sum(
            result["error_type"] == "missing_column" for result in scored_results
        ),
    }


def evaluate_bird_ex(
    predictions_path: str | Path,
    run_name: str,
    output_dir: str | Path,
    timeout_seconds: float = 30.0,
) -> dict:
    """Evaluate all 1,534 Dev predictions with BIRD execution accuracy."""
    safe_run_name = validate_run_name(run_name)
    validated_timeout = validate_timeout_seconds(timeout_seconds)
    prediction_path = Path(predictions_path)
    output_directory = Path(output_dir)
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory

    annotations = _validate_dev_annotations(
        load_strict_jsonl(
            DEV_ANNOTATIONS_PATH,
            description="Dev annotations",
        )
    )
    predictions_by_question_id = _validate_predictions(
        load_strict_jsonl(
            prediction_path,
            description="predictions",
        ),
        annotations,
        prediction_path,
    )
    database_paths = _load_dev_database_paths(annotations)

    scored_results = _score_predictions(
        annotations,
        predictions_by_question_id,
        database_paths,
        validated_timeout,
    )
    metrics = _build_metrics(safe_run_name, scored_results)

    return publish_evaluation_outputs(
        report_root=output_directory.parent,
        run_name=output_directory.name,
        scored_results=scored_results,
        metrics=metrics,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full BIRD Dev prediction JSONL with execution accuracy."
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help=(
            "Prediction JSONL path. Each row must contain question_id, db_id, "
            "raw_output, predicted_sql, extraction_status, and format_compliance."
        ),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Experiment name stored in metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that receives scored_results.jsonl and metrics.json.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-query timeout for both predicted and gold SQL (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = evaluate_bird_ex(
        predictions_path=args.predictions_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
