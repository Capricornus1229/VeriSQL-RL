from __future__ import annotations

import argparse
import math
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.common import (
    load_json,
    load_jsonl,
    open_sqlite_read_only,
    save_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "annotations" / "train" / "bird23_train_filtered.jsonl"
)
DEV_ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "annotations" / "dev" / "bird_sql_dev_20251106.json"
)
TRAIN_SCHEMA_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "schema_catalog_train.json"
)
DEV_SCHEMA_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "schema_catalog_dev.json"
)
GOLD_SQL_EXECUTION_PATH = (
    PROJECT_ROOT / "data" / "interim" / "gold_sql_execution.jsonl"
)

_DEFAULT_BUSY_TIMEOUT_SECONDS = 5.0
_PROGRESS_HANDLER_INSTRUCTIONS = 1_000
_EXPECTED_ANNOTATION_COUNTS = {"train": 6_601, "dev": 1_534}
_DEV_DIFFICULTIES = {"simple", "moderate", "challenging"}
_ALLOWED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_RECURSIVE,
    sqlite3.SQLITE_SELECT,
}
_UNSAFE_FUNCTIONS = {
    "edit",
    "load_extension",
    "readfile",
    "writefile",
}


def _validate_timeout_seconds(timeout_seconds: object) -> float:
    """Validate and normalize a public timeout argument."""
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise TypeError(
            "timeout_seconds must be a finite positive number, got "
            f"{timeout_seconds!r} ({type(timeout_seconds).__name__})."
        )

    try:
        normalized_timeout = float(timeout_seconds)
    except OverflowError as error:
        raise ValueError(
            "timeout_seconds must be a finite positive number, got "
            f"{timeout_seconds!r}."
        ) from error

    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
        raise ValueError(
            "timeout_seconds must be a finite positive number, got "
            f"{timeout_seconds!r}."
        )

    return normalized_timeout


def _validate_max_rows(max_rows: object) -> int | None:
    """Validate and normalize the optional public result-row cap."""
    if max_rows is None:
        return None
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError(
            "max_rows must be a positive integer and cannot be boolean, "
            f"got {max_rows!r}."
        )
    return max_rows


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed wall-clock time in milliseconds."""
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _execution_failure(
    status: str,
    error_type: str,
    error_message: str,
    started_at: float,
    *,
    row_count: int = 0,
    column_count: int = 0,
) -> dict:
    """Build the stable result shape used by all failed executions."""
    return {
        "status": status,
        "elapsed_ms": _elapsed_ms(started_at),
        "rows": None,
        "row_count": row_count,
        "column_count": column_count,
        "empty_result": False,
        "error_type": error_type,
        "error_message": error_message,
    }


def _classify_sqlite_error(error: sqlite3.Error) -> str:
    """Classify common SQLite failures without relying on Python 3.11 APIs."""
    message = str(error).lower()

    if "no such table" in message:
        return "missing_table"
    if "no such column" in message:
        return "missing_column"
    if "ambiguous column name" in message:
        return "ambiguous_column"
    if "no such function" in message:
        return "missing_function"
    if "database is locked" in message or "database table is locked" in message:
        return "database_locked"
    if (
        "syntax error" in message
        or "incomplete input" in message
        or "unrecognized token" in message
        or "near " in message
    ):
        return "syntax_error"
    if (
        "unable to open database" in message
        or "file is not a database" in message
        or "database disk image is malformed" in message
    ):
        return "database_open_error"

    return "sqlite_error"


def _authorizer_action_description(
    action_code: int,
    first_argument: str | None,
    second_argument: str | None,
    database_name: str | None,
) -> str:
    """Describe an operation rejected by the read-only SQL authorizer."""
    return (
        f"SQLite authorizer rejected operation code {action_code} "
        f"with arguments ({first_argument!r}, {second_argument!r}) "
        f"on database {database_name!r}."
    )


def _execute_on_connection(
    connection: sqlite3.Connection,
    sql: str,
    timeout_seconds: float,
    started_at: float,
    *,
    max_rows: int | None,
    timeout_deadline: float | None,
) -> dict:
    """Execute one result-producing statement on a dedicated connection."""
    denied_operation: list[str] = []

    def authorize(
        action_code: int,
        first_argument: str | None,
        second_argument: str | None,
        database_name: str | None,
        trigger_or_view: str | None,
    ) -> int:
        del trigger_or_view

        function_name = (
            second_argument
            if action_code == sqlite3.SQLITE_FUNCTION
            else None
        )
        function_is_unsafe = (
            isinstance(function_name, str)
            and function_name.lower() in _UNSAFE_FUNCTIONS
        )
        if action_code in _ALLOWED_AUTHORIZER_ACTIONS and not function_is_unsafe:
            return sqlite3.SQLITE_OK

        if not denied_operation:
            denied_operation.append(
                _authorizer_action_description(
                    action_code,
                    first_argument,
                    second_argument,
                    database_name,
                )
            )
        return sqlite3.SQLITE_DENY

    timeout_triggered = threading.Event()
    deadline = (
        time.perf_counter() + timeout_seconds
        if timeout_deadline is None
        else timeout_deadline
    )

    def check_deadline() -> int:
        if time.perf_counter() >= deadline:
            timeout_triggered.set()
            return 1
        return 0

    def interrupt_at_deadline() -> None:
        timeout_triggered.set()
        try:
            connection.interrupt()
        except sqlite3.Error:
            # The owning thread will report the execution outcome.
            pass

    timer: threading.Timer | None = None
    if timeout_deadline is None:
        timer = threading.Timer(
            min(timeout_seconds, threading.TIMEOUT_MAX),
            interrupt_at_deadline,
        )
        timer.daemon = True
    timer_started = False
    caught_error: sqlite3.Error | sqlite3.Warning | None = None
    rows: list[tuple[Any, ...]] | None = None
    column_count: int | None = None

    try:
        connection.set_authorizer(authorize)
        connection.set_progress_handler(
            check_deadline,
            _PROGRESS_HANDLER_INSTRUCTIONS,
        )
        if timer is None:
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                timeout_triggered.set()
                raise sqlite3.OperationalError("interrupted")
            timer = threading.Timer(
                min(remaining_seconds, threading.TIMEOUT_MAX),
                interrupt_at_deadline,
            )
            timer.daemon = True
        timer.start()
        timer_started = True

        cursor = connection.execute(sql)
        if cursor.description is not None:
            column_count = len(cursor.description)
            rows = (
                cursor.fetchall()
                if max_rows is None
                else cursor.fetchmany(max_rows + 1)
            )
    except (sqlite3.Error, sqlite3.Warning) as error:
        caught_error = error
    finally:
        if timer_started:
            timer.cancel()
            timer.join()
        connection.set_progress_handler(None, 0)

    if denied_operation:
        return _execution_failure(
            "unsafe_sql",
            "unsafe_operation",
            str(caught_error) if caught_error is not None else denied_operation[0],
            started_at,
        )

    if isinstance(caught_error, sqlite3.Warning) or (
        caught_error is not None
        and "one statement at a time" in str(caught_error).lower()
    ):
        return _execution_failure(
            "unsafe_sql",
            "multiple_statements",
            str(caught_error),
            started_at,
        )

    if timeout_triggered.is_set() and (
        caught_error is None or "interrupted" in str(caught_error).lower()
    ):
        return _execution_failure(
            "timeout",
            "timeout",
            (
                str(caught_error)
                if caught_error is not None
                else f"SQL execution exceeded {timeout_seconds:g} seconds."
            ),
            started_at,
        )

    if caught_error is not None:
        return _execution_failure(
            "execution_error",
            _classify_sqlite_error(caught_error),
            str(caught_error),
            started_at,
        )

    if rows is None or column_count is None:
        return _execution_failure(
            "execution_error",
            "invalid_sql",
            "SQL must contain one read-only statement that returns a result set.",
            started_at,
        )

    if max_rows is not None and len(rows) > max_rows:
        return _execution_failure(
            "row_limit",
            "row_limit_exceeded",
            f"SQL result exceeds the configured limit of {max_rows} rows.",
            started_at,
            row_count=len(rows),
            column_count=column_count,
        )

    return {
        "status": "success",
        "elapsed_ms": _elapsed_ms(started_at),
        "rows": rows,
        "row_count": len(rows),
        "column_count": column_count,
        "empty_result": not rows,
        "error_type": None,
        "error_message": None,
    }


def execute_sql(
    db_path: str | Path,
    sql: str,
    timeout_seconds: float = 30.0,
    *,
    max_rows: int | None = None,
) -> dict:
    """Execute one read-only SQL statement and return a structured result.

    When ``max_rows`` is set, execution reads one row beyond the limit so an
    exactly-full result remains successful while a larger result is rejected.
    Omitting the cap preserves the evaluator's existing unbounded behavior.
    """
    if not isinstance(db_path, (str, Path)):
        raise TypeError(
            f"db_path must be a string or Path, got {db_path!r} "
            f"({type(db_path).__name__})."
        )
    if not isinstance(sql, str):
        raise TypeError(
            f"sql must be a string, got {sql!r} ({type(sql).__name__})."
        )
    normalized_timeout = _validate_timeout_seconds(timeout_seconds)
    normalized_max_rows = _validate_max_rows(max_rows)

    started_at = time.perf_counter()
    try:
        database_path = Path(db_path).resolve()
        database_exists = database_path.exists()
        database_is_file = database_path.is_file()
    except (OSError, RuntimeError, ValueError) as error:
        return _execution_failure(
            "execution_error",
            "database_open_error",
            str(error),
            started_at,
        )

    if not database_exists:
        return _execution_failure(
            "execution_error",
            "database_not_found",
            f"SQLite database does not exist: {database_path}.",
            started_at,
        )
    if not database_is_file:
        return _execution_failure(
            "execution_error",
            "database_open_error",
            f"SQLite database path is not a regular file: {database_path}.",
            started_at,
        )

    timeout_deadline = (
        time.perf_counter() + normalized_timeout
        if normalized_max_rows is not None
        else None
    )
    busy_timeout_seconds = min(
        _DEFAULT_BUSY_TIMEOUT_SECONDS,
        normalized_timeout,
    )
    try:
        with open_sqlite_read_only(
            database_path,
            busy_timeout_seconds=busy_timeout_seconds,
        ) as connection:
            return _execute_on_connection(
                connection,
                sql,
                normalized_timeout,
                started_at,
                max_rows=normalized_max_rows,
                timeout_deadline=timeout_deadline,
            )
    except sqlite3.Warning as error:
        if normalized_max_rows is not None:
            return _execution_failure(
                "unsafe_sql",
                "multiple_statements",
                str(error),
                started_at,
            )
        raise
    except sqlite3.Error as error:
        if normalized_max_rows is not None:
            return _execution_failure(
                "execution_error",
                _classify_sqlite_error(error),
                str(error),
                started_at,
            )
        return _execution_failure(
            "execution_error",
            "database_open_error",
            str(error),
            started_at,
        )
    except (OSError, ValueError) as error:
        return _execution_failure(
            "execution_error",
            "database_open_error",
            str(error),
            started_at,
        )


def _batch_input_error(
    split_name: str,
    message: str,
    *,
    source_index: int | None = None,
    db_id: object | None = None,
) -> ValueError:
    """Create a contextual error for invalid batch input data."""
    context = [f"split={split_name}"]
    if source_index is not None:
        context.append(f"source_index={source_index}")
    if db_id is not None:
        context.append(f"db_id={db_id!r}")
    return ValueError(f"Invalid gold SQL batch input ({', '.join(context)}): {message}")


def _load_schema_catalog_index(
    split_name: str,
    catalog_path: Path,
) -> dict[str, dict]:
    """Load and validate the execution fields in one Schema Catalog."""
    database_root = (
        PROJECT_ROOT / "data" / "raw" / "databases" / split_name
    ).resolve()
    catalog = load_json(catalog_path)
    if not isinstance(catalog, list):
        raise _batch_input_error(
            split_name,
            f"Schema Catalog must be a list, got {type(catalog).__name__}.",
        )

    catalog_by_db_id: dict[str, dict] = {}
    for catalog_index, database_schema in enumerate(catalog):
        if not isinstance(database_schema, dict):
            raise _batch_input_error(
                split_name,
                "Schema Catalog entry "
                f"{catalog_index} must be an object, got "
                f"{type(database_schema).__name__}.",
            )

        db_id = database_schema.get("db_id")
        sqlite_path = database_schema.get("sqlite_path")
        if not isinstance(db_id, str) or not db_id.strip():
            raise _batch_input_error(
                split_name,
                f"Schema Catalog entry {catalog_index} has an invalid db_id.",
                db_id=db_id,
            )
        if db_id in catalog_by_db_id:
            raise _batch_input_error(
                split_name,
                "Schema Catalog contains a duplicate db_id.",
                db_id=db_id,
            )
        if not isinstance(sqlite_path, str) or not sqlite_path.strip():
            raise _batch_input_error(
                split_name,
                f"Schema Catalog entry {catalog_index} has an invalid sqlite_path.",
                db_id=db_id,
            )

        relative_path = Path(sqlite_path)
        if relative_path.is_absolute():
            raise _batch_input_error(
                split_name,
                "Schema Catalog sqlite_path must be project-relative, "
                f"got {sqlite_path!r}.",
                db_id=db_id,
            )
        resolved_path = (PROJECT_ROOT / relative_path).resolve()
        try:
            resolved_path.relative_to(database_root)
        except (OSError, ValueError) as error:
            raise _batch_input_error(
                split_name,
                "Schema Catalog sqlite_path resolves outside its database split: "
                f"{sqlite_path!r}.",
                db_id=db_id,
            ) from error

        expected_path = (database_root / db_id / f"{db_id}.sqlite").resolve()
        if resolved_path != expected_path:
            raise _batch_input_error(
                split_name,
                "Schema Catalog sqlite_path does not match the expected database "
                f"path {expected_path}: {sqlite_path!r}.",
                db_id=db_id,
            )

        catalog_by_db_id[db_id] = database_schema

    return catalog_by_db_id


def _validate_annotation_sample(
    sample: object,
    split_name: str,
    source_index: int,
) -> dict:
    """Validate fields copied from one Train or Dev annotation."""
    if not isinstance(sample, dict):
        raise _batch_input_error(
            split_name,
            f"annotation must be an object, got {type(sample).__name__}.",
            source_index=source_index,
        )

    required_string_fields = ["db_id", "question", "evidence", "SQL"]
    if split_name == "dev":
        required_string_fields.append("difficulty")

    for field_name in required_string_fields:
        field_value = sample.get(field_name)
        if not isinstance(field_value, str):
            raise _batch_input_error(
                split_name,
                f"field {field_name!r} must be a string, got "
                f"{field_value!r} ({type(field_value).__name__}).",
                source_index=source_index,
                db_id=sample.get("db_id"),
            )

    if not sample["db_id"].strip():
        raise _batch_input_error(
            split_name,
            "field 'db_id' must not be empty.",
            source_index=source_index,
            db_id=sample["db_id"],
        )

    if split_name == "dev":
        question_id = sample.get("question_id")
        if type(question_id) is not int:
            raise _batch_input_error(
                split_name,
                "field 'question_id' must be an integer, got "
                f"{question_id!r} ({type(question_id).__name__}).",
                source_index=source_index,
                db_id=sample["db_id"],
            )
        if sample["difficulty"] not in _DEV_DIFFICULTIES:
            raise _batch_input_error(
                split_name,
                "field 'difficulty' must be 'simple', 'moderate', or "
                f"'challenging', got {sample['difficulty']!r}.",
                source_index=source_index,
                db_id=sample["db_id"],
            )

    return sample


def _validate_annotation_collection(
    samples: list[dict],
    split_name: str,
) -> list[dict]:
    """Validate a complete annotation split before executing any SQL."""
    expected_count = _EXPECTED_ANNOTATION_COUNTS[split_name]
    if len(samples) != expected_count:
        raise _batch_input_error(
            split_name,
            f"expected {expected_count} annotation records, got {len(samples)}.",
        )

    validated_samples = [
        _validate_annotation_sample(sample, split_name, source_index)
        for source_index, sample in enumerate(samples)
    ]
    if split_name == "dev":
        question_ids = [sample["question_id"] for sample in validated_samples]
        if len(set(question_ids)) != len(question_ids):
            raise _batch_input_error(
                split_name,
                "annotation question_id values must be unique.",
            )
        expected_question_ids = set(range(expected_count))
        actual_question_ids = set(question_ids)
        if actual_question_ids != expected_question_ids:
            raise _batch_input_error(
                split_name,
                "annotation question_id values must be exactly 0 through "
                f"{expected_count - 1}; missing="
                f"{sorted(expected_question_ids - actual_question_ids)!r}, "
                f"unexpected={sorted(actual_question_ids - expected_question_ids)!r}.",
            )

    return validated_samples


def _synthetic_execution(error_type: str, error_message: str) -> dict:
    """Return a zero-duration execution failure for a missing prerequisite."""
    return {
        "status": "execution_error",
        "elapsed_ms": 0.0,
        "rows": None,
        "row_count": 0,
        "column_count": 0,
        "empty_result": False,
        "error_type": error_type,
        "error_message": error_message,
    }


def _execution_without_rows(execution: dict) -> dict:
    """Return the JSONL-safe execution summary in its stable field order."""
    return {
        "status": execution["status"],
        "elapsed_ms": execution["elapsed_ms"],
        "row_count": execution["row_count"],
        "column_count": execution["column_count"],
        "empty_result": execution["empty_result"],
        "error_type": execution["error_type"],
        "error_message": execution["error_message"],
    }


def _resolve_catalog_database_path(sqlite_path: str) -> Path:
    """Resolve a catalog SQLite path relative to the project root."""
    return (PROJECT_ROOT / sqlite_path).resolve()


def _build_batch_record(
    sample: dict,
    split_name: str,
    source_index: int,
    catalog_by_db_id: dict[str, dict],
    timeout_seconds: float,
) -> dict:
    """Execute one annotation's gold SQL and build its persisted record."""
    db_id = sample["db_id"]
    database_schema = catalog_by_db_id.get(db_id)
    database_available = False

    if database_schema is None:
        sqlite_path: str | None = None
        execution = _synthetic_execution(
            "missing_schema",
            f"No Schema Catalog entry exists for db_id={db_id!r}.",
        )
    else:
        sqlite_path = database_schema["sqlite_path"]
        database_path = _resolve_catalog_database_path(sqlite_path)
        database_available = database_path.is_file()
        if not database_available:
            execution = _synthetic_execution(
                "database_not_found",
                f"SQLite database does not exist: {database_path}.",
            )
        else:
            execution = execute_sql(
                database_path,
                sample["SQL"],
                timeout_seconds,
            )

    record = {
        "sample_id": f"{split_name}_{source_index:06d}",
        "source_index": source_index,
    }
    if split_name == "dev":
        record["question_id"] = sample["question_id"]
        record["difficulty"] = sample["difficulty"]
    record.update(
        {
            "db_id": db_id,
            "question": sample["question"],
            "evidence": sample["evidence"],
            "gold_sql": sample["SQL"],
            "sqlite_path": sqlite_path,
            "gold_execution": _execution_without_rows(execution),
            "usable": (
                database_available and execution["status"] == "success"
            ),
        }
    )
    return record


def _publish_jsonl_atomically(records: list[dict], output_path: Path) -> list[dict]:
    """Write, verify, and atomically publish one JSONL artifact with rollback."""
    temporary_path = Path(f"{output_path}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_symlink() or (
        output_path.exists() and not output_path.is_file()
    ):
        raise ValueError(
            "Gold SQL execution output must be a regular, non-symlink file "
            f"when it already exists, got {output_path}."
        )

    try:
        temporary_path.unlink(missing_ok=True)
        save_jsonl(records, temporary_path)
        staged_records = load_jsonl(temporary_path)
        if staged_records != records:
            raise RuntimeError(
                "Staged gold SQL execution JSONL does not match in-memory records."
            )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    try:
        previous_content = output_path.read_bytes()
    except FileNotFoundError:
        previous_content = None
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    replaced = False
    try:
        temporary_path.replace(output_path)
        replaced = True
        persisted_records = load_jsonl(output_path)
        if persisted_records != records:
            raise RuntimeError(
                "Published gold SQL execution JSONL does not match in-memory records."
            )
    except BaseException as publication_error:
        rollback_error: OSError | None = None
        if replaced:
            try:
                if previous_content is None:
                    output_path.unlink(missing_ok=True)
                else:
                    temporary_path.write_bytes(previous_content)
                    temporary_path.replace(output_path)
            except OSError as error:
                rollback_error = error

        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if rollback_error is None:
                rollback_error = cleanup_error

        if rollback_error is not None:
            raise RuntimeError(
                "Gold SQL execution report publication failed and rollback was "
                f"incomplete. Publication error: {publication_error}. "
                f"Rollback error: {rollback_error}."
            ) from publication_error
        if not isinstance(publication_error, Exception):
            raise
        raise RuntimeError(
            "Gold SQL execution report publication failed; the previous output "
            f"was restored. Publication error: {publication_error}."
        ) from publication_error

    return persisted_records


def execute_gold_sql_batch(
    timeout_seconds: float = 30.0,
    output_path: str | Path = GOLD_SQL_EXECUTION_PATH,
) -> list[dict]:
    """Execute all Train and Dev gold SQL and atomically save summaries."""
    normalized_timeout = _validate_timeout_seconds(timeout_seconds)
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            f"output_path must be a string or Path, got {output_path!r} "
            f"({type(output_path).__name__})."
        )

    split_sources = {
        "train": (TRAIN_ANNOTATION_PATH, TRAIN_SCHEMA_CATALOG_PATH),
        "dev": (DEV_ANNOTATION_PATH, DEV_SCHEMA_CATALOG_PATH),
    }
    prepared_inputs: dict[str, tuple[list[dict], dict[str, dict]]] = {}

    for split_name, (annotation_path, catalog_path) in split_sources.items():
        samples = load_jsonl(annotation_path)
        validated_samples = _validate_annotation_collection(samples, split_name)
        catalog_by_db_id = _load_schema_catalog_index(split_name, catalog_path)
        prepared_inputs[split_name] = (validated_samples, catalog_by_db_id)

    records = []
    for split_name in ("train", "dev"):
        samples, catalog_by_db_id = prepared_inputs[split_name]
        for source_index, sample in enumerate(samples):
            records.append(
                _build_batch_record(
                    sample,
                    split_name,
                    source_index,
                    catalog_by_db_id,
                    normalized_timeout,
                )
            )

    return _publish_jsonl_atomically(records, Path(output_path).resolve())


def main() -> None:
    """Run the gold SQL batch executor from the command line."""
    parser = argparse.ArgumentParser(
        description="Execute all Train and Dev gold SQL in read-only mode."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum execution time for each SQL statement (default: 30).",
    )
    arguments = parser.parse_args()

    records = execute_gold_sql_batch(arguments.timeout_seconds)
    status_counts = Counter(
        record["gold_execution"]["status"] for record in records
    )
    usable_count = sum(record["usable"] for record in records)

    print(f"Gold SQL execution completed: {len(records)} total samples")
    print(f"usable={usable_count}")
    for status in ("success", "timeout", "execution_error", "unsafe_sql"):
        print(f"{status}={status_counts.get(status, 0)}")
    print(f"output={GOLD_SQL_EXECUTION_PATH}")


if __name__ == "__main__":
    main()
