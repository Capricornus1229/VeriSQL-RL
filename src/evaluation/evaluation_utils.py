from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.common import load_json, load_jsonl


SCORED_RESULTS_FILENAME = "scored_results.jsonl"
METRICS_FILENAME = "metrics.json"


def load_strict_jsonl(file_path: str | Path, *, description: str) -> list[dict]:
    """Load a JSONL file while requiring every parsed record to be an object."""
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"{description} must be an existing file, got {path}.")

    try:
        records = load_jsonl(path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not load {description} at {path}: {error}.") from error

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{description} record {record_index} must be a JSON object, "
                f"got {type(record).__name__}: {record!r}."
            )

    return records


def load_json_file(file_path: str | Path, *, description: str) -> Any:
    """Load one UTF-8 JSON document with contextual errors."""
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"{description} must be an existing file, got {path}.")

    try:
        return load_json(path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not load {description} at {path}: {error}.") from error


def validate_timeout_seconds(timeout_seconds: float) -> float:
    """Require a finite positive timeout and return it as a float."""
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise ValueError(
            "timeout_seconds must be a finite positive number, "
            f"got {timeout_seconds!r}."
        )
    try:
        normalized_timeout = float(timeout_seconds)
    except OverflowError as error:
        raise ValueError(
            "timeout_seconds must be a finite positive number, "
            f"got {timeout_seconds!r}."
        ) from error
    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
        raise ValueError(
            "timeout_seconds must be a finite positive number, "
            f"got {timeout_seconds!r}."
        )
    return normalized_timeout


def validate_run_name(run_name: str) -> str:
    """Require a non-empty, single-component run directory name."""
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f"run_name must be a non-empty string, got {run_name!r}.")
    if (
        run_name in {".", ".."}
        or "/" in run_name
        or "\\" in run_name
        or "\x00" in run_name
        or Path(run_name).is_absolute()
    ):
        raise ValueError(
            "run_name must be a safe single directory name and cannot be an "
            f"absolute path, '.', '..', or contain '/' or '\\': {run_name!r}."
        )
    return run_name


def percentage(numerator: int, denominator: int) -> float:
    """Return a percentage on the 0-100 scale rounded to two decimals."""
    if denominator == 0:
        return 0.0
    return round(numerator * 100.0 / denominator, 2)


def official_execution_match(
    predicted_rows: list[tuple],
    gold_rows: list[tuple],
) -> int:
    """Apply BIRD's official execution-match rule, which ignores row order."""
    return int(set(predicted_rows) == set(gold_rows))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _remove_files(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"{path}: {error}")
    return errors


def _remove_new_empty_directory(path: Path, *, existed_before: bool) -> None:
    if existed_before:
        return
    try:
        path.rmdir()
    except OSError:
        pass


def publish_evaluation_outputs(
    *,
    report_root: str | Path,
    run_name: str,
    scored_results: list[dict],
    metrics: dict,
) -> dict:
    """Validate, stage, publish, and round-trip verify both evaluation files."""
    safe_run_name = validate_run_name(run_name)
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()

    run_directory = root / safe_run_name
    resolved_run_directory = run_directory.resolve()
    if resolved_run_directory.parent != resolved_root:
        raise ValueError(
            f"run_name resolves outside the evaluation report root: {run_name!r}."
        )
    if run_directory.is_symlink() or (
        run_directory.exists() and not run_directory.is_dir()
    ):
        raise ValueError(
            "The evaluation run path must be a real directory when it already "
            f"exists, got {run_directory}."
        )

    run_directory_existed = run_directory.exists()
    run_directory.mkdir(exist_ok=True)

    token = uuid4().hex
    scored_path = run_directory / SCORED_RESULTS_FILENAME
    metrics_path = run_directory / METRICS_FILENAME
    scored_temporary_path = run_directory / f".{SCORED_RESULTS_FILENAME}.{token}.tmp"
    metrics_temporary_path = run_directory / f".{METRICS_FILENAME}.{token}.tmp"
    temporary_paths = [scored_temporary_path, metrics_temporary_path]

    final_paths = [scored_path, metrics_path]
    invalid_final_paths = [
        path
        for path in final_paths
        if path.is_symlink() or (path.exists() and not path.is_file())
    ]
    if invalid_final_paths:
        _remove_new_empty_directory(
            run_directory,
            existed_before=run_directory_existed,
        )
        raise ValueError(
            "Evaluation output paths must be regular, non-symlink files when "
            f"they already exist: {[str(path) for path in invalid_final_paths]!r}."
        )

    try:
        _write_jsonl(scored_temporary_path, scored_results)
        _write_json(metrics_temporary_path, metrics)

        staged_scored_results = load_strict_jsonl(
            scored_temporary_path,
            description="staged scored results",
        )
        staged_metrics = load_json_file(
            metrics_temporary_path,
            description="staged evaluation metrics",
        )
        if staged_scored_results != scored_results:
            raise RuntimeError(
                "Staged scored results do not match the in-memory results."
            )
        if staged_metrics != metrics:
            raise RuntimeError("Staged metrics do not match the in-memory metrics.")
    except BaseException:
        _remove_files(temporary_paths)
        _remove_new_empty_directory(
            run_directory,
            existed_before=run_directory_existed,
        )
        raise

    previous_contents: dict[Path, bytes | None] = {}
    try:
        for final_path in final_paths:
            try:
                previous_contents[final_path] = final_path.read_bytes()
            except FileNotFoundError:
                previous_contents[final_path] = None
    except BaseException:
        _remove_files(temporary_paths)
        _remove_new_empty_directory(
            run_directory,
            existed_before=run_directory_existed,
        )
        raise

    replaced_paths: list[tuple[Path, Path]] = []
    try:
        for temporary_path, final_path in zip(temporary_paths, final_paths):
            temporary_path.replace(final_path)
            replaced_paths.append((temporary_path, final_path))

        published_scored_results = load_strict_jsonl(
            scored_path,
            description="published scored results",
        )
        published_metrics = load_json_file(
            metrics_path,
            description="published evaluation metrics",
        )
        if published_scored_results != scored_results:
            raise RuntimeError(
                "Published scored results do not match the in-memory results."
            )
        if published_metrics != metrics:
            raise RuntimeError("Published metrics do not match the in-memory metrics.")
    except BaseException as publication_error:
        rollback_errors: list[str] = []
        recovery_paths: list[Path] = []

        for _, final_path in reversed(replaced_paths):
            previous_content = previous_contents[final_path]
            try:
                if previous_content is None:
                    final_path.unlink(missing_ok=True)
                else:
                    recovery_path = run_directory / (
                        f".{final_path.name}.{token}.rollback.tmp"
                    )
                    recovery_paths.append(recovery_path)
                    with recovery_path.open("xb", buffering=0) as recovery_file:
                        recovery_file.write(previous_content)
                        os.fsync(recovery_file.fileno())
                    recovery_path.replace(final_path)
                    recovery_paths.remove(recovery_path)
            except (OSError, ValueError) as rollback_error:
                rollback_errors.append(
                    f"Could not restore {final_path}: {rollback_error}."
                )

        cleanup_errors = _remove_files(temporary_paths + recovery_paths)
        rollback_errors.extend(cleanup_errors)
        _remove_new_empty_directory(
            run_directory,
            existed_before=run_directory_existed,
        )

        if rollback_errors:
            raise RuntimeError(
                "Evaluation output publication failed and rollback was "
                f"incomplete. Publication error: {publication_error}. "
                f"Rollback errors: {rollback_errors!r}."
            ) from publication_error
        if not isinstance(publication_error, Exception):
            raise
        raise RuntimeError(
            "Evaluation output publication failed; previous output files were "
            f"restored. Publication error: {publication_error}."
        ) from publication_error

    return published_metrics
