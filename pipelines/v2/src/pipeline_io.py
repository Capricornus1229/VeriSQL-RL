from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the V2 YAML configuration."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"V2 config must be a YAML object: {config_path}")
    return config


def resolve_project_path(value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_pipeline_path(value: str | Path) -> Path:
    """Resolve a writable path and require it to remain inside the V2 folder."""
    path = Path(value).expanduser()
    resolved = (
        path.resolve() if path.is_absolute() else (PIPELINE_ROOT / path).resolve()
    )
    try:
        resolved.relative_to(PIPELINE_ROOT)
    except ValueError as error:
        raise ValueError(
            f"V2 output path must stay inside {PIPELINE_ROOT}: {resolved}"
        ) from error
    return resolved


def project_relative_path(value: str | Path) -> str:
    """Return a repository-local path suitable for persistent metadata."""
    resolved = resolve_project_path(value)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {resolved}") from error


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}, line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL records must be objects: {path}, line {line_number}"
                )
            records.append(record)
    return records


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = resolve_pipeline_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    destination = resolve_pipeline_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl_flush(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object and make the progress durable on disk."""
    destination = resolve_pipeline_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
