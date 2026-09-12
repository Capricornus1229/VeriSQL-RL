import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def open_sqlite_read_only(
    db_path: str | Path,
    *,
    busy_timeout_seconds: float = 5.0,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database with a read-only, query-only connection."""
    if isinstance(busy_timeout_seconds, bool) or not isinstance(
        busy_timeout_seconds,
        (int, float),
    ):
        raise TypeError(
            "busy_timeout_seconds must be a finite positive number, got "
            f"{busy_timeout_seconds!r} ({type(busy_timeout_seconds).__name__})."
        )
    try:
        normalized_timeout = float(busy_timeout_seconds)
    except OverflowError as error:
        raise ValueError(
            "busy_timeout_seconds must be a finite positive number, got "
            f"{busy_timeout_seconds!r}."
        ) from error
    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
        raise ValueError(
            "busy_timeout_seconds must be a finite positive number, got "
            f"{busy_timeout_seconds!r}."
        )

    database_path = Path(db_path).resolve()
    database_uri = f"{database_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(
        database_uri,
        uri=True,
        timeout=normalized_timeout,
        isolation_level=None,
    )

    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        yield connection
    finally:
        connection.close()


# JSON and JSONL file handling
def load_json(file_path: str | Path) -> Any:
    """Load a JSON file."""
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load a JSONL file and return a list of samples."""
    file_path = Path(file_path)
    samples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Line {line_number} in {file_path} is not valid JSON: {error}"
                ) from error

    return samples


def save_json(data: Any, save_path: str | Path) -> None:
    """Save data to a JSON file."""
    file_path = Path(save_path)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_jsonl(data: list[dict], save_path: str | Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    file_path = Path(save_path)

    with file_path.open("w", encoding="utf-8") as file:
        for sample in data:
            json_line = json.dumps(sample, ensure_ascii=False)
            file.write(json_line + "\n")
