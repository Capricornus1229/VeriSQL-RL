"""Small read-only executor that adds column names for browser display."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.common import open_sqlite_read_only


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<BLOB: {len(value)} bytes>"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def execute_display(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float,
    max_rows: int,
    total_row_count: int,
) -> dict[str, Any]:
    """Re-run a V2-verified SQL query and retain only the display row budget."""
    started_at = time.perf_counter()
    timed_out = threading.Event()
    connection: sqlite3.Connection | None = None

    def interrupt() -> None:
        timed_out.set()
        if connection is not None:
            try:
                connection.interrupt()
            except sqlite3.Error:
                pass

    def check_deadline() -> int:
        if time.perf_counter() >= deadline:
            timed_out.set()
            return 1
        return 0

    timer = threading.Timer(timeout_seconds, interrupt)
    timer.daemon = True
    try:
        with open_sqlite_read_only(
            db_path,
            busy_timeout_seconds=min(timeout_seconds, 5.0),
        ) as connection:
            deadline = time.perf_counter() + timeout_seconds
            connection.set_progress_handler(check_deadline, 1_000)
            timer.start()
            cursor = connection.execute(sql)
            columns = [item[0] for item in (cursor.description or [])]
            fetched = cursor.fetchmany(max_rows + 1)
            rows = [
                [_json_value(value) for value in row]
                for row in fetched[:max_rows]
            ]
            result_truncated = total_row_count > max_rows or len(fetched) > max_rows
            connection.set_progress_handler(None, 0)
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": total_row_count,
            "displayed_row_count": len(rows),
            "result_truncated": result_truncated,
            "elapsed_ms": _elapsed_ms(started_at),
            "error_type": None,
            "error_message": None,
        }
    except sqlite3.Error as error:
        return {
            "status": "timeout" if timed_out.is_set() else "execution_error",
            "columns": [],
            "rows": None,
            "row_count": total_row_count,
            "displayed_row_count": 0,
            "result_truncated": total_row_count > max_rows,
            "elapsed_ms": _elapsed_ms(started_at),
            "error_type": "timeout" if timed_out.is_set() else "display_error",
            "error_message": str(error),
        }
    finally:
        timer.cancel()
        if timer.is_alive():
            timer.join()
