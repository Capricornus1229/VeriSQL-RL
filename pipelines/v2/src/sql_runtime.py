"""SQLite execution, screening scores, advantages, and result voting."""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_ALLOWED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_RECURSIVE,
    sqlite3.SQLITE_SELECT,
}
_UNSAFE_FUNCTIONS = {"edit", "load_extension", "readfile", "writefile"}
_PROGRESS_HANDLER_INSTRUCTIONS = 1_000

DEFAULT_SCREENING_PARTIAL_SCORE_CONFIG = {
    "executable": 0.10,
    "table_f1": 0.10,
    "column_f1": 0.15,
    "column_count_match": 0.05,
    "row_count_match": 0.05,
    "result_jaccard": 0.15,
    "incorrect_cap": 0.55,
}


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _failure(
    *,
    status: str,
    error_type: str,
    error_message: str,
    started_at: float,
    accessed_tables: Sequence[str] = (),
    accessed_columns: Sequence[tuple[str, str]] = (),
    row_count: int = 0,
    column_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "rows": None,
        "row_count": row_count,
        "column_count": column_count,
        "empty_result": False,
        "elapsed_ms": _elapsed_ms(started_at),
        "error_type": error_type,
        "error_message": error_message,
        "accessed_tables": list(accessed_tables),
        "accessed_columns": [
            {"table": table, "column": column}
            for table, column in accessed_columns
        ],
    }


def _sqlite_error_type(error: sqlite3.Error) -> str:
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
    if any(
        marker in message
        for marker in ("syntax error", "incomplete input", "unrecognized token", "near ")
    ):
        return "syntax_error"
    if any(
        marker in message
        for marker in (
            "unable to open database",
            "file is not a database",
            "database disk image is malformed",
        )
    ):
        return "database_open_error"
    return "sqlite_error"


def execute_sql_capped(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 5.0,
    max_rows: int | None = 10_000,
) -> dict[str, Any]:
    """Execute one read-only query and record the real tables/columns it reads."""
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise TypeError("timeout_seconds must be a positive number.")
    if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number.")
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer or None.")
    if not isinstance(sql, str):
        raise TypeError("sql must be a string.")

    started_at = time.perf_counter()
    database_path = Path(db_path).resolve()
    if not database_path.is_file():
        error_type = "database_not_found" if not database_path.exists() else "database_open_error"
        return _failure(
            status="execution_error",
            error_type=error_type,
            error_message=f"SQLite database is not a file: {database_path}.",
            started_at=started_at,
        )

    timeout = float(timeout_seconds)
    deadline = time.perf_counter() + timeout
    timeout_triggered = threading.Event()
    denied_operation: list[str] = []
    tables: list[str] = []
    columns: list[tuple[str, str]] = []
    seen_tables: set[str] = set()
    seen_columns: set[tuple[str, str]] = set()
    connection: sqlite3.Connection | None = None
    timer: threading.Timer | None = None

    def authorize(
        action_code: int,
        first_argument: str | None,
        second_argument: str | None,
        database_name: str | None,
        trigger_or_view: str | None,
    ) -> int:
        del trigger_or_view
        if action_code == sqlite3.SQLITE_READ and first_argument:
            if not first_argument.casefold().startswith("sqlite_"):
                table_key = first_argument.casefold()
                if table_key not in seen_tables:
                    seen_tables.add(table_key)
                    tables.append(first_argument)
                if second_argument:
                    column_key = (table_key, second_argument.casefold())
                    if column_key not in seen_columns:
                        seen_columns.add(column_key)
                        columns.append((first_argument, second_argument))

        function_name = second_argument if action_code == sqlite3.SQLITE_FUNCTION else None
        unsafe_function = (
            isinstance(function_name, str)
            and function_name.casefold() in _UNSAFE_FUNCTIONS
        )
        if action_code in _ALLOWED_AUTHORIZER_ACTIONS and not unsafe_function:
            return sqlite3.SQLITE_OK

        if not denied_operation:
            denied_operation.append(
                f"Rejected SQLite operation {action_code} "
                f"({first_argument!r}, {second_argument!r}) on {database_name!r}."
            )
        return sqlite3.SQLITE_DENY

    def check_deadline() -> int:
        if time.perf_counter() >= deadline:
            timeout_triggered.set()
            return 1
        return 0

    def interrupt() -> None:
        timeout_triggered.set()
        if connection is not None:
            try:
                connection.interrupt()
            except sqlite3.Error:
                pass

    caught_error: sqlite3.Error | sqlite3.Warning | None = None
    rows: list[tuple[Any, ...]] | None = None
    column_count = 0
    try:
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=min(5.0, timeout),
        )
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(authorize)
        connection.set_progress_handler(check_deadline, _PROGRESS_HANDLER_INSTRUCTIONS)

        timer = threading.Timer(min(timeout, threading.TIMEOUT_MAX), interrupt)
        timer.daemon = True
        timer.start()

        cursor = connection.execute(sql)
        if cursor.description is not None:
            column_count = len(cursor.description)
            rows = cursor.fetchall() if max_rows is None else cursor.fetchmany(max_rows + 1)
    except (sqlite3.Error, sqlite3.Warning) as error:
        caught_error = error
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        if connection is not None:
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)
            connection.close()

    failure_context = {
        "started_at": started_at,
        "accessed_tables": tables,
        "accessed_columns": columns,
    }
    if denied_operation:
        return _failure(
            status="unsafe_sql",
            error_type="unsafe_operation",
            error_message=str(caught_error) if caught_error else denied_operation[0],
            **failure_context,
        )
    if caught_error is not None and "one statement at a time" in str(caught_error).lower():
        return _failure(
            status="unsafe_sql",
            error_type="multiple_statements",
            error_message=str(caught_error),
            **failure_context,
        )
    if timeout_triggered.is_set() and (
        caught_error is None or "interrupted" in str(caught_error).lower()
    ):
        return _failure(
            status="timeout",
            error_type="timeout",
            error_message=str(caught_error) if caught_error else "SQL execution timed out.",
            **failure_context,
        )
    if caught_error is not None:
        return _failure(
            status="execution_error",
            error_type=_sqlite_error_type(caught_error),
            error_message=str(caught_error),
            **failure_context,
        )
    if rows is None:
        return _failure(
            status="execution_error",
            error_type="invalid_sql",
            error_message="SQL must be one read-only statement that returns rows.",
            **failure_context,
        )
    if max_rows is not None and len(rows) > max_rows:
        return _failure(
            status="row_limit",
            error_type="row_limit_exceeded",
            error_message=f"SQL result exceeds {max_rows} rows.",
            row_count=len(rows),
            column_count=column_count,
            **failure_context,
        )

    return {
        "status": "success",
        "rows": rows,
        "row_count": len(rows),
        "column_count": column_count,
        "empty_result": not rows,
        "elapsed_ms": _elapsed_ms(started_at),
        "error_type": None,
        "error_message": None,
        "accessed_tables": tables,
        "accessed_columns": [
            {"table": table, "column": column} for table, column in columns
        ],
    }


def _row_set(rows: Sequence[Sequence[Any]]) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in rows}


def official_execution_match(
    predicted_rows: Sequence[Sequence[Any]] | Mapping[str, Any],
    gold_rows: Sequence[Sequence[Any]] | Mapping[str, Any],
) -> int:
    """BIRD EX: row order and duplicate row counts are ignored."""
    if isinstance(predicted_rows, Mapping) or isinstance(gold_rows, Mapping):
        if not isinstance(predicted_rows, Mapping) or not isinstance(gold_rows, Mapping):
            raise TypeError("Compare either two execution results or two row sequences.")
        if predicted_rows.get("status") != "success" or gold_rows.get("status") != "success":
            return 0
        predicted_rows = predicted_rows["rows"]
        gold_rows = gold_rows["rows"]
    return int(_row_set(predicted_rows) == _row_set(gold_rows))


def _f1(predicted: set[Any], gold: set[Any]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def _access_sets(result: Mapping[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    tables = {
        str(table).casefold() for table in result.get("accessed_tables", [])
    }
    columns = {
        (str(item["table"]).casefold(), str(item["column"]).casefold())
        for item in result.get("accessed_columns", [])
    }
    return tables, columns


def _result_jaccard(
    predicted_rows: Sequence[Sequence[Any]],
    gold_rows: Sequence[Sequence[Any]],
) -> float:
    predicted = _row_set(predicted_rows)
    gold = _row_set(gold_rows)
    union = predicted | gold
    return len(predicted & gold) / len(union) if union else 1.0


def compute_screening_partial_score_details(
    prediction_result: Mapping[str, Any],
    gold_result: Mapping[str, Any],
    *,
    score_config: Mapping[str, float] | None = None,
) -> dict[str, float | int]:
    """Return the partial score used only to find useful GRPO prompts."""
    if gold_result.get("status") != "success":
        raise ValueError("Gold execution must succeed before screening score computation.")

    weights = dict(DEFAULT_SCREENING_PARTIAL_SCORE_CONFIG)
    if score_config is not None:
        for name in weights:
            if name in score_config:
                weights[name] = float(score_config[name])

    if prediction_result.get("status") != "success":
        return {
            "screening_score": 0.0,
            "official_ex": 0,
            "table_f1": 0.0,
            "column_f1": 0.0,
            "column_count_match": 0,
            "row_count_match": 0,
            "result_jaccard": 0.0,
        }

    predicted_rows = prediction_result["rows"]
    gold_rows = gold_result["rows"]
    official_ex = official_execution_match(predicted_rows, gold_rows)
    predicted_tables, predicted_columns = _access_sets(prediction_result)
    gold_tables, gold_columns = _access_sets(gold_result)
    table_f1 = _f1(predicted_tables, gold_tables)
    column_f1 = _f1(predicted_columns, gold_columns)
    column_count_match = int(
        prediction_result.get("column_count") == gold_result.get("column_count")
    )
    row_count_match = int(
        prediction_result.get("row_count") == gold_result.get("row_count")
    )
    result_jaccard = (
        _result_jaccard(predicted_rows, gold_rows)
        if column_count_match
        else 0.0
    )

    if official_ex:
        score = 1.0
    else:
        score = (
            weights["executable"]
            + weights["table_f1"] * table_f1
            + weights["column_f1"] * column_f1
            + weights["column_count_match"] * column_count_match
            + weights["row_count_match"] * row_count_match
            + weights["result_jaccard"] * result_jaccard
        )
        score = min(score, weights["incorrect_cap"])

    return {
        "screening_score": round(float(score), 8),
        "official_ex": official_ex,
        "table_f1": round(table_f1, 8),
        "column_f1": round(column_f1, 8),
        "column_count_match": column_count_match,
        "row_count_match": row_count_match,
        "result_jaccard": round(result_jaccard, 8),
    }


def compute_screening_partial_score(
    prediction_result: Mapping[str, Any],
    gold_result: Mapping[str, Any],
    *,
    score_config: Mapping[str, float] | None = None,
) -> float:
    return float(
        compute_screening_partial_score_details(
            prediction_result,
            gold_result,
            score_config=score_config,
        )["screening_score"]
    )


def has_screening_score_variance(
    scores: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Whether partial screening scores distinguish candidates in one group."""
    return bool(scores) and max(scores) - min(scores) > tolerance


def summarize_execution(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove potentially large rows while retaining diagnostics and access trace."""
    return {key: value for key, value in result.items() if key != "rows"}


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-4,
) -> list[float]:
    """Normalize one group by its sample standard deviation."""
    if len(rewards) < 2:
        raise ValueError("A reward group must contain at least two values.")
    values = [float(reward) for reward in rewards]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Rewards must be finite.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    denominator = math.sqrt(variance) + epsilon
    return [(value - mean) / denominator for value in values]


def _result_key(result: Mapping[str, Any]) -> frozenset[tuple[Any, ...]]:
    return frozenset(_row_set(result["rows"]))


def select_by_result_vote(
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_support: int = 2,
) -> dict[str, Any]:
    """Select without Gold using conservative execution-result voting.

    Each candidate must contain an ``execution`` result.  ``is_greedy`` and
    ``average_log_prob`` are optional.  A successful greedy candidate remains
    selected unless a unique result cluster reaches ``minimum_support``.
    """
    if type(minimum_support) is not int or minimum_support <= 0:
        raise ValueError("minimum_support must be a positive integer.")

    def log_prob(candidate: Mapping[str, Any]) -> float:
        value = candidate.get("average_log_prob")
        return float(value) if isinstance(value, (int, float)) else -math.inf

    successful = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.get("execution", {}).get("status") == "success"
    ]
    non_empty = [
        item for item in successful if item[1]["execution"].get("rows")
    ]
    voting_candidates = non_empty if non_empty else successful
    greedy = next(
        (
            (index, candidate)
            for index, candidate in voting_candidates
            if candidate.get("is_greedy") is True
        ),
        None,
    )
    if not successful:
        return {
            "selected_index": None,
            "selected_sql": "",
            "support": 0,
            "cluster_count": 0,
            "executable_count": 0,
            "used_vote": False,
            "reason": "no_executable_candidate",
        }

    clusters: dict[frozenset[tuple[Any, ...]], list[tuple[int, Mapping[str, Any]]]] = {}
    for item in voting_candidates:
        clusters.setdefault(_result_key(item[1]["execution"]), []).append(item)

    largest = max(len(cluster) for cluster in clusters.values())
    top_clusters = [cluster for cluster in clusters.values() if len(cluster) == largest]
    greedy_cluster = next(
        (
            cluster
            for cluster in clusters.values()
            if greedy is not None and any(index == greedy[0] for index, _ in cluster)
        ),
        None,
    )

    def cluster_log_prob(
        cluster: Sequence[tuple[int, Mapping[str, Any]]],
    ) -> float:
        scores = [log_prob(candidate) for _, candidate in cluster]
        return (
            sum(scores) / len(scores)
            if all(math.isfinite(score) for score in scores)
            else -math.inf
        )

    if largest >= minimum_support and len(top_clusters) == 1:
        chosen_cluster = top_clusters[0]
        used_vote = greedy is None or all(index != greedy[0] for index, _ in chosen_cluster)
        reason = "unique_result_majority"
    elif largest >= minimum_support and any(
        greedy_cluster is cluster for cluster in top_clusters
    ):
        chosen_cluster = greedy_cluster
        used_vote = False
        reason = "tied_majority_kept_greedy"
    elif largest >= minimum_support:
        chosen_cluster = max(top_clusters, key=cluster_log_prob)
        used_vote = True
        reason = "tied_majority_log_prob"
    elif greedy_cluster is not None:
        chosen_cluster = greedy_cluster
        used_vote = False
        reason = "kept_greedy_without_majority"
    else:
        chosen_cluster = max(top_clusters, key=cluster_log_prob)
        used_vote = False
        reason = "log_prob_fallback"

    chosen_index, chosen_candidate = max(
        chosen_cluster,
        key=lambda item: (
            item[1].get("is_greedy") is True,
            log_prob(item[1]),
            -item[0],
        ),
    )
    return {
        "selected_index": chosen_index,
        "selected_sql": str(chosen_candidate.get("predicted_sql", "")),
        "support": len(chosen_cluster),
        "cluster_count": len(clusters),
        "executable_count": len(successful),
        "used_vote": used_vote,
        "reason": reason,
    }


def choose_execution_vote(
    candidates: Sequence[Mapping[str, Any]],
    minimum_support: int = 2,
    greedy_index: int = 0,
) -> dict[str, Any]:
    """Compatibility wrapper used by V2 inference and diagnostics.

    ``mean_logprob`` is the shared generator field; internally the selector
    calls it ``average_log_prob``.  ``greedy_index`` may point at a failed
    candidate, in which case the best executable fallback is selected.
    """
    prepared: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["is_greedy"] = index == greedy_index
        if "average_log_prob" not in item and "mean_logprob" in item:
            item["average_log_prob"] = item["mean_logprob"]
        prepared.append(item)
    result = select_by_result_vote(prepared, minimum_support=minimum_support)
    return {
        "chosen_index": result["selected_index"],
        "reason": result["reason"],
        "support": result["support"],
        "cluster_count": result["cluster_count"],
        "executable_count": result["executable_count"],
        "used_vote": result["used_vote"],
    }
