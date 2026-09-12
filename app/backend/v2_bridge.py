"""Thin adapters over the frozen V2 prompt, execution, and vote functions."""

from __future__ import annotations

from typing import Any

from pipelines.v2.src.prompt import build_prompt, render_schema, retrieve_grounding
from pipelines.v2.src.sql_runtime import (
    choose_execution_vote,
    execute_sql_capped,
    summarize_execution,
)


def retrieve_for_query(
    registry_entry: dict[str, Any],
    question: str,
    evidence: str | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grounding = config["grounding"]
    return retrieve_grounding(
        config["paths"]["grounding_index"],
        split=registry_entry["split"],
        db_id=registry_entry["db_id"],
        question=question,
        evidence=evidence or "",
        top_columns=grounding["top_columns"],
        values_per_column=grounding["values_per_column"],
    )


def build_v2_messages(
    registry_entry: dict[str, Any],
    question: str,
    evidence: str | None,
    grounding: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return build_prompt(
        registry_entry["schema_text"],
        question,
        evidence or "",
        [hit["line"] for hit in grounding],
    )


def execute_candidate(
    registry_entry: dict[str, Any],
    sql: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    execution = config["execution"]
    return execute_sql_capped(
        registry_entry["sqlite_path"],
        sql,
        timeout_seconds=execution["timeout_seconds"],
        max_rows=execution["vote_max_rows"],
    )


def select_accurate_candidate(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return choose_execution_vote(
        candidates,
        minimum_support=config["execution"]["minimum_vote_support"],
        greedy_index=0,
    )


__all__ = [
    "build_v2_messages",
    "execute_candidate",
    "render_schema",
    "retrieve_for_query",
    "select_accurate_candidate",
    "summarize_execution",
]
