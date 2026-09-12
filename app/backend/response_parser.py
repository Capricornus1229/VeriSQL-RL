"""Deployment copy of the frozen V2 model-response parser semantics."""

from __future__ import annotations

import re
from typing import Any


_FENCE = "```"
_THINK_SQL_RE = re.compile(
    rf"\A<think>\s*(?P<reasoning>.*?)\s*</think>\s*"
    rf"{re.escape(_FENCE)}sql[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SQL_FENCE_RE = re.compile(
    rf"\A{re.escape(_FENCE)}sql[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_FENCE_RE = re.compile(
    rf"\A{re.escape(_FENCE)}[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.DOTALL,
)
_RAW_SQL_RE = re.compile(r"\A(?:SELECT|WITH)\b", re.IGNORECASE)


def extract_reasoning_sql(raw_output: str) -> dict[str, Any]:
    """Extract complete supported output forms without repairing model SQL."""
    stripped = raw_output.strip()
    match = _THINK_SQL_RE.fullmatch(stripped)
    if match:
        reasoning = match.group("reasoning").strip()
        sql = match.group("sql").strip()
        if reasoning and sql and _FENCE not in reasoning and _FENCE not in sql:
            return {
                "reasoning": reasoning,
                "predicted_sql": sql,
                "extraction_status": "reasoning_sql_fence",
                "format_compliance": True,
            }

    match = _SQL_FENCE_RE.fullmatch(stripped)
    if match:
        sql = match.group("sql").strip()
        if sql and _FENCE not in sql:
            return {
                "reasoning": "",
                "predicted_sql": sql,
                "extraction_status": "sql_fence",
                "format_compliance": False,
            }

    match = _GENERIC_FENCE_RE.fullmatch(stripped)
    if match:
        sql = match.group("sql").strip()
        if sql and _FENCE not in sql:
            return {
                "reasoning": "",
                "predicted_sql": sql,
                "extraction_status": "generic_fence",
                "format_compliance": False,
            }

    if _RAW_SQL_RE.match(stripped):
        return {
            "reasoning": "",
            "predicted_sql": stripped,
            "extraction_status": "raw_sql",
            "format_compliance": False,
        }
    return {
        "reasoning": "",
        "predicted_sql": "",
        "extraction_status": "failed",
        "format_compliance": False,
    }
