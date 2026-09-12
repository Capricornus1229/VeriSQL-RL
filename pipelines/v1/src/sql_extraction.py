from __future__ import annotations

import re


EXTRACTION_STATUSES = {
    "sql_fence",
    "generic_fence",
    "raw_sql",
    "failed",
}

SQL_FENCE_PATTERN = re.compile(
    r"\A```sql[ \t]*\r?\n(?P<sql>.*?)\r?\n```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
GENERIC_FENCE_PATTERN = re.compile(
    r"\A```[ \t]*\r?\n(?P<sql>.*?)\r?\n```\Z",
    flags=re.DOTALL,
)
RAW_SQL_PATTERN = re.compile(r"\A(?:SELECT|WITH)\b", flags=re.IGNORECASE)


def extract_predicted_sql(raw_output: str) -> tuple[str, str, bool]:
    """Extract one SQL statement using the project's conservative rules."""
    if not isinstance(raw_output, str):
        raise TypeError(
            f"raw_output must be a string, got {raw_output!r} "
            f"({type(raw_output).__name__})."
        )

    stripped_output = raw_output.strip()
    for pattern, status, format_compliance in (
        (SQL_FENCE_PATTERN, "sql_fence", True),
        (GENERIC_FENCE_PATTERN, "generic_fence", False),
    ):
        match = pattern.fullmatch(stripped_output)
        if match is None:
            continue
        sql = match.group("sql")
        if "```" not in sql and sql.strip():
            return sql.strip(), status, format_compliance
        return "", "failed", False

    if RAW_SQL_PATTERN.match(stripped_output):
        return stripped_output, "raw_sql", False
    return "", "failed", False
