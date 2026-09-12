import pytest

from app.backend.response_parser import extract_reasoning_sql


@pytest.mark.parametrize(
    ("raw_output", "reasoning", "sql", "status", "compliant"),
    [
        (
            "<think>Use the users table.</think>\n```sql\nSELECT * FROM users\n```",
            "Use the users table.",
            "SELECT * FROM users",
            "reasoning_sql_fence",
            True,
        ),
        ("```sql\nSELECT 1\n```", "", "SELECT 1", "sql_fence", False),
        ("```\nSELECT 1\n```", "", "SELECT 1", "generic_fence", False),
        ("SELECT 1", "", "SELECT 1", "raw_sql", False),
        (
            "WITH answer AS (SELECT 1) SELECT * FROM answer",
            "",
            "WITH answer AS (SELECT 1) SELECT * FROM answer",
            "raw_sql",
            False,
        ),
        ("Here is the answer: SELECT 1", "", "", "failed", False),
        ("", "", "", "failed", False),
    ],
)
def test_response_parser_matches_v2_semantics(
    raw_output,
    reasoning,
    sql,
    status,
    compliant,
):
    assert extract_reasoning_sql(raw_output) == {
        "reasoning": reasoning,
        "predicted_sql": sql,
        "extraction_status": status,
        "format_compliance": compliant,
    }
