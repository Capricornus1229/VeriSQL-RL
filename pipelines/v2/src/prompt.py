from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence


SYSTEM_PROMPT = """You are an expert SQLite query generator.

Given a database schema, optional evidence, relevant database information, and a question, reason briefly about tables, joins, filters, aggregation, and ordering, then generate one valid read-only SQLite query.

Return exactly one <think>...</think> block followed by one SQL code block and no other text."""

_DESCRIPTION_TABLE_ALIASES = {
    ("student_loan", "filed_for_bankrupcy"): "filed_for_bankruptcy",
}
_VALUE_TYPE_MARKERS = ("TEXT", "CHAR", "CLOB", "DATE", "DATETIME")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "or",
    "please",
    "show",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def render_schema(database_schema: dict[str, Any]) -> str:
    """Render the complete catalog schema without pruning tables or columns."""
    table_sections: list[str] = []
    for table in database_schema["tables"]:
        table_name = table["name"]
        header = f"TABLE {quote_identifier(table_name)}"
        if table.get("display_name") and table["display_name"] != table_name:
            header += f" -- {_one_line(table['display_name'])}"

        lines = [header]
        primary_keys = table["primary_keys"]
        for column in table["columns"]:
            column_name = column["name"]
            line = f"- {quote_identifier(column_name)} {str(column['type']).upper()}"
            if len(primary_keys) == 1 and primary_keys[0] == column_name:
                line += " [PRIMARY KEY]"
            if column.get("display_name") and column["display_name"] != column_name:
                line += f" -- {_one_line(column['display_name'])}"
            lines.append(line)

        if len(primary_keys) > 1:
            keys = ", ".join(quote_identifier(name) for name in primary_keys)
            lines.append(f"PRIMARY KEY ({keys})")
        table_sections.append("\n".join(lines))

    if database_schema["foreign_keys"]:
        foreign_keys = ["FOREIGN KEY"]
        for key in database_schema["foreign_keys"]:
            foreign_keys.append(
                f"{quote_identifier(key['source_table'])}."
                f"{quote_identifier(key['source_column'])} -> "
                f"{quote_identifier(key['target_table'])}."
                f"{quote_identifier(key['target_column'])}"
            )
        table_sections.append("\n".join(foreign_keys))

    return "\n\n".join(table_sections)


def _read_description_csv(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                rows: list[dict[str, str]] = []
                for source_row in reader:
                    row = {
                        str(key).strip(): _one_line(value)
                        for key, value in source_row.items()
                        if key is not None
                    }
                    rows.append(row)
                return rows
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not decode database description {path}: {last_error}")


def _description_file_for_table(
    db_id: str,
    table: dict[str, Any],
    parsed_files: list[tuple[Path, list[dict[str, str]], set[str]]],
    used_paths: set[Path],
) -> tuple[Path, list[dict[str, str]], set[str]]:
    alias = _DESCRIPTION_TABLE_ALIASES.get((db_id, table["name"]), table["name"])
    exact = [
        item
        for item in parsed_files
        if item[0] not in used_paths and item[0].stem.casefold() == alias.casefold()
    ]
    if len(exact) == 1:
        return exact[0]

    catalog_columns = {column["name"].casefold() for column in table["columns"]}
    scored = [
        (len(catalog_columns & item[2]), item)
        for item in parsed_files
        if item[0] not in used_paths
    ]
    best_score = max((score for score, _ in scored), default=0)
    best = [item for score, item in scored if score == best_score and score > 0]
    if len(best) != 1:
        raise ValueError(
            f"Could not uniquely map a database description CSV for "
            f"db_id={db_id!r}, table={table['name']!r}."
        )
    return best[0]


def load_database_descriptions(
    database_directory: str | Path,
    database_schema: dict[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    """Map BIRD database-description rows to Catalog table and column names."""
    description_directory = Path(database_directory) / "database_description"
    parsed_files: list[tuple[Path, list[dict[str, str]], set[str]]] = []
    for path in sorted(description_directory.glob("*.csv")):
        rows = _read_description_csv(path)
        original_names = {
            row.get("original_column_name", "").casefold()
            for row in rows
            if row.get("original_column_name")
        }
        parsed_files.append((path, rows, original_names))

    result: dict[str, dict[str, dict[str, str]]] = {}
    used_paths: set[Path] = set()
    db_id = database_schema["db_id"]
    for table in database_schema["tables"]:
        path, rows, _ = _description_file_for_table(
            db_id,
            table,
            parsed_files,
            used_paths,
        )
        used_paths.add(path)
        rows_by_column = {
            row.get("original_column_name", "").casefold(): row
            for row in rows
            if row.get("original_column_name")
        }
        table_result: dict[str, dict[str, str]] = {}
        for column in table["columns"]:
            row = rows_by_column.get(column["name"].casefold(), {})
            table_result[column["name"]] = {
                "column_description": row.get("column_description", ""),
                "value_description": row.get("value_description", ""),
                "data_format": row.get("data_format", ""),
            }
        result[table["name"]] = table_result
    return result


def _is_value_column(declared_type: object) -> bool:
    normalized = str(declared_type or "").upper()
    return any(marker in normalized for marker in _VALUE_TYPE_MARKERS)


def _normalize_value(value: object, max_value_chars: int) -> str | None:
    if isinstance(value, bytes):
        return None
    normalized = _one_line(value)
    if not normalized or len(normalized) > max_value_chars:
        return None
    return normalized


def _column_search_text(
    table: dict[str, Any],
    column: dict[str, Any],
    description: dict[str, str],
) -> str:
    return " ".join(
        part
        for part in (
            table["name"],
            table.get("display_name", ""),
            column["name"],
            column.get("display_name", ""),
            description["column_description"],
            description["value_description"],
            description["data_format"],
        )
        if part
    )


def build_value_index(
    catalogs_by_split: dict[str, list[dict[str, Any]]],
    database_roots: dict[str, Path],
    index_path: str | Path,
    *,
    scan_rows_per_column: int,
    max_unique_values_per_column: int,
    max_value_chars: int,
) -> dict[str, int]:
    """Build the V2 FTS5 column/value index from read-only BIRD databases."""
    destination = Path(index_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)

    stats = {
        "database_count": 0,
        "column_count": 0,
        "value_column_count": 0,
        "scanned_value_count": 0,
        "indexed_unique_value_count": 0,
    }
    index_connection: sqlite3.Connection | None = None
    try:
        index_connection = sqlite3.connect(temporary)
        index_connection.execute("PRAGMA journal_mode = DELETE")
        index_connection.execute(
            """
            CREATE TABLE column_info (
                split TEXT NOT NULL,
                db_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                table_order INTEGER NOT NULL,
                column_order INTEGER NOT NULL,
                column_description TEXT NOT NULL,
                value_description TEXT NOT NULL,
                data_format TEXT NOT NULL,
                PRIMARY KEY (split, db_id, table_name, column_name)
            )
            """
        )
        index_connection.execute(
            """
            CREATE VIRTUAL TABLE grounding_fts USING fts5(
                split UNINDEXED,
                db_id UNINDEXED,
                table_name UNINDEXED,
                column_name UNINDEXED,
                row_kind UNINDEXED,
                value UNINDEXED,
                search_text,
                tokenize = 'unicode61'
            )
            """
        )

        for split, catalog in catalogs_by_split.items():
            database_root = Path(database_roots[split])
            for database_schema in catalog:
                stats["database_count"] += 1
                db_id = database_schema["db_id"]
                database_directory = database_root / db_id
                descriptions = load_database_descriptions(
                    database_directory,
                    database_schema,
                )
                sqlite_path = database_directory / f"{db_id}.sqlite"
                uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
                with sqlite3.connect(uri, uri=True, isolation_level=None) as source:
                    source.execute("PRAGMA query_only = ON")
                    for table_order, table in enumerate(database_schema["tables"]):
                        table_name = table["name"]
                        for column_order, column in enumerate(table["columns"]):
                            stats["column_count"] += 1
                            column_name = column["name"]
                            description = descriptions[table_name][column_name]
                            index_connection.execute(
                                """
                                INSERT INTO column_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    split,
                                    db_id,
                                    table_name,
                                    column_name,
                                    table_order,
                                    column_order,
                                    description["column_description"],
                                    description["value_description"],
                                    description["data_format"],
                                ),
                            )
                            index_connection.execute(
                                "INSERT INTO grounding_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    split,
                                    db_id,
                                    table_name,
                                    column_name,
                                    "metadata",
                                    "",
                                    _column_search_text(table, column, description),
                                ),
                            )

                            if not _is_value_column(column["type"]):
                                continue
                            stats["value_column_count"] += 1
                            sql = (
                                f"SELECT {quote_identifier(column_name)} "
                                f"FROM {quote_identifier(table_name)} "
                                f"WHERE {quote_identifier(column_name)} IS NOT NULL "
                                "LIMIT ?"
                            )
                            seen: set[str] = set()
                            for (raw_value,) in source.execute(
                                sql,
                                (scan_rows_per_column,),
                            ):
                                stats["scanned_value_count"] += 1
                                value = _normalize_value(raw_value, max_value_chars)
                                if value is None or value.casefold() in seen:
                                    continue
                                seen.add(value.casefold())
                                index_connection.execute(
                                    "INSERT INTO grounding_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        split,
                                        db_id,
                                        table_name,
                                        column_name,
                                        "value",
                                        value,
                                        value,
                                    ),
                                )
                                stats["indexed_unique_value_count"] += 1
                                if len(seen) >= max_unique_values_per_column:
                                    break

        index_connection.execute(
            "INSERT INTO grounding_fts(grounding_fts) VALUES('optimize')"
        )
        index_connection.commit()
        index_connection.close()
        index_connection = None
        os.replace(temporary, destination)
    finally:
        if index_connection is not None:
            index_connection.close()
        temporary.unlink(missing_ok=True)
    return stats


def _fts_query(question: str, evidence: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\w+", f"{question} {evidence}".casefold()):
        token = token.strip("_")
        if not token or token in _QUERY_STOPWORDS or token in seen:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        seen.add(token)
        terms.append(token)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _render_grounding_line(
    table_name: str,
    column_name: str,
    column_description: str,
    value_description: str,
    matched_values: Sequence[str],
) -> str:
    line = f"- {quote_identifier(table_name)}.{quote_identifier(column_name)}"
    details: list[str] = []
    if column_description:
        details.append(column_description)
    if value_description and value_description != column_description:
        details.append(f"value notes: {value_description}")
    if matched_values:
        rendered_values = ", ".join(
            json.dumps(value, ensure_ascii=False) for value in matched_values
        )
        details.append(f"matched values: {rendered_values}")
    if details:
        line += ": " + " | ".join(details)
    return line


def retrieve_grounding(
    index_path: str | Path,
    *,
    split: str,
    db_id: str,
    question: str,
    evidence: str,
    top_columns: int,
    values_per_column: int,
) -> list[dict[str, Any]]:
    """Retrieve a small set of relevant columns and matched database values."""
    match_query = _fts_query(question, evidence)
    if not match_query:
        return []

    uri = f"{Path(index_path).resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT table_name, column_name, row_kind, value,
                   bm25(grounding_fts) AS score
            FROM grounding_fts
            WHERE grounding_fts MATCH ? AND split = ? AND db_id = ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, split, db_id, max(256, top_columns * 64)),
        ).fetchall()

        ranked_columns: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["table_name"], row["column_name"])
            hit = ranked_columns.setdefault(
                key,
                {"score": float(row["score"]), "values": []},
            )
            hit["score"] = min(hit["score"], float(row["score"]))
            value = row["value"]
            if (
                row["row_kind"] == "value"
                and value
                and value not in hit["values"]
                and len(hit["values"]) < values_per_column
            ):
                hit["values"].append(value)

        selected = sorted(
            ranked_columns,
            key=lambda key: (ranked_columns[key]["score"], key[0], key[1]),
        )[:top_columns]
        results: list[dict[str, Any]] = []
        for table_name, column_name in selected:
            info = connection.execute(
                """
                SELECT column_description, value_description, data_format
                FROM column_info
                WHERE split = ? AND db_id = ? AND table_name = ? AND column_name = ?
                """,
                (split, db_id, table_name, column_name),
            ).fetchone()
            values = ranked_columns[(table_name, column_name)]["values"]
            line = _render_grounding_line(
                table_name,
                column_name,
                info["column_description"],
                info["value_description"],
                values,
            )
            results.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "description": info["column_description"],
                    "value_description": info["value_description"],
                    "data_format": info["data_format"],
                    "matched_values": values,
                    "line": line,
                }
            )
    return results


def build_prompt(
    schema_text: str,
    question: str,
    evidence: str,
    relevant_database_lines: Sequence[str],
) -> list[dict[str, str]]:
    """Build the shared V2 prompt without accepting Gold SQL as an input."""
    sections = [f"Database schema:\n{schema_text}"]
    if relevant_database_lines:
        sections.append(
            "Relevant database information:\n"
            + "\n".join(relevant_database_lines)
        )
    if evidence.strip():
        sections.append(f"Evidence:\n{evidence.strip()}")
    sections.append(f"Question:\n{question.strip()}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
