from __future__ import annotations

from pathlib import Path

from src.common import load_json, load_jsonl, save_jsonl

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]

TRAIN_ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "annotations" / "train" / "bird23_train_filtered.jsonl"
)
DEV_ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "annotations" / "dev" / "bird_sql_dev_20251106.json"
)
TRAIN_SCHEMA_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "schema_catalog_train.json"
)
DEV_SCHEMA_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "interim" / "schema_catalog_dev.json"
)

SFT_TRAIN_PATH = PIPELINE_ROOT / "artifacts" / "data" / "sft_train.jsonl"
SFT_VAL_PATH = PIPELINE_ROOT / "artifacts" / "data" / "sft_val.jsonl"
DEV_EVAL_PATH = PIPELINE_ROOT / "artifacts" / "data" / "dev_eval.jsonl"

VALIDATION_DB_IDS = frozenset(
    {
        "airline",
        "authors",
        "bike_share_1",
        "cs_semester",
        "donor",
        "music_tracker",
        "simpson_episodes",
    }
)

SYSTEM_PROMPT = """You are an expert SQLite query generator.

Given a database schema, optional evidence, and a question,
generate one valid read-only SQLite query.

Return only one SQL code block and no explanation."""

_EXPECTED_ANNOTATION_COUNTS = {"train": 6601, "dev": 1534}
_EXPECTED_DATABASE_COUNTS = {"train": 69, "dev": 11}
_EXPECTED_OUTPUT_COUNTS = {
    "sft_train": 6013,
    "sft_val": 588,
    "dev_eval": 1534,
}
_MINIMUM_VALIDATION_RATIO = 0.08
_MAXIMUM_VALIDATION_RATIO = 0.12


def _validation_error(context: str, message: str) -> ValueError:
    """Create a consistently formatted dataset validation error."""
    return ValueError(f"SFT dataset validation failed: {context}: {message}")


def _require_non_empty_string(value: object, context: str) -> str:
    """Require a non-empty string without changing its contents."""
    if not isinstance(value, str) or not value.strip():
        raise _validation_error(
            context,
            f"expected a non-empty string, got {value!r} "
            f"({type(value).__name__}).",
        )
    return value


def _find_duplicates(values: list[str]) -> list[str]:
    """Return duplicate strings once, in their first duplicate order."""
    seen = set()
    duplicates = []
    duplicate_set = set()

    for value in values:
        if value in seen and value not in duplicate_set:
            duplicates.append(value)
            duplicate_set.add(value)
        seen.add(value)

    return duplicates


def _quote_identifier(identifier: str) -> str:
    """Quote an SQLite identifier, including any embedded double quotes."""
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _display_comment(display_name: str) -> str:
    """Keep an auxiliary display name on one compact comment line."""
    return " ".join(display_name.split())


def _validate_database_schema(database_schema: object, context: str) -> dict:
    """Validate one Catalog database and all references used by rendering."""
    if not isinstance(database_schema, dict):
        raise _validation_error(
            context,
            f"database schema must be an object, got "
            f"{type(database_schema).__name__}.",
        )

    db_id = _require_non_empty_string(
        database_schema.get("db_id"),
        f"{context}, field='db_id'",
    )
    tables = database_schema.get("tables")
    foreign_keys = database_schema.get("foreign_keys")
    if not isinstance(tables, list):
        raise _validation_error(
            f"{context}, db_id={db_id!r}, field='tables'",
            f"expected a list, got {tables!r} ({type(tables).__name__}).",
        )
    if not tables:
        raise _validation_error(
            f"{context}, db_id={db_id!r}, field='tables'",
            "a database schema must contain at least one real table.",
        )
    if not isinstance(foreign_keys, list):
        raise _validation_error(
            f"{context}, db_id={db_id!r}, field='foreign_keys'",
            "expected a list, got "
            f"{foreign_keys!r} ({type(foreign_keys).__name__}).",
        )

    table_names = []
    columns_by_table: dict[str, set[str]] = {}
    for table_index, table in enumerate(tables):
        table_context = (
            f"{context}, db_id={db_id!r}, table_index={table_index}"
        )
        if not isinstance(table, dict):
            raise _validation_error(
                table_context,
                f"table must be an object, got {type(table).__name__}.",
            )

        table_name = _require_non_empty_string(
            table.get("name"),
            f"{table_context}, field='name'",
        )
        _require_non_empty_string(
            table.get("display_name"),
            f"{table_context}, table={table_name!r}, field='display_name'",
        )
        columns = table.get("columns")
        primary_keys = table.get("primary_keys")
        if not isinstance(columns, list):
            raise _validation_error(
                f"{table_context}, table={table_name!r}, field='columns'",
                f"expected a list, got {columns!r} "
                f"({type(columns).__name__}).",
            )
        if not columns:
            raise _validation_error(
                f"{table_context}, table={table_name!r}, field='columns'",
                "a real table must contain at least one column.",
            )
        if not isinstance(primary_keys, list):
            raise _validation_error(
                f"{table_context}, table={table_name!r}, "
                "field='primary_keys'",
                f"expected a list, got {primary_keys!r} "
                f"({type(primary_keys).__name__}).",
            )

        column_names = []
        for column_index, column in enumerate(columns):
            column_context = (
                f"{table_context}, table={table_name!r}, "
                f"column_index={column_index}"
            )
            if not isinstance(column, dict):
                raise _validation_error(
                    column_context,
                    f"column must be an object, got "
                    f"{type(column).__name__}.",
                )

            column_name = _require_non_empty_string(
                column.get("name"),
                f"{column_context}, field='name'",
            )
            _require_non_empty_string(
                column.get("display_name"),
                f"{column_context}, column={column_name!r}, "
                "field='display_name'",
            )
            _require_non_empty_string(
                column.get("type"),
                f"{column_context}, column={column_name!r}, field='type'",
            )
            column_names.append(column_name)

        duplicate_columns = _find_duplicates(column_names)
        if duplicate_columns:
            raise _validation_error(
                f"{table_context}, table={table_name!r}, field='columns'",
                f"duplicate column names: {duplicate_columns!r}.",
            )

        for primary_key_index, primary_key in enumerate(primary_keys):
            primary_key = _require_non_empty_string(
                primary_key,
                f"{table_context}, table={table_name!r}, "
                f"field='primary_keys[{primary_key_index}]'",
            )
            if primary_key not in column_names:
                raise _validation_error(
                    f"{table_context}, table={table_name!r}, "
                    f"field='primary_keys[{primary_key_index}]'",
                    f"column {primary_key!r} does not exist in the table.",
                )
        duplicate_primary_keys = _find_duplicates(primary_keys)
        if duplicate_primary_keys:
            raise _validation_error(
                f"{table_context}, table={table_name!r}, "
                "field='primary_keys'",
                f"duplicate column references: {duplicate_primary_keys!r}.",
            )

        table_names.append(table_name)
        columns_by_table[table_name] = set(column_names)

    duplicate_tables = _find_duplicates(table_names)
    if duplicate_tables:
        raise _validation_error(
            f"{context}, db_id={db_id!r}, field='tables'",
            f"duplicate table names: {duplicate_tables!r}.",
        )

    seen_foreign_keys = set()
    required_foreign_key_fields = (
        "source_table",
        "source_column",
        "target_table",
        "target_column",
    )
    for foreign_key_index, foreign_key in enumerate(foreign_keys):
        foreign_key_context = (
            f"{context}, db_id={db_id!r}, "
            f"foreign_key_index={foreign_key_index}"
        )
        if not isinstance(foreign_key, dict):
            raise _validation_error(
                foreign_key_context,
                f"foreign key must be an object, got "
                f"{type(foreign_key).__name__}.",
            )

        values = tuple(
            _require_non_empty_string(
                foreign_key.get(field_name),
                f"{foreign_key_context}, field={field_name!r}",
            )
            for field_name in required_foreign_key_fields
        )
        source_table, source_column, target_table, target_column = values
        if source_table not in columns_by_table:
            raise _validation_error(
                foreign_key_context,
                f"source table {source_table!r} does not exist.",
            )
        if target_table not in columns_by_table:
            raise _validation_error(
                foreign_key_context,
                f"target table {target_table!r} does not exist.",
            )
        if source_column not in columns_by_table[source_table]:
            raise _validation_error(
                foreign_key_context,
                f"source column {source_table}.{source_column} does not exist.",
            )
        if target_column not in columns_by_table[target_table]:
            raise _validation_error(
                foreign_key_context,
                f"target column {target_table}.{target_column} does not exist.",
            )
        if values in seen_foreign_keys:
            raise _validation_error(
                foreign_key_context,
                f"duplicate foreign key reference: {values!r}.",
            )
        seen_foreign_keys.add(values)

    return database_schema


def render_schema(database_schema: dict) -> str:
    """Render one validated Schema Catalog database as compact text."""
    database_schema = _validate_database_schema(
        database_schema,
        "render_schema input",
    )
    rendered_tables = []

    for table in database_schema["tables"]:
        table_name = table["name"]
        table_display_name = table["display_name"]
        table_header = f"TABLE {_quote_identifier(table_name)}"
        if table_display_name != table_name:
            table_header += f" -- {_display_comment(table_display_name)}"

        table_lines = [table_header]
        primary_keys = table["primary_keys"]
        for column in table["columns"]:
            column_name = column["name"]
            column_display_name = column["display_name"]
            column_line = (
                f"- {_quote_identifier(column_name)} "
                f"{column['type'].upper()}"
            )
            if len(primary_keys) == 1 and column_name == primary_keys[0]:
                column_line += " [PRIMARY KEY]"
            if column_display_name != column_name:
                column_line += f" -- {_display_comment(column_display_name)}"
            table_lines.append(column_line)

        if len(primary_keys) > 1:
            quoted_primary_keys = ", ".join(
                _quote_identifier(column_name)
                for column_name in primary_keys
            )
            table_lines.append(f"PRIMARY KEY ({quoted_primary_keys})")

        rendered_tables.append("\n".join(table_lines))

    sections = list(rendered_tables)
    foreign_keys = database_schema["foreign_keys"]
    if foreign_keys:
        foreign_key_lines = ["FOREIGN KEY"]
        for foreign_key in foreign_keys:
            foreign_key_lines.append(
                f"{_quote_identifier(foreign_key['source_table'])}."
                f"{_quote_identifier(foreign_key['source_column'])} -> "
                f"{_quote_identifier(foreign_key['target_table'])}."
                f"{_quote_identifier(foreign_key['target_column'])}"
            )
        sections.append("\n".join(foreign_key_lines))

    return "\n\n".join(sections)


def _build_prompt(schema_text: str, question: str, evidence: str) -> list[dict]:
    """Build the shared conversational prompt for every data split."""
    question_text = question.strip()
    evidence_text = evidence.strip()
    user_sections = [f"Database schema:\n{schema_text}"]
    if evidence_text:
        user_sections.append(f"Evidence:\n{evidence_text}")
    user_sections.append(f"Question:\n{question_text}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]


def _build_completion(gold_sql: str) -> list[dict]:
    """Wrap an unchanged gold SQL string in one assistant SQL code block."""
    return [
        {
            "role": "assistant",
            "content": f"```sql\n{gold_sql}\n```",
        }
    ]


def _load_schema_catalog(
    path: Path,
    split_name: str,
) -> tuple[list[dict], dict[str, dict]]:
    """Load, validate, and index one Schema Catalog by db_id."""
    try:
        catalog = load_json(path)
    except (OSError, ValueError) as error:
        raise _validation_error(
            f"split={split_name!r}, path={str(path)!r}",
            f"Schema Catalog could not be loaded: {error}.",
        ) from error
    if not isinstance(catalog, list):
        raise _validation_error(
            f"split={split_name!r}, path={str(path)!r}",
            f"Schema Catalog top level must be a list, got "
            f"{type(catalog).__name__}.",
        )

    catalog_by_db_id = {}
    for database_index, database_schema in enumerate(catalog):
        context = f"split={split_name!r}, database_index={database_index}"
        database_schema = _validate_database_schema(database_schema, context)
        db_id = database_schema["db_id"]
        if db_id in catalog_by_db_id:
            raise _validation_error(
                f"split={split_name!r}, db_id={db_id!r}",
                "Schema Catalog contains a duplicate db_id.",
            )
        catalog_by_db_id[db_id] = database_schema

    expected_database_count = _EXPECTED_DATABASE_COUNTS[split_name]
    if len(catalog_by_db_id) != expected_database_count:
        raise _validation_error(
            f"split={split_name!r}, path={str(path)!r}",
            f"expected {expected_database_count} Catalog databases, got "
            f"{len(catalog_by_db_id)}.",
        )

    return catalog, catalog_by_db_id


def _load_annotations(path: Path, split_name: str) -> list[dict]:
    """Load one BIRD annotation file and validate its sample fields."""
    try:
        samples = load_jsonl(path)
    except (OSError, ValueError) as error:
        raise _validation_error(
            f"split={split_name!r}, path={str(path)!r}",
            f"annotations could not be loaded: {error}.",
        ) from error

    expected_count = _EXPECTED_ANNOTATION_COUNTS[split_name]
    if len(samples) != expected_count:
        raise _validation_error(
            f"split={split_name!r}, path={str(path)!r}",
            f"expected {expected_count} annotation samples, got {len(samples)}.",
        )

    for source_index, sample in enumerate(samples):
        context = f"split={split_name!r}, source_index={source_index}"
        if not isinstance(sample, dict):
            raise _validation_error(
                context,
                f"sample must be an object, got {type(sample).__name__}.",
            )

        db_id = _require_non_empty_string(
            sample.get("db_id"),
            f"{context}, field='db_id'",
        )
        _require_non_empty_string(
            sample.get("question"),
            f"{context}, db_id={db_id!r}, field='question'",
        )
        evidence = sample.get("evidence")
        if not isinstance(evidence, str):
            raise _validation_error(
                f"{context}, db_id={db_id!r}, field='evidence'",
                f"expected a string, got {evidence!r} "
                f"({type(evidence).__name__}).",
            )
        _require_non_empty_string(
            sample.get("SQL"),
            f"{context}, db_id={db_id!r}, field='SQL'",
        )

        if split_name == "dev":
            question_id = sample.get("question_id")
            if type(question_id) is not int:
                raise _validation_error(
                    f"{context}, db_id={db_id!r}, field='question_id'",
                    f"expected a non-boolean integer, got {question_id!r} "
                    f"({type(question_id).__name__}).",
                )
            if question_id != source_index:
                raise _validation_error(
                    f"{context}, db_id={db_id!r}, field='question_id'",
                    f"expected {source_index}, got {question_id!r}.",
                )
            _require_non_empty_string(
                sample.get("difficulty"),
                f"{context}, db_id={db_id!r}, field='difficulty'",
            )

    return samples


def _require_annotation_catalog_alignment(
    annotations: list[dict],
    catalog_by_db_id: dict[str, dict],
    split_name: str,
) -> set[str]:
    """Require bidirectional db_id coverage between annotations and Catalog."""
    annotation_db_ids = {sample["db_id"] for sample in annotations}
    catalog_db_ids = set(catalog_by_db_id)
    missing_from_catalog = sorted(annotation_db_ids - catalog_db_ids)
    extra_in_catalog = sorted(catalog_db_ids - annotation_db_ids)
    if missing_from_catalog or extra_in_catalog:
        raise _validation_error(
            f"split={split_name!r}, field='db_id'",
            "annotations and Schema Catalog must contain identical databases; "
            f"missing_from_catalog={missing_from_catalog!r}, "
            f"extra_in_catalog={extra_in_catalog!r}.",
        )
    return annotation_db_ids


def _build_sft_record(
    sample: dict,
    source_index: int,
    schema_text: str,
) -> dict:
    """Convert one Train annotation into conversational prompt-completion."""
    return {
        "sample_id": f"train_{source_index:06d}",
        "source_index": source_index,
        "db_id": sample["db_id"],
        "prompt": _build_prompt(
            schema_text,
            sample["question"],
            sample["evidence"],
        ),
        "completion": _build_completion(sample["SQL"]),
    }


def _build_dev_record(
    sample: dict,
    schema_text: str,
) -> dict:
    """Convert one Dev annotation into a prompt-only evaluation record."""
    return {
        "question_id": sample["question_id"],
        "db_id": sample["db_id"],
        "difficulty": sample["difficulty"],
        "prompt": _build_prompt(
            schema_text,
            sample["question"],
            sample["evidence"],
        ),
        "gold_sql": sample["SQL"],
    }


def _validate_prompt(
    prompt: object,
    expected_prompt: list[dict],
    gold_sql: str,
    context: str,
) -> None:
    """Validate the exact shared prompt shape and guard against SQL leakage."""
    if prompt != expected_prompt:
        raise _validation_error(
            context,
            "prompt does not match the shared prompt builder output.",
        )
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise _validation_error(
            context,
            "prompt must contain exactly one system and one user message.",
        )
    if prompt[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise _validation_error(context, "system prompt is not canonical.")
    if prompt[1].get("role") != "user" or not isinstance(
        prompt[1].get("content"), str
    ):
        raise _validation_error(context, "second prompt message must be user text.")
    if gold_sql in prompt[1]["content"]:
        raise _validation_error(
            context,
            "user prompt contains the complete gold SQL string.",
        )


def _validate_sft_record(
    record: object,
    sample: dict,
    source_index: int,
    schema_text: str,
    expected_subset: str,
) -> None:
    """Validate one SFT Train or Validation output against its source."""
    context = (
        f"output={expected_subset!r}, source_index={source_index}, "
        f"db_id={sample['db_id']!r}"
    )
    if not isinstance(record, dict):
        raise _validation_error(
            context,
            f"record must be an object, got {type(record).__name__}.",
        )

    expected_keys = {
        "sample_id",
        "source_index",
        "db_id",
        "prompt",
        "completion",
    }
    if set(record) != expected_keys:
        raise _validation_error(
            context,
            f"record keys must be {sorted(expected_keys)!r}, got "
            f"{sorted(record, key=repr)!r}.",
        )
    expected_identity = {
        "sample_id": f"train_{source_index:06d}",
        "source_index": source_index,
        "db_id": sample["db_id"],
    }
    for field_name, expected_value in expected_identity.items():
        if record[field_name] != expected_value:
            raise _validation_error(
                context,
                f"field {field_name!r} must be {expected_value!r}, got "
                f"{record[field_name]!r}.",
            )

    expected_prompt = _build_prompt(
        schema_text,
        sample["question"],
        sample["evidence"],
    )
    _validate_prompt(record["prompt"], expected_prompt, sample["SQL"], context)

    completion = record["completion"]
    expected_completion = _build_completion(sample["SQL"])
    if completion != expected_completion:
        raise _validation_error(
            context,
            "completion does not preserve the source gold SQL exactly.",
        )
    if not isinstance(completion, list) or len(completion) != 1:
        raise _validation_error(
            context,
            "completion must contain exactly one assistant message.",
        )
    completion_message = completion[0]
    if not isinstance(completion_message, dict) or set(completion_message) != {
        "role",
        "content",
    }:
        raise _validation_error(
            context,
            "completion message must contain only role and content.",
        )
    completion_content = completion_message.get("content")
    if completion_message.get("role") != "assistant" or not isinstance(
        completion_content, str
    ):
        raise _validation_error(
            context,
            "completion must be one assistant text message.",
        )
    prefix = "```sql\n"
    suffix = "\n```"
    if (
        not completion_content.startswith(prefix)
        or not completion_content.endswith(suffix)
        or completion_content.count("```") != 2
    ):
        raise _validation_error(
            context,
            "completion must contain exactly one SQL code block.",
        )
    extracted_sql = completion_content[len(prefix) : -len(suffix)]
    if extracted_sql != sample["SQL"]:
        raise _validation_error(
            context,
            "SQL extracted from completion differs from the source gold SQL.",
        )


def _validate_dev_record(
    record: object,
    sample: dict,
    source_index: int,
    schema_text: str,
) -> None:
    """Validate one prompt-only Dev output against its source."""
    context = (
        f"output='dev_eval', source_index={source_index}, "
        f"db_id={sample['db_id']!r}"
    )
    if not isinstance(record, dict):
        raise _validation_error(
            context,
            f"record must be an object, got {type(record).__name__}.",
        )

    expected_keys = {
        "question_id",
        "db_id",
        "difficulty",
        "prompt",
        "gold_sql",
    }
    if set(record) != expected_keys:
        raise _validation_error(
            context,
            f"record keys must be {sorted(expected_keys)!r}, got "
            f"{sorted(record, key=repr)!r}.",
        )
    expected_identity = {
        "question_id": sample["question_id"],
        "db_id": sample["db_id"],
        "difficulty": sample["difficulty"],
        "gold_sql": sample["SQL"],
    }
    for field_name, expected_value in expected_identity.items():
        if record[field_name] != expected_value:
            raise _validation_error(
                context,
                f"field {field_name!r} must preserve the source value "
                f"{expected_value!r}, got {record[field_name]!r}.",
            )
    if "completion" in record:
        raise _validation_error(
            context,
            "prompt-only Dev records must not contain completion.",
        )

    expected_prompt = _build_prompt(
        schema_text,
        sample["question"],
        sample["evidence"],
    )
    _validate_prompt(record["prompt"], expected_prompt, sample["SQL"], context)


def _validate_built_datasets(
    datasets: dict[str, list[dict]],
    train_annotations: list[dict],
    dev_annotations: list[dict],
    train_schema_texts: dict[str, str],
    dev_schema_texts: dict[str, str],
) -> None:
    """Validate counts, database isolation, ordering, and record contents."""
    if set(datasets) != set(_EXPECTED_OUTPUT_COUNTS):
        raise _validation_error(
            "output collections",
            "expected exactly sft_train, sft_val, and dev_eval, got "
            f"{sorted(datasets, key=repr)!r}.",
        )
    for output_name, expected_count in _EXPECTED_OUTPUT_COUNTS.items():
        records = datasets[output_name]
        if not isinstance(records, list) or len(records) != expected_count:
            actual_count = len(records) if isinstance(records, list) else None
            raise _validation_error(
                f"output={output_name!r}",
                f"expected {expected_count} records, got {actual_count!r}.",
            )

    expected_train_pairs = [
        (source_index, sample)
        for source_index, sample in enumerate(train_annotations)
        if sample["db_id"] not in VALIDATION_DB_IDS
    ]
    expected_val_pairs = [
        (source_index, sample)
        for source_index, sample in enumerate(train_annotations)
        if sample["db_id"] in VALIDATION_DB_IDS
    ]
    if len(expected_train_pairs) != len(datasets["sft_train"]):
        raise _validation_error(
            "output='sft_train'",
            "records do not match the fixed database-level split.",
        )
    if len(expected_val_pairs) != len(datasets["sft_val"]):
        raise _validation_error(
            "output='sft_val'",
            "records do not match the fixed database-level split.",
        )

    for record, (source_index, sample) in zip(
        datasets["sft_train"], expected_train_pairs
    ):
        _validate_sft_record(
            record,
            sample,
            source_index,
            train_schema_texts[sample["db_id"]],
            "sft_train",
        )
    for record, (source_index, sample) in zip(
        datasets["sft_val"], expected_val_pairs
    ):
        _validate_sft_record(
            record,
            sample,
            source_index,
            train_schema_texts[sample["db_id"]],
            "sft_val",
        )
    for source_index, (record, sample) in enumerate(
        zip(datasets["dev_eval"], dev_annotations)
    ):
        _validate_dev_record(
            record,
            sample,
            source_index,
            dev_schema_texts[sample["db_id"]],
        )

    sft_train_db_ids = {record["db_id"] for record in datasets["sft_train"]}
    sft_val_db_ids = {record["db_id"] for record in datasets["sft_val"]}
    dev_db_ids = {record["db_id"] for record in datasets["dev_eval"]}
    if len(sft_train_db_ids) != 62:
        raise _validation_error(
            "output='sft_train', field='db_id'",
            f"expected 62 databases, got {len(sft_train_db_ids)}.",
        )
    if sft_val_db_ids != set(VALIDATION_DB_IDS):
        raise _validation_error(
            "output='sft_val', field='db_id'",
            "Validation databases differ from the fixed split; "
            f"expected={sorted(VALIDATION_DB_IDS)!r}, "
            f"actual={sorted(sft_val_db_ids)!r}.",
        )
    if len(dev_db_ids) != _EXPECTED_DATABASE_COUNTS["dev"]:
        raise _validation_error(
            "output='dev_eval', field='db_id'",
            f"expected {_EXPECTED_DATABASE_COUNTS['dev']} databases, got "
            f"{len(dev_db_ids)}.",
        )
    if sft_train_db_ids & sft_val_db_ids:
        raise _validation_error(
            "SFT database split",
            "SFT Train and Validation share db_id values: "
            f"{sorted(sft_train_db_ids & sft_val_db_ids)!r}.",
        )
    sft_db_ids = sft_train_db_ids | sft_val_db_ids
    if sft_db_ids & dev_db_ids:
        raise _validation_error(
            "Train/Dev database isolation",
            "SFT and Dev share db_id values: "
            f"{sorted(sft_db_ids & dev_db_ids)!r}.",
        )

    validation_ratio = len(datasets["sft_val"]) / len(train_annotations)
    if not (
        _MINIMUM_VALIDATION_RATIO
        <= validation_ratio
        <= _MAXIMUM_VALIDATION_RATIO
    ):
        raise _validation_error(
            "output='sft_val'",
            "Validation sample ratio must be between 8% and 12%, got "
            f"{validation_ratio:.2%}.",
        )


def _remove_temporary_files(paths: list[Path]) -> list[str]:
    """Remove exact temporary files and return any cleanup errors."""
    cleanup_errors = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(f"{path}: {error}")
    return cleanup_errors


def _reload_and_verify_outputs(
    expected_datasets: dict[str, list[dict]],
    paths: dict[str, Path],
) -> dict[str, list[dict]]:
    """Reload JSONL artifacts and require exact in-memory equality."""
    loaded_datasets = {}
    for output_name in ("sft_train", "sft_val", "dev_eval"):
        path = paths[output_name]
        try:
            records = load_jsonl(path)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"SFT dataset output {output_name!r} could not be read from "
                f"{path}: {error}."
            ) from error
        if records != expected_datasets[output_name]:
            raise RuntimeError(
                f"SFT dataset output {output_name!r} at {path} does not "
                "match the in-memory data."
            )
        loaded_datasets[output_name] = records
    return loaded_datasets


def _publish_outputs(
    expected_datasets: dict[str, list[dict]],
    final_paths: dict[str, Path],
    temporary_paths: dict[str, Path],
) -> dict[str, list[dict]]:
    """Publish all three JSONL files and restore every old output on failure."""
    temporary_path_list = list(temporary_paths.values())
    invalid_final_paths = [
        path
        for path in final_paths.values()
        if path.is_symlink() or (path.exists() and not path.is_file())
    ]
    if invalid_final_paths:
        cleanup_errors = _remove_temporary_files(temporary_path_list)
        cleanup_details = (
            f" Temporary cleanup also failed: {cleanup_errors!r}."
            if cleanup_errors
            else ""
        )
        raise RuntimeError(
            "SFT dataset final paths must be regular, non-symlink files when "
            f"they already exist: {[str(path) for path in invalid_final_paths]!r}."
            f"{cleanup_details}"
        )

    previous_contents: dict[str, bytes | None] = {}
    for output_name, final_path in final_paths.items():
        try:
            previous_contents[output_name] = final_path.read_bytes()
        except FileNotFoundError:
            previous_contents[output_name] = None
        except BaseException as error:
            cleanup_errors = _remove_temporary_files(temporary_path_list)
            cleanup_details = (
                f" Temporary cleanup also failed: {cleanup_errors!r}."
                if cleanup_errors
                else ""
            )
            if not isinstance(error, Exception) and not cleanup_errors:
                raise
            raise RuntimeError(
                f"Existing SFT dataset could not be backed up before "
                f"publication: {final_path}: {error}.{cleanup_details}"
            ) from error

    replaced_outputs = []
    try:
        for output_name in ("sft_train", "sft_val", "dev_eval"):
            temporary_paths[output_name].replace(final_paths[output_name])
            replaced_outputs.append(output_name)
        published_datasets = _reload_and_verify_outputs(
            expected_datasets,
            final_paths,
        )
    except BaseException as publication_error:
        rollback_errors = []
        recovery_paths = set()
        for output_name in reversed(replaced_outputs):
            final_path = final_paths[output_name]
            temporary_path = temporary_paths[output_name]
            try:
                previous_content = previous_contents[output_name]
                if previous_content is None:
                    final_path.unlink(missing_ok=True)
                else:
                    temporary_path.write_bytes(previous_content)
                    try:
                        temporary_path.replace(final_path)
                    except OSError:
                        recovery_paths.add(temporary_path)
                        raise
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{output_name}: could not restore {final_path}: "
                    f"{rollback_error}."
                )

        rollback_errors.extend(
            _remove_temporary_files(
                [
                    path
                    for path in temporary_path_list
                    if path not in recovery_paths
                ]
            )
        )
        if rollback_errors:
            recovery_details = (
                " Recovery copies were retained at "
                f"{sorted(str(path) for path in recovery_paths)!r}."
                if recovery_paths
                else ""
            )
            raise RuntimeError(
                "SFT dataset publication failed and rollback was incomplete. "
                f"Publication error: {publication_error}. "
                f"Rollback errors: {rollback_errors!r}.{recovery_details}"
            ) from publication_error
        if not isinstance(publication_error, Exception):
            raise
        raise RuntimeError(
            "SFT dataset publication failed; all previous outputs were "
            f"restored. Publication error: {publication_error}."
        ) from publication_error

    return published_datasets


def build_sft_dataset() -> dict[str, list[dict]]:
    """Build, validate, atomically publish, and return all SFT datasets."""
    train_annotations = _load_annotations(TRAIN_ANNOTATION_PATH, "train")
    dev_annotations = _load_annotations(DEV_ANNOTATION_PATH, "dev")
    _, train_catalog_by_db_id = _load_schema_catalog(
        TRAIN_SCHEMA_CATALOG_PATH,
        "train",
    )
    _, dev_catalog_by_db_id = _load_schema_catalog(
        DEV_SCHEMA_CATALOG_PATH,
        "dev",
    )

    train_db_ids = _require_annotation_catalog_alignment(
        train_annotations,
        train_catalog_by_db_id,
        "train",
    )
    dev_db_ids = _require_annotation_catalog_alignment(
        dev_annotations,
        dev_catalog_by_db_id,
        "dev",
    )
    missing_validation_db_ids = sorted(VALIDATION_DB_IDS - train_db_ids)
    if missing_validation_db_ids:
        raise _validation_error(
            "fixed Validation databases",
            f"not present in Train annotations: {missing_validation_db_ids!r}.",
        )
    if train_db_ids & dev_db_ids:
        raise _validation_error(
            "Train/Dev database isolation",
            f"shared db_id values: {sorted(train_db_ids & dev_db_ids)!r}.",
        )

    train_schema_texts = {
        db_id: render_schema(database_schema)
        for db_id, database_schema in train_catalog_by_db_id.items()
    }
    dev_schema_texts = {
        db_id: render_schema(database_schema)
        for db_id, database_schema in dev_catalog_by_db_id.items()
    }

    datasets: dict[str, list[dict]] = {
        "sft_train": [],
        "sft_val": [],
        "dev_eval": [],
    }
    for source_index, sample in enumerate(train_annotations):
        output_name = (
            "sft_val"
            if sample["db_id"] in VALIDATION_DB_IDS
            else "sft_train"
        )
        datasets[output_name].append(
            _build_sft_record(
                sample,
                source_index,
                train_schema_texts[sample["db_id"]],
            )
        )
    for source_index, sample in enumerate(dev_annotations):
        datasets["dev_eval"].append(
            _build_dev_record(
                sample,
                dev_schema_texts[sample["db_id"]],
            )
        )

    _validate_built_datasets(
        datasets,
        train_annotations,
        dev_annotations,
        train_schema_texts,
        dev_schema_texts,
    )

    final_paths = {
        "sft_train": SFT_TRAIN_PATH,
        "sft_val": SFT_VAL_PATH,
        "dev_eval": DEV_EVAL_PATH,
    }
    temporary_paths = {
        output_name: Path(f"{path}.tmp")
        for output_name, path in final_paths.items()
    }
    cleanup_errors = _remove_temporary_files(list(temporary_paths.values()))
    if cleanup_errors:
        raise RuntimeError(
            "SFT dataset staging could not remove stale temporary files: "
            f"{cleanup_errors!r}."
        )

    try:
        for output_name in ("sft_train", "sft_val", "dev_eval"):
            temporary_path = temporary_paths[output_name]
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(datasets[output_name], temporary_path)
        staged_datasets = _reload_and_verify_outputs(
            datasets,
            temporary_paths,
        )
    except BaseException as error:
        cleanup_errors = _remove_temporary_files(list(temporary_paths.values()))
        if cleanup_errors:
            raise RuntimeError(
                f"{error} Temporary-file cleanup also failed: "
                f"{cleanup_errors!r}."
            ) from error
        raise

    return _publish_outputs(staged_datasets, final_paths, temporary_paths)


def _count_output_databases(records: list[dict]) -> int:
    """Count unique databases in one generated output."""
    return len({record["db_id"] for record in records})


def main() -> None:
    datasets = build_sft_dataset()

    print("SFT dataset validation passed:")
    for output_name in ("sft_train", "sft_val", "dev_eval"):
        records = datasets[output_name]
        print(
            f"{output_name}={len(records)} samples, "
            f"{_count_output_databases(records)} databases"
        )


if __name__ == "__main__":
    main()
