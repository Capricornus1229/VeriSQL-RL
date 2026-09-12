from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Literal

from src.common import load_json, load_jsonl, open_sqlite_read_only, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SET_PATH = PROJECT_ROOT / "data" / "raw" / "annotations" / "train" / "bird23_train_filtered.jsonl"
TRAIN_DATABASE_PATH = PROJECT_ROOT / "data" / "raw" / "databases" / "train"
TRAIN_TABLES_PATH = PROJECT_ROOT / "data" / "raw" / "official_release" / "train" / "train_tables.json"

DEV_SET_PATH = PROJECT_ROOT / "data" / "raw" / "annotations" / "dev" / "bird_sql_dev_20251106.json"
DEV_DATABASE_PATH = PROJECT_ROOT / "data" / "raw" / "databases" / "dev"
DEV_TABLES_PATH = PROJECT_ROOT / "data" / "raw" / "official_release" / "dev" / "dev_tables.json"

TRAIN_SCHEMA_CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "schema_catalog_train.json"
DEV_SCHEMA_CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "schema_catalog_dev.json"
TRAIN_SCHEMA_CATALOG_TEMP_PATH = Path(f"{TRAIN_SCHEMA_CATALOG_PATH}.tmp")
DEV_SCHEMA_CATALOG_TEMP_PATH = Path(f"{DEV_SCHEMA_CATALOG_PATH}.tmp")

_MINIMUM_SCHEMA_VALIDATION_SQLITE_VERSION = (3, 37, 0)
_ASCII_LOWERCASE_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)


def _collect_db_ids(samples: list[dict], split_name: Literal["train", "dev"]) -> set[str]:
    """Collect and validate the unique database IDs used by a split."""

    if split_name not in ("train", "dev"):
        raise ValueError(
            f"split_name must be 'train' or 'dev', got {split_name!r}."
        )
    
    db_ids = set()

    for sample_index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(
                f"{split_name} sample {sample_index} must be a JSON object, "
                f"but got {type(sample).__name__}."
            )

        db_id = sample.get("db_id")
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(
                f"{split_name} sample {sample_index} has an invalid db_id: {db_id!r}."
            )

        db_ids.add(db_id)

    return db_ids


def _index_sqlite_files(database_root: Path) -> dict[str, list[Path]]:
    """Index SQLite files by filename with a single directory scan."""
    sqlite_files_by_name: dict[str, list[Path]] = {}

    for database_path in database_root.rglob("*.sqlite"):
        sqlite_files_by_name.setdefault(database_path.name, []).append(database_path)

    return sqlite_files_by_name


def _has_user_table(database_path: Path) -> bool:
    """Open a SQLite database in read-only mode and check for a user table."""
    with open_sqlite_read_only(database_path) as connection:
        user_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT GLOB 'sqlite_*'
            LIMIT 1
            """
        ).fetchone()

    return user_table is not None


def build_database_catalog() -> dict[str, dict[str, Path]]:
    """Resolve and validate the SQLite database used by each data split."""
    split_sources = {
        "train": (TRAIN_SET_PATH, TRAIN_DATABASE_PATH),
        "dev": (DEV_SET_PATH, DEV_DATABASE_PATH),
    }
    database_catalog: dict[str, dict[str, Path]] = {}
    db_ids_by_split: dict[str, set[str]] = {}
    validation_issues = []

    for split_name, (annotation_path, database_root) in split_sources.items():
        samples = load_jsonl(annotation_path)
        db_ids = _collect_db_ids(samples, split_name)
        db_ids_by_split[split_name] = db_ids
        sqlite_files_by_name = _index_sqlite_files(database_root)
        split_catalog: dict[str, Path] = {}

        for db_id in sorted(db_ids):
            expected_filename = f"{db_id}.sqlite"
            database_matches = sorted(sqlite_files_by_name.get(expected_filename, []))

            if not database_matches:
                validation_issues.append(
                    f"[{split_name}] No SQLite database found for db_id={db_id!r}; "
                    f"expected one file named {expected_filename!r} under {database_root}."
                )
                continue

            if len(database_matches) > 1:
                matched_paths = ", ".join(str(path) for path in database_matches)
                validation_issues.append(
                    f"[{split_name}] Multiple SQLite databases found for db_id={db_id!r}: "
                    f"{matched_paths}."
                )
                continue

            database_path = database_matches[0]
            expected_path = database_root / db_id / expected_filename
            database_is_valid = True

            if database_path != expected_path:
                validation_issues.append(
                    f"[{split_name}] SQLite database for db_id={db_id!r} is at "
                    f"{database_path}, but the expected path is {expected_path}."
                )
                database_is_valid = False

            try:
                has_user_table = _has_user_table(database_path)
            except sqlite3.Error as error:
                validation_issues.append(
                    f"[{split_name}] Cannot open SQLite database for db_id={db_id!r} "
                    f"at {database_path}: {error}."
                )
                database_is_valid = False
            else:
                if not has_user_table:
                    validation_issues.append(
                        f"[{split_name}] SQLite database for db_id={db_id!r} at "
                        f"{database_path} contains no user tables."
                    )
                    database_is_valid = False

            if database_is_valid:
                split_catalog[db_id] = database_path

        database_catalog[split_name] = split_catalog

    shared_db_ids = db_ids_by_split["train"] & db_ids_by_split["dev"]
    if shared_db_ids:
        formatted_db_ids = ", ".join(
            repr(db_id) for db_id in sorted(shared_db_ids)
        )
        validation_issues.append(
            "[split isolation] Train and dev annotations share the following "
            f"db_id values: {formatted_db_ids}."
        )

    if validation_issues:
        issue_list = "\n".join(f"- {issue}" for issue in validation_issues)
        raise RuntimeError(f"Database validation failed:\n{issue_list}")

    return database_catalog


def _schema_metadata_error(
    split_name: str,
    db_id: object,
    field_name: str,
    message: str,
) -> ValueError:
    """Create a consistently formatted official schema metadata error."""
    return ValueError(
        f"[{split_name}] db_id={db_id!r}, field={field_name!r}: {message}"
    )


def _build_database_schema(
    official_schema: dict,
    database_path: Path,
    split_name: str,
) -> dict:
    """Convert one official BIRD schema entry into the catalog format."""
    db_id = official_schema["db_id"]

    table_names_original = official_schema.get("table_names_original")
    table_names = official_schema.get("table_names")
    column_names_original = official_schema.get("column_names_original")
    column_names = official_schema.get("column_names")
    column_types = official_schema.get("column_types")
    primary_keys = official_schema.get("primary_keys")
    foreign_keys = official_schema.get("foreign_keys")

    list_fields = {
        "table_names_original": table_names_original,
        "table_names": table_names,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
    for field_name, field_value in list_fields.items():
        if not isinstance(field_value, list):
            raise _schema_metadata_error(
                split_name,
                db_id,
                field_name,
                f"expected a list, got {field_value!r} "
                f"({type(field_value).__name__}).",
            )

    if len(table_names_original) != len(table_names):
        raise _schema_metadata_error(
            split_name,
            db_id,
            "table_names",
            "table_names_original and table_names must have the same length, "
            f"got {len(table_names_original)} and {len(table_names)}.",
        )

    column_count = len(column_names_original)
    if column_count != len(column_names) or column_count != len(column_types):
        raise _schema_metadata_error(
            split_name,
            db_id,
            "column_names",
            "column_names_original, column_names, and column_types must have "
            "the same length, got "
            f"{len(column_names_original)}, {len(column_names)}, and "
            f"{len(column_types)}.",
        )

    if not column_names_original or not column_names:
        raise _schema_metadata_error(
            split_name,
            db_id,
            "column_names",
            "column arrays must contain the [-1, '*'] placeholder at index 0, "
            f"got original={column_names_original!r}, display={column_names!r}.",
        )

    original_placeholder = column_names_original[0]
    if (
        not isinstance(original_placeholder, list)
        or len(original_placeholder) != 2
        or type(original_placeholder[0]) is not int
        or original_placeholder[0] != -1
        or original_placeholder[1] != "*"
    ):
        raise _schema_metadata_error(
            split_name,
            db_id,
            "column_names_original[0]",
            f"expected [-1, '*'], got {original_placeholder!r}.",
        )

    display_placeholder = column_names[0]
    if (
        not isinstance(display_placeholder, list)
        or len(display_placeholder) != 2
        or type(display_placeholder[0]) is not int
        or display_placeholder[0] != -1
        or display_placeholder[1] != "*"
    ):
        raise _schema_metadata_error(
            split_name,
            db_id,
            "column_names[0]",
            f"expected [-1, '*'], got {display_placeholder!r}.",
        )

    tables = []
    for table_index, (table_name, table_display_name) in enumerate(
        zip(table_names_original, table_names)
    ):
        if not isinstance(table_name, str):
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"table_names_original[{table_index}]",
                f"expected a string, got {table_name!r} "
                f"({type(table_name).__name__}).",
            )
        if not isinstance(table_display_name, str):
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"table_names[{table_index}]",
                f"expected a string, got {table_display_name!r} "
                f"({type(table_display_name).__name__}).",
            )

        tables.append(
            {
                "name": table_name,
                "display_name": table_display_name,
                "columns": [],
                "primary_keys": [],
            }
        )

    table_count = len(tables)
    for column_index in range(1, column_count):
        original_column = column_names_original[column_index]
        display_column = column_names[column_index]

        if not isinstance(original_column, list) or len(original_column) != 2:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}]",
                f"expected [table_index, column_name], got {original_column!r}.",
            )
        if not isinstance(display_column, list) or len(display_column) != 2:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names[{column_index}]",
                f"expected [table_index, display_name], got {display_column!r}.",
            )

        original_table_index, column_name = original_column
        display_table_index, column_display_name = display_column

        if type(original_table_index) is not int:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}][0]",
                f"expected an integer table index, got {original_table_index!r}.",
            )
        if type(display_table_index) is not int:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names[{column_index}][0]",
                f"expected an integer table index, got {display_table_index!r}.",
            )
        if original_table_index != display_table_index:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}][0]/"
                f"column_names[{column_index}][0]",
                "original and display columns must reference the same table index, "
                f"got original={original_table_index!r}, "
                f"display={display_table_index!r}.",
            )
        if not 0 <= original_table_index < table_count:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}][0]",
                f"table index {original_table_index!r} is outside the valid range "
                f"0..{table_count - 1}; only column index 0 may use table index -1.",
            )
        if not isinstance(column_name, str):
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}][1]",
                f"expected a string, got {column_name!r} "
                f"({type(column_name).__name__}).",
            )
        if not isinstance(column_display_name, str):
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names[{column_index}][1]",
                f"expected a string, got {column_display_name!r} "
                f"({type(column_display_name).__name__}).",
            )
        if column_name == "*":
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names_original[{column_index}][1]",
                "only column index 0 may contain the '*' placeholder, "
                f"got {column_name!r}.",
            )
        if column_display_name == "*":
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_names[{column_index}][1]",
                "only column index 0 may contain the '*' placeholder, "
                f"got {column_display_name!r}.",
            )
        if not isinstance(column_types[column_index], str):
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"column_types[{column_index}]",
                f"expected a string, got {column_types[column_index]!r} "
                f"({type(column_types[column_index]).__name__}).",
            )

        tables[original_table_index]["columns"].append(
            {
                "name": column_name,
                "display_name": column_display_name,
                "type": column_types[column_index],
            }
        )

    for primary_key_index, primary_key in enumerate(primary_keys):
        if type(primary_key) is int:
            primary_key_columns = [primary_key]
        elif isinstance(primary_key, list):
            if not primary_key:
                raise _schema_metadata_error(
                    split_name,
                    db_id,
                    f"primary_keys[{primary_key_index}]",
                    f"a composite primary key must not be empty, got {primary_key!r}.",
                )
            if any(type(column_index) is not int for column_index in primary_key):
                raise _schema_metadata_error(
                    split_name,
                    db_id,
                    f"primary_keys[{primary_key_index}]",
                    f"all composite primary-key indexes must be integers, got {primary_key!r}.",
                )
            if len(set(primary_key)) != len(primary_key):
                raise _schema_metadata_error(
                    split_name,
                    db_id,
                    f"primary_keys[{primary_key_index}]",
                    f"a composite primary key must not repeat a column, got {primary_key!r}.",
                )
            primary_key_columns = primary_key
        else:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"primary_keys[{primary_key_index}]",
                "expected an integer column index or a non-empty list of integer "
                f"column indexes, got {primary_key!r}.",
            )

        for column_index in primary_key_columns:
            if not 0 < column_index < column_count:
                raise _schema_metadata_error(
                    split_name,
                    db_id,
                    f"primary_keys[{primary_key_index}]",
                    f"column index {column_index!r} is invalid; primary keys cannot "
                    "reference the '*' placeholder at index 0.",
                )

        primary_key_table_indexes = {
            column_names_original[column_index][0]
            for column_index in primary_key_columns
        }
        if len(primary_key_table_indexes) != 1:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"primary_keys[{primary_key_index}]",
                f"all columns in a composite primary key must belong to one table, "
                f"got indexes {primary_key_columns!r}.",
            )

        table_index = next(iter(primary_key_table_indexes))
        for column_index in primary_key_columns:
            tables[table_index]["primary_keys"].append(
                column_names_original[column_index][1]
            )

    catalog_foreign_keys = []
    seen_foreign_keys = set()
    for foreign_key_index, foreign_key in enumerate(foreign_keys):
        if not isinstance(foreign_key, list) or len(foreign_key) != 2:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"foreign_keys[{foreign_key_index}]",
                f"expected [source_column_index, target_column_index], got {foreign_key!r}.",
            )

        source_column_index, target_column_index = foreign_key
        if type(source_column_index) is not int or type(target_column_index) is not int:
            raise _schema_metadata_error(
                split_name,
                db_id,
                f"foreign_keys[{foreign_key_index}]",
                f"foreign-key indexes must be integers, got {foreign_key!r}.",
            )

        for role, column_index in (
            ("source", source_column_index),
            ("target", target_column_index),
        ):
            if not 0 < column_index < column_count:
                raise _schema_metadata_error(
                    split_name,
                    db_id,
                    f"foreign_keys[{foreign_key_index}]",
                    f"{role} column index {column_index!r} is invalid; foreign keys "
                    "cannot reference the '*' placeholder at index 0.",
                )

        source_table_index, source_column = column_names_original[source_column_index]
        target_table_index, target_column = column_names_original[target_column_index]
        foreign_key_record = {
            "source_table": table_names_original[source_table_index],
            "source_column": source_column,
            "target_table": table_names_original[target_table_index],
            "target_column": target_column,
        }
        foreign_key_identity = (
            foreign_key_record["source_table"],
            foreign_key_record["source_column"],
            foreign_key_record["target_table"],
            foreign_key_record["target_column"],
        )
        if foreign_key_identity not in seen_foreign_keys:
            catalog_foreign_keys.append(foreign_key_record)
            seen_foreign_keys.add(foreign_key_identity)

    try:
        relative_database_path = (
            Path(database_path)
            .resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix()
        )
    except (OSError, TypeError, ValueError) as error:
        raise _schema_metadata_error(
            split_name,
            db_id,
            "sqlite_path",
            f"database path must be inside the project root: {error}.",
        ) from error

    return {
        "db_id": db_id,
        "sqlite_path": relative_database_path,
        "tables": tables,
        "foreign_keys": catalog_foreign_keys,
    }


def _build_split_schema_catalog(
    official_tables: object,
    database_paths: dict[str, Path],
    split_name: str,
) -> list[dict]:
    """Validate and convert one split of official BIRD schema metadata."""
    if not isinstance(official_tables, list):
        raise _schema_metadata_error(
            split_name,
            None,
            "official_tables",
            f"expected a list, got {official_tables!r} "
            f"({type(official_tables).__name__}).",
        )
    if not isinstance(database_paths, dict):
        raise _schema_metadata_error(
            split_name,
            None,
            f"database_catalog[{split_name!r}]",
            f"expected a dictionary, got {database_paths!r} "
            f"({type(database_paths).__name__}).",
        )

    official_db_ids = []
    seen_db_ids = set()
    for database_index, official_schema in enumerate(official_tables):
        if not isinstance(official_schema, dict):
            raise _schema_metadata_error(
                split_name,
                None,
                f"official_tables[{database_index}]",
                f"expected a JSON object, got {official_schema!r} "
                f"({type(official_schema).__name__}).",
            )

        db_id = official_schema.get("db_id")
        if not isinstance(db_id, str) or not db_id.strip():
            raise _schema_metadata_error(
                split_name,
                db_id,
                "db_id",
                "expected a non-empty string.",
            )
        if db_id in seen_db_ids:
            raise _schema_metadata_error(
                split_name,
                db_id,
                "db_id",
                "duplicate db_id in official tables metadata.",
            )

        official_db_ids.append(db_id)
        seen_db_ids.add(db_id)

    catalog_db_ids = set(database_paths)
    official_db_id_set = set(official_db_ids)
    if catalog_db_ids != official_db_id_set:
        missing_from_official = sorted(
            catalog_db_ids - official_db_id_set,
            key=repr,
        )
        missing_from_database_catalog = sorted(
            official_db_id_set - catalog_db_ids,
            key=repr,
        )
        raise _schema_metadata_error(
            split_name,
            None,
            "db_id",
            "official/database-catalog coverage mismatch: "
            f"missing from official tables={missing_from_official!r}; "
            f"missing from database catalog={missing_from_database_catalog!r}.",
        )

    return [
        _build_database_schema(
            official_schema,
            database_paths[official_schema["db_id"]],
            split_name,
        )
        for official_schema in official_tables
    ]


def _schema_catalog_paths() -> dict[str, dict[str, Path]]:
    """Return the fixed final and staging paths for each split."""
    return {
        "train": {
            "final": TRAIN_SCHEMA_CATALOG_PATH,
            "temporary": TRAIN_SCHEMA_CATALOG_TEMP_PATH,
        },
        "dev": {
            "final": DEV_SCHEMA_CATALOG_PATH,
            "temporary": DEV_SCHEMA_CATALOG_TEMP_PATH,
        },
    }


def _schema_catalog_validation_error(
    split_name: str,
    message: str,
    *,
    db_id: object | None = None,
    table_name: object | None = None,
    column_name: object | None = None,
    path: object | None = None,
) -> RuntimeError:
    """Create a validation error with stable database-schema context."""
    context = [f"split={split_name}"]
    if db_id is not None:
        context.append(f"db_id={db_id!r}")
    if table_name is not None:
        context.append(f"table={table_name!r}")
    if column_name is not None:
        context.append(f"column={column_name!r}")
    if path is not None:
        context.append(f"path={str(path)!r}")

    return RuntimeError(
        f"Schema catalog validation failed: {', '.join(context)}: {message}"
    )


def _ascii_identifier_key(identifier: str) -> str:
    """Fold only ASCII letter case, matching SQLite NOCASE identifier behavior."""
    return identifier.translate(_ASCII_LOWERCASE_TRANSLATION)


def _find_duplicate_names(names: list[str]) -> list[str]:
    """Return exact duplicate names in first-repeated order."""
    seen_names = set()
    duplicate_names = []

    for name in names:
        if name in seen_names and name not in duplicate_names:
            duplicate_names.append(name)
        seen_names.add(name)

    return duplicate_names


def _find_ascii_case_collisions(names: list[str]) -> list[list[str]]:
    """Return groups of distinct names that differ only by ASCII letter case."""
    names_by_key: dict[str, list[str]] = {}

    for name in names:
        names_by_key.setdefault(_ascii_identifier_key(name), []).append(name)

    return [
        grouped_names
        for grouped_names in names_by_key.values()
        if len(grouped_names) > 1
    ]


def _match_schema_names(
    catalog_names: list[str],
    sqlite_names: list[str],
    *,
    split_name: str,
    db_id: str,
    object_name: str,
    table_name: str | None = None,
) -> list[str]:
    """Match names bijectively, preferring exact then unique ASCII-case matches."""
    catalog_duplicates = _find_duplicate_names(catalog_names)
    if catalog_duplicates:
        raise _schema_catalog_validation_error(
            split_name,
            f"Catalog contains duplicate {object_name} names: "
            f"{catalog_duplicates!r}.",
            db_id=db_id,
            table_name=table_name,
        )

    sqlite_duplicates = _find_duplicate_names(sqlite_names)
    if sqlite_duplicates:
        raise _schema_catalog_validation_error(
            split_name,
            f"SQLite contains duplicate {object_name} names: "
            f"{sqlite_duplicates!r}.",
            db_id=db_id,
            table_name=table_name,
        )

    catalog_collisions = _find_ascii_case_collisions(catalog_names)
    if catalog_collisions:
        raise _schema_catalog_validation_error(
            split_name,
            f"Catalog contains ambiguous ASCII-case {object_name} names: "
            f"{catalog_collisions!r}.",
            db_id=db_id,
            table_name=table_name,
        )

    sqlite_collisions = _find_ascii_case_collisions(sqlite_names)
    if sqlite_collisions:
        raise _schema_catalog_validation_error(
            split_name,
            f"SQLite contains ambiguous ASCII-case {object_name} names: "
            f"{sqlite_collisions!r}.",
            db_id=db_id,
            table_name=table_name,
        )

    sqlite_index_by_name = {
        sqlite_name: sqlite_index
        for sqlite_index, sqlite_name in enumerate(sqlite_names)
    }
    matched_names: list[str | None] = [None] * len(catalog_names)
    used_sqlite_indexes = set()

    for catalog_index, catalog_name in enumerate(catalog_names):
        sqlite_index = sqlite_index_by_name.get(catalog_name)
        if sqlite_index is not None:
            matched_names[catalog_index] = sqlite_names[sqlite_index]
            used_sqlite_indexes.add(sqlite_index)

    remaining_sqlite_indexes_by_key: dict[str, list[int]] = {}
    for sqlite_index, sqlite_name in enumerate(sqlite_names):
        if sqlite_index not in used_sqlite_indexes:
            remaining_sqlite_indexes_by_key.setdefault(
                _ascii_identifier_key(sqlite_name),
                [],
            ).append(sqlite_index)

    for catalog_index, catalog_name in enumerate(catalog_names):
        if matched_names[catalog_index] is not None:
            continue

        candidate_indexes = remaining_sqlite_indexes_by_key.get(
            _ascii_identifier_key(catalog_name),
            [],
        )
        if not candidate_indexes:
            if object_name == "table":
                raise _schema_catalog_validation_error(
                    split_name,
                    "exists in Catalog but not in SQLite.",
                    db_id=db_id,
                    table_name=catalog_name,
                )
            raise _schema_catalog_validation_error(
                split_name,
                "exists in Catalog but not in SQLite.",
                db_id=db_id,
                table_name=table_name,
                column_name=catalog_name,
            )
        if len(candidate_indexes) != 1:
            candidates = [sqlite_names[index] for index in candidate_indexes]
            raise _schema_catalog_validation_error(
                split_name,
                f"has ambiguous ASCII-case SQLite matches: {candidates!r}.",
                db_id=db_id,
                table_name=(catalog_name if object_name == "table" else table_name),
                column_name=(catalog_name if object_name == "column" else None),
            )

        sqlite_index = candidate_indexes.pop()
        matched_names[catalog_index] = sqlite_names[sqlite_index]
        used_sqlite_indexes.add(sqlite_index)

    unmatched_sqlite_names = [
        sqlite_name
        for sqlite_index, sqlite_name in enumerate(sqlite_names)
        if sqlite_index not in used_sqlite_indexes
    ]
    if unmatched_sqlite_names:
        unmatched_name = unmatched_sqlite_names[0]
        if object_name == "table":
            raise _schema_catalog_validation_error(
                split_name,
                "exists in SQLite but is missing from Catalog.",
                db_id=db_id,
                table_name=unmatched_name,
            )
        raise _schema_catalog_validation_error(
            split_name,
            "exists in SQLite but is missing from Catalog.",
            db_id=db_id,
            table_name=table_name,
            column_name=unmatched_name,
        )

    return [name for name in matched_names if name is not None]


def _read_sqlite_business_schema(
    database_path: Path,
    split_name: str,
    db_id: str,
) -> dict[str, list[str]]:
    """Read business tables and visible/generated columns from SQLite."""
    try:
        with open_sqlite_read_only(database_path) as connection:
            table_rows = connection.execute(
                """
                SELECT name, type
                FROM pragma_table_list
                WHERE schema = 'main'
                """
            ).fetchall()

            table_names = [
                table_name
                for table_name, table_type in table_rows
                if table_type in ("table", "virtual")
                and not _ascii_identifier_key(table_name).startswith("sqlite_")
            ]
            columns_by_table = {}

            for table_name in table_names:
                column_rows = connection.execute(
                    """
                    SELECT name, hidden
                    FROM pragma_table_xinfo(?)
                    ORDER BY cid
                    """,
                    (table_name,),
                ).fetchall()
                columns_by_table[table_name] = [
                    column_name
                    for column_name, hidden in column_rows
                    if hidden in (0, 2, 3)
                ]

    except (OSError, ValueError, sqlite3.Error) as error:
        raise _schema_catalog_validation_error(
            split_name,
            f"SQLite could not be opened or its schema could not be read: {error}.",
            db_id=db_id,
            path=database_path,
        ) from error

    return columns_by_table


def _validate_database_file_coverage(
    database_paths: dict[str, Path],
    database_root: Path,
    split_name: str,
) -> None:
    """Require every SQLite file under a split to have exactly one catalog entry."""
    try:
        sqlite_files = [path.resolve() for path in database_root.rglob("*.sqlite")]
        expected_files_by_db_id = {
            db_id: Path(path).resolve()
            for db_id, path in database_paths.items()
        }
        expected_files = list(expected_files_by_db_id.values())
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise _schema_catalog_validation_error(
            split_name,
            f"SQLite file coverage could not be inspected: {error}.",
            path=database_root,
        ) from error

    duplicate_sqlite_files = [
        path
        for path in _find_duplicate_names([str(path) for path in sqlite_files])
    ]
    if duplicate_sqlite_files:
        raise _schema_catalog_validation_error(
            split_name,
            f"multiple SQLite paths resolve to the same file: "
            f"{duplicate_sqlite_files!r}.",
            path=database_root,
        )

    duplicate_expected_files = [
        path
        for path in _find_duplicate_names([str(path) for path in expected_files])
    ]
    if duplicate_expected_files:
        raise _schema_catalog_validation_error(
            split_name,
            f"multiple db_id values reference the same SQLite file: "
            f"{duplicate_expected_files!r}.",
        )

    sqlite_file_set = set(sqlite_files)
    expected_file_set = set(expected_files)
    missing_files = sorted(expected_file_set - sqlite_file_set)
    extra_files = sorted(sqlite_file_set - expected_file_set)

    if missing_files:
        missing_path = missing_files[0]
        missing_db_id = next(
            (
                db_id
                for db_id, path in expected_files_by_db_id.items()
                if path == missing_path
            ),
            None,
        )
        raise _schema_catalog_validation_error(
            split_name,
            "is referenced by Catalog but does not exist under the split "
            "database directory.",
            db_id=missing_db_id,
            path=missing_path,
        )

    if extra_files:
        extra_path = extra_files[0]
        raise _schema_catalog_validation_error(
            split_name,
            "exists under the split database directory but is missing from Catalog.",
            db_id=extra_path.stem,
            path=extra_path,
        )


def _normalize_database_schema_names(
    database_schema: dict,
    sqlite_schema: dict[str, list[str]],
    split_name: str,
) -> None:
    """Validate bidirectional name coverage and apply SQLite identifier spelling."""
    db_id = database_schema["db_id"]
    tables = database_schema["tables"]
    catalog_table_names = [table["name"] for table in tables]
    sqlite_table_names = list(sqlite_schema)
    matched_table_names = _match_schema_names(
        catalog_table_names,
        sqlite_table_names,
        split_name=split_name,
        db_id=db_id,
        object_name="table",
    )
    table_name_mapping = dict(zip(catalog_table_names, matched_table_names))
    column_name_mappings: dict[str, dict[str, str]] = {}

    for table, catalog_table_name, sqlite_table_name in zip(
        tables,
        catalog_table_names,
        matched_table_names,
    ):
        catalog_column_names = [column["name"] for column in table["columns"]]
        sqlite_column_names = sqlite_schema[sqlite_table_name]
        matched_column_names = _match_schema_names(
            catalog_column_names,
            sqlite_column_names,
            split_name=split_name,
            db_id=db_id,
            object_name="column",
            table_name=sqlite_table_name,
        )
        column_name_mapping = dict(
            zip(catalog_column_names, matched_column_names)
        )
        column_name_mappings[catalog_table_name] = column_name_mapping

        normalized_primary_keys = []
        for primary_key in table["primary_keys"]:
            if primary_key not in column_name_mapping:
                raise _schema_catalog_validation_error(
                    split_name,
                    "primary key does not reference a Catalog column.",
                    db_id=db_id,
                    table_name=catalog_table_name,
                    column_name=primary_key,
                )
            normalized_primary_keys.append(column_name_mapping[primary_key])

        table["name"] = sqlite_table_name
        table["primary_keys"] = normalized_primary_keys
        for column, sqlite_column_name in zip(
            table["columns"],
            matched_column_names,
        ):
            column["name"] = sqlite_column_name

    normalized_foreign_keys = []
    seen_foreign_keys = set()
    for foreign_key in database_schema["foreign_keys"]:
        source_table = foreign_key["source_table"]
        target_table = foreign_key["target_table"]
        source_column = foreign_key["source_column"]
        target_column = foreign_key["target_column"]

        if source_table not in table_name_mapping:
            raise _schema_catalog_validation_error(
                split_name,
                "foreign key source table is missing from Catalog tables.",
                db_id=db_id,
                table_name=source_table,
            )
        if target_table not in table_name_mapping:
            raise _schema_catalog_validation_error(
                split_name,
                "foreign key target table is missing from Catalog tables.",
                db_id=db_id,
                table_name=target_table,
            )
        if source_column not in column_name_mappings[source_table]:
            raise _schema_catalog_validation_error(
                split_name,
                "foreign key source column is missing from Catalog columns.",
                db_id=db_id,
                table_name=source_table,
                column_name=source_column,
            )
        if target_column not in column_name_mappings[target_table]:
            raise _schema_catalog_validation_error(
                split_name,
                "foreign key target column is missing from Catalog columns.",
                db_id=db_id,
                table_name=target_table,
                column_name=target_column,
            )

        normalized_foreign_key = {
            "source_table": table_name_mapping[source_table],
            "source_column": column_name_mappings[source_table][source_column],
            "target_table": table_name_mapping[target_table],
            "target_column": column_name_mappings[target_table][target_column],
        }
        foreign_key_identity = (
            normalized_foreign_key["source_table"],
            normalized_foreign_key["source_column"],
            normalized_foreign_key["target_table"],
            normalized_foreign_key["target_column"],
        )
        if foreign_key_identity not in seen_foreign_keys:
            normalized_foreign_keys.append(normalized_foreign_key)
            seen_foreign_keys.add(foreign_key_identity)

    database_schema["foreign_keys"] = normalized_foreign_keys


def _validate_and_normalize_schema_catalog(
    schema_catalog: dict[str, list[dict]],
    database_catalog: dict[str, dict[str, Path]],
) -> dict[str, list[dict]]:
    """Validate staged catalog data against all SQLite files in both directions."""
    if sqlite3.sqlite_version_info < _MINIMUM_SCHEMA_VALIDATION_SQLITE_VERSION:
        required_version = ".".join(
            str(part) for part in _MINIMUM_SCHEMA_VALIDATION_SQLITE_VERSION
        )
        actual_version = sqlite3.sqlite_version
        raise RuntimeError(
            "Schema catalog validation requires SQLite "
            f">={required_version}, but the active version is {actual_version}."
        )

    database_roots = {
        "train": TRAIN_DATABASE_PATH,
        "dev": DEV_DATABASE_PATH,
    }
    normalized_catalog = deepcopy(schema_catalog)

    for split_name in ("train", "dev"):
        database_paths = database_catalog[split_name]
        _validate_database_file_coverage(
            database_paths,
            database_roots[split_name],
            split_name,
        )

        databases = normalized_catalog[split_name]
        catalog_db_ids = [database["db_id"] for database in databases]
        duplicate_db_ids = _find_duplicate_names(catalog_db_ids)
        if duplicate_db_ids:
            raise _schema_catalog_validation_error(
                split_name,
                f"Catalog contains duplicate db_id values: {duplicate_db_ids!r}.",
            )

        catalog_db_id_set = set(catalog_db_ids)
        database_catalog_db_id_set = set(database_paths)
        missing_db_ids = sorted(database_catalog_db_id_set - catalog_db_id_set)
        extra_db_ids = sorted(catalog_db_id_set - database_catalog_db_id_set)
        if missing_db_ids:
            raise _schema_catalog_validation_error(
                split_name,
                "exists in database_catalog but is missing from the saved Schema Catalog.",
                db_id=missing_db_ids[0],
            )
        if extra_db_ids:
            raise _schema_catalog_validation_error(
                split_name,
                "exists in the saved Schema Catalog but is missing from database_catalog.",
                db_id=extra_db_ids[0],
            )

        for database_schema in databases:
            db_id = database_schema["db_id"]
            try:
                database_path = Path(database_paths[db_id]).resolve()
                expected_relative_path = database_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                raise _schema_catalog_validation_error(
                    split_name,
                    "database_catalog path is invalid or outside the project root.",
                    db_id=db_id,
                    path=database_paths[db_id],
                ) from error

            sqlite_path = database_schema.get("sqlite_path")
            if sqlite_path != expected_relative_path:
                raise _schema_catalog_validation_error(
                    split_name,
                    f"sqlite_path must be {expected_relative_path!r}, "
                    f"got {sqlite_path!r}.",
                    db_id=db_id,
                    path=sqlite_path,
                )
            if not database_path.is_file():
                raise _schema_catalog_validation_error(
                    split_name,
                    "SQLite path does not exist or is not a file.",
                    db_id=db_id,
                    path=database_path,
                )

            sqlite_schema = _read_sqlite_business_schema(
                database_path,
                split_name,
                db_id,
            )
            _normalize_database_schema_names(
                database_schema,
                sqlite_schema,
                split_name,
            )

    return normalized_catalog


def _load_and_verify_schema_catalog_files(
    expected_catalog: dict[str, list[dict]],
    catalog_paths: dict[str, Path],
) -> dict[str, list[dict]]:
    """Reload both JSON files and require exact round-trip equality."""
    loaded_catalog: dict[str, list[dict]] = {}

    for split_name in ("train", "dev"):
        catalog_path = catalog_paths[split_name]
        try:
            loaded_split = load_json(catalog_path)
        except (OSError, ValueError) as error:
            raise _schema_catalog_validation_error(
                split_name,
                f"saved JSON could not be read: {error}.",
                path=catalog_path,
            ) from error

        if loaded_split != expected_catalog[split_name]:
            raise _schema_catalog_validation_error(
                split_name,
                "saved JSON does not match the in-memory Schema Catalog.",
                path=catalog_path,
            )
        loaded_catalog[split_name] = loaded_split

    return loaded_catalog


def _remove_catalog_sidecars(paths: list[Path]) -> list[str]:
    """Remove exact temporary paths and return any cleanup errors."""
    cleanup_errors = []

    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(f"{path}: {error}")

    return cleanup_errors


def _publish_schema_catalog_files(
    expected_catalog: dict[str, list[dict]],
    catalog_paths: dict[str, dict[str, Path]],
) -> dict[str, list[dict]]:
    """Replace both final files and restore their old versions on failure."""
    temporary_paths = [
        paths["temporary"] for paths in catalog_paths.values()
    ]
    invalid_final_paths = [
        paths["final"]
        for paths in catalog_paths.values()
        if paths["final"].is_symlink()
        or (paths["final"].exists() and not paths["final"].is_file())
    ]
    if invalid_final_paths:
        cleanup_errors = _remove_catalog_sidecars(temporary_paths)
        cleanup_details = (
            f" Temporary-file cleanup also failed: {cleanup_errors!r}."
            if cleanup_errors
            else ""
        )
        raise RuntimeError(
            "Schema catalog final paths must be regular, non-symlink files "
            "when they already exist: "
            f"{[str(path) for path in invalid_final_paths]!r}."
            f"{cleanup_details}"
        )

    previous_contents: dict[str, bytes | None] = {}
    for split_name in ("train", "dev"):
        final_path = catalog_paths[split_name]["final"]
        try:
            previous_contents[split_name] = final_path.read_bytes()
        except FileNotFoundError:
            previous_contents[split_name] = None
        except BaseException as error:
            cleanup_errors = _remove_catalog_sidecars(temporary_paths)
            cleanup_details = (
                f" Temporary-file cleanup also failed: {cleanup_errors!r}."
                if cleanup_errors
                else ""
            )
            if not isinstance(error, Exception) and not cleanup_errors:
                raise
            raise RuntimeError(
                f"Existing Schema Catalog could not be read before publication: "
                f"{final_path}: {error}.{cleanup_details}"
            ) from error

    replaced_splits = []
    try:
        for split_name in ("train", "dev"):
            catalog_paths[split_name]["temporary"].replace(
                catalog_paths[split_name]["final"]
            )
            replaced_splits.append(split_name)

        final_paths = {
            split_name: paths["final"]
            for split_name, paths in catalog_paths.items()
        }
        published_catalog = _load_and_verify_schema_catalog_files(
            expected_catalog,
            final_paths,
        )
    except BaseException as publication_error:
        rollback_errors = []
        recovery_paths = set()
        for split_name in reversed(replaced_splits):
            final_path = catalog_paths[split_name]["final"]
            temporary_path = catalog_paths[split_name]["temporary"]
            try:
                previous_content = previous_contents[split_name]
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
                    f"{split_name}: could not restore {final_path}: "
                    f"{rollback_error}."
                )

        disposable_temporary_paths = [
            path for path in temporary_paths if path not in recovery_paths
        ]
        rollback_errors.extend(
            _remove_catalog_sidecars(disposable_temporary_paths)
        )

        if rollback_errors:
            recovery_details = ""
            if recovery_paths:
                recovery_details = (
                    " Recovery copies containing the previous output were "
                    "retained at: "
                    f"{sorted(str(path) for path in recovery_paths)!r}."
                )
            raise RuntimeError(
                "Schema catalog publication failed and rollback was incomplete. "
                f"Publication error: {publication_error}. "
                f"Rollback errors: {rollback_errors!r}."
                f"{recovery_details}"
            ) from publication_error
        if not isinstance(publication_error, Exception):
            raise
        raise RuntimeError(
            "Schema catalog publication failed; previous output files were "
            f"restored. Publication error: {publication_error}."
        ) from publication_error

    return published_catalog


def build_schema_catalog(
    database_catalog: dict[str, dict[str, Path]],
) -> dict[str, list[dict]]:
    """Build, stage, validate, publish, and return both schema catalogs."""
    if not isinstance(database_catalog, dict):
        raise ValueError(
            f"database_catalog must be a dictionary, got {database_catalog!r} "
            f"({type(database_catalog).__name__})."
        )

    expected_splits = {"train", "dev"}
    actual_splits = set(database_catalog)
    if actual_splits != expected_splits:
        missing_splits = sorted(expected_splits - actual_splits)
        unexpected_splits = sorted(actual_splits - expected_splits, key=repr)
        raise ValueError(
            f"database_catalog must contain exactly train and dev; "
            f"missing={missing_splits!r}, unexpected={unexpected_splits!r}."
        )

    catalog_paths = _schema_catalog_paths()
    temporary_paths = {
        split_name: paths["temporary"]
        for split_name, paths in catalog_paths.items()
    }
    stale_temp_cleanup_errors = _remove_catalog_sidecars(
        list(temporary_paths.values())
    )
    if stale_temp_cleanup_errors:
        raise RuntimeError(
            "Schema catalog staging could not remove stale temporary files: "
            f"{stale_temp_cleanup_errors!r}."
        )

    tables_paths = {
        "train": TRAIN_TABLES_PATH,
        "dev": DEV_TABLES_PATH,
    }
    schema_catalog: dict[str, list[dict]] = {}

    for split_name, tables_path in tables_paths.items():
        official_tables = load_json(tables_path)
        schema_catalog[split_name] = _build_split_schema_catalog(
            official_tables,
            database_catalog[split_name],
            split_name,
        )

    try:
        for split_name in ("train", "dev"):
            temporary_path = temporary_paths[split_name]
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(schema_catalog[split_name], temporary_path)

        staged_catalog = _load_and_verify_schema_catalog_files(
            schema_catalog,
            temporary_paths,
        )
        normalized_catalog = _validate_and_normalize_schema_catalog(
            staged_catalog,
            database_catalog,
        )

        for split_name in ("train", "dev"):
            save_json(normalized_catalog[split_name], temporary_paths[split_name])

        validated_catalog = _load_and_verify_schema_catalog_files(
            normalized_catalog,
            temporary_paths,
        )
    except BaseException as error:
        cleanup_errors = _remove_catalog_sidecars(
            list(temporary_paths.values())
        )
        if cleanup_errors:
            raise RuntimeError(
                f"{error} Temporary-file cleanup also failed: "
                f"{cleanup_errors!r}."
            ) from error
        raise

    return _publish_schema_catalog_files(
        validated_catalog,
        catalog_paths,
    )


def _count_schema_catalog_objects(
    schema_catalog: dict[str, list[dict]],
) -> dict[str, dict[str, int]]:
    """Count databases, tables, and real columns for the CLI summary."""
    return {
        split_name: {
            "databases": len(databases),
            "tables": sum(len(database["tables"]) for database in databases),
            "columns": sum(
                len(table["columns"])
                for database in databases
                for table in database["tables"]
            ),
        }
        for split_name, databases in schema_catalog.items()
    }


def main() -> None:
    database_catalog = build_database_catalog()
    schema_catalog = build_schema_catalog(database_catalog)
    counts = _count_schema_catalog_objects(schema_catalog)

    print("Schema catalog validation passed:")
    for split_name in ("train", "dev"):
        split_counts = counts[split_name]
        print(
            f"{split_name}={split_counts['databases']} databases, "
            f"{split_counts['tables']} tables, "
            f"{split_counts['columns']} columns"
        )


if __name__ == "__main__":
    main()
