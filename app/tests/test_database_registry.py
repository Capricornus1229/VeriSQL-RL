import json
import sqlite3

import pytest

from app.backend.database_registry import DatabaseRegistry, UnknownDatabaseError


def _catalog_entry(db_id: str, sqlite_path: str) -> dict:
    return {
        "db_id": db_id,
        "sqlite_path": sqlite_path,
        "tables": [
            {
                "name": "items",
                "display_name": "items",
                "columns": [
                    {"name": "id", "display_name": "id", "type": "integer"}
                ],
                "primary_keys": ["id"],
            }
        ],
        "foreign_keys": [],
    }


def test_registry_merges_catalogs_and_hides_paths(tmp_path):
    train_db = tmp_path / "train.sqlite"
    dev_db = tmp_path / "dev.sqlite"
    for path in (train_db, dev_db):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

    train_catalog = tmp_path / "train.json"
    dev_catalog = tmp_path / "dev.json"
    demos = tmp_path / "demos.json"
    train_catalog.write_text(
        json.dumps([_catalog_entry("train_db", str(train_db))]),
        encoding="utf-8",
    )
    dev_catalog.write_text(
        json.dumps([_catalog_entry("dev_db", str(dev_db))]),
        encoding="utf-8",
    )
    demos.write_text(
        json.dumps(
            [
                {
                    "db_id": "dev_db",
                    "question": "How many items are there?",
                    "evidence": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    registry = DatabaseRegistry.from_catalogs(
        train_catalog,
        dev_catalog,
        demo_queries=demos,
    )

    assert len(registry) == 2
    assert registry.get_database("train_db")["split"] == "train"
    assert registry.get_database("dev_db")["split"] == "dev"
    assert 'TABLE "items"' in registry.get_schema_text("dev_db")
    public_json = json.dumps(registry.list_databases())
    assert str(tmp_path) not in public_json
    assert "sqlite_path" not in public_json

    with pytest.raises(UnknownDatabaseError, match="missing"):
        registry.get_database("missing")
