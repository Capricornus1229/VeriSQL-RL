"""Read-only registry of the Train and Dev BIRD databases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common import load_json

from .config import PROJECT_ROOT
from .v2_bridge import render_schema


class UnknownDatabaseError(KeyError):
    """Raised when a request names a database outside the catalogs."""


class DatabaseRegistry:
    def __init__(self, entries: dict[str, dict[str, Any]]) -> None:
        self._entries = entries

    @classmethod
    def from_catalogs(
        cls,
        train_catalog: str | Path,
        dev_catalog: str | Path,
        *,
        demo_queries: str | Path,
    ) -> "DatabaseRegistry":
        entries: dict[str, dict[str, Any]] = {}
        for split, catalog_path in (
            ("train", train_catalog),
            ("dev", dev_catalog),
        ):
            catalog = load_json(catalog_path)
            for schema_entry in catalog:
                db_id = schema_entry["db_id"]
                if db_id in entries:
                    raise ValueError(f"Duplicate database id across catalogs: {db_id}")

                sqlite_path = (PROJECT_ROOT / schema_entry["sqlite_path"]).resolve()
                if not sqlite_path.is_file():
                    raise FileNotFoundError(
                        f"SQLite database does not exist for db_id={db_id!r}: "
                        f"{sqlite_path}"
                    )
                entries[db_id] = {
                    **schema_entry,
                    "split": split,
                    "sqlite_path": sqlite_path,
                    "schema_catalog_entry": schema_entry,
                    "schema_text": render_schema(schema_entry),
                    "table_count": len(schema_entry["tables"]),
                    "column_count": sum(
                        len(table["columns"])
                        for table in schema_entry["tables"]
                    ),
                    "example_queries": [],
                }

        for example in load_json(demo_queries):
            entry = entries.get(example["db_id"])
            if entry is not None:
                entry["example_queries"].append(
                    {
                        "question": example["question"],
                        "evidence": example.get("evidence", ""),
                    }
                )
        return cls(entries)

    def list_databases(self) -> list[dict[str, Any]]:
        """Return browser-safe database metadata without filesystem paths."""
        return [
            {
                "db_id": entry["db_id"],
                "split": entry["split"],
                "table_count": entry["table_count"],
                "column_count": entry["column_count"],
                "example_queries": entry["example_queries"],
            }
            for entry in self._entries.values()
        ]

    def get_database(self, db_id: str) -> dict[str, Any]:
        try:
            return self._entries[db_id]
        except KeyError as error:
            raise UnknownDatabaseError(db_id) from error

    def get_schema_text(self, db_id: str) -> str:
        return self.get_database(db_id)["schema_text"]

    def __len__(self) -> int:
        return len(self._entries)
