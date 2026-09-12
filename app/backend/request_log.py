"""Lightweight append-only request logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_request(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
