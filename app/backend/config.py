"""Deployment configuration loading and project-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "app/configs/serve.yaml"


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_serve_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load serving configuration and resolve paths used by the application."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Serving config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Serving config must be a YAML object.")

    model = config["model"]
    paths = config["paths"]
    app = config["app"]

    for key in ("base_model_path", "adapter_path", "tokenizer_path"):
        model[key] = str(resolve_project_path(model[key]))
    for key in (
        "train_schema_catalog",
        "dev_schema_catalog",
        "train_database_root",
        "dev_database_root",
        "grounding_index",
        "demo_queries",
    ):
        paths[key] = str(resolve_project_path(paths[key]))
    app["request_log_path"] = str(resolve_project_path(app["request_log_path"]))

    required_paths = [
        model["base_model_path"],
        model["adapter_path"],
        model["tokenizer_path"],
        paths["train_schema_catalog"],
        paths["dev_schema_catalog"],
        paths["train_database_root"],
        paths["dev_database_root"],
        paths["grounding_index"],
        paths["demo_queries"],
    ]
    missing = [value for value in required_paths if not Path(value).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing deployment assets: " + ", ".join(missing)
        )

    generation = config["generation"]
    execution = config["execution"]
    positive_values = {
        "app.max_concurrent_queries": app["max_concurrent_queries"],
        "model.request_timeout_seconds": model["request_timeout_seconds"],
        "model.max_model_len": model["max_model_len"],
        "generation.max_new_tokens": generation["max_new_tokens"],
        "generation.fast.candidates": generation["fast"]["candidates"],
        "generation.accurate.sampled_candidates": generation["accurate"][
            "sampled_candidates"
        ],
        "execution.timeout_seconds": execution["timeout_seconds"],
        "execution.vote_max_rows": execution["vote_max_rows"],
        "execution.display_max_rows": execution["display_max_rows"],
        "execution.minimum_vote_support": execution["minimum_vote_support"],
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(
            "Serving limits must be positive: " + ", ".join(invalid)
        )
    return config
