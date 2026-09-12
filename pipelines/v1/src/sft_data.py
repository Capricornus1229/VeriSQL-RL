from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.common import load_jsonl


def _require_message(
    message: object,
    *,
    expected_role: str | None,
    context: str,
) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"{context} must be an object.")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError(f"{context}.role must be a non-empty string.")
    if expected_role is not None and role != expected_role:
        raise ValueError(
            f"{context} must have role={expected_role!r}, "
            f"got {role!r}."
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{context}.content must be a non-empty string.")


def validate_sft_record(record: object, *, context: str = "record") -> None:
    """Validate only the conversational fields required for SFT."""
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be an object.")

    prompt = record.get("prompt")
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise ValueError(
            f"{context}.prompt must contain at least a system and a user message."
        )
    for index, message in enumerate(prompt):
        expected_role = None
        if index == 0:
            expected_role = "system"
        elif index == len(prompt) - 1:
            expected_role = "user"
        _require_message(
            message,
            expected_role=expected_role,
            context=f"{context}.prompt[{index}]",
        )

    completion = record.get("completion")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(
            f"{context}.completion must contain exactly one assistant message."
        )
    _require_message(
        completion[0],
        expected_role="assistant",
        context=f"{context}.completion[0]",
    )


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and lightly validate conversational SFT records from JSONL."""
    file_path = Path(path)
    records = load_jsonl(file_path)
    if not records:
        raise ValueError(f"SFT data file is empty: {file_path}")

    for index, record in enumerate(records):
        validate_sft_record(record, context=f"{file_path}, record={index}")
    return records


def _as_token_ids(token_ids: object, *, context: str) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"{context} did not produce a non-empty token ID list.")
    if any(type(token_id) is not int for token_id in token_ids):
        raise ValueError(f"{context} produced a non-integer token ID.")
    return token_ids


def build_completion_only_labels(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    """Mask the prompt so loss is computed only on the assistant completion."""
    if len(prompt_ids) > len(full_ids):
        raise ValueError("Prompt token sequence is longer than the full conversation.")
    labels = list(full_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    return labels


def tokenize_sft_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Apply the chat template and build completion-only causal-LM labels."""
    validate_sft_record(record)
    if type(max_length) is not int or max_length <= 0:
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}.")

    prompt = record["prompt"]
    completion = record["completion"]
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        context="Prompt chat template",
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            [*prompt, *completion],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
        context="Full-conversation chat template",
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        sample_id = record.get("sample_id", "<unknown>")
        raise ValueError(
            "The prompt tokens are not a strict prefix of the full conversation "
            f"for sample_id={sample_id!r}; completion-only masking is unsafe."
        )
    if len(full_ids) == len(prompt_ids):
        sample_id = record.get("sample_id", "<unknown>")
        raise ValueError(
            f"sample_id={sample_id!r} has no completion tokens after templating."
        )
    if len(full_ids) > max_length:
        sample_id = record.get("sample_id", "<unknown>")
        raise ValueError(
            f"sample_id={sample_id!r} has {len(full_ids)} tokens, which exceeds "
            f"max_length={max_length}; training data is not truncated silently."
        )

    labels = build_completion_only_labels(prompt_ids, full_ids)
    supervised_token_count = sum(label != -100 for label in labels[1:])
    if supervised_token_count <= 0:
        sample_id = record.get("sample_id", "<unknown>")
        raise ValueError(
            f"sample_id={sample_id!r} has no supervised token after causal shift."
        )

    return {
        "input_ids": full_ids,
        "labels": labels,
        "sample_id": record.get("sample_id"),
        "db_id": record.get("db_id"),
        "prompt_token_count": len(prompt_ids),
        "supervised_token_count": supervised_token_count,
    }


class SFTDataset(Dataset):
    """Tokenize SFT records on demand without creating a disk cache."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if not records:
            raise ValueError("SFTDataset requires at least one record.")
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return tokenize_sft_record(
            self.records[index],
            self.tokenizer,
            self.max_length,
        )


class SFTDataCollator:
    """Dynamically right-pad completion-only SFT batches."""

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer.padding_side != "right":
            raise ValueError(
                "SFT training requires tokenizer.padding_side='right', got "
                f"{tokenizer.padding_side!r}."
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id before collation.")
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty SFT batch.")

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            feature_input_ids = feature["input_ids"]
            feature_labels = feature["labels"]
            if len(feature_input_ids) != len(feature_labels):
                raise ValueError(
                    f"Feature {row} has different input_ids and labels lengths."
                )
            sequence_length = len(feature_input_ids)
            input_ids[row, :sequence_length] = torch.tensor(
                feature_input_ids,
                dtype=torch.long,
            )
            attention_mask[row, :sequence_length] = 1
            labels[row, :sequence_length] = torch.tensor(
                feature_labels,
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "supervised_token_count": torch.tensor(
                sum(feature["supervised_token_count"] for feature in features),
                dtype=torch.long,
            ),
            "sample_ids": [feature["sample_id"] for feature in features],
            "db_ids": [feature["db_id"] for feature in features],
        }


def inspect_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight token and masking details for preflight diagnostics."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have the same 2D shape.")
    if labels.shape != input_ids.shape:
        raise ValueError("labels and input_ids must have the same shape.")

    real_tokens = attention_mask.to(dtype=torch.bool)
    prompt_token_counts = ((labels == -100) & real_tokens).sum(dim=1)
    supervised_token_counts = (labels[:, 1:] != -100).sum(dim=1)
    masked_token_counts = (labels == -100).sum(dim=1)

    return {
        "batch_shape": list(input_ids.shape),
        "padded_sequence_length": input_ids.shape[1],
        "sequence_lengths": attention_mask.sum(dim=1).detach().cpu().tolist(),
        "prompt_token_counts": prompt_token_counts.detach().cpu().tolist(),
        "supervised_token_counts": supervised_token_counts.detach().cpu().tolist(),
        "masked_token_counts": masked_token_counts.detach().cpu().tolist(),
    }
