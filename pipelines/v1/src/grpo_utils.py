"""GRPO math, execution rewards, and durable training-state writers."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.nn.functional as F

from src.evaluation.evaluation_utils import official_execution_match


def _validate_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a finite positive number, got {value!r} "
            f"({type(value).__name__})."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(
            f"{name} must be a finite positive number, got {value!r}."
        )
    return normalized


def _validate_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer and cannot be boolean, "
            f"got {value!r}."
        )
    return value


def execution_reward(
    prediction_result: dict[str, Any],
    gold_result: dict[str, Any],
) -> float:
    """Return the binary official EX reward for two executor results."""
    if (
        prediction_result.get("status") != "success"
        or gold_result.get("status") != "success"
    ):
        return 0.0
    return float(
        official_execution_match(
            prediction_result["rows"],
            gold_result["rows"],
        )
    )


def compute_group_advantages(
    rewards: Sequence[float] | torch.Tensor,
    *,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Normalize one reward group with its sample standard deviation."""
    normalized_epsilon = _validate_positive_number(epsilon, "epsilon")
    if torch.is_tensor(rewards):
        reward_tensor = rewards.to(dtype=torch.float32)
    else:
        reward_tensor = torch.tensor(list(rewards), dtype=torch.float32)
    if reward_tensor.ndim != 1 or reward_tensor.numel() < 2:
        raise ValueError(
            "rewards must be a one-dimensional group containing at least two "
            f"values, got shape={tuple(reward_tensor.shape)}."
        )
    if not torch.isfinite(reward_tensor).all():
        raise ValueError("rewards must contain only finite values.")

    mean = reward_tensor.mean()
    standard_deviation = reward_tensor.std(correction=1)
    return (reward_tensor - mean) / (
        standard_deviation + normalized_epsilon
    )


def build_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int | Sequence[int],
    *,
    exclude_truncated: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask tokens through the first EOS and identify completions without EOS.

    The returned mask includes the first EOS token.  Rows without any EOS are
    marked as truncated; by default their masks are all false so they cannot
    contribute to the policy loss.
    """
    if not torch.is_tensor(completion_ids) or completion_ids.ndim != 2:
        raise ValueError(
            "completion_ids must be a two-dimensional tensor with shape "
            "(batch, completion_length)."
        )
    if completion_ids.shape[1] == 0:
        raise ValueError("completion_ids must contain at least one token.")

    if type(eos_token_id) is int:
        eos_ids = [eos_token_id]
    elif isinstance(eos_token_id, Sequence) and not isinstance(
        eos_token_id,
        (str, bytes),
    ):
        eos_ids = list(eos_token_id)
    else:
        raise TypeError("eos_token_id must be an integer or a sequence of integers.")
    if not eos_ids or any(type(token_id) is not int for token_id in eos_ids):
        raise ValueError(
            "eos_token_id must contain at least one non-boolean integer."
        )

    eos_tensor = torch.tensor(
        eos_ids,
        dtype=completion_ids.dtype,
        device=completion_ids.device,
    )
    eos_hits = (completion_ids.unsqueeze(-1) == eos_tensor).any(dim=-1)
    has_eos = eos_hits.any(dim=-1)
    truncated = ~has_eos
    first_eos_position = eos_hits.to(torch.int64).argmax(dim=-1)
    positions = torch.arange(
        completion_ids.shape[1],
        device=completion_ids.device,
    ).unsqueeze(0)
    mask = positions <= first_eos_position.unsqueeze(1)
    if exclude_truncated:
        mask = mask & has_eos.unsqueeze(1)
    else:
        mask = torch.where(has_eos.unsqueeze(1), mask, torch.ones_like(mask))
    return mask, truncated


def compute_completion_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    completion_length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return differentiable log-probabilities for final Completion tokens.

    Qwen computes only the final ``completion_length + 1`` logits.  Dropping
    the final next-token logit then aligns the remaining positions with the
    final ``completion_length`` token IDs under the usual causal shift.
    """
    normalized_completion_length = _validate_positive_integer(
        completion_length,
        "completion_length",
    )
    normalized_temperature = _validate_positive_number(temperature, "temperature")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a two-dimensional tensor.")
    if input_ids.shape[0] != 1:
        raise ValueError(
            "Completion log-probabilities require per-device batch size 1, "
            f"got {input_ids.shape[0]}."
        )
    if not torch.is_tensor(attention_mask) or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must be a tensor with the same shape as input_ids."
        )
    if normalized_completion_length >= input_ids.shape[1]:
        raise ValueError(
            "completion_length must leave at least one preceding Prompt token, "
            f"got completion_length={normalized_completion_length}, "
            f"sequence_length={input_ids.shape[1]}."
        )
    if not bool(attention_mask[:, -(normalized_completion_length + 1) :].bool().all()):
        raise ValueError(
            "The predictor position and all Completion tokens must be attended."
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=normalized_completion_length + 1,
        use_cache=False,
    )
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(
            "The model returned logits with an unexpected shape: "
            f"{tuple(logits.shape)}."
        )
    if logits.shape[1] < normalized_completion_length + 1:
        raise RuntimeError(
            "The model returned too few logits for causal alignment: "
            f"needed {normalized_completion_length + 1}, got {logits.shape[1]}."
        )

    aligned_logits = logits[:, -(normalized_completion_length + 1) : -1, :]
    aligned_logits = aligned_logits / normalized_temperature
    target_ids = input_ids[:, -normalized_completion_length:]

    if aligned_logits.dtype in (torch.float32, torch.float64):
        selected_logits = aligned_logits.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        log_normalizers = torch.stack(
            [torch.logsumexp(row_logits, dim=-1) for row_logits in aligned_logits]
        )
        return selected_logits - log_normalizers

    per_row_log_probs: list[torch.Tensor] = []
    for row_logits, row_targets in zip(
        aligned_logits,
        target_ids,
        strict=True,
    ):
        row_log_probs = F.log_softmax(row_logits, dim=-1)
        per_row_log_probs.append(
            row_log_probs.gather(
                dim=-1,
                index=row_targets.unsqueeze(-1),
            ).squeeze(-1)
        )
    return torch.stack(per_row_log_probs)


def compute_grpo_token_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    advantage: float | torch.Tensor,
    *,
    epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO-clipped GRPO loss per token and its clipping mask."""
    normalized_epsilon = _validate_positive_number(epsilon, "epsilon")
    if not torch.is_tensor(current_log_probs) or current_log_probs.ndim != 2:
        raise ValueError(
            "current_log_probs must have shape (batch, completion_length)."
        )
    if old_log_probs is None:
        old_log_probs = current_log_probs.detach()
    if not torch.is_tensor(old_log_probs) or old_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            "old_log_probs must be a tensor with the same shape as "
            "current_log_probs."
        )

    if torch.is_tensor(advantage):
        advantage_tensor = advantage.to(
            device=current_log_probs.device,
            dtype=current_log_probs.dtype,
        )
    else:
        advantage_tensor = torch.tensor(
            advantage,
            device=current_log_probs.device,
            dtype=current_log_probs.dtype,
        )
    if advantage_tensor.numel() != 1 or not torch.isfinite(advantage_tensor).all():
        raise ValueError("advantage must be one finite scalar.")

    log_ratio = current_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - normalized_epsilon,
        1.0 + normalized_epsilon,
    )
    unclipped_objective = ratio * advantage_tensor
    clipped_objective = clipped_ratio * advantage_tensor
    token_loss = -torch.minimum(unclipped_objective, clipped_objective)
    clipped_mask = unclipped_objective > clipped_objective
    return token_loss, clipped_mask


def scale_dapo_loss(
    local_loss_sum: torch.Tensor,
    *,
    global_valid_token_count: int | torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Scale a local token-loss sum for DDP's gradient averaging.

    DDP averages gradients across ranks.  Multiplication by ``world_size``
    therefore makes the accumulated gradient equal to the global valid-token
    mean after every rank divides by the same global token count.
    """
    normalized_world_size = _validate_positive_integer(world_size, "world_size")
    if not torch.is_tensor(local_loss_sum) or local_loss_sum.numel() != 1:
        raise ValueError("local_loss_sum must be a scalar tensor.")
    if torch.is_tensor(global_valid_token_count):
        if global_valid_token_count.numel() != 1:
            raise ValueError("global_valid_token_count must be scalar.")
        denominator = global_valid_token_count.to(
            device=local_loss_sum.device,
            dtype=torch.float32,
        )
        if not bool((denominator > 0).item()):
            raise ValueError("global_valid_token_count must be positive.")
    else:
        normalized_count = _validate_positive_integer(
            global_valid_token_count,
            "global_valid_token_count",
        )
        denominator = torch.tensor(
            normalized_count,
            device=local_loss_sum.device,
            dtype=torch.float32,
        )
    return local_loss_sum.float() * normalized_world_size / denominator


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _atomic_path(destination)
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _atomic_path(destination)
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_jsonl_flush(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record and make it durable before returning."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
