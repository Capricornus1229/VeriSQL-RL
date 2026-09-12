"""Train the frozen V1 Qwen3-8B BIRD GRPO policy.

The completed run publishes only the final Adapter and compact reports.  Large
screening shards, online rollouts, and resumable checkpoints live in the
configured transient work directory and are removed after successful
publication.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import torch
import yaml
from accelerate import Accelerator
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from pipelines.v1.src.grpo_utils import (
    append_jsonl_flush,
    atomic_write_json,
    atomic_write_jsonl,
    build_completion_mask,
    compute_completion_log_probs,
    compute_grpo_token_loss,
    compute_group_advantages,
    execution_reward,
    scale_dapo_loss,
)
from pipelines.v1.src.sql_extraction import extract_predicted_sql
from src.common import load_json, load_jsonl
from src.evaluation.evaluation_utils import percentage, validate_run_name
from src.execution.sql_executor import execute_sql


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "grpo.yaml"
EXPECTED_TRAIN_SAMPLES = 6_013
EXPECTED_VALIDATION_SAMPLES = 588
EXPECTED_WORLD_SIZE = 2
FINAL_ADAPTER_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}
GOLD_COMPLETION_PATTERN = re.compile(
    r"\A```sql\r?\n(?P<sql>[\s\S]*)\r?\n```\Z",
    flags=re.IGNORECASE,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _portable_project_path(value: str | Path) -> str:
    path = _project_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer, got {value!r}.")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"GRPO config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("GRPO config must be a YAML object.")
    return config


def validate_config(config: dict[str, Any]) -> None:
    sections = ("run", "model", "data", "generation", "reward", "grpo", "evaluation")
    for section in sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"GRPO config is missing object section {section!r}.")

    run = config["run"]
    model = config["model"]
    data = config["data"]
    generation = config["generation"]
    reward = config["reward"]
    grpo = config["grpo"]
    evaluation = config["evaluation"]

    run_name = validate_run_name(run.get("run_name"))
    if run.get("expected_num_processes") != EXPECTED_WORLD_SIZE:
        raise ValueError("run.expected_num_processes must be 2.")
    _positive_int(run.get("seed"), "run.seed")
    expected_model_dir = (PIPELINE_ROOT / "runs" / "grpo" / "model").resolve()
    expected_report_dir = (PIPELINE_ROOT / "runs" / "grpo").resolve()
    expected_work_dir = expected_model_dir / ".work"
    model_dir = _project_path(run.get("model_output_dir", ""))
    report_dir = _project_path(run.get("training_report_dir", ""))
    work_dir = _project_path(run.get("work_dir", ""))
    if model_dir != expected_model_dir:
        raise ValueError(
            f"run.model_output_dir must resolve to {expected_model_dir}."
        )
    if report_dir != expected_report_dir:
        raise ValueError(
            f"run.training_report_dir must resolve to {expected_report_dir}."
        )
    if work_dir != expected_work_dir:
        raise ValueError(f"run.work_dir must resolve to {expected_work_dir}.")

    for key in ("model_path", "tokenizer_path", "sft_adapter_path"):
        path = _project_path(model.get(key, ""))
        if not path.exists():
            raise FileNotFoundError(f"model.{key} does not exist: {path}")
    if model.get("dtype") != "bfloat16":
        raise ValueError("model.dtype must be 'bfloat16'.")
    if model.get("attention_implementation") != "sdpa":
        raise ValueError("model.attention_implementation must be 'sdpa'.")
    if model.get("enable_thinking") is not False:
        raise ValueError("model.enable_thinking must be false.")
    if model.get("gradient_checkpointing") is not True:
        raise ValueError("model.gradient_checkpointing must be true.")
    if float(model.get("lora_dropout", -1)) != 0.0:
        raise ValueError("model.lora_dropout must be 0.0 for on-policy scoring.")
    _positive_int(model.get("max_context_length"), "model.max_context_length")

    for key in (
        "sft_train_path",
        "sft_val_path",
        "train_schema_catalog_path",
        "gold_execution_path",
        "sft_baseline_metrics_path",
    ):
        path = _project_path(data.get(key, ""))
        if not path.is_file():
            raise FileNotFoundError(f"data.{key} does not exist: {path}")

    if generation.get("num_generations") != 4:
        raise ValueError("generation.num_generations must be 4.")
    micro_batch_size = _positive_int(
        generation.get("micro_batch_size"), "generation.micro_batch_size"
    )
    if 4 % micro_batch_size != 0:
        raise ValueError("generation.micro_batch_size must divide 4.")
    _positive_int(generation.get("max_new_tokens"), "generation.max_new_tokens")
    _positive_number(generation.get("temperature"), "generation.temperature")
    if float(generation.get("top_p", -1)) != 1.0:
        raise ValueError("generation.top_p must be 1.0 for this experiment.")
    if generation.get("top_k") != 0:
        raise ValueError("generation.top_k must be 0 for this experiment.")

    _positive_number(reward.get("timeout_seconds"), "reward.timeout_seconds")
    _positive_int(reward.get("max_rows"), "reward.max_rows")
    if reward.get("exclude_empty_gold") is not True:
        raise ValueError("reward.exclude_empty_gold must be true.")

    if grpo.get("num_epochs") != 1 or grpo.get("num_iterations") != 1:
        raise ValueError("This experiment requires one epoch and num_iterations=1.")
    if grpo.get("loss_type") != "dapo" or float(grpo.get("beta", -1)) != 0.0:
        raise ValueError("This experiment requires DAPO normalization and beta=0.")
    for key in (
        "learning_rate",
        "adam_epsilon",
        "clip_epsilon",
        "max_grad_norm",
    ):
        _positive_number(grpo.get(key), f"grpo.{key}")
    if not 0 <= float(grpo.get("warmup_ratio", -1)) < 1:
        raise ValueError("grpo.warmup_ratio must be in [0, 1).")
    if grpo.get("scheduler_type") != "cosine":
        raise ValueError("grpo.scheduler_type must be 'cosine'.")
    _positive_int(
        grpo.get("checkpoint_every_optimizer_steps"),
        "grpo.checkpoint_every_optimizer_steps",
    )
    _positive_int(grpo.get("keep_last_checkpoints"), "grpo.keep_last_checkpoints")

    _positive_int(evaluation.get("max_new_tokens"), "evaluation.max_new_tokens")
    _positive_number(evaluation.get("timeout_seconds"), "evaluation.timeout_seconds")
    baseline_dev_ex = evaluation.get("baseline_dev_ex")
    if (
        isinstance(baseline_dev_ex, bool)
        or not isinstance(baseline_dev_ex, (int, float))
        or not 0 <= float(baseline_dev_ex) <= 100
    ):
        raise ValueError("evaluation.baseline_dev_ex must be between 0 and 100.")


def _resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    for key in ("model_output_dir", "training_report_dir", "work_dir"):
        resolved["run"][key] = _portable_project_path(config["run"][key])
    for key in (
        "model_path",
        "tokenizer_path",
        "sft_adapter_path",
    ):
        resolved["model"][key] = _portable_project_path(config["model"][key])
    for key in config["data"]:
        resolved["data"][key] = _portable_project_path(config["data"][key])
    return resolved


def _build_accelerator(config: dict[str, Any]) -> Accelerator:
    accelerator = Accelerator(mixed_precision="bf16")
    expected = config["run"]["expected_num_processes"]
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"GRPO requires {expected} processes, found {accelerator.num_processes}. "
            "Use pipelines/v1/scripts/run_grpo.sh train."
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError("GRPO requires CUDA GPUs.")
    return accelerator


def _publish_run_config(config_path: Path, destination: Path) -> None:
    payload = config_path.read_bytes()
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise ValueError(f"Published GRPO config differs from {config_path}.")
        return
    if destination.exists():
        raise ValueError(f"GRPO run config destination is not a file: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_work_directory(
    accelerator: Accelerator,
    config: dict[str, Any],
    config_path: Path,
    *,
    resume: bool,
) -> Path:
    work_dir = _project_path(config["run"]["work_dir"])
    resolved = _resolved_config(config)
    resolved_path = work_dir / "resolved_config.json"
    run_config_path = (
        _project_path(config["run"]["model_output_dir"]) / "run_config.yaml"
    )
    if accelerator.is_main_process:
        _publish_run_config(config_path, run_config_path)
        if resume:
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume because transient config is missing: {resolved_path}"
                )
            if load_json(resolved_path) != resolved:
                raise ValueError(
                    "Current GRPO config differs from the config used by this run."
                )
        else:
            if work_dir.exists() and any(work_dir.iterdir()):
                raise FileExistsError(
                    f"Fresh GRPO work directory is not empty: {work_dir}. Use --resume."
                )
            work_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(resolved_path, resolved)
    accelerator.wait_for_everyone()
    return work_dir


def _set_seed(seed: int, rank: int = 0) -> None:
    rank_seed = seed + rank * 100_003
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def _validate_prompt(prompt: object, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"{sample_id}: prompt must contain system and user messages.")
    result: list[dict[str, str]] = []
    for index, role in enumerate(("system", "user")):
        message = prompt[index]
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"{sample_id}: prompt[{index}] must have role {role!r}.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{sample_id}: prompt[{index}] content is empty.")
        result.append({"role": role, "content": content})
    return result


def _extract_gold_sql(record: dict[str, Any]) -> str:
    sample_id = record["sample_id"]
    completion = record.get("completion")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"{sample_id}: completion must contain one assistant message.")
    message = completion[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError(f"{sample_id}: completion role must be assistant.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{sample_id}: completion content must be a string.")
    match = GOLD_COMPLETION_PATTERN.fullmatch(content)
    if match is None or not match.group("sql").strip():
        raise ValueError(f"{sample_id}: completion must contain exactly one SQL fence.")
    return match.group("sql")


def _load_sft_records(path: Path, expected_count: int, split: str) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if len(records) != expected_count:
        raise ValueError(f"{split} expected {expected_count} samples, got {len(records)}.")
    seen: set[str] = set()
    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{split}[{index}] has invalid sample_id {sample_id!r}.")
        if sample_id in seen:
            raise ValueError(f"{split} contains duplicate sample_id {sample_id!r}.")
        seen.add(sample_id)
        if type(record.get("source_index")) is not int:
            raise ValueError(f"{sample_id}: source_index must be a non-boolean integer.")
        if not isinstance(record.get("db_id"), str) or not record["db_id"]:
            raise ValueError(f"{sample_id}: db_id must be a non-empty string.")
        _validate_prompt(record.get("prompt"), sample_id)
        _extract_gold_sql(record)
    return records


def _catalog_database_paths(path: Path) -> dict[str, Path]:
    catalog = load_json(path)
    if not isinstance(catalog, list):
        raise ValueError(f"Schema Catalog must be a list: {path}")
    result: dict[str, Path] = {}
    for row in catalog:
        if not isinstance(row, dict):
            raise ValueError(f"Schema Catalog contains a non-object row: {path}")
        db_id = row.get("db_id")
        sqlite_path = row.get("sqlite_path")
        if not isinstance(db_id, str) or not isinstance(sqlite_path, str):
            raise ValueError(f"Schema Catalog has invalid db_id/sqlite_path: {row!r}")
        if db_id in result:
            raise ValueError(f"Schema Catalog contains duplicate db_id {db_id!r}.")
        database_path = _project_path(sqlite_path)
        if not database_path.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {database_path}")
        result[db_id] = database_path
    return result


def _load_gold_report(path: Path) -> dict[str, dict[str, Any]]:
    reports = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for report in reports:
        sample_id = report.get("sample_id")
        if isinstance(sample_id, str) and sample_id.startswith("train_"):
            if sample_id in result:
                raise ValueError(f"Gold report contains duplicate {sample_id!r}.")
            result[sample_id] = report
    return result


def _build_training_inputs(
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Path],
    dict[str, dict[str, Any]],
]:
    data = config["data"]
    train = _load_sft_records(
        _project_path(data["sft_train_path"]), EXPECTED_TRAIN_SAMPLES, "sft_train"
    )
    validation = _load_sft_records(
        _project_path(data["sft_val_path"]),
        EXPECTED_VALIDATION_SAMPLES,
        "sft_val",
    )
    train_databases = _catalog_database_paths(
        _project_path(data["train_schema_catalog_path"])
    )
    gold_report = _load_gold_report(_project_path(data["gold_execution_path"]))

    train_db_ids = {record["db_id"] for record in train + validation}
    if train_db_ids != set(train_databases):
        raise ValueError("SFT Train/Validation db_id values do not match Train Catalog.")
    if len(set(gold_report) & {record["sample_id"] for record in train}) != len(train):
        raise ValueError("Gold execution report does not cover every SFT Train sample.")
    return train, validation, train_databases, gold_report


def _gold_report_exclusion_reason(
    record: dict[str, Any],
    report: dict[str, Any],
    config: dict[str, Any],
    database_path: Path,
) -> str | None:
    execution = report.get("gold_execution")
    if not isinstance(execution, dict):
        return "invalid_gold_report"
    if report.get("db_id") != record["db_id"]:
        raise ValueError(f"{record['sample_id']}: Gold report db_id mismatch.")
    if report.get("gold_sql") != _extract_gold_sql(record):
        raise ValueError(f"{record['sample_id']}: Gold report SQL mismatch.")
    if _project_path(report.get("sqlite_path", "")) != database_path:
        raise ValueError(f"{record['sample_id']}: Gold report SQLite path mismatch.")
    if report.get("usable") is not True or execution.get("status") != "success":
        return f"gold_report_{execution.get('status', 'unusable')}"
    if config["reward"]["exclude_empty_gold"] and execution.get("empty_result") is True:
        return "empty_gold_result"
    if float(execution.get("elapsed_ms", math.inf)) > (
        float(config["reward"]["timeout_seconds"]) * 1_000
    ):
        return "gold_report_timeout_limit"
    if int(execution.get("row_count", config["reward"]["max_rows"] + 1)) > int(
        config["reward"]["max_rows"]
    ):
        return "gold_report_row_limit"
    return None


def _screen_candidates(
    train: list[dict[str, Any]],
    train_databases: dict[str, Path],
    gold_report: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    excluded: dict[str, str] = {}
    eligible: list[dict[str, Any]] = []
    for record in train:
        reason = _gold_report_exclusion_reason(
            record,
            gold_report[record["sample_id"]],
            config,
            train_databases[record["db_id"]],
        )
        if reason is None:
            eligible.append(record)
        else:
            excluded[record["sample_id"]] = reason

    return eligible, excluded


def _load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        _project_path(config["model"]["tokenizer_path"]), use_fast=True
    )
    if not tokenizer.chat_template:
        raise ValueError("Configured Tokenizer has no Chat Template.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Configured Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _disable_lora_dropout(model: PeftModel) -> None:
    for peft_config in model.peft_config.values():
        peft_config.lora_dropout = 0.0
    for module in model.modules():
        dropout_modules = getattr(module, "lora_dropout", None)
        if dropout_modules is None:
            continue
        for dropout in dropout_modules.values():
            if isinstance(dropout, torch.nn.Dropout):
                dropout.p = 0.0


def _load_model(
    config: dict[str, Any], adapter_path: Path
) -> tuple[PeftModel, dict[str, int | float]]:
    base = AutoModelForCausalLM.from_pretrained(
        _project_path(config["model"]["model_path"]),
        dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    _disable_lora_dropout(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if not trainable or unexpected:
        raise RuntimeError(
            "Only LoRA parameters may be trainable; unexpected="
            f"{unexpected[:1]!r}."
        )
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    return model, {
        "total_parameters": total,
        "trainable_parameters": trainable_count,
        "trainable_ratio": 100.0 * trainable_count / total,
    }


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    grpo = config["grpo"]
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(grpo["learning_rate"]),
        weight_decay=float(grpo["weight_decay"]),
        betas=(float(grpo["beta1"]), float(grpo["beta2"])),
        eps=float(grpo["adam_epsilon"]),
    )


def _summary_execution(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "rows"}


def _prompt_encoding(tokenizer: Any, record: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        record["prompt"],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise RuntimeError(f"{record.get('sample_id', record.get('question_id'))}: bad Prompt encoding.")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _eos_ids(tokenizer: Any, model: Any) -> list[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    return [value] if type(value) is int else list(value)


def _trim_completion(row: torch.Tensor, eos_ids: Sequence[int]) -> tuple[torch.Tensor, bool]:
    values = row.tolist()
    eos_set = set(eos_ids)
    first_eos = next((index for index, token in enumerate(values) if token in eos_set), None)
    if first_eos is None:
        return row, True
    return row[: first_eos + 1], False


def _sampling_seed(base_seed: int, source_index: int, phase: int, offset: int = 0) -> int:
    return int((base_seed * 1_000_003 + source_index * 9_176 + phase * 97_409 + offset) % (2**31 - 1))


def _generate_sampled_group(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    unwrapped = accelerator.unwrap_model(model)
    encoded = _prompt_encoding(tokenizer, record, accelerator.device)
    prompt_length = encoded["input_ids"].shape[1]
    settings = config["generation"]
    max_new_tokens = int(settings["max_new_tokens"])
    context_limit = min(
        int(config["model"]["max_context_length"]),
        int(getattr(unwrapped.config, "max_position_embeddings", 2**31 - 1)),
    )
    if prompt_length + max_new_tokens > context_limit:
        raise ValueError(
            f"{record['sample_id']}: prompt={prompt_length} plus generation="
            f"{max_new_tokens} exceeds context={context_limit}."
        )

    was_training = model.training
    unwrapped.eval()
    eos_ids = _eos_ids(tokenizer, unwrapped)
    generated: list[dict[str, Any]] = []
    micro_batch = int(settings["micro_batch_size"])
    try:
        for start in range(0, 4, micro_batch):
            count = min(micro_batch, 4 - start)
            _set_seed(seed + start, accelerator.process_index)
            with torch.inference_mode():
                output = unwrapped.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    num_return_sequences=count,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            new_ids = output[:, prompt_length:]
            for row in new_ids:
                completion_ids, truncated = _trim_completion(row, eos_ids)
                raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)
                sql, status, compliance = extract_predicted_sql(raw_output)
                generated.append(
                    {
                        "ids": completion_ids.detach(),
                        "truncated": truncated,
                        "raw_output": raw_output,
                        "predicted_sql": sql,
                        "extraction_status": status,
                        "format_compliance": compliance,
                    }
                )
    finally:
        if was_training:
            model.train()
    if len(generated) != 4:
        raise RuntimeError(f"Expected four completions, generated {len(generated)}.")
    return encoded, generated


def _score_generated_group(
    record: dict[str, Any],
    generated: list[dict[str, Any]],
    database_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]]]:
    reward_config = config["reward"]
    gold_result = execute_sql(
        database_path,
        _extract_gold_sql(record),
        timeout_seconds=reward_config["timeout_seconds"],
        max_rows=reward_config["max_rows"],
    )
    if gold_result["status"] != "success":
        return gold_result, [], [
            {
                "raw_output": completion["raw_output"],
                "predicted_sql": completion["predicted_sql"],
                "extraction_status": completion["extraction_status"],
                "format_compliance": completion["format_compliance"],
                "truncated": completion["truncated"],
                "completion_tokens": int(completion["ids"].numel()),
                "reward": None,
                "execution": None,
            }
            for completion in generated
        ]
    if reward_config["exclude_empty_gold"] and gold_result["empty_result"]:
        return gold_result, [], [
            {
                "raw_output": completion["raw_output"],
                "predicted_sql": completion["predicted_sql"],
                "extraction_status": completion["extraction_status"],
                "format_compliance": completion["format_compliance"],
                "truncated": completion["truncated"],
                "completion_tokens": int(completion["ids"].numel()),
                "reward": None,
                "execution": None,
            }
            for completion in generated
        ]

    rewards: list[float] = []
    scored: list[dict[str, Any]] = []
    for completion in generated:
        prediction_result = execute_sql(
            database_path,
            completion["predicted_sql"],
            timeout_seconds=reward_config["timeout_seconds"],
            max_rows=reward_config["max_rows"],
        )
        reward = execution_reward(prediction_result, gold_result)
        rewards.append(reward)
        scored.append(
            {
                "raw_output": completion["raw_output"],
                "predicted_sql": completion["predicted_sql"],
                "extraction_status": completion["extraction_status"],
                "format_compliance": completion["format_compliance"],
                "truncated": completion["truncated"],
                "completion_tokens": int(completion["ids"].numel()),
                "reward": reward,
                "execution": _summary_execution(prediction_result),
            }
        )
    return gold_result, rewards, scored


def _validate_screen_prefix(
    rows: list[dict[str, Any]], assigned: list[dict[str, Any]], path: Path
) -> None:
    if len(rows) > len(assigned):
        raise ValueError(f"Screen shard has too many rows: {path}")
    for index, row in enumerate(rows):
        if row.get("sample_id") != assigned[index]["sample_id"]:
            raise ValueError(
                f"Screen shard {path} is not its deterministic prefix at row {index}."
            )


def _run_screening(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    candidates: list[dict[str, Any]],
    prefilter_exclusions: dict[str, str],
    train_databases: dict[str, Path],
    config: dict[str, Any],
    run_dir: Path,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    screen_dir = run_dir / "screening"
    merged_path = screen_dir / "pass_at_4.jsonl"
    mixed_path = screen_dir / "mixed_prompts.jsonl"
    summary_path = screen_dir / "summary.json"
    if merged_path.is_file() and mixed_path.is_file() and summary_path.is_file():
        mixed = load_jsonl(mixed_path)
        mixed_ids = [row.get("sample_id") for row in mixed]
        if (
            len(mixed_ids) != len(set(mixed_ids))
            or any(row.get("pass_count") not in (1, 2, 3) for row in mixed)
        ):
            raise ValueError(f"Existing mixed prompt file is invalid: {mixed_path}")
        accelerator.wait_for_everyone()
        return mixed, load_json(summary_path)

    screen_dir.mkdir(parents=True, exist_ok=True)
    rank = accelerator.process_index
    assigned = candidates[rank:: accelerator.num_processes]
    shard_path = screen_dir / f"pass_at_4.rank_{rank:03d}.partial.jsonl"
    existing = load_jsonl(shard_path) if shard_path.is_file() else []
    if existing and not resume:
        raise FileExistsError(f"Fresh screening found an existing shard: {shard_path}")
    _validate_screen_prefix(existing, assigned, shard_path)

    for local_index, record in enumerate(assigned[len(existing) :], start=len(existing)):
        source_index = record["source_index"]
        seed = _sampling_seed(config["run"]["seed"], source_index, phase=1)
        _, generated = _generate_sampled_group(
            accelerator,
            model,
            tokenizer,
            record,
            config,
            seed=seed,
        )
        gold_result, rewards, scored = _score_generated_group(
            record,
            generated,
            train_databases[record["db_id"]],
            config,
        )
        eligible = bool(rewards)
        exclusion_reason = None
        if not eligible:
            exclusion_reason = (
                "empty_gold_result"
                if gold_result.get("status") == "success"
                and gold_result.get("empty_result")
                else f"gold_runtime_{gold_result.get('status', 'invalid')}"
            )
        append_jsonl_flush(
            shard_path,
            {
                "sample_id": record["sample_id"],
                "source_index": source_index,
                "db_id": record["db_id"],
                "eligible": eligible,
                "exclusion_reason": exclusion_reason,
                "pass_count": int(sum(rewards)) if eligible else None,
                "gold_execution": _summary_execution(gold_result),
                "candidates": scored,
            },
        )
        if (local_index + 1) % 25 == 0:
            accelerator.print(
                f"Screening progress per rank: rank={rank}, "
                f"completed={local_index + 1}/{len(assigned)}"
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        by_id: dict[str, dict[str, Any]] = {}
        for process_rank in range(accelerator.num_processes):
            rows = load_jsonl(
                screen_dir / f"pass_at_4.rank_{process_rank:03d}.partial.jsonl"
            )
            expected = candidates[process_rank:: accelerator.num_processes]
            _validate_screen_prefix(rows, expected, shard_path)
            if len(rows) != len(expected):
                raise RuntimeError(
                    f"Rank {process_rank} screening is incomplete: "
                    f"{len(rows)}/{len(expected)}."
                )
            for row in rows:
                if row["sample_id"] in by_id:
                    raise RuntimeError(f"Duplicate screened sample {row['sample_id']!r}.")
                by_id[row["sample_id"]] = row
        merged = [by_id[record["sample_id"]] for record in candidates]
        buckets = Counter(
            row["pass_count"] for row in merged if row["eligible"] is True
        )
        mixed = [
            {
                "sample_id": row["sample_id"],
                "source_index": row["source_index"],
                "db_id": row["db_id"],
                "pass_count": row["pass_count"],
            }
            for row in merged
            if row["pass_count"] in (1, 2, 3)
        ]
        summary = {
            "source_train_samples": EXPECTED_TRAIN_SAMPLES,
            "prefilter_excluded_count": len(prefilter_exclusions),
            "prefilter_exclusion_reasons": dict(
                sorted(Counter(prefilter_exclusions.values()).items())
            ),
            "screen_candidate_count": len(candidates),
            "runtime_eligible_count": sum(row["eligible"] for row in merged),
            "runtime_excluded_count": sum(not row["eligible"] for row in merged),
            "pass_at_4_buckets": {str(value): buckets[value] for value in range(5)},
            "mixed_prompt_count": len(mixed),
        }
        atomic_write_jsonl(merged_path, merged)
        atomic_write_jsonl(mixed_path, mixed)
        atomic_write_json(summary_path, summary)
    accelerator.wait_for_everyone()
    return load_jsonl(mixed_path), load_json(summary_path)


def _checkpoint_number(path: Path) -> int:
    return int(path.name.removeprefix("step_"))


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in checkpoint_dir.glob("step_*")
            if path.is_dir() and path.name.removeprefix("step_").isdigit()
        ],
        key=_checkpoint_number,
    )


def _latest_checkpoint(run_dir: Path) -> Path | None:
    paths = _checkpoint_paths(run_dir)
    return paths[-1] if paths else None


def _save_adapter_atomic(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    destination: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if destination.exists():
            raise FileExistsError(f"Adapter destination already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            accelerator.unwrap_model(model).save_pretrained(
                temporary, safe_serialization=True
            )
            tokenizer.save_pretrained(temporary)
            (temporary / "README.md").unlink(missing_ok=True)
            actual_files = {path.name for path in temporary.iterdir() if path.is_file()}
            if actual_files != FINAL_ADAPTER_FILES:
                raise RuntimeError(
                    "Final Adapter file contract changed: "
                    f"expected={sorted(FINAL_ADAPTER_FILES)!r}, "
                    f"actual={sorted(actual_files)!r}."
                )
            os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    accelerator.wait_for_everyone()


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer: Any,
    run_dir: Path,
    config: dict[str, Any],
    *,
    optimizer_step: int,
    next_pair_cursor: int,
    rollout_rows_per_rank: int,
) -> Path:
    destination = run_dir / "checkpoints" / f"step_{optimizer_step:06d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            adapter_dir = temporary / "adapter"
            accelerator.unwrap_model(model).save_pretrained(
                adapter_dir, safe_serialization=True
            )
            tokenizer.save_pretrained(adapter_dir)
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            atomic_write_json(
                temporary / "trainer_state.json",
                {
                    "optimizer_step": optimizer_step,
                    "next_pair_cursor": next_pair_cursor,
                    "rollout_rows_per_rank": rollout_rows_per_rank,
                },
            )
            atomic_write_json(temporary / "resolved_config.json", config)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

        checkpoints = _checkpoint_paths(run_dir)
        keep = int(config["grpo"]["keep_last_checkpoints"])
        for obsolete in checkpoints[:-keep]:
            shutil.rmtree(obsolete)
    accelerator.wait_for_everyone()
    return destination


def _load_checkpoint_state(checkpoint: Path) -> dict[str, int]:
    state = load_json(checkpoint / "trainer_state.json")
    required = ("optimizer_step", "next_pair_cursor", "rollout_rows_per_rank")
    if any(type(state.get(key)) is not int or state[key] < 0 for key in required):
        raise ValueError(f"Invalid trainer state in {checkpoint}.")
    return state


def _restore_optimizer_scheduler(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    )


def _ordered_mixed_ids(
    mixed: list[dict[str, Any]], seed: int
) -> tuple[list[str], bool]:
    sample_ids = [row["sample_id"] for row in mixed]
    random.Random(seed + 71_011).shuffle(sample_ids)
    duplicated = len(sample_ids) % EXPECTED_WORLD_SIZE != 0
    if duplicated:
        sample_ids.append(sample_ids[-1])
    return sample_ids, duplicated


def _truncate_rollout_log(path: Path, count: int) -> None:
    rows = load_jsonl(path) if path.is_file() else []
    if len(rows) < count:
        raise RuntimeError(
            f"Rollout log {path} has {len(rows)} rows but checkpoint needs {count}."
        )
    if len(rows) != count:
        atomic_write_jsonl(path, rows[:count])


def _truncate_main_training_metrics(
    accelerator: Accelerator, path: Path, next_pair_cursor: int
) -> None:
    if accelerator.is_main_process and path.is_file():
        rows = load_jsonl(path)
        retained = [
            row for row in rows if int(row.get("pair_cursor", -1)) <= next_pair_cursor
        ]
        if len(retained) != len(rows):
            atomic_write_jsonl(path, retained)


def _reduce_sum(accelerator: Accelerator, value: int | float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=accelerator.device)
    return float(accelerator.reduce(tensor, reduction="sum").item())


def _run_grpo_training(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    mixed: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    train_databases: dict[str, Path],
    config: dict[str, Any],
    run_dir: Path,
    *,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    final_adapter = run_dir / "final_adapter"
    summary_path = run_dir / "training_summary.json"
    if final_adapter.is_dir() and not summary_path.is_file() and resume_checkpoint:
        if accelerator.is_main_process:
            shutil.rmtree(final_adapter)
        accelerator.wait_for_everyone()
    elif final_adapter.is_dir() != summary_path.is_file():
        raise RuntimeError(
            "Final Adapter and training summary must either both exist or both "
            f"be absent in {run_dir}."
        )
    if final_adapter.is_dir() and summary_path.is_file():
        accelerator.wait_for_everyone()
        return load_json(summary_path)

    ordered_ids, duplicated_last_prompt = _ordered_mixed_ids(
        mixed, int(config["run"]["seed"])
    )
    pair_count = len(ordered_ids) // accelerator.num_processes
    if pair_count == 0:
        raise RuntimeError("No mixed prompts are available for GRPO training.")
    planned_scheduler_steps = pair_count
    warmup_steps = math.ceil(
        planned_scheduler_steps * float(config["grpo"]["warmup_ratio"])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_scheduler_steps,
    )

    rank = accelerator.process_index
    rollout_path = run_dir / "training" / f"online_rollouts.rank_{rank:03d}.jsonl"
    train_metrics_path = run_dir / "training" / "train_metrics.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_step = 0
    pair_cursor = 0
    if resume_checkpoint is not None:
        state = _load_checkpoint_state(resume_checkpoint)
        optimizer_step = state["optimizer_step"]
        pair_cursor = state["next_pair_cursor"]
        _restore_optimizer_scheduler(resume_checkpoint, optimizer, scheduler)
        _truncate_rollout_log(rollout_path, state["rollout_rows_per_rank"])
        _truncate_main_training_metrics(
            accelerator, train_metrics_path, state["next_pair_cursor"]
        )
    else:
        # No model checkpoint means that partial online updates cannot be replayed.
        if rollout_path.exists():
            atomic_write_jsonl(rollout_path, [])
        if accelerator.is_main_process:
            atomic_write_jsonl(train_metrics_path, [])
    accelerator.wait_for_everyone()

    prior_rollouts = load_jsonl(rollout_path) if rollout_path.is_file() else []

    tracked_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    tracked_before = tracked_parameter.detach().clone()
    base_frozen = all(
        ("lora_" in name) or not parameter.requires_grad
        for name, parameter in model.named_parameters()
    )
    resumed_with_updates = resume_checkpoint is not None and optimizer_step > 0
    gradient_seen = resumed_with_updates
    zero_signal_local_groups = sum(
        not row.get("local_has_signal", row["update_performed"])
        for row in prior_rollouts
    )
    globally_skipped_pairs = sum(
        not row["update_performed"] for row in prior_rollouts
    )
    processed_pairs = pair_cursor
    total_local_completions = sum(len(row.get("candidates", [])) for row in prior_rollouts)
    total_local_truncated = sum(
        sum(bool(candidate.get("truncated")) for candidate in row.get("candidates", []))
        for row in prior_rollouts
    )
    started_at = time.perf_counter()
    grpo_config = config["grpo"]

    while pair_cursor < pair_count:
        sample_id = ordered_ids[pair_cursor * accelerator.num_processes + rank]
        record = records_by_id[sample_id]
        seed = _sampling_seed(
            int(config["run"]["seed"]),
            int(record["source_index"]),
            phase=2,
            offset=pair_cursor * accelerator.num_processes + rank,
        )
        prompt_encoding, generated = _generate_sampled_group(
            accelerator,
            model,
            tokenizer,
            record,
            config,
            seed=seed,
        )
        gold_result, rewards, scored = _score_generated_group(
            record,
            generated,
            train_databases[record["db_id"]],
            config,
        )
        scorable = len(rewards) == 4
        if scorable:
            advantages = compute_group_advantages(rewards).to(accelerator.device)
        else:
            rewards = [0.0] * 4
            advantages = torch.zeros(4, dtype=torch.float32, device=accelerator.device)

        masks: list[torch.Tensor] = []
        eos_ids = _eos_ids(tokenizer, accelerator.unwrap_model(model))
        for completion in generated:
            ids = completion["ids"].unsqueeze(0)
            mask, truncated = build_completion_mask(
                ids,
                eos_ids,
                exclude_truncated=True,
            )
            if not scorable:
                mask = torch.zeros_like(mask)
            masks.append(mask)
            total_local_completions += 1
            total_local_truncated += int(truncated.item())

        local_has_signal = scorable and any(
            bool(mask.any().item())
            and bool((advantages[index] != 0).item())
            for index, mask in enumerate(masks)
        )
        zero_signal_local_groups += int(not local_has_signal)
        global_signal_groups = int(round(_reduce_sum(accelerator, int(local_has_signal))))
        local_valid_tokens = sum(int(mask.sum().item()) for mask in masks)
        global_valid_tokens = int(round(_reduce_sum(accelerator, local_valid_tokens)))
        update_performed = global_signal_groups > 0
        local_loss_value = 0.0
        local_clipped_tokens = 0
        gradient_norm_value: float | None = None

        if update_performed:
            if global_valid_tokens <= 0:
                raise RuntimeError("A signal-bearing GRPO step has no valid tokens.")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for completion_index, (completion, mask) in enumerate(
                zip(generated, masks, strict=True)
            ):
                context = (
                    accelerator.no_sync(model)
                    if completion_index < len(generated) - 1
                    else nullcontext()
                )
                with context:
                    completion_ids = completion["ids"].unsqueeze(0)
                    input_ids = torch.cat(
                        [prompt_encoding["input_ids"], completion_ids], dim=1
                    )
                    attention_mask = torch.cat(
                        [
                            prompt_encoding["attention_mask"],
                            torch.ones_like(completion_ids),
                        ],
                        dim=1,
                    )
                    current_log_probs = compute_completion_log_probs(
                        model,
                        input_ids,
                        attention_mask,
                        completion_length=completion_ids.shape[1],
                        temperature=float(config["generation"]["temperature"]),
                    )
                    token_loss, clipped = compute_grpo_token_loss(
                        current_log_probs,
                        None,
                        advantages[completion_index],
                        epsilon=float(grpo_config["clip_epsilon"]),
                    )
                    local_numerator = (token_loss.float() * mask).sum()
                    loss = scale_dapo_loss(
                        local_numerator,
                        global_valid_token_count=global_valid_tokens,
                        world_size=accelerator.num_processes,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite GRPO loss at pair_cursor={pair_cursor}."
                        )
                    accelerator.backward(loss)
                    local_loss_value += float(loss.detach())
                    local_clipped_tokens += int((clipped & mask).sum().item())

            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            local_gradient_ok = bool(gradients) and all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            )
            gradient_seen = gradient_seen or local_gradient_ok
            gradient_norm = accelerator.clip_grad_norm_(
                model.parameters(), float(grpo_config["max_grad_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at pair_cursor={pair_cursor}."
                )
            gradient_norm_value = float(gradient_norm.detach())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        else:
            globally_skipped_pairs += 1

        rollout_record = {
            "pair_cursor": pair_cursor,
            "rank": rank,
            "sample_id": sample_id,
            "source_index": record["source_index"],
            "db_id": record["db_id"],
            "duplicated_prompt": (
                duplicated_last_prompt
                and pair_cursor == pair_count - 1
                and rank == accelerator.num_processes - 1
            ),
            "scorable": scorable,
            "gold_execution": _summary_execution(gold_result),
            "rewards": rewards,
            "advantages": [float(value) for value in advantages.cpu().tolist()],
            "global_signal_groups": global_signal_groups,
            "local_has_signal": local_has_signal,
            "local_valid_tokens": local_valid_tokens,
            "global_valid_tokens": global_valid_tokens,
            "update_performed": update_performed,
            "optimizer_step": optimizer_step,
            "learning_rate": scheduler.get_last_lr()[0],
            "local_scaled_loss": local_loss_value,
            "local_clipped_tokens": local_clipped_tokens,
            "gradient_norm": gradient_norm_value,
            "candidates": scored,
        }
        append_jsonl_flush(rollout_path, rollout_record)
        pair_cursor += 1
        processed_pairs = pair_cursor

        if accelerator.is_main_process:
            append_jsonl_flush(
                train_metrics_path,
                {
                    "pair_cursor": pair_cursor,
                    "optimizer_step": optimizer_step,
                    "global_signal_groups": global_signal_groups,
                    "global_valid_tokens": global_valid_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.perf_counter() - started_at,
                },
            )

        checkpoint_interval = int(
            grpo_config["checkpoint_every_optimizer_steps"]
        )
        if update_performed and optimizer_step % checkpoint_interval == 0:
            _save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                tokenizer,
                run_dir,
                _resolved_config(config),
                optimizer_step=optimizer_step,
                next_pair_cursor=pair_cursor,
                rollout_rows_per_rank=pair_cursor,
            )

    latest = _latest_checkpoint(run_dir)
    latest_state = _load_checkpoint_state(latest) if latest is not None else None
    if latest_state is None or latest_state["next_pair_cursor"] != pair_cursor:
        _save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            tokenizer,
            run_dir,
            _resolved_config(config),
            optimizer_step=optimizer_step,
            next_pair_cursor=pair_cursor,
            rollout_rows_per_rank=pair_cursor,
        )

    if not final_adapter.exists():
        _save_adapter_atomic(
            accelerator,
            model,
            tokenizer,
            final_adapter,
        )

    parameter_changed = resumed_with_updates or not torch.equal(
        tracked_before, tracked_parameter.detach()
    )
    check_values = accelerator.reduce(
        torch.tensor(
            [int(base_frozen), int(gradient_seen), int(parameter_changed)],
            dtype=torch.int64,
            device=accelerator.device,
        ),
        reduction="sum",
    )
    all_base_frozen, all_gradient_seen, all_parameter_changed = [
        int(value) == accelerator.num_processes for value in check_values.tolist()
    ]
    if not (all_base_frozen and all_gradient_seen and all_parameter_changed):
        raise RuntimeError(
            "GRPO failed a frozen-base, finite-gradient, or parameter-update check."
        )

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    reload_name = "grpo_reload_check"
    unwrapped.load_adapter(final_adapter, adapter_name=reload_name, is_trainable=False)
    unwrapped.delete_adapter(reload_name)
    accelerator.wait_for_everyone()

    totals = accelerator.reduce(
        torch.tensor(
            [
                zero_signal_local_groups,
                globally_skipped_pairs,
                total_local_completions,
                total_local_truncated,
            ],
            dtype=torch.int64,
            device=accelerator.device,
        ),
        reduction="sum",
    ).tolist()
    summary = {
        "status": "completed",
        "mixed_prompt_count": len(mixed),
        "ordered_prompt_count": len(ordered_ids),
        "duplicated_last_prompt": duplicated_last_prompt,
        "processed_global_prompt_pairs": processed_pairs,
        "optimizer_steps": optimizer_step,
        "zero_signal_local_group_count": int(totals[0]),
        "zero_signal_local_group_rate": percentage(
            int(totals[0]), processed_pairs * accelerator.num_processes
        ),
        "globally_skipped_optimizer_step_count": int(totals[1])
        // accelerator.num_processes,
        "generated_completion_count": int(totals[2]),
        "truncated_completion_count": int(totals[3]),
        "duration_seconds": time.perf_counter() - started_at,
        "final_adapter": str(
            _project_path(config["run"]["model_output_dir"])
            .joinpath("final_adapter")
            .relative_to(PROJECT_ROOT)
        ),
    }
    if accelerator.is_main_process:
        atomic_write_json(summary_path, summary)
    accelerator.wait_for_everyone()
    return load_json(summary_path)


def _generate_greedy(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    unwrapped = accelerator.unwrap_model(model)
    encoded = _prompt_encoding(tokenizer, record, accelerator.device)
    prompt_length = encoded["input_ids"].shape[1]
    max_new_tokens = int(config["evaluation"]["max_new_tokens"])
    context_limit = min(
        int(config["model"]["max_context_length"]),
        int(getattr(unwrapped.config, "max_position_embeddings", 2**31 - 1)),
    )
    if prompt_length + max_new_tokens > context_limit:
        identifier = record.get("sample_id", record.get("question_id"))
        raise ValueError(
            f"Evaluation prompt {identifier!r} plus generation exceeds context."
        )
    eos_ids = _eos_ids(tokenizer, unwrapped)
    was_training = model.training
    unwrapped.eval()
    try:
        with torch.inference_mode():
            output = unwrapped.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
    finally:
        if was_training:
            model.train()
    completion_ids, truncated = _trim_completion(output[0, prompt_length:], eos_ids)
    raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)
    sql, status, compliance = extract_predicted_sql(raw_output)
    return {
        "raw_output": raw_output,
        "predicted_sql": sql,
        "extraction_status": status,
        "format_compliance": compliance,
        "truncated": truncated,
        "completion_tokens": int(completion_ids.numel()),
    }


def _validate_evaluation_prefix(
    rows: list[dict[str, Any]],
    assigned: list[dict[str, Any]],
    identifier_name: str,
    path: Path,
) -> None:
    if len(rows) > len(assigned):
        raise ValueError(f"Evaluation shard has too many rows: {path}")
    for index, row in enumerate(rows):
        if row.get(identifier_name) != assigned[index].get(identifier_name):
            raise ValueError(
                f"Evaluation shard {path} is not its deterministic prefix at row {index}."
            )


def _run_internal_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    validation: list[dict[str, Any]],
    train_databases: dict[str, Path],
    config: dict[str, Any],
    run_dir: Path,
    *,
    label: str,
    resume: bool,
) -> dict[str, Any]:
    phase_dir = run_dir / "evaluation" / label
    metrics_path = phase_dir / "metrics.json"
    if metrics_path.is_file():
        accelerator.wait_for_everyone()
        return load_json(metrics_path)

    selected = list(validation)
    rank = accelerator.process_index
    assigned = selected[rank:: accelerator.num_processes]
    shard_path = phase_dir / f"scored.rank_{rank:03d}.partial.jsonl"
    phase_dir.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(shard_path) if shard_path.is_file() else []
    _validate_evaluation_prefix(existing, assigned, "sample_id", shard_path)
    if existing and not resume:
        raise FileExistsError(f"Fresh validation found an existing shard: {shard_path}")

    timeout = float(config["evaluation"]["timeout_seconds"])
    for record in assigned[len(existing) :]:
        prediction = _generate_greedy(
            accelerator, model, tokenizer, record, config
        )
        database_path = train_databases[record["db_id"]]
        prediction_result = execute_sql(
            database_path,
            prediction["predicted_sql"],
            timeout_seconds=timeout,
        )
        gold_result = execute_sql(
            database_path,
            _extract_gold_sql(record),
            timeout_seconds=timeout,
        )
        append_jsonl_flush(
            shard_path,
            {
                "sample_id": record["sample_id"],
                "source_index": record["source_index"],
                "db_id": record["db_id"],
                **prediction,
                "prediction_execution": _summary_execution(prediction_result),
                "gold_execution": _summary_execution(gold_result),
                "official_ex": int(execution_reward(prediction_result, gold_result)),
            },
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        by_id: dict[str, dict[str, Any]] = {}
        for process_rank in range(accelerator.num_processes):
            rows = load_jsonl(
                phase_dir / f"scored.rank_{process_rank:03d}.partial.jsonl"
            )
            expected = selected[process_rank:: accelerator.num_processes]
            _validate_evaluation_prefix(rows, expected, "sample_id", shard_path)
            if len(rows) != len(expected):
                raise RuntimeError(
                    f"Validation rank {process_rank} is incomplete: "
                    f"{len(rows)}/{len(expected)}."
                )
            for row in rows:
                if row["sample_id"] in by_id:
                    raise RuntimeError(f"Duplicate validation row {row['sample_id']!r}.")
                by_id[row["sample_id"]] = row
        merged = [by_id[record["sample_id"]] for record in selected]
        metrics = {
            "label": label,
            "sample_count": len(merged),
            "official_ex": percentage(
                sum(row["official_ex"] for row in merged), len(merged)
            ),
            "executable_rate": percentage(
                sum(
                    row["prediction_execution"]["status"] == "success"
                    for row in merged
                ),
                len(merged),
            ),
            "format_compliance_rate": percentage(
                sum(row["format_compliance"] for row in merged), len(merged)
            ),
            "gold_failure_count": sum(
                row["gold_execution"]["status"] != "success" for row in merged
            ),
            "prediction_timeout_count": sum(
                row["prediction_execution"]["status"] == "timeout" for row in merged
            ),
        }
        atomic_write_jsonl(phase_dir / "scored_results.jsonl", merged)
        atomic_write_json(metrics_path, metrics)
    accelerator.wait_for_everyone()
    return load_json(metrics_path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _adapter_directories_match(source: Path, destination: Path) -> bool:
    if not source.is_dir() or not destination.is_dir():
        return False
    source_files = {path.name for path in source.iterdir() if path.is_file()}
    destination_files = {
        path.name for path in destination.iterdir() if path.is_file()
    }
    if source_files != FINAL_ADAPTER_FILES or destination_files != FINAL_ADAPTER_FILES:
        return False
    return all(
        filecmp.cmp(source / filename, destination / filename, shallow=False)
        for filename in FINAL_ADAPTER_FILES
    )


def _publish_final_adapter(source: Path, destination: Path) -> None:
    if destination.exists():
        if _adapter_directories_match(source, destination):
            return
        raise FileExistsError(
            f"A different final Adapter already exists at {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copytree(source, temporary)
        if not _adapter_directories_match(source, temporary):
            raise RuntimeError("Staged final Adapter differs from the trained Adapter.")
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _completed_training_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    model_output_dir = _project_path(config["run"]["model_output_dir"])
    report_dir = _project_path(config["run"]["training_report_dir"])
    work_dir = _project_path(config["run"]["work_dir"])
    final_adapter = model_output_dir / "final_adapter"
    train_metrics_path = report_dir / "train_metrics.jsonl"
    summary_path = report_dir / "summary.json"
    if not (
        final_adapter.is_dir()
        and train_metrics_path.is_file()
        and summary_path.is_file()
    ):
        return None
    if {path.name for path in final_adapter.iterdir() if path.is_file()} != FINAL_ADAPTER_FILES:
        raise ValueError(f"Published final Adapter has an invalid file set: {final_adapter}")
    summary = load_json(summary_path)
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise ValueError(f"Published GRPO summary is invalid: {summary_path}")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and work_dir.exists():
        shutil.rmtree(work_dir)
    return summary


def _publish_training_outputs(
    accelerator: Accelerator,
    config: dict[str, Any],
    work_dir: Path,
    summary: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model_output_dir = _project_path(config["run"]["model_output_dir"])
        report_dir = _project_path(config["run"]["training_report_dir"])
        source_adapter = work_dir / "final_adapter"
        source_metrics = work_dir / "training" / "train_metrics.jsonl"
        if not source_adapter.is_dir() or not source_metrics.is_file():
            raise RuntimeError("Completed GRPO work products are missing.")
        metric_rows = load_jsonl(source_metrics)
        expected_rows = int(summary["training"]["processed_global_prompt_pairs"])
        if len(metric_rows) != expected_rows:
            raise RuntimeError(
                f"Training metrics contain {len(metric_rows)} rows; expected {expected_rows}."
            )

        _publish_final_adapter(source_adapter, model_output_dir / "final_adapter")
        _copy_file_atomic(source_metrics, report_dir / "train_metrics.jsonl")
        atomic_write_json(report_dir / "summary.json", summary)
        shutil.rmtree(work_dir)
    accelerator.wait_for_everyone()


def run_experiment(
    config: dict[str, Any],
    config_path: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    completed_summary = _completed_training_summary(config)
    if completed_summary is not None:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "GRPO training is already complete: "
                f"report_dir={_project_path(config['run']['training_report_dir'])}",
                flush=True,
            )
        return completed_summary

    accelerator = _build_accelerator(config)
    try:
        _set_seed(int(config["run"]["seed"]), accelerator.process_index)
        work_dir = _prepare_work_directory(
            accelerator,
            config,
            config_path,
            resume=resume,
        )
        train, validation, train_databases, gold_report = _build_training_inputs(config)
        records_by_id = {record["sample_id"]: record for record in train}
        candidates, prefilter_exclusions = _screen_candidates(
            train,
            train_databases,
            gold_report,
            config,
        )

        latest_checkpoint = _latest_checkpoint(work_dir) if resume else None
        transient_adapter = work_dir / "final_adapter"
        transient_training_summary = work_dir / "training_summary.json"
        training_complete = (
            transient_adapter.is_dir() and transient_training_summary.is_file()
        )
        screen_complete = all(
            (work_dir / "screening" / filename).is_file()
            for filename in ("pass_at_4.jsonl", "mixed_prompts.jsonl", "summary.json")
        )
        before_metrics_path = (
            work_dir
            / "evaluation"
            / "internal_validation_before"
            / "metrics.json"
        )
        if latest_checkpoint is not None and not screen_complete:
            raise RuntimeError(
                "A GRPO checkpoint exists but pass@4 screening is incomplete; "
                "refusing to mix a trained policy into initial screening."
            )
        if (
            latest_checkpoint is not None or transient_adapter.is_dir()
        ) and not before_metrics_path.is_file():
            raise RuntimeError(
                "Training progress exists but the pre-training Validation result "
                "is missing, so it can no longer be reconstructed fairly."
            )

        if training_complete:
            adapter_path = transient_adapter
        elif latest_checkpoint is not None:
            adapter_path = latest_checkpoint / "adapter"
        else:
            adapter_path = _project_path(config["model"]["sft_adapter_path"])

        tokenizer = _load_tokenizer(config)
        model, parameter_summary = _load_model(config, adapter_path)
        optimizer = _build_optimizer(model, config)
        model, optimizer = accelerator.prepare(model, optimizer)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        accelerator.print(
            "GRPO setup: "
            f"processes={accelerator.num_processes}, "
            f"screen_candidates={len(candidates)}, "
            f"prefilter_excluded={len(prefilter_exclusions)}, "
            f"trainable_parameters={parameter_summary['trainable_parameters']:,}"
        )

        mixed, screening_summary = _run_screening(
            accelerator,
            model,
            tokenizer,
            candidates,
            prefilter_exclusions,
            train_databases,
            config,
            work_dir,
            resume=resume,
        )
        before_metrics = _run_internal_validation(
            accelerator,
            model,
            tokenizer,
            validation,
            train_databases,
            config,
            work_dir,
            label="internal_validation_before",
            resume=resume,
        )
        training_summary = _run_grpo_training(
            accelerator,
            model,
            optimizer,
            tokenizer,
            mixed,
            records_by_id,
            train_databases,
            config,
            work_dir,
            resume_checkpoint=(latest_checkpoint if not training_complete else None),
        )
        after_metrics = _run_internal_validation(
            accelerator,
            model,
            tokenizer,
            validation,
            train_databases,
            config,
            work_dir,
            label="internal_validation_after",
            resume=resume,
        )

        peak_local = torch.tensor(
            [torch.cuda.max_memory_allocated(accelerator.device) / 1024**3],
            dtype=torch.float64,
            device=accelerator.device,
        )
        peak_memory = [
            round(value, 3)
            for value in accelerator.gather(peak_local).cpu().tolist()
        ]
        summary = {
            "status": "completed",
            "screening": screening_summary,
            "training": training_summary,
            "internal_validation_before": before_metrics,
            "internal_validation_after": after_metrics,
            "internal_validation_ex_delta": round(
                after_metrics["official_ex"] - before_metrics["official_ex"], 2
            ),
            "dev_metrics": None,
            "sft_baseline_dev_metrics": None,
            "dev_ex_delta_from_sft": None,
            "parameter_summary": parameter_summary,
            "peak_memory_gb_per_process": peak_memory,
        }
        _publish_training_outputs(accelerator, config, work_dir, summary)
        accelerator.print(
            f"GRPO training completed: mixed_prompts={len(mixed)}, "
            f"optimizer_steps={training_summary['optimizer_steps']}, "
            f"validation_ex={after_metrics['official_ex']}, "
            f"report_dir={_project_path(config['run']['training_report_dir'])}"
        )
        return summary
    finally:
        accelerator.end_training()


def finalize_dev_summary(config: dict[str, Any]) -> dict[str, Any]:
    run_name = config["run"]["run_name"]
    summary_path = (
        _project_path(config["run"]["training_report_dir"]) / "summary.json"
    )
    predictions_path = _project_path(config["evaluation"]["predictions_path"])
    metrics_path = _project_path(config["evaluation"]["metrics_path"])
    summary = load_json(summary_path)
    predictions = load_jsonl(predictions_path)
    dev_metrics = load_json(metrics_path)
    baseline_metrics = load_json(
        _project_path(config["data"]["sft_baseline_metrics_path"])
    )

    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise ValueError(f"GRPO training summary is invalid: {summary_path}")
    if len(predictions) != 1_534:
        raise ValueError(
            f"Dev predictions must contain 1534 rows, got {len(predictions)}."
        )
    if (
        not isinstance(dev_metrics, dict)
        or dev_metrics.get("run_name") != run_name
        or dev_metrics.get("total") != 1_534
    ):
        raise ValueError(f"GRPO Dev metrics are invalid: {metrics_path}")
    if not isinstance(baseline_metrics, dict) or float(
        baseline_metrics.get("overall_ex", math.nan)
    ) != float(config["evaluation"]["baseline_dev_ex"]):
        raise ValueError(
            "Configured SFT baseline EX does not match the baseline metrics file."
        )

    summary["dev_metrics"] = dev_metrics
    summary["sft_baseline_dev_metrics"] = baseline_metrics
    summary["dev_ex_delta_from_sft"] = round(
        float(dev_metrics["overall_ex"])
        - float(baseline_metrics["overall_ex"]),
        2,
    )
    atomic_write_json(summary_path, summary)
    print(
        "GRPO Dev summary finalized: "
        f"sft_ex={baseline_metrics['overall_ex']}, "
        f"grpo_ex={dev_metrics['overall_ex']}, "
        f"delta={summary['dev_ex_delta_from_sft']:+.2f}",
        flush=True,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Qwen3-8B BIRD execution-reward GRPO Adapter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished screening, Validation, or the latest checkpoint.",
    )
    action.add_argument(
        "--finalize-dev",
        action="store_true",
        help="Merge completed mainline Dev metrics into the training summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)
    if args.finalize_dev:
        finalize_dev_summary(config)
        return
    run_experiment(config, config_path, resume=args.resume)


if __name__ == "__main__":
    main()
