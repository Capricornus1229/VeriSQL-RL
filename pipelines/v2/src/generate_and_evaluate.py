"""Shared Qwen generation, SQLite execution scoring, and pass@8 voting."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.multiprocessing as mp
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines.v2.src.pipeline_io import (
    append_jsonl_flush,
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    load_json,
    load_jsonl,
    project_relative_path,
    resolve_project_path,
)
from pipelines.v2.src.sql_runtime import (
    choose_execution_vote,
    execute_sql_capped,
    official_execution_match,
    summarize_execution,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"
DEFAULT_DEVICES = ("cuda:0", "cuda:1")

_FENCE = chr(96) * 3
_THINK_SQL_RE = re.compile(
    rf"\A<think>\s*(?P<reasoning>.*?)\s*</think>\s*"
    rf"{re.escape(_FENCE)}sql[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SQL_FENCE_RE = re.compile(
    rf"\A{re.escape(_FENCE)}sql[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_FENCE_RE = re.compile(
    rf"\A{re.escape(_FENCE)}[ \t]*\r?\n(?P<sql>.*?)\r?\n?"
    rf"{re.escape(_FENCE)}[ \t]*\Z",
    re.DOTALL,
)
_RAW_SQL_RE = re.compile(r"\A(?:SELECT|WITH)\b", re.IGNORECASE)


def extract_reasoning_sql(raw_output: str) -> dict[str, Any]:
    """Extract a complete reasoning-plus-SQL response without repairing it."""
    stripped = raw_output.strip()
    match = _THINK_SQL_RE.fullmatch(stripped)
    if match:
        reasoning = match.group("reasoning").strip()
        sql = match.group("sql").strip()
        if reasoning and sql and _FENCE not in reasoning and _FENCE not in sql:
            return {
                "reasoning": reasoning,
                "predicted_sql": sql,
                "extraction_status": "reasoning_sql_fence",
                "format_compliance": True,
            }

    match = _SQL_FENCE_RE.fullmatch(stripped)
    if match:
        sql = match.group("sql").strip()
        if sql and _FENCE not in sql:
            return {
                "reasoning": "",
                "predicted_sql": sql,
                "extraction_status": "sql_fence",
                "format_compliance": False,
            }

    match = _GENERIC_FENCE_RE.fullmatch(stripped)
    if match:
        sql = match.group("sql").strip()
        if sql and _FENCE not in sql:
            return {
                "reasoning": "",
                "predicted_sql": sql,
                "extraction_status": "generic_fence",
                "format_compliance": False,
            }

    if _RAW_SQL_RE.match(stripped):
        return {
            "reasoning": "",
            "predicted_sql": stripped,
            "extraction_status": "raw_sql",
            "format_compliance": False,
        }
    return {
        "reasoning": "",
        "predicted_sql": "",
        "extraction_status": "failed",
        "format_compliance": False,
    }


def load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_project_path(config["model"]["tokenizer_path"]),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_inference_model(
    config: dict[str, Any], adapter_path: str | Path, device: str
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(config)
    base = AutoModelForCausalLM.from_pretrained(
        resolve_project_path(config["model"]["base_model_path"]),
        dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        resolve_project_path(adapter_path),
        is_trainable=False,
    ).to(torch.device(device))
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def encode_prompt(
    tokenizer: Any, prompt: list[dict[str, str]], device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    return [value] if type(value) is int else list(value)


def _trim_at_eos(
    token_ids: torch.Tensor, eos_ids: Sequence[int]
) -> tuple[torch.Tensor, bool]:
    eos_set = set(eos_ids)
    values = token_ids.tolist()
    position = next((i for i, value in enumerate(values) if value in eos_set), None)
    if position is None:
        return token_ids, True
    return token_ids[: position + 1], False


def completion_mean_logprob(
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_attention: torch.Tensor,
    completion_ids: torch.Tensor,
    temperature: float,
) -> float:
    """Score a sampled completion under the temperature-matched policy."""
    if completion_ids.numel() == 0:
        return float("-inf")
    completion = completion_ids.unsqueeze(0)
    input_ids = torch.cat([prompt_ids, completion], dim=1)
    attention_mask = torch.cat(
        [prompt_attention, torch.ones_like(completion)], dim=1
    )
    length = int(completion.shape[1])
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=length + 1,
            use_cache=False,
        ).logits[:, -(length + 1) : -1]
        selected = torch.log_softmax(logits.float() / temperature, dim=-1).gather(
            -1, completion.unsqueeze(-1)
        ).squeeze(-1)
    return round(float(selected.mean().item()), 8)


def generate_candidates(
    model: Any,
    tokenizer: Any,
    prompt: list[dict[str, str]],
    config: dict[str, Any],
    *,
    sampled_count: int,
    include_greedy: bool,
    seed: int,
    score_logprobs: bool = False,
) -> list[dict[str, Any]]:
    """Generate an optional greedy candidate followed by sampled candidates."""
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    prompt_length = int(encoded["input_ids"].shape[1])
    settings = config["generation"]
    max_new_tokens = int(settings["max_new_tokens"])
    context_limit = min(
        int(config["model"]["max_context_length"]),
        int(getattr(model.config, "max_position_embeddings", 2**31 - 1)),
    )
    if prompt_length + max_new_tokens > context_limit:
        raise ValueError(
            f"Prompt length {prompt_length} plus max_new_tokens {max_new_tokens} "
            f"exceeds context length {context_limit}."
        )

    eos_ids = _eos_ids(model, tokenizer)
    rows: list[torch.Tensor] = []
    kinds: list[str] = []
    if include_greedy:
        with torch.inference_mode():
            generated = model.generate(
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
        rows.append(generated[0, prompt_length:])
        kinds.append("greedy")

    micro_batch = int(settings["micro_batch_size"])
    for start in range(0, sampled_count, micro_batch):
        count = min(micro_batch, sampled_count - start)
        torch.manual_seed(seed + start)
        torch.cuda.manual_seed_all(seed + start)
        with torch.inference_mode():
            generated = model.generate(
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
        rows.extend(generated[:, prompt_length:])
        kinds.extend(["sampled"] * count)

    candidates: list[dict[str, Any]] = []
    for kind, row in zip(kinds, rows, strict=True):
        completion_ids, truncated = _trim_at_eos(row, eos_ids)
        raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)
        candidates.append(
            {
                "kind": kind,
                "raw_output": raw_output,
                **extract_reasoning_sql(raw_output),
                "truncated": truncated,
                "completion_tokens": int(completion_ids.numel()),
                "mean_logprob": (
                    completion_mean_logprob(
                        model,
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        completion_ids,
                        float(settings["temperature"]),
                    )
                    if score_logprobs
                    else None
                ),
                "token_ids": completion_ids.detach().cpu().tolist(),
            }
        )
    return candidates


def _identity(record: dict[str, Any]) -> str | int:
    return record.get("sample_id", record.get("question_id"))


def generation_seed(
    config: dict[str, Any], record: dict[str, Any], phase: int
) -> int:
    source = int(record.get("source_index", record.get("question_id", 0)))
    return int(
        (int(config["run"]["seed"]) * 1_000_003 + source * 9_176 + phase * 97_409)
        % (2**31 - 1)
    )


def _generation_state(
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    adapter_path: str | Path,
    *,
    num_candidates: int,
    include_greedy: bool,
    score_logprobs: bool,
    seed_phase: int,
    devices: Sequence[str],
) -> dict[str, Any]:
    return {
        "record_ids": [_identity(record) for record in records],
        "adapter_path": project_relative_path(adapter_path),
        "generation": config["generation"],
        "num_candidates": num_candidates,
        "include_greedy": include_greedy,
        "score_logprobs": score_logprobs,
        "seed_phase": seed_phase,
        "devices": list(devices),
    }


def _worker_generate(
    rank: int,
    devices: tuple[str, ...],
    config_path: str,
    records_path: str,
    adapter_path: str,
    output_dir: str,
    num_candidates: int,
    include_greedy: bool,
    score_logprobs: bool,
    seed_phase: int,
    resume: bool,
    persist_token_ids: bool,
) -> None:
    config = load_config(config_path)
    records = load_jsonl(records_path)
    assigned = records[rank::len(devices)]
    shard_path = Path(output_dir) / f"generated.rank_{rank:03d}.partial.jsonl"
    existing = load_jsonl(shard_path) if shard_path.is_file() else []
    if existing and not resume:
        raise FileExistsError(f"Existing generation shard requires --resume: {shard_path}")
    if [_identity(row) for row in existing] != [
        _identity(row) for row in assigned[: len(existing)]
    ]:
        raise ValueError(f"Generation shard is not a prefix: {shard_path}")
    if len(existing) == len(assigned):
        return

    torch.cuda.set_device(torch.device(devices[rank]))
    model, tokenizer = load_inference_model(config, adapter_path, devices[rank])
    sampled_count = num_candidates - int(include_greedy)
    for record in assigned[len(existing):]:
        candidates = generate_candidates(
            model,
            tokenizer,
            record["prompt"],
            config,
            sampled_count=sampled_count,
            include_greedy=include_greedy,
            seed=generation_seed(config, record, seed_phase),
            score_logprobs=score_logprobs,
        )
        if not persist_token_ids:
            for candidate in candidates:
                candidate.pop("token_ids")
        append_jsonl_flush(
            shard_path,
            {
                **{
                    key: record[key]
                    for key in (
                        "sample_id",
                        "source_index",
                        "question_id",
                        "db_id",
                        "difficulty",
                    )
                    if key in record
                },
                "candidates": candidates,
            },
        )


def run_sharded_generation(
    config_path: str | Path,
    records: Sequence[dict[str, Any]],
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    num_candidates: int,
    include_greedy: bool,
    seed_phase: int,
    resume: bool,
    devices: Sequence[str] = DEFAULT_DEVICES,
    score_logprobs: bool = False,
    persist_token_ids: bool = False,
) -> list[dict[str, Any]]:
    """Generate deterministic stride shards and merge them in source order."""
    config_path = Path(config_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_records.jsonl"
    state_path = output / "generation_state.json"
    merged_path = output / "generated.jsonl"
    records = list(records)
    if not records:
        raise ValueError("Generation data is empty.")

    expected_state = _generation_state(
        load_config(config_path),
        records,
        adapter_path,
        num_candidates=num_candidates,
        include_greedy=include_greedy,
        score_logprobs=score_logprobs,
        seed_phase=seed_phase,
        devices=devices,
    )
    if state_path.is_file():
        if load_json(state_path) != expected_state:
            raise ValueError(f"Generation settings differ from {state_path}.")
    else:
        atomic_write_json(state_path, expected_state)
        atomic_write_jsonl(selected_path, records)

    if merged_path.is_file():
        merged = load_jsonl(merged_path)
        if [_identity(row) for row in merged] == [_identity(row) for row in records]:
            return merged
    elif state_path.is_file() and not resume:
        shard_paths = list(output.glob("generated.rank_*.partial.jsonl"))
        if shard_paths:
            raise FileExistsError(f"Existing generation requires --resume: {output}")

    normalized_devices = tuple(devices)
    arguments = (
        normalized_devices,
        str(config_path),
        str(selected_path),
        str(resolve_project_path(adapter_path)),
        str(output),
        num_candidates,
        include_greedy,
        score_logprobs,
        seed_phase,
        resume,
        persist_token_ids,
    )
    if len(normalized_devices) == 1:
        _worker_generate(0, *arguments)
    else:
        mp.spawn(_worker_generate, args=arguments, nprocs=len(normalized_devices), join=True)

    merged_by_id: dict[str | int, dict[str, Any]] = {}
    for rank in range(len(normalized_devices)):
        shard = load_jsonl(output / f"generated.rank_{rank:03d}.partial.jsonl")
        expected = records[rank::len(normalized_devices)]
        if [_identity(row) for row in shard] != [_identity(row) for row in expected]:
            raise RuntimeError(f"Generation shard {rank} is incomplete.")
        for row in shard:
            merged_by_id[_identity(row)] = row
    merged = [merged_by_id[_identity(record)] for record in records]
    atomic_write_jsonl(merged_path, merged)
    return merged


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _system_metrics(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    correct = sum(int(row["official_ex"]) for row in scored)
    executable = sum(
        row["prediction_execution"]["status"] == "success" for row in scored
    )
    compliant = sum(bool(row["format_compliance"]) for row in scored)
    metrics: dict[str, Any] = {
        "sample_count": total,
        "correct_count": correct,
        "official_ex": _percentage(correct, total),
        "executable_rate": _percentage(executable, total),
        "format_compliance_rate": _percentage(compliant, total),
        "gold_failure_count": sum(
            row["gold_execution"]["status"] != "success" for row in scored
        ),
    }
    difficulties = ("simple", "moderate", "challenging")
    if any("difficulty" in row for row in scored):
        metrics["difficulty"] = {}
        for difficulty in difficulties:
            rows = [row for row in scored if row.get("difficulty") == difficulty]
            difficulty_correct = sum(int(row["official_ex"]) for row in rows)
            metrics["difficulty"][difficulty] = {
                "sample_count": len(rows),
                "correct_count": difficulty_correct,
                "official_ex": _percentage(difficulty_correct, len(rows)),
            }
    return metrics


def _record_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("sample_id", "source_index", "question_id", "db_id", "difficulty")
        if key in record
    }


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"token_ids", "execution"}
    }


def _execute_candidate_for_vote(
    record: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    execution = execute_sql_capped(
        resolve_project_path(record["sqlite_path"]),
        candidate["predicted_sql"],
        timeout_seconds=float(config["evaluation"]["vote_timeout_seconds"]),
        max_rows=int(config["evaluation"]["vote_max_rows"]),
    )
    return {**candidate, "execution": execution}


def _score_selected_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    system: str,
    candidate_index: int,
    vote_decision: dict[str, Any] | None,
    config: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    database_path = resolve_project_path(record["sqlite_path"])
    timeout = float(config["evaluation"]["official_timeout_seconds"])
    if prediction is None:
        prediction = execute_sql_capped(
            database_path,
            candidate["predicted_sql"],
            timeout_seconds=timeout,
            max_rows=None,
        )
    if gold is None:
        gold = execute_sql_capped(
            database_path,
            record["gold_sql"],
            timeout_seconds=timeout,
            max_rows=None,
        )
    return {
        **_record_fields(record),
        "system": system,
        "candidate_index": candidate_index,
        "vote_decision": vote_decision,
        **_candidate_public(candidate),
        "selection_execution": summarize_execution(candidate["execution"]),
        "prediction_execution": summarize_execution(prediction),
        "gold_execution": summarize_execution(gold),
        "official_ex": official_execution_match(prediction, gold),
    }


def score_pass8_generation(
    records: Sequence[dict[str, Any]],
    generated_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Score candidate zero as greedy and execution-vote over all eight."""
    generated_by_id = {_identity(row): row for row in generated_rows}
    greedy_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    for record in records:
        candidates = [
            _execute_candidate_for_vote(record, candidate, config)
            for candidate in generated_by_id[_identity(record)]["candidates"]
        ]
        vote = choose_execution_vote(
            candidates,
            minimum_support=int(config["evaluation"]["minimum_vote_support"]),
            greedy_index=0,
        )
        chosen_index = vote["chosen_index"]
        if chosen_index is None:
            chosen_index = 0
        database_path = resolve_project_path(record["sqlite_path"])
        timeout = float(config["evaluation"]["official_timeout_seconds"])
        gold = execute_sql_capped(
            database_path,
            record["gold_sql"],
            timeout_seconds=timeout,
            max_rows=None,
        )
        prediction_cache: dict[str, dict[str, Any]] = {}
        for index in {0, chosen_index}:
            sql = candidates[index]["predicted_sql"]
            if sql not in prediction_cache:
                prediction_cache[sql] = execute_sql_capped(
                    database_path,
                    sql,
                    timeout_seconds=timeout,
                    max_rows=None,
                )
        greedy_rows.append(
            _score_selected_candidate(
                record,
                candidates[0],
                system="greedy",
                candidate_index=0,
                vote_decision=None,
                config=config,
                prediction=prediction_cache[candidates[0]["predicted_sql"]],
                gold=gold,
            )
        )
        vote_rows.append(
            _score_selected_candidate(
                record,
                candidates[chosen_index],
                system="vote",
                candidate_index=chosen_index,
                vote_decision=vote,
                config=config,
                prediction=prediction_cache[
                    candidates[chosen_index]["predicted_sql"]
                ],
                gold=gold,
            )
        )
    return {
        "greedy": (greedy_rows, _system_metrics(greedy_rows)),
        "vote": (vote_rows, _system_metrics(vote_rows)),
    }


def score_greedy_generation(
    records: Sequence[dict[str, Any]],
    generated_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated_by_id = {_identity(row): row for row in generated_rows}
    scored: list[dict[str, Any]] = []
    for record in records:
        candidate = generated_by_id[_identity(record)]["candidates"][0]
        database_path = resolve_project_path(record["sqlite_path"])
        timeout = float(config["evaluation"]["official_timeout_seconds"])
        prediction = execute_sql_capped(
            database_path,
            candidate["predicted_sql"],
            timeout_seconds=timeout,
            max_rows=None,
        )
        gold = execute_sql_capped(
            database_path,
            record["gold_sql"],
            timeout_seconds=timeout,
            max_rows=None,
        )
        executed = {**candidate, "execution": prediction}
        scored.append(
            _score_selected_candidate(
                record,
                executed,
                system="greedy",
                candidate_index=0,
                vote_decision=None,
                config=config,
                prediction=prediction,
                gold=gold,
            )
        )
    return scored, _system_metrics(scored)


def _prediction_row(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: scored[key]
            for key in ("sample_id", "source_index", "question_id", "db_id", "difficulty")
            if key in scored
        },
        "candidate_index": scored["candidate_index"],
        "raw_output": scored["raw_output"],
        "predicted_sql": scored["predicted_sql"],
        "extraction_status": scored["extraction_status"],
        "format_compliance": scored["format_compliance"],
        "vote_decision": scored["vote_decision"],
    }


def _publish_system(
    output_dir: Path,
    scored: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    atomic_write_jsonl(output_dir / "predictions.jsonl", map(_prediction_row, scored))
    atomic_write_jsonl(output_dir / "scored_results.jsonl", scored)
    atomic_write_json(output_dir / "metrics.json", metrics)


def evaluate_adapter_greedy(
    *,
    config_path: str | Path,
    adapter_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    work_dir: str | Path | None = None,
    resume: bool = False,
    devices: Sequence[str] = DEFAULT_DEVICES,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Generate and score one deterministic candidate per record."""
    output = Path(output_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        return load_json(metrics_path)
    records = load_jsonl(data_path)
    if max_samples is not None:
        records = records[:max_samples]
    work = Path(work_dir) if work_dir is not None else output / ".generation"
    generated = run_sharded_generation(
        config_path,
        records,
        adapter_path,
        work,
        num_candidates=1,
        include_greedy=True,
        seed_phase=41,
        resume=resume,
        devices=devices,
    )
    scored, metrics = score_greedy_generation(records, generated, load_config(config_path))
    _publish_system(output, scored, metrics)
    shutil.rmtree(work, ignore_errors=True)
    return metrics


def evaluate_adapter_pass8(
    *,
    config_path: str | Path,
    adapter_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    work_dir: str | Path,
    resume: bool = False,
    devices: Sequence[str] = DEFAULT_DEVICES,
) -> dict[str, dict[str, Any]]:
    """Generate one greedy plus seven sampled candidates and score both systems."""
    output = Path(output_dir)
    metric_paths = {
        system: output / system / "metrics.json" for system in ("greedy", "vote")
    }
    if all(path.is_file() for path in metric_paths.values()):
        return {system: load_json(path) for system, path in metric_paths.items()}
    records = load_jsonl(data_path)
    generated = run_sharded_generation(
        config_path,
        records,
        adapter_path,
        work_dir,
        num_candidates=8,
        include_greedy=True,
        seed_phase=int(load_config(config_path)["evaluation"]["vote_seed_phase"]),
        resume=resume,
        devices=devices,
        score_logprobs=True,
    )
    results = score_pass8_generation(records, generated, load_config(config_path))
    metrics: dict[str, dict[str, Any]] = {}
    for system, (scored, system_metrics) in results.items():
        _publish_system(output / system, scored, system_metrics)
        metrics[system] = system_metrics
    shutil.rmtree(work_dir, ignore_errors=True)
    return metrics


# The SFT training callback keeps this short historical name.
evaluate_adapter = evaluate_adapter_greedy


def evaluate_dev(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    resume: bool = False,
    devices: Sequence[str] = DEFAULT_DEVICES,
) -> dict[str, Any]:
    """Evaluate selected SFT greedy and Exact-GRPO greedy/pass@8 Vote."""
    dev_data_path = PIPELINE_ROOT / "artifacts" / "data" / "dev.jsonl"
    dev_work_root = PIPELINE_ROOT / ".work" / "dev"

    sft_selection = load_json(PIPELINE_ROOT / "runs" / "sft" / "selection.json")
    sft_adapter_path = resolve_project_path(sft_selection["adapter_path"])
    sft_greedy = evaluate_adapter_greedy(
        config_path=config_path,
        adapter_path=sft_adapter_path,
        data_path=dev_data_path,
        output_dir=PIPELINE_ROOT / "runs" / "dev" / "sft_greedy",
        work_dir=dev_work_root / "sft_greedy",
        resume=resume,
        devices=devices,
    )

    selection = load_json(PIPELINE_ROOT / "runs" / "grpo" / "selection.json")
    adapter_path = resolve_project_path(selection["best_adapter_path"])
    metrics = evaluate_adapter_pass8(
        config_path=config_path,
        adapter_path=adapter_path,
        data_path=dev_data_path,
        output_dir=PIPELINE_ROOT / "runs" / "dev",
        work_dir=dev_work_root / "grpo_pass8",
        resume=resume,
        devices=devices,
    )
    greedy_rows = load_jsonl(
        PIPELINE_ROOT / "runs" / "dev" / "greedy" / "scored_results.jsonl"
    )
    vote_rows = load_jsonl(
        PIPELINE_ROOT / "runs" / "dev" / "vote" / "scored_results.jsonl"
    )
    wrong_to_right = sum(
        not greedy["official_ex"] and vote["official_ex"]
        for greedy, vote in zip(greedy_rows, vote_rows, strict=True)
    )
    right_to_wrong = sum(
        greedy["official_ex"] and not vote["official_ex"]
        for greedy, vote in zip(greedy_rows, vote_rows, strict=True)
    )
    summary = {
        "sft_adapter_path": project_relative_path(sft_adapter_path),
        "sft_greedy": sft_greedy,
        "adapter_path": project_relative_path(adapter_path),
        "greedy": metrics["greedy"],
        "vote": metrics["vote"],
        "grpo_over_sft_correct_gain": (
            int(metrics["greedy"]["correct_count"])
            - int(sft_greedy["correct_count"])
        ),
        "grpo_over_sft_ex_gain_points": round(
            float(metrics["greedy"]["official_ex"])
            - float(sft_greedy["official_ex"]),
            2,
        ),
        "vote_correct_gain": (
            int(metrics["vote"]["correct_count"])
            - int(metrics["greedy"]["correct_count"])
        ),
        "vote_ex_gain_points": round(
            float(metrics["vote"]["official_ex"])
            - float(metrics["greedy"]["official_ex"]),
            2,
        ),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
    }
    atomic_write_json(PIPELINE_ROOT / "runs" / "dev" / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("greedy", "pass8", "dev"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "dev":
        result = evaluate_dev(
            args.config,
            resume=args.resume,
            devices=args.devices,
        )
    else:
        if not all((args.adapter_path, args.data_path, args.output_dir, args.work_dir)):
            raise ValueError(
                "greedy/pass8 require --adapter-path, --data-path, "
                "--output-dir, and --work-dir."
            )
        function = (
            evaluate_adapter_greedy
            if args.command == "greedy"
            else evaluate_adapter_pass8
        )
        result = function(
            config_path=args.config,
            adapter_path=args.adapter_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            resume=args.resume,
            devices=args.devices,
        )
    print(result)


if __name__ == "__main__":
    main()
