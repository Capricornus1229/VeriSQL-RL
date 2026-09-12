"""Train the single V2 Exact-only GRPO policy on screened Train prompts."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from pipelines.v2.src.generate_and_evaluate import encode_prompt, generate_candidates
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
    compute_group_advantages,
    execute_sql_capped,
    official_execution_match,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"
POLICY_ADAPTER = "default"
SFT_SELECTION_PATH = PIPELINE_ROOT / "runs" / "sft" / "selection.json"
SCREEN_PATH = PIPELINE_ROOT / "runs" / "screen" / "selected_prompts.jsonl"
TRAIN_PATH = PIPELINE_ROOT / "artifacts" / "data" / "train.jsonl"
FORMAL_RUN_ROOT = PIPELINE_ROOT / "runs" / "grpo"
FORMAL_WORK_ROOT = PIPELINE_ROOT / ".work" / "grpo" / "train"


def grpo_token_objective(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantage: float,
    *,
    clip_epsilon: float,
) -> torch.Tensor:
    ratio = torch.exp(current_logps - old_logps)
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * advantage
    return -torch.minimum(unclipped, clipped)


def dapo_local_loss(
    per_token_objective: torch.Tensor,
    *,
    global_signal_tokens: int,
    world_size: int,
) -> torch.Tensor:
    return per_token_objective.sum() * world_size / global_signal_tokens


def _selected_sft_adapter() -> Path:
    selection = load_json(SFT_SELECTION_PATH)
    return resolve_project_path(selection["adapter_path"])


def _load_training_records() -> list[dict[str, Any]]:
    selected = load_jsonl(SCREEN_PATH)
    records_by_id = {row["sample_id"]: row for row in load_jsonl(TRAIN_PATH)}
    records = [records_by_id[row["sample_id"]] for row in selected]
    if not records:
        raise ValueError("The GRPO screen selected no prompts.")
    return records


def _disable_lora_dropout(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _load_policy(
    config: dict[str, Any], adapter_path: Path
) -> tuple[PeftModel, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_project_path(config["model"]["tokenizer_path"]),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        resolve_project_path(config["model"]["base_model_path"]),
        dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=True,
        adapter_name=POLICY_ADAPTER,
    )
    model.set_adapter(POLICY_ADAPTER, inference_mode=False)
    _disable_lora_dropout(model)
    return model, tokenizer


def _completion_token_logps(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_attention: torch.Tensor,
    completion_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    completion = completion_ids.unsqueeze(0)
    input_ids = torch.cat([prompt_ids, completion], dim=1)
    attention_mask = torch.cat(
        [prompt_attention, torch.ones_like(completion)], dim=1
    )
    length = int(completion.shape[1])
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=length + 1,
        use_cache=False,
    ).logits[:, -(length + 1): -1]
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(
        -1, completion.unsqueeze(-1)
    ).squeeze(0).squeeze(-1)


def _global_sum(accelerator: Accelerator, value: int | float) -> float:
    tensor = torch.tensor(float(value), device=accelerator.device)
    return float(accelerator.reduce(tensor, reduction="sum").item())


def _save_adapter(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    destination: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_adapter(POLICY_ADAPTER, inference_mode=False)
        unwrapped.save_pretrained(
            destination,
            selected_adapters=[POLICY_ADAPTER],
            safe_serialization=True,
        )
        tokenizer.save_pretrained(destination)
        (destination / "README.md").unlink(missing_ok=True)
    accelerator.wait_for_everyone()


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    destination: Path,
    *,
    optimizer_step: int,
    prompt_cursor: int,
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if accelerator.is_main_process:
        shutil.rmtree(temporary, ignore_errors=True)
    accelerator.wait_for_everyone()
    _save_adapter(accelerator, model, tokenizer, temporary / "adapter")
    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        atomic_write_json(
            temporary / "training_state.json",
            {"optimizer_step": optimizer_step, "prompt_cursor": prompt_cursor},
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    accelerator.wait_for_everyone()


def _latest_checkpoint(run_root: Path) -> Path | None:
    candidates = list((run_root / "checkpoints").glob("step_*"))
    candidates.append(run_root / "final_checkpoint")
    checkpoints = [
        path for path in candidates if (path / "training_state.json").is_file()
    ]
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda path: int(load_json(path / "training_state.json")["prompt_cursor"]),
    )


def _score_group(
    record: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[float]:
    database_path = resolve_project_path(record["sqlite_path"])
    timeout = float(config["execution"]["timeout_seconds"])
    max_rows = int(config["execution"]["max_rows"])
    gold = execute_sql_capped(
        database_path,
        record["gold_sql"],
        timeout_seconds=timeout,
        max_rows=max_rows,
    )
    if gold["status"] != "success" or gold["empty_result"]:
        raise RuntimeError(
            f"Screened prompt {record['sample_id']} has an ineligible Gold result."
        )
    rewards: list[float] = []
    for candidate in candidates:
        prediction = execute_sql_capped(
            database_path,
            candidate["predicted_sql"],
            timeout_seconds=timeout,
            max_rows=max_rows,
        )
        rewards.append(float(official_execution_match(prediction, gold)))
    return rewards


def _metric_row(
    rank: int,
    cursor: int,
    optimizer_step: int,
    record: dict[str, Any],
    group_type: str,
    valid_count: int,
    exact_count: int,
    truncated_count: int,
    rewards: Sequence[float],
    loss: float,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "prompt_cursor": cursor,
        "rank": rank,
        "optimizer_step": optimizer_step,
        "sample_id": record["sample_id"],
        "db_id": record["db_id"],
        "group_type": group_type,
        "valid_candidate_count": valid_count,
        "exact_pass_count": exact_count,
        "truncated_candidate_count": truncated_count,
        "mean_reward": round(sum(rewards) / len(rewards), 8),
        "loss": round(loss, 8),
        "learning_rate": learning_rate,
    }


def _merge_training_metrics(work_root: Path, output_path: Path, world_size: int) -> None:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = work_root / f"train_metrics.rank_{rank:03d}.jsonl"
        if path.is_file():
            rows.extend(load_jsonl(path))
    rows.sort(key=lambda row: (row["prompt_cursor"], row["rank"]))
    atomic_write_jsonl(output_path, rows)


def train_exact_grpo(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    mode: str,
    resume: bool = False,
) -> None:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    accelerator = Accelerator(
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    if accelerator.num_processes != int(config["run"]["expected_num_processes"]):
        raise RuntimeError("Exact GRPO must run with exactly two processes.")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Exact GRPO requires CUDA.")

    run_root = (
        FORMAL_RUN_ROOT
        if mode == "train"
        else PIPELINE_ROOT / ".work" / "grpo" / "smoke"
    )
    work_root = FORMAL_WORK_ROOT if mode == "train" else run_root / "work"
    if mode == "train" and (run_root / "training_summary.json").is_file():
        raise FileExistsError(f"GRPO training is already complete: {run_root}")

    records = _load_training_records()
    random.Random(int(config["run"]["seed"])).shuffle(records)
    original_prompt_count = len(records)
    duplicated_prompt: str | None = None
    if len(records) % accelerator.num_processes:
        duplicated_prompt = records[-1]["sample_id"]
        records.append(records[-1])
    global_prompt_batches = len(records) // accelerator.num_processes

    checkpoint = _latest_checkpoint(run_root) if resume else None
    initial_adapter = (
        checkpoint / "adapter" if checkpoint is not None else _selected_sft_adapter()
    )
    model, tokenizer = _load_policy(config, initial_adapter)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["grpo"]["learning_rate"]),
        betas=(float(config["grpo"]["beta1"]), float(config["grpo"]["beta2"])),
        eps=float(config["grpo"]["adam_epsilon"]),
        weight_decay=float(config["grpo"]["weight_decay"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(
            global_prompt_batches * float(config["grpo"]["warmup_ratio"])
        ),
        num_training_steps=global_prompt_batches,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    optimizer_step = 0
    prompt_cursor = 0
    if checkpoint is not None:
        state = load_json(checkpoint / "training_state.json")
        optimizer_step = int(state["optimizer_step"])
        prompt_cursor = int(state["prompt_cursor"])
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location=accelerator.device)
        )
        scheduler.load_state_dict(
            torch.load(checkpoint / "scheduler.pt", map_location=accelerator.device)
        )

    metrics_path = work_root / (
        f"train_metrics.rank_{accelerator.process_index:03d}.jsonl"
    )
    retained_metrics: list[dict[str, Any]] = []
    if resume and metrics_path.is_file():
        retained_metrics = [
            row
            for row in load_jsonl(metrics_path)
            if int(row["prompt_cursor"]) < prompt_cursor
        ]
        atomic_write_jsonl(metrics_path, retained_metrics)
    elif metrics_path.exists():
        raise FileExistsError(f"Existing GRPO work requires --resume: {metrics_path}")

    counters = {
        "processed": len(retained_metrics),
        "mixed": sum(row["group_type"] == "mixed" for row in retained_metrics),
        "all_wrong": sum(row["group_type"] == "all_wrong" for row in retained_metrics),
        "all_correct": sum(row["group_type"] == "all_correct" for row in retained_metrics),
        "insufficient_valid": sum(
            row["group_type"] == "insufficient_valid" for row in retained_metrics
        ),
        "truncated": sum(
            int(row["truncated_candidate_count"]) for row in retained_metrics
        ),
    }
    unwrapped = accelerator.unwrap_model(model)
    last_checkpoint_step = optimizer_step
    started = time.perf_counter()

    for cursor in range(prompt_cursor, global_prompt_batches):
        record = records[
            cursor * accelerator.num_processes + accelerator.process_index
        ]
        unwrapped.set_adapter(POLICY_ADAPTER, inference_mode=False)
        unwrapped.eval()
        unwrapped.config.use_cache = True
        candidates = generate_candidates(
            unwrapped,
            tokenizer,
            record["prompt"],
            config,
            sampled_count=int(config["grpo"]["num_generations"]),
            include_greedy=False,
            seed=(
                int(config["run"]["seed"])
                + cursor * 10_007
                + accelerator.process_index
            ),
        )
        unwrapped.config.use_cache = False
        unwrapped.train()

        rewards = _score_group(record, candidates, config)
        valid_indices = [
            index
            for index, candidate in enumerate(candidates)
            if not candidate["truncated"]
        ]
        exact_count = sum(int(rewards[index]) for index in valid_indices)
        valid_count = len(valid_indices)
        if valid_count < 2:
            group_type = "insufficient_valid"
        elif exact_count == 0:
            group_type = "all_wrong"
        elif exact_count == valid_count:
            group_type = "all_correct"
        else:
            group_type = "mixed"
        local_signal = group_type == "mixed"

        advantages = [0.0] * len(candidates)
        if local_signal:
            valid_advantages = compute_group_advantages(
                [rewards[index] for index in valid_indices]
            )
            for index, advantage in zip(
                valid_indices, valid_advantages, strict=True
            ):
                advantages[index] = advantage

        global_signal_groups = int(_global_sum(accelerator, int(local_signal)))
        local_signal_tokens = (
            sum(len(candidates[index]["token_ids"]) for index in valid_indices)
            if local_signal
            else 0
        )
        global_signal_tokens = int(_global_sum(accelerator, local_signal_tokens))
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        if global_signal_groups:
            encoded = encode_prompt(tokenizer, record["prompt"], accelerator.device)
            for candidate_index, (candidate, advantage) in enumerate(
                zip(candidates, advantages, strict=True)
            ):
                completion_ids = torch.tensor(
                    candidate["token_ids"],
                    dtype=torch.long,
                    device=accelerator.device,
                )
                sync_context = (
                    nullcontext()
                    if candidate_index == len(candidates) - 1
                    else accelerator.no_sync(model)
                )
                with sync_context:
                    current_logps = _completion_token_logps(
                        model,
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        completion_ids,
                        float(config["generation"]["temperature"]),
                    )
                    objective = grpo_token_objective(
                        current_logps,
                        current_logps.detach(),
                        advantage,
                        clip_epsilon=float(config["grpo"]["clip_epsilon"]),
                    )
                    loss = dapo_local_loss(
                        objective,
                        global_signal_tokens=global_signal_tokens,
                        world_size=accelerator.num_processes,
                    )
                    accelerator.backward(loss)
                    loss_value += float(loss.detach())
            accelerator.clip_grad_norm_(
                trainable, float(config["grpo"]["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer_step += 1

        counters["processed"] += 1
        counters[group_type] += 1
        counters["truncated"] += len(candidates) - valid_count
        append_jsonl_flush(
            metrics_path,
            _metric_row(
                accelerator.process_index,
                cursor,
                optimizer_step,
                record,
                group_type,
                valid_count,
                exact_count,
                len(candidates) - valid_count,
                rewards,
                loss_value,
                float(scheduler.get_last_lr()[0]),
            ),
        )
        prompt_cursor = cursor + 1

        checkpoint_interval = int(
            config["grpo"]["checkpoint_every_optimizer_steps"]
        )
        if (
            mode == "train"
            and optimizer_step > last_checkpoint_step
            and optimizer_step % checkpoint_interval == 0
        ):
            _save_checkpoint(
                accelerator,
                model,
                tokenizer,
                optimizer,
                scheduler,
                run_root / "checkpoints" / f"step_{optimizer_step:06d}",
                optimizer_step=optimizer_step,
                prompt_cursor=prompt_cursor,
            )
            last_checkpoint_step = optimizer_step

        if (
            mode == "smoke"
            and optimizer_step >= int(config["grpo"]["smoke_optimizer_steps"])
        ):
            break

    _save_checkpoint(
        accelerator,
        model,
        tokenizer,
        optimizer,
        scheduler,
        run_root / "final_checkpoint",
        optimizer_step=optimizer_step,
        prompt_cursor=prompt_cursor,
    )

    local = torch.tensor(
        [
            counters["processed"],
            counters["mixed"],
            counters["all_wrong"],
            counters["all_correct"],
            counters["insufficient_valid"],
            counters["truncated"],
        ],
        dtype=torch.long,
        device=accelerator.device,
    )
    totals = list(map(int, accelerator.reduce(local, reduction="sum").tolist()))
    elapsed = torch.tensor(time.perf_counter() - started, device=accelerator.device)
    duration_seconds = float(accelerator.reduce(elapsed, reduction="max").item())
    if accelerator.is_main_process:
        if mode == "train":
            _merge_training_metrics(
                work_root,
                run_root / "train_metrics.jsonl",
                accelerator.num_processes,
            )
        summary = {
            "source_prompt_count": original_prompt_count,
            "duplicated_prompt": duplicated_prompt,
            "processed_groups": totals[0],
            "exact_mixed_groups": totals[1],
            "all_wrong_groups": totals[2],
            "all_correct_groups": totals[3],
            "insufficient_valid_groups": totals[4],
            "truncated_candidate_count": totals[5],
            "optimizer_steps": optimizer_step,
            "prompt_cursor": prompt_cursor,
            "duration_seconds": round(duration_seconds, 3),
            "final_checkpoint": project_relative_path(
                run_root / "final_checkpoint"
            ),
        }
        atomic_write_json(
            run_root / (
                "training_summary.json" if mode == "train" else "summary.json"
            ),
            summary,
        )
        if mode == "train":
            shutil.rmtree(work_root, ignore_errors=True)
        print(
            f"Exact GRPO {mode} completed: optimizer_steps={optimizer_step}, "
            f"exact_mixed_groups={totals[1]}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_exact_grpo(args.config, mode=args.mode, resume=args.resume)


if __name__ == "__main__":
    main()
