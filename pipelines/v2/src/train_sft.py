"""Qwen3-8B reasoning SFT with LoRA."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"
DATA_ROOT = PIPELINE_ROOT / "artifacts" / "data"
SFT_DATA_PATHS = (
    DATA_ROOT / "sft_train.jsonl",
    DATA_ROOT / "sft_validation.jsonl",
)
SFT_RUN_ROOT = PIPELINE_ROOT / "runs" / "sft"
SFT_WORK_ROOT = PIPELINE_ROOT / ".work" / "sft"

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("V2 config must be a YAML object.")
    for section in ("run", "model", "sft"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"V2 config is missing {section!r}.")
    if config["run"].get("expected_num_processes") != 2:
        raise ValueError("V2 SFT requires run.expected_num_processes=2.")
    if config["generation"].get("enable_thinking") is not True:
        raise ValueError("Reasoning SFT requires generation.enable_thinking=true.")
    if config["sft"].get("train_batch_size_per_device") != 1:
        raise ValueError("Completion-only logits require train batch size 1.")
    if config["sft"].get("validation_batch_size_per_device") != 1:
        raise ValueError("Completion-only logits require validation batch size 1.")
    return config


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object.")
                records.append(record)
    if not records:
        raise ValueError(f"SFT dataset is empty: {path}")
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("Chat template returned no token IDs.")
    return value


def validate_reasoning_record(record: dict[str, Any]) -> None:
    prompt = record.get("prompt")
    completion = record.get("completion")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("SFT prompt must contain exactly system and user messages.")
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError("SFT prompt roles must be system then user.")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("SFT completion must contain one assistant message.")
    assistant = completion[0]
    if assistant.get("role") != "assistant":
        raise ValueError("SFT completion role must be assistant.")
    if not isinstance(assistant.get("reasoning_content"), str):
        raise ValueError("SFT completion requires string reasoning_content.")
    if not isinstance(assistant.get("content"), str) or not assistant["content"]:
        raise ValueError("SFT completion requires SQL content.")


def tokenize_reasoning_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize native Qwen thinking plus exact SQL as completion supervision."""
    validate_reasoning_record(record)
    prompt = record["prompt"]
    full_messages = [*prompt, *record["completion"]]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"sample_id={record.get('sample_id')!r} prompt is not a prefix of "
            "the reasoning conversation."
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"sample_id={record.get('sample_id')!r} has {len(full_ids)} tokens, "
            f"above max_length={max_length}."
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    supervised_count = sum(label != -100 for label in labels[1:])
    if supervised_count == 0:
        raise ValueError("SFT record has no supervised completion tokens.")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "supervised_token_count": supervised_count,
        "sample_id": record.get("sample_id"),
    }


class ReasoningDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return tokenize_reasoning_record(
            self.records[index], self.tokenizer, self.max_length
        )


class BatchOneCollator:
    """The V2 memory optimization intentionally fixes each device batch to one."""

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("V2 SFT collator requires batch size 1.")
        feature = features[0]
        return {
            "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [feature["attention_mask"]], dtype=torch.long
            ),
            "labels": torch.tensor([feature["labels"]], dtype=torch.long),
            "supervised_token_count": torch.tensor(
                feature["supervised_token_count"], dtype=torch.long
            ),
            "sample_ids": [feature["sample_id"]],
        }


def compute_completion_only_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Compute standard causal loss while projecting only completion logits."""
    labels = batch["labels"]
    if labels.shape[0] != 1:
        raise ValueError("Completion-only loss requires per-device batch size 1.")
    supervised_count = int(batch["supervised_token_count"].item())
    if supervised_count <= 0:
        raise ValueError("Completion-only loss requires supervised tokens.")
    mask = labels != -100
    expected = torch.zeros_like(mask)
    expected[:, -supervised_count:] = True
    if not torch.equal(mask, expected):
        raise ValueError("Supervised labels must form one continuous suffix.")

    shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
    shift_labels = shift_labels[:, -(supervised_count + 1) :].contiguous()
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=labels,
        logits_to_keep=supervised_count + 1,
        shift_labels=shift_labels,
    ).loss


def load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(
    config: dict[str, Any],
    resume_adapter_path: Path | None = None,
) -> PeftModel:
    settings = config["model"]
    base = AutoModelForCausalLM.from_pretrained(
        settings["base_model_path"],
        dtype=torch.bfloat16,
        attn_implementation=settings["attention_implementation"],
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    if resume_adapter_path is not None:
        model = PeftModel.from_pretrained(
            base, resume_adapter_path, is_trainable=True
        )
    else:
        sft = config["sft"]
        model = get_peft_model(
            base,
            LoraConfig(
                r=sft["lora_r"],
                lora_alpha=sft["lora_alpha"],
                lora_dropout=sft["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=sft["target_modules"],
            ),
        )
    model.config.use_cache = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable.")
    return model


def build_accelerator(config: dict[str, Any]) -> Accelerator:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config["sft"]["gradient_accumulation_steps"],
        step_scheduler_with_optimizer=False,
    )
    if accelerator.num_processes != config["run"]["expected_num_processes"]:
        raise RuntimeError("Launch V2 SFT with exactly two Accelerate processes.")
    if accelerator.device.type != "cuda":
        raise RuntimeError("V2 SFT requires CUDA.")
    return accelerator


def build_dataloaders(
    tokenizer: Any,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, ReasoningDataset]:
    train_path, validation_path = SFT_DATA_PATHS
    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    max_length = int(config["sft"]["max_length"])
    train_dataset = ReasoningDataset(train_records, tokenizer, max_length)
    validation_dataset = ReasoningDataset(validation_records, tokenizer, max_length)
    collator = BatchOneCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, validation_loader, train_dataset


def audit_sft_datasets(
    train_dataset: ReasoningDataset,
    validation_dataset: ReasoningDataset,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tokenize every SFT record before model loading and report the maxima."""

    def scan(
        dataset: ReasoningDataset,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        longest = dataset[0]
        for index in range(1, len(dataset)):
            candidate = dataset[index]
            if len(candidate["input_ids"]) > len(longest["input_ids"]):
                longest = candidate
        return {
            "sample_count": len(dataset),
            "longest_sample_id": longest["sample_id"],
            "max_sequence_length": len(longest["input_ids"]),
            "longest_sample_supervised_tokens": longest[
                "supervised_token_count"
            ],
        }, longest

    train_summary, longest_train = scan(train_dataset)
    validation_summary, _ = scan(validation_dataset)
    return {
        "train": train_summary,
        "validation": validation_summary,
    }, longest_train


def build_preflight_loader(
    dataset: ReasoningDataset,
    config: dict[str, Any],
    longest: dict[str, Any] | None = None,
) -> tuple[DataLoader, dict[str, Any]]:
    """Repeat the longest tokenized sample for one full accumulation window."""
    if longest is None:
        longest = dataset[0]
        for index in range(1, len(dataset)):
            candidate = dataset[index]
            if len(candidate["input_ids"]) > len(longest["input_ids"]):
                longest = candidate
    repeats = (
        int(config["run"]["expected_num_processes"])
        * int(config["sft"]["gradient_accumulation_steps"])
    )
    loader = DataLoader(
        [longest] * repeats,
        batch_size=1,
        shuffle=False,
        collate_fn=BatchOneCollator(),
        num_workers=0,
        pin_memory=True,
    )
    return loader, {
        "sample_id": longest["sample_id"],
        "sequence_length": len(longest["input_ids"]),
        "supervised_tokens": longest["supervised_token_count"],
    }


def _reduce_pair(
    accelerator: Accelerator, loss_sum: float, tokens: int
) -> tuple[float, int]:
    values = torch.tensor(
        [loss_sum, float(tokens)], dtype=torch.float64, device=accelerator.device
    )
    values = accelerator.reduce(values, reduction="sum")
    return float(values[0].item()), int(values[1].item())


@torch.no_grad()
def evaluate_loss(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        loss = compute_completion_only_loss(model, batch)
        supervised = int(batch["supervised_token_count"].item())
        loss_sum += float(loss.detach()) * supervised
        token_count += supervised
    loss_sum, token_count = _reduce_pair(accelerator, loss_sum, token_count)
    model.train()
    validation_loss = loss_sum / token_count
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(validation_loss),
        "validation_supervised_tokens": token_count,
    }


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> Any:
    sft = config["sft"]
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=sft["learning_rate"],
        weight_decay=0.0,
    )


def _training_steps(record_count: int, config: dict[str, Any]) -> int:
    sft = config["sft"]
    per_update = (
        config["run"]["expected_num_processes"]
        * sft["train_batch_size_per_device"]
        * sft["gradient_accumulation_steps"]
    )
    return math.ceil(record_count / per_update) * int(sft["epochs"])


def _save_adapter(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    destination: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            destination, safe_serialization=True
        )
        tokenizer.save_pretrained(destination)
    accelerator.wait_for_everyone()


def save_epoch_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run_root: Path,
    *,
    epoch: int,
    global_step: int,
    validation_metrics: dict[str, Any],
) -> Path:
    checkpoint = run_root / "checkpoints" / f"epoch_{epoch:03d}"
    _save_adapter(accelerator, model, tokenizer, checkpoint / "adapter")
    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        _write_json(
            checkpoint / "trainer_state.json",
            {"epoch": epoch, "global_step": global_step},
        )
        _write_json(checkpoint / "validation" / "loss_metrics.json", validation_metrics)
    accelerator.wait_for_everyone()
    return checkpoint


def _train_epoch(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    stop_at_step: int | None = None,
) -> tuple[int, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    token_count = 0
    started = time.perf_counter()
    for batch in loader:
        with accelerator.accumulate(model):
            loss = compute_completion_only_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("SFT loss is not finite.")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config["sft"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 20 == 0 and accelerator.is_main_process:
                    print(
                        f"SFT update: epoch={epoch}, step={global_step}, "
                        f"loss={float(loss.detach()):.6f}, "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
        supervised = int(batch["supervised_token_count"].item())
        loss_sum += float(loss.detach()) * supervised
        token_count += supervised
        if stop_at_step is not None and global_step >= stop_at_step:
            break
    loss_sum, token_count = _reduce_pair(accelerator, loss_sum, token_count)
    return global_step, {
        "epoch": epoch,
        "train_loss": loss_sum / token_count,
        "train_supervised_tokens": token_count,
        "duration_seconds": time.perf_counter() - started,
    }


def _load_resume(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, int]:
    with (checkpoint / "trainer_state.json").open("r", encoding="utf-8") as file:
        state = json.load(file)
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu"))
    global_step = int(state["global_step"])
    return int(state["epoch"]) + 1, global_step


def run_training(
    config: dict[str, Any],
    *,
    mode: str,
    resume_from: Path | None,
    run_root: Path | None = None,
) -> None:
    output_root = SFT_RUN_ROOT
    if mode in {"preflight", "smoke"}:
        output_root = run_root or (SFT_WORK_ROOT / mode)
    if mode == "train" and resume_from is None:
        checkpoints = output_root / "checkpoints"
        if checkpoints.exists() and any(checkpoints.iterdir()):
            raise FileExistsError(f"Fresh SFT checkpoints already exist: {checkpoints}")

    accelerator = build_accelerator(config)
    seed = int(config["run"]["seed"])
    random.seed(seed + accelerator.process_index)
    torch.manual_seed(seed + accelerator.process_index)

    tokenizer = load_tokenizer(config)
    train_loader, validation_loader, train_dataset = build_dataloaders(
        tokenizer, config
    )
    data_audit, longest_train = audit_sft_datasets(
        train_dataset, validation_loader.dataset
    )
    if accelerator.is_main_process:
        print(
            "SFT token audit passed: "
            f"train_max={data_audit['train']['max_sequence_length']}, "
            "validation_max="
            f"{data_audit['validation']['max_sequence_length']}",
            flush=True,
        )
    preflight_details = None
    if mode == "preflight":
        train_loader, preflight_details = build_preflight_loader(
            train_dataset, config, longest_train
        )
    model = build_model(
        config,
        resume_adapter_path=(resume_from / "adapter") if resume_from else None,
    )
    optimizer = _optimizer(model, config)
    total_steps = _training_steps(len(train_dataset), config)
    warmup_steps = math.ceil(total_steps * float(config["sft"]["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )

    start_epoch = 1
    global_step = 0
    if resume_from is not None:
        start_epoch, global_step = _load_resume(resume_from, optimizer, scheduler)

    epochs = int(config["sft"]["epochs"])
    if mode == "preflight":
        epochs = 1
    epoch_summaries: list[dict[str, Any]] = []
    summary_path = output_root / "summary.json"
    if resume_from is not None and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as file:
            previous_summary = json.load(file)
        if "training" in previous_summary:
            previous_summary = previous_summary["training"]
        epoch_summaries = list(previous_summary.get("epochs", []))

    if start_epoch > epochs:
        if accelerator.is_main_process:
            print("SFT training is already complete; continuing to model selection.")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    for epoch in range(start_epoch, epochs + 1):
        stop_at = None
        if mode in {"preflight", "smoke"}:
            requested_steps = 1 if mode == "preflight" else int(
                config["sft"]["smoke_optimizer_steps"]
            )
            stop_at = global_step + requested_steps
        global_step, train_metrics = _train_epoch(
            accelerator,
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            epoch=epoch,
            global_step=global_step,
            stop_at_step=stop_at,
        )
        validation_metrics = evaluate_loss(
            accelerator,
            model,
            validation_loader,
            max_batches=32 if mode != "train" else None,
        )
        combined = {**train_metrics, **validation_metrics, "global_step": global_step}
        epoch_summaries.append(combined)

        if mode == "train":
            checkpoint = save_epoch_checkpoint(
                accelerator,
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_root,
                epoch=epoch,
                global_step=global_step,
                validation_metrics=combined,
            )
        elif mode == "smoke":
            _save_adapter(accelerator, model, tokenizer, output_root / "last_adapter")
        break_after_epoch = mode in {"preflight", "smoke"}
        if break_after_epoch:
            break

    if accelerator.is_main_process:
        _write_json(
            output_root / "summary.json",
            {
                "mode": mode,
                "global_step": global_step,
                "epochs": epoch_summaries,
                "data_audit": data_audit,
                "preflight_sample": preflight_details,
            },
        )
        print(f"SFT {mode} completed: optimizer_steps={global_step}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train"), default="train"
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Output directory for the internal preflight or smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.resume_from is not None and args.mode != "train":
        raise ValueError("--resume-from is supported only in train mode.")
    run_training(
        config,
        mode=args.mode,
        resume_from=args.resume_from,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
