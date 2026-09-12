from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from pipelines.v1.src.sft_data import (
    SFTDataCollator,
    SFTDataset,
    inspect_batch,
    load_sft_records,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "sft.yaml"
LONGEST_SAMPLE_ID = "train_004837"


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _summary_path(path: str | Path) -> str:
    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the plain YAML configuration used by the training script."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML object: {path}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check only settings that directly affect whether training can run."""
    required_sections = ("run", "model", "data", "lora", "optimizer", "training")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Training config is missing section {section!r}.")

    run = config["run"]
    model = config["model"]
    data = config["data"]
    lora = config["lora"]
    optimizer = config["optimizer"]
    training = config["training"]

    for section_name, section, keys in (
        ("run", run, ("run_name", "expected_num_processes", "output_dir", "report_dir")),
        ("model", model, ("model_path", "tokenizer_path")),
        ("data", data, ("train_path", "validation_path")),
    ):
        for key in keys:
            if not section.get(key):
                raise ValueError(f"Training config requires {section_name}.{key}.")

    for key in ("train_path", "validation_path"):
        path = _project_path(data[key])
        if not path.is_file():
            raise FileNotFoundError(f"Configured data file does not exist: {path}")

    positive_integer_fields = (
        ("run.expected_num_processes", run.get("expected_num_processes")),
        ("data.max_length", data.get("max_length")),
        ("data.train_batch_size_per_device", data.get("train_batch_size_per_device")),
        (
            "data.validation_batch_size_per_device",
            data.get("validation_batch_size_per_device"),
        ),
        ("training.num_epochs", training.get("num_epochs")),
        (
            "training.gradient_accumulation_steps",
            training.get("gradient_accumulation_steps"),
        ),
        ("training.log_every_steps", training.get("log_every_steps")),
        (
            "training.keep_last_checkpoints",
            training.get("keep_last_checkpoints"),
        ),
        ("training.smoke_optimizer_steps", training.get("smoke_optimizer_steps")),
        (
            "training.smoke_validation_batches",
            training.get("smoke_validation_batches"),
        ),
    )
    for name, value in positive_integer_fields:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if run["expected_num_processes"] != 2:
        raise ValueError("run.expected_num_processes must be 2 for this training run.")

    if data.get("padding_side") != "right":
        raise ValueError("data.padding_side must be 'right' for SFT training.")
    if model.get("dtype") != "bfloat16":
        raise ValueError("model.dtype must be 'bfloat16'.")
    if model.get("enable_thinking") is not False:
        raise ValueError("model.enable_thinking must be false.")
    if model.get("use_cache") is not False:
        raise ValueError("model.use_cache must be false during training.")
    if model.get("gradient_checkpointing") is not True:
        raise ValueError("model.gradient_checkpointing must be true.")
    if not isinstance(lora.get("target_modules"), list) or not lora["target_modules"]:
        raise ValueError("lora.target_modules must be a non-empty list.")
    if optimizer.get("learning_rate", 0) <= 0:
        raise ValueError("optimizer.learning_rate must be positive.")
    if training.get("scheduler_type") != "cosine":
        raise ValueError("The first training version supports scheduler_type='cosine'.")
    if not 0.0 <= training.get("warmup_ratio", -1.0) < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1).")
    if training.get("evaluate_each_epoch") is not True:
        raise ValueError("training.evaluate_each_epoch must be true.")


def build_accelerator(config: dict[str, Any]) -> Accelerator:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],
    )
    expected = config["run"]["expected_num_processes"]
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"This run requires {expected} Accelerate processes, but found "
            f"{accelerator.num_processes}. Use "
            "pipelines/v1/scripts/run_sft.sh."
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Qwen3-8B SFT requires CUDA GPUs.")
    return accelerator


def set_seed(config: dict[str, Any]) -> None:
    seed = int(config["run"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"])
    if not tokenizer.chat_template:
        raise ValueError("The configured tokenizer does not contain a chat template.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model_with_lora(
    config: dict[str, Any],
    resume_adapter_path: str | Path | None = None,
) -> PeftModel:
    model_config = config["model"]
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_path"],
        dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    if resume_adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(resume_adapter_path),
            is_trainable=True,
        )
    else:
        lora = config["lora"]
        lora_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            bias=lora["bias"],
            task_type=lora["task_type"],
            target_modules=lora["target_modules"],
        )
        model = get_peft_model(base_model, lora_config)

    model.config.use_cache = False
    return model


def summarize_trainable_parameters(model: torch.nn.Module) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    if trainable_parameters == 0:
        raise RuntimeError("LoRA injection produced no trainable parameters.")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "Only LoRA parameters may be trainable; unexpected parameter: "
            f"{unexpected[0]}"
        )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": 100.0 * trainable_parameters / total_parameters,
        "lora_parameter_tensors": len(trainable),
    }


def build_dataloaders(
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[DataLoader, DataLoader, SFTDataset, SFTDataset]:
    data = config["data"]
    train_records = load_sft_records(_project_path(data["train_path"]))
    validation_records = load_sft_records(_project_path(data["validation_path"]))
    train_dataset = SFTDataset(train_records, tokenizer, data["max_length"])
    validation_dataset = SFTDataset(
        validation_records,
        tokenizer,
        data["max_length"],
    )
    collator = SFTDataCollator(tokenizer)
    common_loader_options = {
        "num_workers": data["dataloader_num_workers"],
        "pin_memory": data["pin_memory"],
        "collate_fn": collator,
    }
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data["train_batch_size_per_device"],
        shuffle=True,
        **common_loader_options,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=data["validation_batch_size_per_device"],
        shuffle=False,
        **common_loader_options,
    )
    return train_dataloader, validation_dataloader, train_dataset, validation_dataset


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    settings = config["optimizer"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=(settings["beta1"], settings["beta2"]),
        eps=settings["epsilon"],
    )


def compute_training_schedule(
    train_dataloader: DataLoader,
    config: dict[str, Any],
) -> dict[str, int]:
    accumulation_steps = config["training"]["gradient_accumulation_steps"]
    optimizer_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    total_optimizer_steps = (
        optimizer_steps_per_epoch * config["training"]["num_epochs"]
    )
    warmup_steps = math.ceil(
        total_optimizer_steps * config["training"]["warmup_ratio"]
    )
    return {
        "local_micro_batches_per_epoch": len(train_dataloader),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": warmup_steps,
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def compute_completion_only_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Compute the standard causal-LM loss without materializing prompt logits."""
    labels = batch["labels"]
    if labels.shape[0] != 1:
        raise ValueError(
            "Completion-only logits require per-device batch size 1, got "
            f"{labels.shape[0]}."
        )

    supervised_token_count = int(batch["supervised_token_count"].item())
    if supervised_token_count <= 0:
        raise ValueError("Completion-only loss requires supervised tokens.")

    supervised_mask = labels != -100
    expected_mask = torch.zeros_like(supervised_mask)
    expected_mask[:, -supervised_token_count:] = True
    if (
        int(supervised_mask.sum().item()) != supervised_token_count
        or not torch.equal(supervised_mask, expected_mask)
    ):
        raise ValueError(
            "Completion-only loss requires supervised labels to form one "
            "continuous suffix."
        )

    shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
    shift_labels = shift_labels[:, -(supervised_token_count + 1) :].contiguous()
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=labels,
        logits_to_keep=supervised_token_count + 1,
        shift_labels=shift_labels,
    ).loss


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _reduce_training_totals(
    accelerator: Accelerator,
    values: list[float],
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=accelerator.device)
    return accelerator.reduce(tensor, reduction="sum").tolist()


def _peak_memory_gb_per_process(accelerator: Accelerator) -> list[float]:
    peak_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
    gathered = accelerator.gather(
        torch.tensor([peak_gb], dtype=torch.float64, device=accelerator.device)
    )
    return [round(value, 3) for value in gathered.cpu().tolist()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_adapter(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    destination: Path,
    config_path: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        shutil.copy2(config_path, destination / "run_config.yaml")
    accelerator.wait_for_everyone()


def _checkpoint_step(path: Path) -> int:
    return int(path.name.removeprefix("step_"))


def _prune_checkpoints(checkpoints_dir: Path, keep_count: int) -> None:
    checkpoints = sorted(
        (
            path
            for path in checkpoints_dir.glob("step_*")
            if path.is_dir() and path.name.removeprefix("step_").isdigit()
        ),
        key=_checkpoint_step,
    )
    if len(checkpoints) <= keep_count:
        return

    epoch_boundaries = []
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        if state_path.is_file():
            with state_path.open("r", encoding="utf-8") as file:
                if json.load(file).get("micro_batches_seen_in_epoch") == 0:
                    epoch_boundaries.append(checkpoint)

    keep: list[Path] = []
    if epoch_boundaries:
        keep.append(epoch_boundaries[-1])
    for checkpoint in reversed(checkpoints):
        if checkpoint not in keep and len(keep) < keep_count:
            keep.append(checkpoint)
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    micro_batches_seen_in_epoch: int,
    best_validation_loss: float | None,
    best_epoch: int | None,
    keep_count: int,
) -> Path:
    destination = output_dir / "checkpoints" / f"step_{global_step:06d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            destination / "adapter",
            safe_serialization=True,
        )
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        torch.save(scheduler.state_dict(), destination / "scheduler.pt")
        _write_json(
            destination / "trainer_state.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "micro_batches_seen_in_epoch": micro_batches_seen_in_epoch,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
            },
        )
        _prune_checkpoints(destination.parent, keep_count)
    accelerator.wait_for_everyone()
    return destination


def load_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter"
    if not state_path.is_file() or not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint must contain trainer_state.json and adapter/: {checkpoint}"
        )
    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    if state.get("micro_batches_seen_in_epoch") != 0:
        raise ValueError(
            "This first training version resumes only from epoch-boundary "
            f"checkpoints (micro_batches_seen_in_epoch=0): {checkpoint}"
        )
    return state


def restore_optimizer_and_scheduler(
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    optimizer.load_state_dict(
        torch.load(checkpoint_path / "optimizer.pt", map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(checkpoint_path / "scheduler.pt", map_location="cpu")
    )


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    validation_dataloader: DataLoader,
    *,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    started_at = time.perf_counter()
    local_loss_sum = 0.0
    local_supervised_tokens = 0

    for batch_index, batch in enumerate(validation_dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        loss = compute_completion_only_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is NaN or infinite.")
        supervised_tokens = int(batch["supervised_token_count"].item())
        local_loss_sum += float(loss.detach()) * supervised_tokens
        local_supervised_tokens += supervised_tokens

    global_loss_sum, global_tokens = _reduce_training_totals(
        accelerator,
        [local_loss_sum, local_supervised_tokens],
    )
    if global_tokens <= 0:
        raise RuntimeError("Validation produced no supervised tokens.")
    validation_loss = global_loss_sum / global_tokens
    model.train()
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(validation_loss),
        "validation_supervised_tokens": int(global_tokens),
        "validation_duration_seconds": time.perf_counter() - started_at,
    }


def train_one_epoch(
    accelerator: Accelerator,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    report_dir: Path,
    *,
    epoch: int,
    global_step: int,
    max_optimizer_steps: int | None = None,
) -> tuple[int, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started_at = time.perf_counter()
    log_started_at = started_at
    epoch_loss_sum = 0.0
    epoch_supervised_tokens = 0
    epoch_samples = 0
    epoch_input_tokens = 0
    window_loss_sum = 0.0
    window_supervised_tokens = 0
    window_samples = 0
    window_input_tokens = 0
    last_gradient_norm = 0.0
    micro_batches_seen = 0

    def write_window() -> None:
        nonlocal window_loss_sum, window_supervised_tokens
        nonlocal window_samples, window_input_tokens, log_started_at
        totals = _reduce_training_totals(
            accelerator,
            [
                window_loss_sum,
                window_supervised_tokens,
                window_samples,
                window_input_tokens,
            ],
        )
        loss_sum, supervised_tokens, _, input_tokens = totals
        elapsed = time.perf_counter() - log_started_at
        if accelerator.is_main_process and supervised_tokens > 0:
            _append_jsonl(
                report_dir / "train_metrics.jsonl",
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": loss_sum / supervised_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": last_gradient_norm,
                    "supervised_tokens": int(supervised_tokens),
                    "tokens_per_second": input_tokens / elapsed,
                    "elapsed_seconds": elapsed,
                },
            )
        window_loss_sum = 0.0
        window_supervised_tokens = 0
        window_samples = 0
        window_input_tokens = 0
        log_started_at = time.perf_counter()

    for batch_index, batch in enumerate(train_dataloader):
        micro_batches_seen = batch_index + 1
        supervised_tokens = int(batch["supervised_token_count"].item())
        input_tokens = int(batch["attention_mask"].sum().item())
        batch_size = len(batch["sample_ids"])

        with accelerator.accumulate(model):
            loss = compute_completion_only_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Training loss is NaN or infinite at epoch={epoch}, "
                    f"micro_batch={micro_batches_seen}."
                )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                gradient_norm = accelerator.clip_grad_norm_(
                    model.parameters(),
                    config["optimizer"]["max_grad_norm"],
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Gradient norm is NaN or infinite at epoch={epoch}."
                    )
                last_gradient_norm = float(gradient_norm.detach())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        detached_loss = float(loss.detach())
        epoch_loss_sum += detached_loss * supervised_tokens
        epoch_supervised_tokens += supervised_tokens
        epoch_samples += batch_size
        epoch_input_tokens += input_tokens
        window_loss_sum += detached_loss * supervised_tokens
        window_supervised_tokens += supervised_tokens
        window_samples += batch_size
        window_input_tokens += input_tokens

        if accelerator.sync_gradients:
            if global_step % config["training"]["log_every_steps"] == 0:
                write_window()
            if max_optimizer_steps is not None and global_step >= max_optimizer_steps:
                break

    if window_supervised_tokens:
        write_window()

    totals = _reduce_training_totals(
        accelerator,
        [
            epoch_loss_sum,
            epoch_supervised_tokens,
            epoch_samples,
            epoch_input_tokens,
        ],
    )
    loss_sum, supervised_tokens, samples, input_tokens = totals
    if supervised_tokens <= 0:
        raise RuntimeError("Training epoch produced no supervised tokens.")
    return global_step, {
        "train_loss": loss_sum / supervised_tokens,
        "supervised_tokens": int(supervised_tokens),
        "samples": int(samples),
        "input_tokens": int(input_tokens),
        "duration_seconds": time.perf_counter() - started_at,
        "micro_batches_seen_in_epoch": micro_batches_seen,
        "epoch_completed": micro_batches_seen == len(train_dataloader),
    }


def run_preflight(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    train_dataset: SFTDataset,
    tokenizer: Any,
    config: dict[str, Any],
    config_path: Path,
    report_dir: Path,
    parameter_summary: dict[str, Any],
) -> None:
    model.train()
    try:
        longest_index = next(
            index
            for index, record in enumerate(train_dataset.records)
            if record.get("sample_id") == LONGEST_SAMPLE_ID
        )
    except StopIteration as error:
        raise ValueError(
            f"Preflight sample {LONGEST_SAMPLE_ID!r} is missing from training data."
        ) from error

    collator = SFTDataCollator(tokenizer)
    longest_batch = collator([train_dataset[longest_index]])
    batch_details = inspect_batch(longest_batch)
    longest_batch = _move_batch_to_device(longest_batch, accelerator.device)
    normal_batches = iter(train_dataloader)

    tracked_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    parameter_before = tracked_parameter.detach().clone()
    longest_loss = 0.0
    lora_gradient_ok = False
    base_parameters_frozen = all(
        ("lora_" in name) or (not parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = config["training"]["gradient_accumulation_steps"]
    for micro_step in range(accumulation_steps):
        batch = longest_batch if micro_step == 0 else next(normal_batches)
        with accelerator.accumulate(model):
            loss = compute_completion_only_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Preflight loss is NaN or infinite.")
            accelerator.backward(loss)
            if micro_step == 0:
                longest_loss = float(loss.detach())
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and "lora_" in name
                    and parameter.grad is not None
                ]
                lora_gradient_ok = bool(gradients) and all(
                    torch.isfinite(gradient).all().item() for gradient in gradients
                ) and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
                base_parameters_frozen = base_parameters_frozen and all(
                    parameter.grad is None
                    for name, parameter in model.named_parameters()
                    if "lora_" not in name
                )
            if accelerator.sync_gradients:
                gradient_norm = accelerator.clip_grad_norm_(
                    model.parameters(),
                    config["optimizer"]["max_grad_norm"],
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Preflight gradient norm is not finite.")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    local_checks = accelerator.reduce(
        torch.tensor(
            [
                int(lora_gradient_ok),
                int(base_parameters_frozen),
                int(not torch.equal(parameter_before, tracked_parameter.detach())),
            ],
            dtype=torch.int64,
            device=accelerator.device,
        ),
        reduction="sum",
    )
    expected = accelerator.num_processes
    lora_gradient_ok, base_parameters_frozen, parameter_updated = [
        int(value) == expected for value in local_checks.tolist()
    ]
    if not (lora_gradient_ok and base_parameters_frozen and parameter_updated):
        raise RuntimeError(
            "Preflight failed a LoRA gradient, frozen-base, or parameter-update check."
        )

    temporary_adapter = report_dir / "temporary_adapter"
    save_adapter(
        accelerator,
        model,
        tokenizer,
        temporary_adapter,
        config_path,
    )
    unwrapped_model = accelerator.unwrap_model(model)
    reload_name = "preflight_reload"
    unwrapped_model.load_adapter(
        temporary_adapter,
        adapter_name=reload_name,
        is_trainable=False,
    )
    unwrapped_model.delete_adapter(reload_name)
    accelerator.wait_for_everyone()

    summary = {
        "status": "passed",
        "longest_sample_id": LONGEST_SAMPLE_ID,
        "sequence_length": batch_details["sequence_lengths"][0],
        "loss": longest_loss,
        "trainable_parameters": parameter_summary["trainable_parameters"],
        "lora_gradient_ok": lora_gradient_ok,
        "base_parameters_frozen": base_parameters_frozen,
        "parameter_updated": parameter_updated,
        "adapter_reload_ok": True,
        "peak_memory_gb_per_process": _peak_memory_gb_per_process(accelerator),
    }
    if accelerator.is_main_process:
        shutil.rmtree(temporary_adapter)
        _write_json(report_dir / "summary.json", summary)
    accelerator.wait_for_everyone()
    accelerator.print(
        f"Preflight passed: sample={LONGEST_SAMPLE_ID}, "
        f"sequence_length={summary['sequence_length']}"
    )


def _mode_paths(config: dict[str, Any], mode: str) -> tuple[Path | None, Path]:
    output_dir = _project_path(config["run"]["output_dir"])
    report_dir = _project_path(config["run"]["report_dir"])
    work_dir = output_dir / ".work"
    if mode == "smoke":
        return (
            work_dir / "smoke" / "model",
            work_dir / "smoke" / "report",
        )
    if mode == "preflight":
        return None, work_dir / "preflight"
    return output_dir, report_dir


def _prepare_automatic_work_directory(
    accelerator: Accelerator,
    config: dict[str, Any],
) -> None:
    output_dir = _project_path(config["run"]["output_dir"])
    report_dir = _project_path(config["run"]["report_dir"])
    work_dir = output_dir / ".work"
    if accelerator.is_main_process:
        output_entries = (
            [path for path in output_dir.iterdir() if path.name != ".work"]
            if output_dir.is_dir()
            else []
        )
        report_entries = (
            [
                path
                for path in report_dir.iterdir()
                if path.resolve() != output_dir
            ]
            if report_dir.is_dir()
            else []
        )
        report_has_results = bool(report_entries)
        if output_entries or report_has_results:
            raise FileExistsError(
                "Fresh SFT output already exists. Remove it or resume an "
                "epoch-boundary checkpoint before starting another train run."
            )
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
    accelerator.wait_for_everyone()


def _remove_automatic_work_directory(
    accelerator: Accelerator,
    config: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = _project_path(config["run"]["output_dir"])
        shutil.rmtree(output_dir / ".work")
        output_dir.rmdir()
    accelerator.wait_for_everyone()


def _prepare_run_directories(
    accelerator: Accelerator,
    output_dir: Path | None,
    report_dir: Path,
    config_path: Path,
    *,
    resume: bool,
) -> None:
    if accelerator.is_main_process:
        if not resume and output_dir is not None:
            for path in (output_dir, report_dir):
                if path.is_dir() and any(path.iterdir()):
                    raise FileExistsError(
                        f"Fresh run output is not empty: {path}. Choose a new run "
                        "directory or resume an epoch-boundary checkpoint."
                    )
        report_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, output_dir / "run_config.yaml")
    accelerator.wait_for_everyone()


def _write_eval_record(
    accelerator: Accelerator,
    report_dir: Path,
    *,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    is_best: bool,
) -> None:
    if accelerator.is_main_process:
        _append_jsonl(
            report_dir / "eval_metrics.jsonl",
            {
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
                "is_best": is_best,
            },
        )


def _run_training(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    tokenizer: Any,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    report_dir: Path,
    parameter_summary: dict[str, Any],
    *,
    mode: str,
    resume_state: dict[str, Any] | None,
) -> None:
    started_at = time.perf_counter()
    start_epoch = int(resume_state["epoch"]) if resume_state else 0
    global_step = int(resume_state["global_step"]) if resume_state else 0
    best_validation_loss = (
        resume_state.get("best_validation_loss") if resume_state else None
    )
    best_epoch = resume_state.get("best_epoch") if resume_state else None
    final_train_loss: float | None = None

    smoke_target = (
        config["training"]["smoke_optimizer_steps"] if mode == "smoke" else None
    )
    for epoch_index in range(start_epoch, config["training"]["num_epochs"]):
        epoch_number = epoch_index + 1
        global_step, train_metrics = train_one_epoch(
            accelerator,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            config,
            report_dir,
            epoch=epoch_number,
            global_step=global_step,
            max_optimizer_steps=smoke_target,
        )
        final_train_loss = train_metrics["train_loss"]

        validation_metrics = evaluate(
            accelerator,
            model,
            validation_dataloader,
            max_batches=(
                config["training"]["smoke_validation_batches"]
                if mode == "smoke"
                else None
            ),
        )
        current_validation_loss = validation_metrics["validation_loss"]
        is_best = (
            best_validation_loss is None
            or current_validation_loss < best_validation_loss
        )
        if is_best:
            best_validation_loss = current_validation_loss
            best_epoch = epoch_number
            if mode == "train":
                save_adapter(
                    accelerator,
                    model,
                    tokenizer,
                    output_dir / "best_adapter",
                    config_path,
                )
        _write_eval_record(
            accelerator,
            report_dir,
            epoch=epoch_number,
            global_step=global_step,
            metrics=validation_metrics,
            is_best=is_best,
        )

        if mode == "smoke":
            break

        if not train_metrics["epoch_completed"]:
            raise RuntimeError("Formal training stopped before the epoch completed.")
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            output_dir,
            epoch=epoch_number,
            global_step=global_step,
            micro_batches_seen_in_epoch=0,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            keep_count=config["training"]["keep_last_checkpoints"],
        )

    last_adapter = output_dir / "last_adapter"
    save_adapter(accelerator, model, tokenizer, last_adapter, config_path)
    peak_memory = _peak_memory_gb_per_process(accelerator)
    summary = {
        "run_name": config["run"]["run_name"],
        "status": "completed",
        "mode": mode,
        "base_model": config["model"]["model_path"],
        "train_samples": len(train_dataloader.dataset),
        "validation_samples": len(validation_dataloader.dataset),
        "num_processes": accelerator.num_processes,
        "precision": "bf16",
        "max_length": config["data"]["max_length"],
        "lora_config": config["lora"],
        **parameter_summary,
        "micro_batch_size": config["data"]["train_batch_size_per_device"],
        "gradient_accumulation_steps": config["training"][
            "gradient_accumulation_steps"
        ],
        "effective_batch_size": (
            config["data"]["train_batch_size_per_device"]
            * accelerator.num_processes
            * config["training"]["gradient_accumulation_steps"]
        ),
        "num_epochs": config["training"]["num_epochs"],
        "optimizer_steps": global_step,
        "final_train_loss": final_train_loss,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "best_adapter_path": (
            _summary_path(output_dir / "best_adapter")
            if mode == "train"
            else None
        ),
        "last_adapter_path": _summary_path(last_adapter),
        "training_duration_seconds": time.perf_counter() - started_at,
        "peak_memory_gb_per_process": peak_memory,
    }
    if accelerator.is_main_process:
        _write_json(report_dir / "summary.json", summary)
    accelerator.wait_for_everyone()
    accelerator.print(
        f"SFT {mode} completed: optimizer_steps={global_step}, "
        f"last_adapter={last_adapter}"
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-8B on VeriSQL-RL using hand-written SFT and LoRA."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "train"),
        required=True,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume train mode from an epoch-boundary checkpoint.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)
    if args.resume_from is not None and args.mode != "train":
        raise ValueError("--resume-from is supported only with --mode train.")

    resume_path = args.resume_from.resolve() if args.resume_from else None
    resume_state = load_resume_state(resume_path) if resume_path else None
    accelerator = build_accelerator(config)
    set_seed(config)
    if args.mode == "preflight":
        _prepare_automatic_work_directory(accelerator, config)
    output_dir, report_dir = _mode_paths(config, args.mode)
    _prepare_run_directories(
        accelerator,
        output_dir,
        report_dir,
        config_path,
        resume=resume_path is not None,
    )

    tokenizer = load_tokenizer(config)
    model = build_model_with_lora(
        config,
        resume_adapter_path=(resume_path / "adapter" if resume_path else None),
    )
    parameter_summary = summarize_trainable_parameters(model)
    train_dataloader, validation_dataloader, train_dataset, _ = build_dataloaders(
        config,
        tokenizer,
    )
    optimizer = build_optimizer(model, config)
    model, optimizer, train_dataloader, validation_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
    )
    schedule = compute_training_schedule(train_dataloader, config)
    if args.mode == "preflight":
        scheduler = build_scheduler(optimizer, total_steps=1, warmup_steps=0)
    else:
        scheduler = build_scheduler(
            optimizer,
            schedule["total_optimizer_steps"],
            schedule["warmup_steps"],
        )
    if resume_path is not None:
        restore_optimizer_and_scheduler(resume_path, optimizer, scheduler)

    accelerator.print(
        "Training setup: "
        f"processes={accelerator.num_processes}, "
        f"trainable_parameters={parameter_summary['trainable_parameters']:,}, "
        f"local_micro_batches_per_epoch="
        f"{schedule['local_micro_batches_per_epoch']}, "
        f"optimizer_steps_per_epoch={schedule['optimizer_steps_per_epoch']}, "
        f"total_optimizer_steps={schedule['total_optimizer_steps']}"
    )
    torch.cuda.reset_peak_memory_stats(accelerator.device)

    if args.mode == "preflight":
        run_preflight(
            accelerator,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            train_dataset,
            tokenizer,
            config,
            config_path,
            report_dir,
            parameter_summary,
        )
        return

    if output_dir is None:
        raise RuntimeError("Training output directory is missing.")
    _run_training(
        accelerator,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        validation_dataloader,
        tokenizer,
        config,
        config_path,
        output_dir,
        report_dir,
        parameter_summary,
        mode=args.mode,
        resume_state=resume_state,
    )
    if args.mode == "smoke":
        _remove_automatic_work_directory(accelerator, config)


if __name__ == "__main__":
    main()
