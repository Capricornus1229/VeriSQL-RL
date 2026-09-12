"""
Main pipeline for generating SQL predictions on the Dev dataset using a specified model and tokenizer.

*************************************
dev_eval.jsonl
      ↓
load_dev_dataset()
      ↓
_load_tokenizer()
_load_model()
      ↓
_generate_batch()
      ├── apply_chat_template()
      ├── tokenize
      ├── model.generate()
      └── batch_decode()
      ↓
extract_predicted_sql()
      ↓
_build_prediction()
      ↓
predictions.jsonl
      ↓
bird_evaluation.py
      ↓
Execute SQL and Calculate EX
*************************************

"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines.v1.src.sql_extraction import (
    EXTRACTION_STATUSES,
    extract_predicted_sql,
)
from src.common import load_json, load_jsonl


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEV_DATA_PATH = PIPELINE_ROOT / "artifacts" / "data" / "dev_eval.jsonl"

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
EXPECTED_DEV_SAMPLE_COUNT = 1_534
PREDICTIONS_FILENAME = "predictions.jsonl"
SHARD_PREDICTIONS_FILENAME_TEMPLATE = "predictions.rank_{rank:03d}.partial.jsonl"
MERGED_PREDICTIONS_FILENAME = "predictions.merged.tmp"
RESUME_STATE_FILENAME = "resume_state.json"
PREDICTION_FIELDS = {
    "question_id",
    "db_id",
    "raw_output",
    "predicted_sql",
    "extraction_status",
    "format_compliance",
}


def _generation_error(context: str, message: str) -> ValueError:
    return ValueError(f"Dev generation validation failed: {context}: {message}")


def _require_non_empty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _generation_error(
            context,
            f"expected a non-empty string, got {value!r} "
            f"({type(value).__name__}).",
        )
    return value


def _validate_prompt(prompt: object, context: str) -> list[dict]:
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise _generation_error(
            context,
            "prompt must be a two-message list containing system and user "
            f"messages, got {prompt!r}.",
        )

    for message_index, expected_role in enumerate(("system", "user")):
        message = prompt[message_index]
        message_context = f"{context}[{message_index}]"
        if not isinstance(message, dict):
            raise _generation_error(
                message_context,
                f"expected an object, got {message!r} "
                f"({type(message).__name__}).",
            )
        if message.get("role") != expected_role:
            raise _generation_error(
                f"{message_context}, field='role'",
                f"expected {expected_role!r}, got {message.get('role')!r}.",
            )
        _require_non_empty_string(
            message.get("content"),
            f"{message_context}, field='content'",
        )
    return prompt


def load_dev_dataset(
    data_path: str | Path = DEV_DATA_PATH,
) -> list[dict]:
    """Load and strictly validate the complete processed Dev dataset."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dev generation input does not exist: {path}")

    try:
        records = load_jsonl(path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load Dev generation input at {path}: {error}."
        ) from error

    if len(records) != EXPECTED_DEV_SAMPLE_COUNT:
        raise _generation_error(
            f"path={path}",
            f"expected {EXPECTED_DEV_SAMPLE_COUNT} records, got {len(records)}.",
        )

    for source_index, record in enumerate(records):
        context = f"path={path}, source_index={source_index}"
        if not isinstance(record, dict):
            raise _generation_error(
                context,
                f"record must be an object, got {record!r} "
                f"({type(record).__name__}).",
            )

        question_id = record.get("question_id")
        if type(question_id) is not int:
            raise _generation_error(
                f"{context}, field='question_id'",
                f"expected an integer that is not boolean, got {question_id!r}.",
            )
        if question_id != source_index:
            raise _generation_error(
                f"{context}, field='question_id'",
                f"expected {source_index}, got {question_id}.",
            )

        _require_non_empty_string(
            record.get("db_id"),
            f"{context}, field='db_id'",
        )
        _validate_prompt(record.get("prompt"), f"{context}, field='prompt'")
        if "completion" in record:
            raise _generation_error(
                f"{context}, field='completion'",
                "Dev generation input must remain prompt-only.",
            )

    return records


def _validate_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer and cannot be boolean, "
            f"got {value!r}."
        )
    return value


def _validate_run_name(run_name: object) -> str:
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f"run_name must be a non-empty string, got {run_name!r}.")
    if (
        run_name in {".", ".."}
        or "/" in run_name
        or "\\" in run_name
        or "\x00" in run_name
        or Path(run_name).is_absolute()
    ):
        raise ValueError(
            "run_name must be a safe single directory name and cannot be an "
            f"absolute path, '.', '..', or contain '/' or '\\': {run_name!r}."
        )
    return run_name


def _normalize_source(source: str | Path | None, name: str) -> str | None:
    if source is None:
        return None
    if not isinstance(source, (str, Path)):
        raise TypeError(
            f"{name} must be a string or Path, got {source!r} "
            f"({type(source).__name__})."
        )

    source_text = str(source)
    if not source_text.strip():
        raise ValueError(f"{name} must not be empty.")

    candidate_path = Path(source_text).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())
    if isinstance(source, Path):
        raise FileNotFoundError(f"{name} does not exist: {candidate_path}")
    return source_text


def _portable_source(source: str | None) -> str | None:
    if source is None:
        return None
    candidate = Path(source)
    if not candidate.is_absolute():
        return source
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source


def _normalize_devices(devices: Sequence[str]) -> tuple[str, ...]:
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise TypeError(
            "devices must be a non-empty sequence such as "
            "('cuda:0', 'cuda:1')."
        )
    if not devices:
        raise ValueError("devices must contain at least one CUDA device.")

    normalized_devices: list[str] = []
    for device_index, device in enumerate(devices):
        if not isinstance(device, str) or not device.strip():
            raise ValueError(
                f"devices[{device_index}] must be a non-empty string, "
                f"got {device!r}."
            )
        try:
            parsed_device = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"Invalid CUDA device at devices[{device_index}]: "
                f"{device!r}: {error}."
            ) from error
        if parsed_device.type != "cuda" or parsed_device.index is None:
            raise ValueError(
                f"devices[{device_index}] must be an explicitly indexed CUDA "
                f"device such as 'cuda:0', got {device!r}."
            )
        normalized_devices.append(f"cuda:{parsed_device.index}")

    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError(
            f"devices must not contain duplicates, got {normalized_devices!r}."
        )
    return tuple(normalized_devices)


def _validate_cuda_devices_available(devices: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Dev generation.")
    device_count = torch.cuda.device_count()
    unavailable_devices = [
        device
        for device in devices
        if torch.device(device).index >= device_count
    ]
    if unavailable_devices:
        raise ValueError(
            f"CUDA devices {unavailable_devices!r} are unavailable; detected "
            f"{device_count} visible device(s)."
        )


def _reject_distributed_launch() -> None:
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as error:
        raise ValueError(
            f"WORLD_SIZE must be an integer, got {raw_world_size!r}."
        ) from error
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}.")
    if world_size > 1:
        raise RuntimeError(
            "generate_dev launches its own per-GPU worker processes. Do not "
            "use torchrun; pass all target GPUs through --devices instead."
        )


def _resolve_cuda_device(device: str) -> torch.device:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(
            "device must be a non-empty CUDA device string, "
            f"got {device!r}."
        )
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid CUDA device {device!r}: {error}.") from error
    if resolved_device.type != "cuda":
        raise ValueError(
            f"Dev generation requires a CUDA device, got {device!r}."
        )
    device_index = (
        resolved_device.index
        if resolved_device.index is not None
        else 0
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Dev generation.")
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device_index} is unavailable; detected "
            f"{torch.cuda.device_count()} device(s)."
        )
    normalized_device = torch.device("cuda", device_index)
    torch.cuda.set_device(normalized_device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"CUDA device {normalized_device} does not support BF16 generation."
        )
    return normalized_device


def _load_tokenizer(
    model_source: str,
    adapter_source: str | None,
    tokenizer_source: str | None,
) -> tuple[Any, str]:
    selected_source = tokenizer_source or adapter_source or model_source
    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_source, use_fast=True)
    except Exception as error:
        source_role = (
            "explicit tokenizer"
            if tokenizer_source is not None
            else "adapter tokenizer"
            if adapter_source is not None
            else "base-model tokenizer"
        )
        raise RuntimeError(
            f"Could not load the {source_role} from {selected_source!r}: "
            f"{type(error).__name__}: {error}."
        ) from error

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("The selected tokenizer has no non-empty Chat Template.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer has no EOS token ID.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("The selected tokenizer has no usable pad token ID.")
    tokenizer.padding_side = "left"
    return tokenizer, selected_source


def _load_model(
    model_source: str,
    adapter_source: str | None,
    device: torch.device,
) -> Any:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            attn_implementation="sdpa", # use scaled dot-product attention for better performance
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load the base model from {model_source!r}: "
            f"{type(error).__name__}: {error}."
        ) from error

    if adapter_source is not None:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                adapter_source,
                is_trainable=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load the LoRA adapter from {adapter_source!r}: "
                f"{type(error).__name__}: {error}."
            ) from error

    model.config.use_cache = True
    model.eval()
    return model


def _model_context_limit(model: Any) -> int:
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if type(context_limit) is not int or context_limit <= 0:
        raise RuntimeError(
            "The loaded model does not expose a positive integer "
            "config.max_position_embeddings."
        )
    return context_limit


def _generate_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    *,
    device: torch.device,
    max_new_tokens: int,
    context_limit: int,
) -> list[str]:
    prompts = [record["prompt"] for record in records]
    try:
        encoded = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            return_dict=True, # return a dictionary containing input_ids and attention_mask
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_generation_prompt=True, # add a generation prompt "<assistant>" to the input
            enable_thinking=False,
        )
    except Exception as error:
        question_ids = [record["question_id"] for record in records]
        raise RuntimeError(
            f"Chat Template tokenization failed for question_id values "
            f"{question_ids!r}: {type(error).__name__}: {error}."
        ) from error

    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError(
            "Chat Template must return a two-dimensional input_ids tensor."
        )
    if (
        not torch.is_tensor(attention_mask)
        or attention_mask.shape != input_ids.shape
    ):
        raise RuntimeError(
            "Chat Template must return attention_mask with the same shape as "
            "input_ids."
        )
    if input_ids.shape[0] != len(records):
        raise RuntimeError(
            "Chat Template returned a batch size inconsistent with the input "
            f"records: expected {len(records)}, got {input_ids.shape[0]}."
        )

    prompt_lengths = attention_mask.sum(dim=1).tolist()
    for record, prompt_length in zip(records, prompt_lengths, strict=True):
        if type(prompt_length) is not int or prompt_length <= 0:
            raise RuntimeError(
                f"question_id={record['question_id']} has invalid Prompt token "
                f"length {prompt_length!r}."
            )
        if prompt_length + max_new_tokens > context_limit:
            raise ValueError(
                "Dev Prompt plus maximum generation exceeds the model context: "
                f"question_id={record['question_id']}, prompt_tokens="
                f"{prompt_length}, max_new_tokens={max_new_tokens}, "
                f"context_limit={context_limit}."
            )

    encoded = encoded.to(device)
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    input_width = encoded["input_ids"].shape[1]
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                do_sample=False, # do not sample, use greedy decoding
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1, # do not use beam search, use greedy decoding
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
    except Exception as error:
        question_ids = [record["question_id"] for record in records]
        raise RuntimeError(
            f"Model generation failed for question_id values {question_ids!r}: "
            f"{type(error).__name__}: {error}."
        ) from error

    if not torch.is_tensor(generated_ids) or generated_ids.ndim != 2:
        raise RuntimeError("model.generate() must return a two-dimensional tensor.")
    if generated_ids.shape[0] != len(records):
        raise RuntimeError(
            "model.generate() returned a batch size inconsistent with the input "
            f"records: expected {len(records)}, got {generated_ids.shape[0]}."
        )
    if generated_ids.shape[1] <= input_width:
        raise RuntimeError(
            "model.generate() returned no new tokens after the Prompt."
        )

    new_token_ids = generated_ids[:, input_width:] # extract only the newly generated tokens after the prompt
    raw_outputs = tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )
    if not isinstance(raw_outputs, list) or len(raw_outputs) != len(records):
        raise RuntimeError(
            "Tokenizer batch_decode() returned an unexpected number of outputs."
        )
    if any(not isinstance(raw_output, str) for raw_output in raw_outputs):
        raise RuntimeError("Tokenizer batch_decode() must return strings.")
    return raw_outputs


def _build_prediction(record: dict, raw_output: str) -> dict:
    predicted_sql, extraction_status, format_compliance = (
        extract_predicted_sql(raw_output)
    )
    return {
        "question_id": record["question_id"],
        "db_id": record["db_id"],
        "raw_output": raw_output,
        "predicted_sql": predicted_sql,
        "extraction_status": extraction_status,
        "format_compliance": format_compliance,
    }


def _validate_predictions(
    predictions: object,
    expected_records: list[dict],
    *,
    require_complete: bool,
    description: str,
) -> list[dict]:
    if not isinstance(predictions, list):
        raise _generation_error(
            description,
            f"expected a list, got {type(predictions).__name__}.",
        )
    if len(predictions) > len(expected_records):
        raise _generation_error(
            description,
            f"contains {len(predictions)} records but only "
            f"{len(expected_records)} are expected.",
        )
    if require_complete and len(predictions) != len(expected_records):
        raise _generation_error(
            description,
            f"expected {len(expected_records)} records, got {len(predictions)}.",
        )

    for record_index, prediction in enumerate(predictions):
        expected_record = expected_records[record_index]
        context = f"{description}, record_index={record_index}"
        if not isinstance(prediction, dict):
            raise _generation_error(
                context,
                f"prediction must be an object, got {prediction!r}.",
            )
        if set(prediction) != PREDICTION_FIELDS:
            raise _generation_error(
                context,
                f"expected exactly fields {sorted(PREDICTION_FIELDS)!r}, got "
                f"{sorted(prediction)!r}.",
            )
        if prediction["question_id"] != expected_record["question_id"]:
            raise _generation_error(
                f"{context}, field='question_id'",
                f"expected {expected_record['question_id']!r}, got "
                f"{prediction['question_id']!r}.",
            )
        if type(prediction["question_id"]) is not int:
            raise _generation_error(
                f"{context}, field='question_id'",
                "question_id must be an integer that is not boolean.",
            )
        if prediction["db_id"] != expected_record["db_id"]:
            raise _generation_error(
                f"{context}, field='db_id'",
                f"expected {expected_record['db_id']!r}, got "
                f"{prediction['db_id']!r}.",
            )
        for field_name in ("raw_output", "predicted_sql"):
            if not isinstance(prediction[field_name], str):
                raise _generation_error(
                    f"{context}, field={field_name!r}",
                    f"expected a string, got {prediction[field_name]!r}.",
                )
        if (
            not isinstance(prediction["extraction_status"], str)
            or prediction["extraction_status"] not in EXTRACTION_STATUSES
        ):
            raise _generation_error(
                f"{context}, field='extraction_status'",
                f"expected one of {sorted(EXTRACTION_STATUSES)!r}, got "
                f"{prediction['extraction_status']!r}.",
            )
        if type(prediction["format_compliance"]) is not bool:
            raise _generation_error(
                f"{context}, field='format_compliance'",
                "expected a boolean value, got "
                f"{prediction['format_compliance']!r}.",
            )

        extracted = extract_predicted_sql(prediction["raw_output"])
        recorded_extraction = (
            prediction["predicted_sql"],
            prediction["extraction_status"],
            prediction["format_compliance"],
        )
        if recorded_extraction != extracted:
            raise _generation_error(
                context,
                "predicted_sql, extraction_status, or format_compliance is "
                "inconsistent with the extraction of raw_output.",
            )
    return predictions


def _load_prediction_jsonl(path: Path, description: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")

    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.endswith("\n"):
                    raise ValueError(
                        f"line {line_number} is incomplete because it does not "
                        "end with a newline."
                    )
                if not line.strip():
                    raise ValueError(f"line {line_number} is blank.")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"line {line_number} must contain a JSON object, got "
                        f"{type(value).__name__}."
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not read {description} at {path}: {error}"
        ) from error
    return records


def _output_paths(output_dir: str | Path, world_size: int) -> dict[str, Any]:
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise TypeError("output_dir must be a non-empty string or Path.")
    run_directory = Path(output_dir).expanduser()
    if not run_directory.is_absolute():
        run_directory = PROJECT_ROOT / run_directory
    if run_directory.is_symlink() or (
        run_directory.exists() and not run_directory.is_dir()
    ):
        raise ValueError(
            "Prediction run path must be a real directory when it already "
            f"exists, got {run_directory}."
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    run_directory = run_directory.resolve()

    paths = {
        "directory": run_directory,
        "final": run_directory / PREDICTIONS_FILENAME,
        "merged": run_directory / MERGED_PREDICTIONS_FILENAME,
        "state": run_directory / RESUME_STATE_FILENAME,
        "shards": [
            run_directory
            / SHARD_PREDICTIONS_FILENAME_TEMPLATE.format(rank=rank)
            for rank in range(world_size)
        ],
    }
    for role in ("final", "merged", "state"):
        path = paths[role]
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(
                f"Prediction {role} path must be a regular, non-symlink file "
                f"when it exists, got {path}."
            )
    for rank, path in enumerate(paths["shards"]):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(
                f"Prediction shard path for rank={rank} must be a regular, "
                f"non-symlink file when it exists, got {path}."
            )
    return paths


def _acquire_run_lock(run_directory: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(run_directory, flags)
    except OSError as error:
        raise RuntimeError(
            f"Could not open prediction run directory {run_directory}: {error}."
        ) from error

    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(file_descriptor)
        raise RuntimeError(
            "Another generation process is already using prediction run "
            f"directory {run_directory}."
        ) from error
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _release_run_lock(file_descriptor: int) -> None:
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(file_descriptor)


def _build_resume_state(
    *,
    run_name: str,
    data_path: Path,
    selected_sample_count: int,
    model_source: str,
    adapter_source: str | None,
    tokenizer_source: str,
    batch_size: int,
    max_new_tokens: int,
    devices: tuple[str, ...],
) -> dict:
    return {
        "run_name": run_name,
        "data_path": _portable_source(str(data_path.resolve())),
        "selected_sample_count": selected_sample_count,
        "model_source": _portable_source(model_source),
        "adapter_source": _portable_source(adapter_source),
        "tokenizer_source": _portable_source(tokenizer_source),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "devices": list(devices),
        "world_size": len(devices),
        "enable_thinking": False,
        "do_sample": False,
    }


def _write_new_resume_files(
    state_path: Path,
    shard_paths: list[Path],
    state: dict,
) -> None:
    state_created = False
    created_shard_paths: list[Path] = []
    try:
        with state_path.open("x", encoding="utf-8") as file:
            state_created = True
            json.dump(state, file, ensure_ascii=False, indent=4)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        for shard_path in shard_paths:
            with shard_path.open("x", encoding="utf-8") as file:
                created_shard_paths.append(shard_path)
                file.flush()
                os.fsync(file.fileno())
    except BaseException:
        for shard_path in created_shard_paths:
            shard_path.unlink(missing_ok=True)
        if state_created:
            state_path.unlink(missing_ok=True)
        raise


def _prepare_progress(
    *,
    paths: dict[str, Any],
    state: dict,
    shard_records: list[list[dict]],
    resume: bool,
) -> list[list[dict]]:
    final_path = paths["final"]
    state_path = paths["state"]
    shard_paths = paths["shards"]
    expected_shard_paths = set(shard_paths)
    existing_shard_paths = set(
        paths["directory"].glob("predictions*.partial.jsonl")
    )
    unexpected_shard_paths = existing_shard_paths - expected_shard_paths
    state_exists = state_path.exists()

    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(
            f"Final predictions already exist at {final_path}; choose a new "
            "output directory or remove the existing predictions explicitly."
        )
    if unexpected_shard_paths:
        raise RuntimeError(
            "Unexpected prediction shard files exist in the run directory: "
            f"{sorted(str(path) for path in unexpected_shard_paths)!r}."
        )

    if resume:
        if existing_shard_paths and not state_exists:
            raise RuntimeError(
                "Cannot resume because prediction shards exist without the "
                "resume state that identifies their generation settings."
            )
        if not state_exists:
            raise FileNotFoundError(
                f"No interrupted generation exists to resume in {paths['directory']}."
            )
        try:
            saved_state = load_json(state_path)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(
                f"Could not read resume state at {state_path}: {error}."
            ) from error
        if saved_state != state:
            if isinstance(saved_state, dict):
                mismatched_fields = sorted(
                    key
                    for key in set(saved_state) | set(state)
                    if saved_state.get(key) != state.get(key)
                )
            else:
                mismatched_fields = ["<top-level structure>"]
            raise ValueError(
                "Resume state does not match this generation request; "
                f"mismatched fields={mismatched_fields!r}."
            )
        paths["merged"].unlink(missing_ok=True)
        for shard_path in shard_paths:
            if not shard_path.exists():
                with shard_path.open("x", encoding="utf-8") as file:
                    file.flush()
                    os.fsync(file.fileno())
        return [
            _validate_predictions(
                _load_prediction_jsonl(
                    shard_path,
                    f"partial predictions for rank={rank}",
                ),
                shard_records[rank],
                require_complete=False,
                description=f"partial predictions for rank={rank}",
            )
            for rank, shard_path in enumerate(shard_paths)
        ]

    if existing_shard_paths or state_exists or paths["merged"].exists():
        raise FileExistsError(
            f"Interrupted generation files exist in {paths['directory']}; pass "
            "resume=True to continue or remove the run directory explicitly "
            "to restart."
        )

    _write_new_resume_files(state_path, shard_paths, state)
    return [[] for _ in shard_paths]


def _append_predictions(path: Path, predictions: list[dict]) -> None:
    if not predictions:
        return
    payload = "".join(
        json.dumps(prediction, ensure_ascii=False) + "\n"
        for prediction in predictions
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _generate_shard_worker(
    rank: int,
    devices: tuple[str, ...],
    data_path_text: str,
    selected_sample_count: int,
    model_source: str,
    adapter_source: str | None,
    explicit_tokenizer_source: str | None,
    batch_size: int,
    max_new_tokens: int,
    shard_path_texts: tuple[str, ...],
) -> None:
    device = devices[rank]
    dev_records = load_dev_dataset(data_path_text)
    selected_records = dev_records[:selected_sample_count]
    rank_records = selected_records[rank::len(devices)]
    shard_path = Path(shard_path_texts[rank])
    predictions = _validate_predictions(
        _load_prediction_jsonl(
            shard_path,
            f"partial predictions for rank={rank}",
        ),
        rank_records,
        require_complete=False,
        description=f"partial predictions for rank={rank}",
    )
    if len(predictions) == len(rank_records):
        print(
            f"[rank={rank} device={device}] shard already complete: "
            f"{len(predictions)}/{len(rank_records)} samples.",
            flush=True,
        )
        return

    resolved_device = _resolve_cuda_device(device)
    tokenizer, _ = _load_tokenizer(
        model_source,
        adapter_source,
        explicit_tokenizer_source,
    )
    model = _load_model(model_source, adapter_source, resolved_device)
    context_limit = _model_context_limit(model)
    started_at = time.perf_counter()

    for batch_start in range(
        len(predictions),
        len(rank_records),
        batch_size,
    ):
        batch_records = rank_records[batch_start : batch_start + batch_size]
        raw_outputs = _generate_batch(
            model,
            tokenizer,
            batch_records,
            device=resolved_device,
            max_new_tokens=max_new_tokens,
            context_limit=context_limit,
        )
        batch_predictions = [
            _build_prediction(record, raw_output)
            for record, raw_output in zip(
                batch_records,
                raw_outputs,
                strict=True,
            )
        ]
        candidate_predictions = [*predictions, *batch_predictions]
        _validate_predictions(
            candidate_predictions,
            rank_records,
            require_complete=False,
            description=f"generated predictions for rank={rank}",
        )
        _append_predictions(shard_path, batch_predictions)
        predictions.extend(batch_predictions)

        elapsed_seconds = time.perf_counter() - started_at
        print(
            f"[rank={rank} device={device}] generated "
            f"{len(predictions)}/{len(rank_records)} shard samples in "
            f"{elapsed_seconds:.1f}s.",
            flush=True,
        )

    _validate_predictions(
        predictions,
        rank_records,
        require_complete=True,
        description=f"completed predictions for rank={rank}",
    )


def _run_generation_workers(
    *,
    devices: tuple[str, ...],
    data_path: Path,
    selected_sample_count: int,
    model_source: str,
    adapter_source: str | None,
    explicit_tokenizer_source: str | None,
    batch_size: int,
    max_new_tokens: int,
    shard_paths: list[Path],
) -> None:
    worker_arguments = (
        devices,
        str(data_path),
        selected_sample_count,
        model_source,
        adapter_source,
        explicit_tokenizer_source,
        batch_size,
        max_new_tokens,
        tuple(str(path) for path in shard_paths),
    )
    if len(devices) == 1:
        _generate_shard_worker(0, *worker_arguments)
        return
    mp.spawn(
        _generate_shard_worker,
        args=worker_arguments,
        nprocs=len(devices),
        join=True,
    )


def _write_predictions_file(path: Path, predictions: list[dict]) -> None:
    payload = "".join(
        json.dumps(prediction, ensure_ascii=False) + "\n"
        for prediction in predictions
    )
    with path.open("x", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _publish_predictions(
    *,
    paths: dict[str, Any],
    selected_records: list[dict],
    shard_records: list[list[dict]],
) -> list[dict]:
    predictions: list[dict] = []
    for rank, shard_path in enumerate(paths["shards"]):
        predictions.extend(
            _validate_predictions(
                _load_prediction_jsonl(
                    shard_path,
                    f"completed prediction shard for rank={rank}",
                ),
                shard_records[rank],
                require_complete=True,
                description=f"completed prediction shard for rank={rank}",
            )
        )

    predictions.sort(key=lambda prediction: prediction["question_id"])
    predictions = _validate_predictions(
        predictions,
        selected_records,
        require_complete=True,
        description="merged predictions",
    )

    try:
        _write_predictions_file(paths["merged"], predictions)
        merged_predictions = _validate_predictions(
            _load_prediction_jsonl(
                paths["merged"],
                "temporary merged predictions",
            ),
            selected_records,
            require_complete=True,
            description="temporary merged predictions",
        )
        if paths["final"].exists() or paths["final"].is_symlink():
            raise FileExistsError(
                "Final predictions appeared while generation was running at "
                f"{paths['final']}; refusing to replace them."
            )
        paths["merged"].replace(paths["final"])
        published_predictions = _validate_predictions(
            _load_prediction_jsonl(paths["final"], "published predictions"),
            selected_records,
            require_complete=True,
            description="published predictions",
        )
        if published_predictions != merged_predictions:
            raise RuntimeError(
                "Published predictions differ from the validated merged data."
            )
    except BaseException:
        paths["merged"].unlink(missing_ok=True)
        raise

    for shard_path in paths["shards"]:
        shard_path.unlink()
    paths["state"].unlink()
    return published_predictions


def generate_dev_predictions(
    model_path: str | Path = DEFAULT_MODEL_ID,
    *,
    adapter_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    run_name: str,
    output_dir: str | Path,
    batch_size: int = 1,
    max_new_tokens: int = 2048,
    devices: Sequence[str] = ("cuda:0",),
    resume: bool = False,
    max_samples: int | None = None,
) -> list[dict]:
    """Generate evaluator-ready Dev predictions across one or more GPUs."""
    _reject_distributed_launch()
    safe_run_name = _validate_run_name(run_name)
    normalized_devices = _normalize_devices(devices)
    validated_batch_size = _validate_positive_integer(batch_size, "batch_size")
    validated_max_new_tokens = _validate_positive_integer(
        max_new_tokens,
        "max_new_tokens",
    )
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean value.")

    data_path = DEV_DATA_PATH
    dev_records = load_dev_dataset(data_path)
    if max_samples is None:
        selected_records = dev_records
    else:
        validated_max_samples = _validate_positive_integer(
            max_samples,
            "max_samples",
        )
        if validated_max_samples > len(dev_records):
            raise ValueError(
                f"max_samples cannot exceed {len(dev_records)}, got "
                f"{validated_max_samples}."
            )
        selected_records = dev_records[:validated_max_samples]

    model_source = _normalize_source(model_path, "model_path")
    adapter_source = _normalize_source(adapter_path, "adapter_path")
    explicit_tokenizer_source = _normalize_source(
        tokenizer_path,
        "tokenizer_path",
    )
    if model_source is None:
        raise ValueError("model_path must not be None.")
    _validate_cuda_devices_available(normalized_devices)

    selected_tokenizer_source = (
        explicit_tokenizer_source or adapter_source or model_source
    )
    shard_records = [
        selected_records[rank::len(normalized_devices)]
        for rank in range(len(normalized_devices))
    ]
    paths = _output_paths(output_dir, len(normalized_devices))
    run_lock = _acquire_run_lock(paths["directory"])
    try:
        state = _build_resume_state(
            run_name=safe_run_name,
            data_path=data_path,
            selected_sample_count=len(selected_records),
            model_source=model_source,
            adapter_source=adapter_source,
            tokenizer_source=selected_tokenizer_source,
            batch_size=validated_batch_size,
            max_new_tokens=validated_max_new_tokens,
            devices=normalized_devices,
        )
        shard_progress = _prepare_progress(
            paths=paths,
            state=state,
            shard_records=shard_records,
            resume=resume,
        )

        if any(
            len(progress) < len(shard_records[rank])
            for rank, progress in enumerate(shard_progress)
        ):
            _run_generation_workers(
                devices=normalized_devices,
                data_path=data_path,
                selected_sample_count=len(selected_records),
                model_source=model_source,
                adapter_source=adapter_source,
                explicit_tokenizer_source=explicit_tokenizer_source,
                batch_size=validated_batch_size,
                max_new_tokens=validated_max_new_tokens,
                shard_paths=paths["shards"],
            )

        return _publish_predictions(
            paths=paths,
            selected_records=selected_records,
            shard_records=shard_records,
        )
    finally:
        _release_run_lock(run_lock)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluator-ready BIRD Dev predictions with Qwen3 on one "
            "or more CUDA GPUs."
        )
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_ID,
        help=(
            "Base model directory or Hugging Face model ID "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT LoRA adapter directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional tokenizer directory or Hugging Face model ID. It takes "
            "priority over the adapter and base model tokenizer."
        ),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Experiment identity recorded in resumable generation state.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory that directly receives predictions.jsonl and any "
            "interrupted-run recovery files."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU generation batch size (default: 1).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens per sample (default: 2048).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help=(
            "Ordered CUDA devices used for sample-parallel generation "
            "(default: cuda:0)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching per-device prediction shard progress.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Generate only the first N Dev samples for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    predictions = generate_dev_predictions(
        model_path=arguments.model_path,
        adapter_path=arguments.adapter_path,
        tokenizer_path=arguments.tokenizer_path,
        run_name=arguments.run_name,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
        max_new_tokens=arguments.max_new_tokens,
        devices=arguments.devices,
        resume=arguments.resume,
        max_samples=arguments.max_samples,
    )
    extraction_counts = {
        status: sum(
            prediction["extraction_status"] == status
            for prediction in predictions
        )
        for status in (
            "sql_fence",
            "generic_fence",
            "raw_sql",
            "failed",
        )
    }
    compliant_count = sum(
        prediction["format_compliance"] for prediction in predictions
    )
    compliance_rate = (
        compliant_count * 100.0 / len(predictions) if predictions else 0.0
    )
    output_directory = Path(arguments.output_dir).expanduser()
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory
    output_path = output_directory.resolve() / PREDICTIONS_FILENAME
    print("Dev generation completed:")
    print(f"samples={len(predictions)}")
    for status, count in extraction_counts.items():
        print(f"{status}={count}")
    print(
        f"format_compliance={compliant_count}/{len(predictions)} "
        f"({compliance_rate:.2f}%)"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
