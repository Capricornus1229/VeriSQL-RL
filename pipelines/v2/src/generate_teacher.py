"""Generate Gold-conditioned rationales and materialize the SFT datasets.

The 14B teacher is used only for a short explanation.  The SQL target is always
copied from the prepared data, so a teacher response can never alter the Gold
SQL used for supervision.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline.yaml"

DATA_ROOT = PIPELINE_ROOT / "artifacts" / "data"
TEACHER_ROOT = PIPELINE_ROOT / "artifacts" / "teacher"
TEACHER_WORK_ROOT = PIPELINE_ROOT / ".work" / "teacher"

BASE_INPUT_PATHS = {
    "train": DATA_ROOT / "train.jsonl",
    "validation": DATA_ROOT / "validation.jsonl",
}
RATIONALE_PATHS = {
    "train": TEACHER_ROOT / "rationales.train.jsonl",
    "validation": TEACHER_ROOT / "rationales.validation.jsonl",
}
SFT_PATHS = {
    "train": DATA_ROOT / "sft_train.jsonl",
    "validation": DATA_ROOT / "sft_validation.jsonl",
}

RATIONALE_PATTERN = re.compile(
    r"\A<RATIONALE>[ \t]*\r?\n(?P<rationale>.+?)\r?\n</RATIONALE>\Z",
    flags=re.DOTALL,
)
SQL_FENCE_PREFIX = "```sql\n"
SQL_FENCE_SUFFIX = "\n```"

TEACHER_SYSTEM_PROMPT = """You are a SQLite reasoning annotator.

Given a database schema, evidence, question, and a verified Gold SQL query,
briefly explain why that exact query answers the question. Focus on tables,
joins, filters, aggregation, and ordering that actually occur in the Gold SQL.
Do not rewrite, correct, or reproduce the SQL.

Return exactly one block in this form and no other text:
<RATIONALE>
1. First concise step.
2. Second concise step.
</RATIONALE>

Use two to four concise numbered steps."""


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("V2 config must be a YAML object.")
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
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def extract_gold_sql(record: dict[str, Any]) -> str:
    gold_sql = record.get("gold_sql")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError(f"sample_id={record.get('sample_id')!r} has no Gold SQL.")

    completion = record.get("completion")
    if completion is not None:
        if not isinstance(completion, list) or len(completion) != 1:
            raise ValueError("Prepared SFT records require one assistant completion.")
        content = completion[0].get("content")
        expected_content = f"{SQL_FENCE_PREFIX}{gold_sql}{SQL_FENCE_SUFFIX}"
        if content != expected_content:
            raise ValueError(
                f"sample_id={record.get('sample_id')!r} completion does not "
                "preserve gold_sql exactly."
            )
    return gold_sql


def parse_rationale(raw_output: str) -> str | None:
    """Parse a single complete teacher rationale block."""
    if not isinstance(raw_output, str):
        return None
    match = RATIONALE_PATTERN.fullmatch(raw_output.strip())
    if match is None:
        return None
    rationale = match.group("rationale").strip()
    return rationale or None


def build_teacher_messages(
    record: dict[str, Any],
    gold_sql: str | None = None,
) -> list[dict[str, str]]:
    """Build the teacher-only prompt; this prompt is never used by the student."""
    if gold_sql is None:
        gold_sql = extract_gold_sql(record)
    prompt = record.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("Prepared records require a system/user prompt pair.")
    user_content = prompt[1].get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError("Prepared records require a non-empty user prompt.")

    teacher_user = (
        f"{user_content}\n\n"
        "Verified Gold SQL (reference only):\n"
        f"<GOLD_SQL>\n{gold_sql}\n</GOLD_SQL>"
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": teacher_user},
    ]


def build_reasoning_record(
    record: dict[str, Any],
    rationale: str | None,
) -> dict[str, Any]:
    """Attach teacher reasoning while copying the exact source Gold SQL."""
    gold_sql = extract_gold_sql(record)
    output = dict(record)
    output["prompt"] = [dict(message) for message in record["prompt"]]
    output["completion"] = [
        {
            "role": "assistant",
            "reasoning_content": rationale or "",
            "content": f"{SQL_FENCE_PREFIX}{gold_sql}{SQL_FENCE_SUFFIX}",
        }
    ]
    output["rationale_status"] = "teacher" if rationale else "sql_only_fallback"
    return output


def _artifact_identity(record: dict[str, Any]) -> tuple[object, object, object]:
    return record.get("sample_id"), record.get("source_index"), record.get("db_id")


def _teacher_artifact(
    source: dict[str, Any],
    raw_output: str,
) -> dict[str, Any]:
    rationale = parse_rationale(raw_output)
    return {
        "sample_id": source.get("sample_id"),
        "source_index": source.get("source_index"),
        "db_id": source.get("db_id"),
        "status": "success" if rationale else "fallback",
        "raw_output": raw_output,
        "rationale": rationale,
    }


def _validate_artifact_prefix(
    source_records: list[dict[str, Any]],
    artifact_records: list[dict[str, Any]],
) -> None:
    if len(artifact_records) > len(source_records):
        raise ValueError("Teacher partial contains more records than its source data.")
    for index, artifact in enumerate(artifact_records):
        if _artifact_identity(artifact) != _artifact_identity(source_records[index]):
            raise ValueError(
                f"Teacher partial stops matching source data at record {index}."
            )
        if artifact.get("status") not in {"success", "fallback"}:
            raise ValueError(f"Teacher artifact {index} has an invalid status.")


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _load_teacher(config: dict[str, Any]) -> tuple[Any, torch.nn.Module]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Balanced teacher generation requires two visible GPUs.")
    model_settings = config["model"]
    teacher_settings = config["teacher"]
    if teacher_settings.get("device_map") != "balanced":
        raise ValueError("teacher.device_map must be 'balanced'.")

    source = model_settings["teacher_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        attn_implementation=model_settings["attention_implementation"],
        device_map="balanced",
        max_memory={0: "30GiB", 1: "30GiB"},
    )
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def _generate_one(
    tokenizer: Any,
    model: torch.nn.Module,
    messages: list[dict[str, str]],
    *,
    max_length: int,
    max_new_tokens: int,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_length = int(inputs["input_ids"].shape[1])
    if input_length + max_new_tokens > max_length:
        raise ValueError(
            f"Teacher input requires {input_length + max_new_tokens} tokens, "
            f"above teacher.max_length={max_length}."
        )
    device = _input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = generated[0, input_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_rationales(
    config: dict[str, Any],
    *,
    source_path: str | Path,
    output_path: str | Path,
    resume: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Generate rationale artifacts with one balanced 14B model."""
    source_records = load_jsonl(source_path)
    if not source_records:
        raise ValueError(f"Teacher source is empty: {source_path}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        source_records = source_records[:max_samples]
    output_path = Path(output_path)
    partial_path = TEACHER_WORK_ROOT / f"{output_path.name}.partial"
    if output_path.exists():
        raise FileExistsError(f"Teacher output already exists: {output_path}")

    completed = load_jsonl(partial_path) if resume and partial_path.exists() else []
    if partial_path.exists() and not resume:
        raise FileExistsError(
            f"Teacher partial exists; pass --resume: {partial_path}"
        )
    _validate_artifact_prefix(source_records, completed)
    if len(completed) == len(source_records):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.replace(output_path)
        return completed

    tokenizer, model = _load_teacher(config)
    max_new_tokens = int(config["teacher"]["max_new_tokens"])
    max_length = int(config["teacher"]["max_length"])
    started = time.perf_counter()
    for record in source_records[len(completed) :]:
        gold_sql = extract_gold_sql(record)
        messages = build_teacher_messages(record, gold_sql)
        raw_output = _generate_one(
            tokenizer,
            model,
            messages,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        artifact = _teacher_artifact(record, raw_output)
        _append_jsonl(partial_path, artifact)
        completed.append(artifact)
        if len(completed) % 25 == 0 or len(completed) == len(source_records):
            print(
                f"Teacher progress: {len(completed)}/{len(source_records)}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    _validate_artifact_prefix(source_records, completed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(output_path)
    return completed


def run_teacher_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Generate the longest Train teacher input without touching full artifacts."""
    source_records = load_jsonl(BASE_INPUT_PATHS["train"])
    tokenizer, model = _load_teacher(config)

    longest_record: dict[str, Any] | None = None
    longest_messages: list[dict[str, str]] | None = None
    longest_input_tokens = -1
    for record in source_records:
        messages = build_teacher_messages(record)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if len(encoded) > longest_input_tokens:
            longest_record = record
            longest_messages = messages
            longest_input_tokens = len(encoded)

    assert longest_record is not None and longest_messages is not None
    started = time.perf_counter()
    raw_output = _generate_one(
        tokenizer,
        model,
        longest_messages,
        max_length=int(config["teacher"]["max_length"]),
        max_new_tokens=int(config["teacher"]["max_new_tokens"]),
    )
    artifact = _teacher_artifact(longest_record, raw_output)
    result = {
        **artifact,
        "input_tokens": longest_input_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gib": [
            round(torch.cuda.max_memory_allocated(index) / (1024**3), 3)
            for index in range(2)
        ],
    }
    _write_json(TEACHER_WORK_ROOT / "smoke.json", result)
    return result


def materialize_reasoning_dataset(
    source_path: str | Path,
    rationale_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    sources = load_jsonl(source_path)
    artifacts = load_jsonl(rationale_path)
    if len(sources) != len(artifacts):
        raise ValueError("Teacher artifact and source data counts differ.")
    _validate_artifact_prefix(sources, artifacts)
    records = [
        build_reasoning_record(source, artifact.get("rationale"))
        for source, artifact in zip(sources, artifacts, strict=True)
    ]
    _write_jsonl(Path(output_path), records)
    return records


def build_sft_dataset() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        records = materialize_reasoning_dataset(
            BASE_INPUT_PATHS[split],
            RATIONALE_PATHS[split],
            SFT_PATHS[split],
        )
        counts[split] = len(records)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "generate-train",
            "generate-validation",
            "build-sft",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate only the longest Train teacher input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        if args.command != "generate-train":
            raise ValueError("--smoke is supported only with generate-train.")
        result = run_teacher_smoke(config)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command == "generate-train":
        output_path = RATIONALE_PATHS["train"]
        if args.max_samples is not None:
            output_path = TEACHER_ROOT / "rationales.train.smoke.jsonl"
        records = generate_rationales(
            config,
            source_path=BASE_INPUT_PATHS["train"],
            output_path=output_path,
            resume=args.resume,
            max_samples=args.max_samples,
        )
        successes = sum(record["status"] == "success" for record in records)
        print(
            f"Teacher train rationales: {len(records)}, "
            f"success={successes}, fallback={len(records) - successes}"
        )
    elif args.command == "generate-validation":
        output_path = RATIONALE_PATHS["validation"]
        if args.max_samples is not None:
            output_path = TEACHER_ROOT / "rationales.validation.smoke.jsonl"
        records = generate_rationales(
            config,
            source_path=BASE_INPUT_PATHS["validation"],
            output_path=output_path,
            resume=args.resume,
            max_samples=args.max_samples,
        )
        successes = sum(record["status"] == "success" for record in records)
        print(
            f"Teacher validation rationales: {len(records)}, "
            f"success={successes}, fallback={len(records) - successes}"
        )
    else:
        print(json.dumps(build_sft_dataset(), ensure_ascii=False))


if __name__ == "__main__":
    main()
