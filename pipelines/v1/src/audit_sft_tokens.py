from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common import load_json, load_jsonl, save_json

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
SFT_TRAIN_PATH = PIPELINE_ROOT / "artifacts" / "data" / "sft_train.jsonl"
SFT_VAL_PATH = PIPELINE_ROOT / "artifacts" / "data" / "sft_val.jsonl"
DEV_EVAL_PATH = PIPELINE_ROOT / "artifacts" / "data" / "dev_eval.jsonl"
TOKEN_STATS_PATH = PIPELINE_ROOT / "artifacts" / "data" / "sft_token_stats.json"

CANDIDATE_MAX_LENGTHS = (4096, 8192, 12288, 16384)
PERCENTILES = (50, 90, 95, 99)
EXPECTED_SAMPLE_COUNTS = {
    "sft_train": 6013,
    "sft_val": 588,
    "dev_eval": 1534,
}
SPLIT_PATHS = {
    "sft_train": SFT_TRAIN_PATH,
    "sft_val": SFT_VAL_PATH,
    "dev_eval": DEV_EVAL_PATH,
}
SFT_SPLITS = ("sft_train", "sft_val")
ALL_SPLITS = (*SFT_SPLITS, "dev_eval")


def _audit_error(context: str, message: str) -> ValueError:
    """Create a consistently formatted token-audit validation error."""
    return ValueError(f"SFT token audit failed: {context}: {message}")


def _require_non_empty_string(value: object, context: str) -> str:
    """Require a non-empty string without changing its contents."""
    if not isinstance(value, str) or not value.strip():
        raise _audit_error(
            context,
            f"expected a non-empty string, got {value!r} "
            f"({type(value).__name__}).",
        )
    return value


def _validate_message(
    message: object,
    *,
    expected_role: str,
    context: str,
) -> dict:
    """Validate one conversational message used by the chat template."""
    if not isinstance(message, dict):
        raise _audit_error(
            context,
            f"expected an object, got {message!r} "
            f"({type(message).__name__}).",
        )
    if message.get("role") != expected_role:
        raise _audit_error(
            f"{context}, field='role'",
            f"expected {expected_role!r}, got {message.get('role')!r}.",
        )
    _require_non_empty_string(message.get("content"), f"{context}, field='content'")
    return message


def _validate_prompt(prompt: object, context: str) -> list[dict]:
    """Require the two-message system/user prompt produced by the builder."""
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise _audit_error(
            context,
            "prompt must be a two-message list containing system and user "
            f"messages, got {prompt!r}.",
        )
    _validate_message(
        prompt[0],
        expected_role="system",
        context=f"{context}[0]",
    )
    _validate_message(
        prompt[1],
        expected_role="user",
        context=f"{context}[1]",
    )
    return prompt


def _validate_completion(completion: object, context: str) -> list[dict]:
    """Require exactly one non-empty assistant completion."""
    if not isinstance(completion, list) or len(completion) != 1:
        raise _audit_error(
            context,
            f"completion must contain exactly one assistant message, got "
            f"{completion!r}.",
        )
    _validate_message(
        completion[0],
        expected_role="assistant",
        context=f"{context}[0]",
    )
    return completion


def _validate_token_ids(token_ids: object, context: str) -> list[int]:
    """Validate the non-batched output of apply_chat_template()."""
    if not isinstance(token_ids, list) or not token_ids:
        raise _audit_error(
            context,
            f"expected a non-empty list of token IDs, got {token_ids!r}.",
        )
    invalid_ids = [token_id for token_id in token_ids if type(token_id) is not int]
    if invalid_ids:
        raise _audit_error(
            context,
            f"token IDs must be integers, got invalid values {invalid_ids[:5]!r}.",
        )
    return token_ids


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    context: str,
) -> list[int]:
    """Tokenize one conversation with the audit's fixed Qwen3 settings."""
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except Exception as error:
        raise _audit_error(
            context,
            f"chat-template tokenization raised {type(error).__name__}: {error}.",
        ) from error
    return _validate_token_ids(token_ids, context)


def _sample_identity(sample: dict, split_name: str, context: str) -> dict:
    """Validate and return the report-safe identity fields for one sample."""
    db_id = _require_non_empty_string(sample.get("db_id"), f"{context}, field='db_id'")
    if split_name in SFT_SPLITS:
        sample_id = _require_non_empty_string(
            sample.get("sample_id"),
            f"{context}, field='sample_id'",
        )
        source_index = sample.get("source_index")
        if type(source_index) is not int or source_index < 0:
            raise _audit_error(
                f"{context}, field='source_index'",
                f"expected a non-negative integer, got {source_index!r}.",
            )
        return {
            "sample_id": sample_id,
            "source_index": source_index,
            "db_id": db_id,
        }

    question_id = sample.get("question_id")
    if type(question_id) is not int or question_id < 0:
        raise _audit_error(
            f"{context}, field='question_id'",
            f"expected a non-negative integer, got {question_id!r}.",
        )
    _require_non_empty_string(sample.get("gold_sql"), f"{context}, field='gold_sql'")
    if "completion" in sample:
        raise _audit_error(
            f"{context}, field='completion'",
            "Dev must remain prompt-only and must not contain completion.",
        )
    return {"question_id": question_id, "db_id": db_id}


def _audit_sample(
    tokenizer: Any,
    sample: object,
    split_name: str,
    source_index: int,
) -> dict:
    """Tokenize one sample and return only identity and length information."""
    context = f"split={split_name}, source_index={source_index}"
    if not isinstance(sample, dict):
        raise _audit_error(
            context,
            f"sample must be an object, got {type(sample).__name__}.",
        )

    identity = _sample_identity(sample, split_name, context)
    prompt = _validate_prompt(sample.get("prompt"), f"{context}, field='prompt'")
    prompt_ids = _apply_chat_template(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        context=f"{context}, prompt",
    )

    if split_name in SFT_SPLITS:
        completion = _validate_completion(
            sample.get("completion"),
            f"{context}, field='completion'",
        )
        full_ids = _apply_chat_template(
            tokenizer,
            prompt + completion,
            add_generation_prompt=False,
            context=f"{context}, full conversation",
        )
        if (
            len(full_ids) <= len(prompt_ids)
            or full_ids[: len(prompt_ids)] != prompt_ids
        ):
            raise _audit_error(
                context,
                "the full conversational token sequence does not have the "
                "generation-ready Prompt token sequence as a strict prefix.",
            )
        completion_token_count = len(full_ids) - len(prompt_ids)
        total_token_count = len(full_ids)
    else:
        completion_token_count = 0
        total_token_count = len(prompt_ids)

    return {
        "split": split_name,
        "source_index_in_file": source_index,
        **identity,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_token_count,
        "total_tokens": total_token_count,
    }


def _load_and_audit_split(tokenizer: Any, split_name: str, path: Path) -> list[dict]:
    """Load, validate, and tokenize one complete processed-data split."""
    if not path.is_file():
        raise FileNotFoundError(f"SFT token audit input does not exist: {path}")
    try:
        samples = load_jsonl(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"SFT token audit could not read split={split_name} from {path}: {error}."
        ) from error

    expected_count = EXPECTED_SAMPLE_COUNTS[split_name]
    if len(samples) != expected_count:
        raise _audit_error(
            f"split={split_name}",
            f"expected {expected_count} samples, got {len(samples)}.",
        )

    audited_samples = [
        _audit_sample(tokenizer, sample, split_name, source_index)
        for source_index, sample in enumerate(samples)
    ]

    if split_name in SFT_SPLITS:
        identities = [sample["sample_id"] for sample in audited_samples]
    else:
        identities = [sample["question_id"] for sample in audited_samples]
    if len(identities) != len(set(identities)):
        raise _audit_error(split_name, "sample identifiers must be unique.")
    return audited_samples


def _nearest_rank(values: list[int], percentile: int) -> int:
    """Return an integer percentile using the deterministic nearest-rank rule."""
    if not values:
        raise ValueError("nearest-rank percentile requires at least one value.")
    if type(percentile) is not int or not 1 <= percentile <= 100:
        raise ValueError(
            f"percentile must be an integer in [1, 100], got {percentile!r}."
        )
    ordered_values = sorted(values)
    rank = (percentile * len(ordered_values) + 99) // 100
    return ordered_values[rank - 1]


def _length_distribution(values: list[int]) -> dict[str, int]:
    """Build the fixed percentile/max summary used throughout the report."""
    return {
        **{
            f"p{percentile}": _nearest_rank(values, percentile)
            for percentile in PERCENTILES
        },
        "max": max(values),
    }


def _retained_distribution(values: list[int]) -> dict[str, int]:
    """Summarize supervised-token counts retained after truncation."""
    return {
        "min": min(values),
        **{
            f"p{percentile}": _nearest_rank(values, percentile)
            for percentile in PERCENTILES
        },
        "max": max(values),
        "total": sum(values),
    }


def _longest_samples(audited_samples: list[dict], limit: int = 10) -> list[dict]:
    """Return the longest samples, with input order as the deterministic tie-break."""
    longest = sorted(
        audited_samples,
        key=lambda sample: (
            -sample["total_tokens"],
            sample["source_index_in_file"],
        ),
    )[:limit]
    result = []
    for sample in longest:
        identity = (
            {"sample_id": sample["sample_id"]}
            if "sample_id" in sample
            else {"question_id": sample["question_id"]}
        )
        result.append(
            {
                "split": sample["split"],
                **identity,
                "db_id": sample["db_id"],
                "prompt_tokens": sample["prompt_tokens"],
                "completion_tokens": sample["completion_tokens"],
                "total_tokens": sample["total_tokens"],
            }
        )
    return result


def _build_split_statistics(audited_samples: list[dict]) -> dict:
    """Aggregate token lengths and threshold counts for one split."""
    prompt_lengths = [sample["prompt_tokens"] for sample in audited_samples]
    completion_lengths = [
        sample["completion_tokens"] for sample in audited_samples
    ]
    total_lengths = [sample["total_tokens"] for sample in audited_samples]
    statistics = {
        "sample_count": len(audited_samples),
        "prompt_tokens": _length_distribution(prompt_lengths),
        "completion_tokens": _length_distribution(completion_lengths),
        "total_tokens": _length_distribution(total_lengths),
    }
    for max_length in CANDIDATE_MAX_LENGTHS:
        statistics[f"over_{max_length}"] = sum(
            total_tokens > max_length for total_tokens in total_lengths
        )
    statistics["longest_samples"] = _longest_samples(audited_samples)
    return statistics


def _simulate_keep_start_truncation(
    audited_samples: list[dict],
    max_length: int,
) -> dict:
    """Simulate TRL keep_start truncation for one SFT split and length."""
    retained_supervised_tokens = []
    truncated_sample_count = 0
    prompt_truncated_count = 0
    completion_truncated_count = 0

    for sample in audited_samples:
        prompt_tokens = sample["prompt_tokens"]
        completion_tokens = sample["completion_tokens"]
        total_tokens = sample["total_tokens"]
        retained_tokens = max(0, min(total_tokens, max_length) - prompt_tokens)
        retained_tokens = min(retained_tokens, completion_tokens)
        retained_supervised_tokens.append(retained_tokens)

        truncated_sample_count += total_tokens > max_length
        prompt_truncated_count += prompt_tokens > max_length
        completion_truncated_count += retained_tokens < completion_tokens

    original_supervised_token_total = sum(
        sample["completion_tokens"] for sample in audited_samples
    )
    retained_supervised_token_total = sum(retained_supervised_tokens)
    if original_supervised_token_total <= 0:
        raise _audit_error(
            "truncation simulation",
            "an SFT split must contain at least one supervised token.",
        )

    return {
        "truncated_sample_count": truncated_sample_count,
        "prompt_truncated_count": prompt_truncated_count,
        "completion_truncated_count": completion_truncated_count,
        "zero_supervised_token_count": sum(
            retained_tokens == 0
            for retained_tokens in retained_supervised_tokens
        ),
        "retained_supervised_tokens": _retained_distribution(
            retained_supervised_tokens
        ),
        "retained_supervised_token_rate": round(
            retained_supervised_token_total / original_supervised_token_total,
            6,
        ),
    }


def _build_candidate_statistics(
    audited_by_split: dict[str, list[dict]],
) -> dict[str, dict[str, dict]]:
    """Build keep_start simulations for every candidate and SFT split."""
    return {
        str(max_length): {
            split_name: _simulate_keep_start_truncation(
                audited_by_split[split_name],
                max_length,
            )
            for split_name in SFT_SPLITS
        }
        for max_length in CANDIDATE_MAX_LENGTHS
    }


def _minimum_safe_candidate(
    candidate_statistics: dict[str, dict[str, dict]],
    metric_name: str,
) -> int | None:
    """Return the first candidate for which both SFT splits have zero risk."""
    for max_length in CANDIDATE_MAX_LENGTHS:
        if all(
            candidate_statistics[str(max_length)][split_name][metric_name] == 0
            for split_name in SFT_SPLITS
        ):
            return max_length
    return None


def _require_monotonic_distribution(distribution: object, context: str) -> None:
    """Validate the shape and ordering of one percentile distribution."""
    if not isinstance(distribution, dict):
        raise _audit_error(context, "token distribution must be an object.")
    keys = [*(f"p{percentile}" for percentile in PERCENTILES), "max"]
    values = []
    for key in keys:
        value = distribution.get(key)
        if type(value) is not int or value < 0:
            raise _audit_error(
                f"{context}, field={key!r}",
                f"expected a non-negative integer, got {value!r}.",
            )
        values.append(value)
    if values != sorted(values):
        raise _audit_error(
            context,
            f"percentiles and max must be non-decreasing, got {values!r}.",
        )


def _validate_report(
    report: object,
    audited_by_split: dict[str, list[dict]],
) -> dict:
    """Recompute key invariants before and after JSON publication."""
    if not isinstance(report, dict):
        raise _audit_error("report", "top level must be an object.")
    if report.get("enable_thinking") is not False:
        raise _audit_error("report", "enable_thinking must be false.")
    if report.get("truncation_mode") != "keep_start":
        raise _audit_error("report", "truncation_mode must be 'keep_start'.")

    split_statistics = report.get("splits")
    if not isinstance(split_statistics, dict) or set(split_statistics) != set(
        ALL_SPLITS
    ):
        raise _audit_error(
            "report, field='splits'",
            f"expected exactly {list(ALL_SPLITS)!r}.",
        )
    for split_name in ALL_SPLITS:
        expected_statistics = _build_split_statistics(
            audited_by_split[split_name]
        )
        actual_statistics = split_statistics[split_name]
        if actual_statistics != expected_statistics:
            raise _audit_error(
                f"report, split={split_name}",
                "statistics do not match the per-sample audit.",
            )
        for length_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            _require_monotonic_distribution(
                actual_statistics[length_name],
                f"report, split={split_name}, field={length_name!r}",
            )

    expected_candidates = _build_candidate_statistics(audited_by_split)
    if report.get("candidate_max_lengths") != expected_candidates:
        raise _audit_error(
            "report, field='candidate_max_lengths'",
            "truncation statistics do not match the per-sample simulation.",
        )
    expected_zero_safe = _minimum_safe_candidate(
        expected_candidates,
        "zero_supervised_token_count",
    )
    expected_no_truncation = _minimum_safe_candidate(
        expected_candidates,
        "truncated_sample_count",
    )
    if report.get("minimum_zero_supervision_safe_max_length") != expected_zero_safe:
        raise _audit_error(
            "report, field='minimum_zero_supervision_safe_max_length'",
            f"expected {expected_zero_safe!r}.",
        )
    if report.get("minimum_no_truncation_max_length") != expected_no_truncation:
        raise _audit_error(
            "report, field='minimum_no_truncation_max_length'",
            f"expected {expected_no_truncation!r}.",
        )
    return report


def _load_tokenizer(tokenizer_source: str | Path) -> tuple[Any, dict[str, str]]:
    """Load the tokenizer and return only non-sensitive report metadata."""
    if not isinstance(tokenizer_source, (str, Path)):
        raise TypeError(
            "tokenizer_source must be a string or Path, got "
            f"{tokenizer_source!r} ({type(tokenizer_source).__name__})."
        )
    source_text = str(tokenizer_source)
    if not source_text.strip():
        raise ValueError("tokenizer_source must not be empty.")

    source_path = Path(source_text).expanduser()
    explicit_local_path = isinstance(tokenizer_source, Path)
    is_local = explicit_local_path or source_path.exists()
    if is_local and not source_path.is_dir():
        raise FileNotFoundError(
            "The requested local tokenizer directory does not exist or is not "
            f"a directory: {source_path}"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "SFT token audit requires transformers; install the project's "
            "declared dependencies first."
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(source_path) if is_local else source_text,
            local_files_only=is_local,
        )
    except Exception as error:
        source_kind = "local" if is_local else "huggingface"
        raise RuntimeError(
            f"Could not load the {source_kind} Qwen3 tokenizer: "
            f"{type(error).__name__}: {error}"
        ) from error

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(
            "The loaded tokenizer must provide one non-empty string chat template."
        )
    metadata = {
        "tokenizer_source": "local" if is_local else "huggingface",
    }
    return tokenizer, metadata


def _publish_report_atomically(report: dict, output_path: Path) -> dict:
    """Stage, verify, publish, and if necessary restore one JSON report."""
    temporary_path = Path(f"{output_path}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_symlink() or (
        output_path.exists() and not output_path.is_file()
    ):
        raise ValueError(
            "SFT token audit output must be a regular, non-symlink file when "
            f"it already exists, got {output_path}."
        )

    try:
        temporary_path.unlink(missing_ok=True)
        save_json(report, temporary_path)
        staged_report = load_json(temporary_path)
        if staged_report != report:
            raise RuntimeError(
                "Staged SFT token audit report does not match in-memory data."
            )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    try:
        previous_content = output_path.read_bytes()
    except FileNotFoundError:
        previous_content = None
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    replaced = False
    try:
        temporary_path.replace(output_path)
        replaced = True
        persisted_report = load_json(output_path)
        if persisted_report != report:
            raise RuntimeError(
                "Published SFT token audit report does not match in-memory data."
            )
    except BaseException as publication_error:
        rollback_error: OSError | None = None
        if replaced:
            try:
                if previous_content is None:
                    output_path.unlink(missing_ok=True)
                else:
                    temporary_path.write_bytes(previous_content)
                    temporary_path.replace(output_path)
            except OSError as error:
                rollback_error = error

        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if rollback_error is None:
                rollback_error = cleanup_error

        if rollback_error is not None:
            raise RuntimeError(
                "SFT token audit publication failed and rollback was incomplete. "
                f"Publication error: {publication_error}. "
                f"Rollback error: {rollback_error}."
            ) from publication_error
        if not isinstance(publication_error, Exception):
            raise
        raise RuntimeError(
            "SFT token audit publication failed; the previous output was "
            f"restored. Publication error: {publication_error}."
        ) from publication_error

    return persisted_report


def audit_sft_tokens(
    tokenizer_source: str | Path = DEFAULT_MODEL_ID,
    output_path: str | Path = TOKEN_STATS_PATH,
) -> dict:
    """Audit all processed splits with Qwen3's non-thinking chat template."""
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            f"output_path must be a string or Path, got {output_path!r} "
            f"({type(output_path).__name__})."
        )
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path must not be empty.")

    tokenizer, tokenizer_metadata = _load_tokenizer(tokenizer_source)
    audited_by_split = {
        split_name: _load_and_audit_split(
            tokenizer,
            split_name,
            SPLIT_PATHS[split_name],
        )
        for split_name in ALL_SPLITS
    }
    candidate_statistics = _build_candidate_statistics(audited_by_split)
    report = {
        "model_id": DEFAULT_MODEL_ID,
        **tokenizer_metadata,
        "enable_thinking": False,
        "truncation_mode": "keep_start",
        "splits": {
            split_name: _build_split_statistics(audited_by_split[split_name])
            for split_name in ALL_SPLITS
        },
        "candidate_max_lengths": candidate_statistics,
        "minimum_zero_supervision_safe_max_length": _minimum_safe_candidate(
            candidate_statistics,
            "zero_supervised_token_count",
        ),
        "minimum_no_truncation_max_length": _minimum_safe_candidate(
            candidate_statistics,
            "truncated_sample_count",
        ),
    }
    _validate_report(report, audited_by_split)

    published_report = _publish_report_atomically(
        report,
        Path(output_path).expanduser().absolute(),
    )
    return _validate_report(published_report, audited_by_split)


def main() -> None:
    """Run the Qwen3 SFT token audit from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit processed SFT and Dev token lengths with Qwen3-8B's "
            "non-thinking chat template."
        )
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional local Qwen3-8B tokenizer directory. When omitted, "
            f"load {DEFAULT_MODEL_ID} from Hugging Face."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(TOKEN_STATS_PATH),
        help=(
            "JSON report path (default: "
            "pipelines/v1/artifacts/data/sft_token_stats.json)."
        ),
    )
    arguments = parser.parse_args()

    tokenizer_source: str | Path = (
        Path(arguments.tokenizer_path)
        if arguments.tokenizer_path is not None
        else DEFAULT_MODEL_ID
    )
    report = audit_sft_tokens(tokenizer_source, arguments.output_path)

    print("SFT token audit passed:")
    for split_name in ALL_SPLITS:
        split_statistics = report["splits"][split_name]
        print(
            f"{split_name}={split_statistics['sample_count']} samples, "
            f"max_total_tokens={split_statistics['total_tokens']['max']}"
        )
    print(
        "minimum_zero_supervision_safe_max_length="
        f"{report['minimum_zero_supervision_safe_max_length']}"
    )
    print(
        "minimum_no_truncation_max_length="
        f"{report['minimum_no_truncation_max_length']}"
    )
    print(f"output={Path(arguments.output_path).expanduser().absolute()}")


if __name__ == "__main__":
    main()
