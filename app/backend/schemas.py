"""Stable HTTP request and response models for VeriSQL Studio."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_id: str
    question: str
    evidence: str | None = None
    mode: Literal["fast", "accurate"] = "fast"

    @field_validator("db_id", "question")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("evidence")
    @classmethod
    def trim_optional_evidence(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class GroundingHit(BaseModel):
    table: str
    column: str
    description: str
    value_description: str
    data_format: str
    matched_values: list[Any] = Field(default_factory=list)
    line: str


class SelectedCandidate(BaseModel):
    index: int
    kind: Literal["greedy", "sampled"]
    reasoning: str
    sql: str
    raw_output: str
    extraction_status: str
    format_compliance: bool
    completion_tokens: int | None
    mean_logprob: float | None
    execution_status: str
    row_count: int
    column_count: int
    empty_result: bool
    execution_elapsed_ms: float
    error_type: str | None
    selected: bool = False
    cluster_support: int = 0


class ExecutionView(BaseModel):
    status: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] | None = None
    row_count: int = 0
    displayed_row_count: int = 0
    result_truncated: bool = False
    elapsed_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    accessed_tables: list[str] = Field(default_factory=list)
    accessed_columns: list[dict[str, str]] = Field(default_factory=list)


class VoteInfo(BaseModel):
    candidate_count: int
    executable_count: int
    chosen_index: int | None
    support: int
    cluster_count: int
    used_vote: bool
    reason: str


class TimingInfo(BaseModel):
    grounding_ms: float
    prompt_ms: float
    generation_ms: float
    candidate_execution_ms: float
    vote_ms: float
    display_execution_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    request_id: str
    status: str
    db_id: str
    split: Literal["train", "dev"]
    mode: Literal["fast", "accurate"]
    question: str
    grounding: list[GroundingHit]
    candidates: list[SelectedCandidate]
    selected_candidate: SelectedCandidate | None
    execution: ExecutionView | None
    vote: VoteInfo | None
    timings: TimingInfo
