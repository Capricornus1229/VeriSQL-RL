"""Fast and Accurate Text-to-SQL request orchestration."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .display_executor import execute_display
from .request_log import write_request
from .response_parser import extract_reasoning_sql
from .schemas import QueryRequest
from .v2_bridge import (
    build_v2_messages,
    execute_candidate,
    retrieve_for_query,
    select_accurate_candidate,
    summarize_execution,
)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _not_executed_result() -> dict[str, Any]:
    return {
        "status": "not_executed",
        "rows": None,
        "row_count": 0,
        "column_count": 0,
        "empty_result": False,
        "elapsed_ms": 0.0,
        "error_type": "no_sql_extracted",
        "error_message": "No supported SQL response was extracted.",
        "accessed_tables": [],
        "accessed_columns": [],
    }


class QueryService:
    def __init__(self, registry: Any, model_client: Any, config: dict[str, Any]):
        self.registry = registry
        self.model_client = model_client
        self.config = config
        self.semaphore = asyncio.Semaphore(
            config["app"]["max_concurrent_queries"]
        )

    @staticmethod
    def _parse_candidate(
        generated: dict[str, Any],
        *,
        index: int,
        kind: str,
    ) -> dict[str, Any]:
        return {
            **generated,
            **extract_reasoning_sql(generated["raw_output"]),
            "index": index,
            "kind": kind,
        }

    async def _execute_candidates(
        self,
        registry_entry: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> None:
        async def execute_one(candidate: dict[str, Any]) -> None:
            if not candidate["predicted_sql"]:
                candidate["execution"] = _not_executed_result()
                return
            candidate["execution"] = await asyncio.to_thread(
                execute_candidate,
                registry_entry,
                candidate["predicted_sql"],
                self.config,
            )

        await asyncio.gather(*(execute_one(candidate) for candidate in candidates))

    async def _run_fast(self, prompt: Any) -> list[dict[str, Any]]:
        generated = await self.model_client.generate_greedy(prompt)
        return [
            self._parse_candidate(generated[0], index=0, kind="greedy")
        ]

    async def _run_accurate(self, prompt: Any) -> list[dict[str, Any]]:
        accurate = self.config["generation"]["accurate"]
        greedy, sampled = await asyncio.gather(
            self.model_client.generate_greedy(prompt),
            self.model_client.generate_sampled(
                prompt,
                accurate["sampled_candidates"],
                accurate["seed"],
            ),
        )
        candidates = [
            self._parse_candidate(greedy[0], index=0, kind="greedy")
        ]
        candidates.extend(
            self._parse_candidate(item, index=index, kind="sampled")
            for index, item in enumerate(sampled, start=1)
        )
        return candidates

    @staticmethod
    def _cluster_supports(candidates: list[dict[str, Any]]) -> dict[int, int]:
        """Compute display-only support counts using the frozen Vote row semantics."""
        successful = [
            candidate
            for candidate in candidates
            if candidate["execution"]["status"] == "success"
        ]
        nonempty = [
            candidate for candidate in successful if candidate["execution"]["rows"]
        ]
        voting_candidates = nonempty if nonempty else successful
        clusters: dict[frozenset[tuple[Any, ...]], list[int]] = {}
        for candidate in voting_candidates:
            result_key = frozenset(
                tuple(row) for row in candidate["execution"]["rows"]
            )
            clusters.setdefault(result_key, []).append(candidate["index"])
        return {
            index: len(indices)
            for indices in clusters.values()
            for index in indices
        }

    @staticmethod
    def _candidate_view(
        candidate: dict[str, Any],
        *,
        selected_index: int | None,
        supports: dict[int, int],
    ) -> dict[str, Any]:
        execution = candidate["execution"]
        return {
            "index": candidate["index"],
            "kind": candidate["kind"],
            "reasoning": candidate["reasoning"],
            "sql": candidate["predicted_sql"],
            "raw_output": candidate["raw_output"],
            "extraction_status": candidate["extraction_status"],
            "format_compliance": candidate["format_compliance"],
            "completion_tokens": candidate["completion_tokens"],
            "mean_logprob": candidate["mean_logprob"],
            "execution_status": execution["status"],
            "row_count": execution["row_count"],
            "column_count": execution["column_count"],
            "empty_result": execution["empty_result"],
            "execution_elapsed_ms": execution["elapsed_ms"],
            "error_type": execution["error_type"],
            "selected": candidate["index"] == selected_index,
            "cluster_support": supports.get(candidate["index"], 0),
        }

    @staticmethod
    def _failed_execution_view(execution: dict[str, Any]) -> dict[str, Any]:
        summary = summarize_execution(execution)
        error_message = summary["error_message"]
        if summary["error_type"] in {"database_not_found", "database_open_error"}:
            error_message = "The registered SQLite database is unavailable."
        return {
            "status": summary["status"],
            "columns": [],
            "rows": None,
            "row_count": summary["row_count"],
            "displayed_row_count": 0,
            "result_truncated": False,
            "elapsed_ms": summary["elapsed_ms"],
            "error_type": summary["error_type"],
            "error_message": error_message,
            "accessed_tables": summary["accessed_tables"],
            "accessed_columns": summary["accessed_columns"],
        }

    async def query(self, request: QueryRequest) -> dict[str, Any]:
        async with self.semaphore:
            return await self._query(request)

    async def _query(self, request: QueryRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        request_id = str(uuid.uuid4())
        registry_entry = self.registry.get_database(request.db_id)

        stage_started = time.perf_counter()
        grounding = await asyncio.to_thread(
            retrieve_for_query,
            registry_entry,
            request.question,
            request.evidence,
            self.config,
        )
        grounding_ms = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        messages = build_v2_messages(
            registry_entry,
            request.question,
            request.evidence,
            grounding,
        )
        rendered_prompt = self.model_client.render_prompt(messages)
        prompt_ms = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        if request.mode == "fast":
            candidates = await self._run_fast(rendered_prompt)
        else:
            candidates = await self._run_accurate(rendered_prompt)
        generation_ms = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        await self._execute_candidates(registry_entry, candidates)
        candidate_execution_ms = _elapsed_ms(stage_started)

        vote_result: dict[str, Any] | None = None
        stage_started = time.perf_counter()
        if request.mode == "accurate":
            vote_result = select_accurate_candidate(candidates, self.config)
            selected_index = vote_result["chosen_index"]
        else:
            selected_index = 0
        vote_ms = _elapsed_ms(stage_started) if vote_result is not None else 0.0

        chosen = (
            candidates[selected_index]
            if selected_index is not None
            else None
        )
        display_execution_ms = 0.0
        execution_view: dict[str, Any] | None = None
        if chosen is not None and chosen["execution"]["status"] == "success":
            stage_started = time.perf_counter()
            display = await asyncio.to_thread(
                execute_display,
                registry_entry["sqlite_path"],
                chosen["predicted_sql"],
                timeout_seconds=self.config["execution"]["timeout_seconds"],
                max_rows=self.config["execution"]["display_max_rows"],
                total_row_count=chosen["execution"]["row_count"],
            )
            display_execution_ms = _elapsed_ms(stage_started)
            display["accessed_tables"] = chosen["execution"]["accessed_tables"]
            display["accessed_columns"] = chosen["execution"]["accessed_columns"]
            execution_view = display
        elif chosen is not None:
            execution_view = self._failed_execution_view(chosen["execution"])

        if request.mode == "accurate" and chosen is None:
            status = "no_executable_candidate"
        elif chosen is not None and not chosen["predicted_sql"]:
            status = "no_sql_extracted"
        elif execution_view is None or execution_view["status"] != "success":
            status = "sql_execution_failed"
        else:
            status = "success"

        supports = self._cluster_supports(candidates)
        candidate_views = [
            self._candidate_view(
                candidate,
                selected_index=selected_index,
                supports=supports,
            )
            for candidate in candidates
        ]
        selected_candidate = (
            candidate_views[selected_index]
            if selected_index is not None
            else None
        )
        vote_view = (
            {
                "candidate_count": len(candidates),
                "executable_count": vote_result["executable_count"],
                "chosen_index": vote_result["chosen_index"],
                "support": vote_result["support"],
                "cluster_count": vote_result["cluster_count"],
                "used_vote": vote_result["used_vote"],
                "reason": vote_result["reason"],
            }
            if vote_result is not None
            else None
        )

        total_ms = _elapsed_ms(started_at)
        timings = {
            "grounding_ms": grounding_ms,
            "prompt_ms": prompt_ms,
            "generation_ms": generation_ms,
            "candidate_execution_ms": candidate_execution_ms,
            "vote_ms": vote_ms,
            "display_execution_ms": display_execution_ms,
            "total_ms": total_ms,
        }
        response = {
            "request_id": request_id,
            "status": status,
            "db_id": request.db_id,
            "split": registry_entry["split"],
            "mode": request.mode,
            "question": request.question,
            "grounding": grounding,
            "candidates": candidate_views,
            "selected_candidate": selected_candidate,
            "execution": execution_view,
            "vote": vote_view,
            "timings": timings,
        }

        executable_count = sum(
            candidate["execution"]["status"] == "success"
            for candidate in candidates
        )
        error_type = (
            execution_view.get("error_type")
            if execution_view is not None
            else status if status != "success" else None
        )
        write_request(
            self.config["app"]["request_log_path"],
            {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "db_id": request.db_id,
                "mode": request.mode,
                "status": status,
                "input_tokens": rendered_prompt.input_tokens,
                "output_tokens": sum(
                    candidate["completion_tokens"] or 0
                    for candidate in candidates
                ),
                "candidate_count": len(candidates),
                "executable_count": executable_count,
                "total_ms": total_ms,
                "generation_ms": generation_ms,
                "execution_ms": round(
                    candidate_execution_ms + display_execution_ms,
                    3,
                ),
                "error_type": error_type,
            },
        )
        return response
