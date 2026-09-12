import asyncio
from types import SimpleNamespace

import app.backend.query_service as service_module
from app.backend.query_service import QueryService
from app.backend.schemas import QueryRequest, QueryResponse


class FakeRegistry:
    def get_database(self, db_id):
        return {
            "db_id": db_id,
            "split": "dev",
            "sqlite_path": "/unused.sqlite",
            "schema_text": 'TABLE "items"\n- "id" INTEGER',
        }


class FakeModelClient:
    def __init__(self, *, valid=True):
        self.valid = valid
        self.greedy_calls = 0
        self.sampled_calls = 0

    def render_prompt(self, messages):
        assert messages
        return SimpleNamespace(text="rendered", input_tokens=123)

    async def generate_greedy(self, prompt):
        self.greedy_calls += 1
        raw = (
            "<think>Count rows.</think>\n```sql\nSELECT 1\n```"
            if self.valid
            else "I cannot answer this question."
        )
        return [
            {
                "raw_output": raw,
                "completion_tokens": 12,
                "mean_logprob": -0.1,
            }
        ]

    async def generate_sampled(self, prompt, count, seed):
        self.sampled_calls += 1
        assert count == 7
        assert seed == 42
        if not self.valid:
            return [
                {
                    "raw_output": "No SQL available.",
                    "completion_tokens": 4,
                    "mean_logprob": -1.0,
                }
                for _ in range(count)
            ]
        sqls = [
            "SELECT 1",
            "SELECT 1",
            "SELECT 2",
            "SELECT 3",
            "SELECT 4",
            "SELECT 5",
            "SELECT 6",
        ]
        return [
            {
                "raw_output": f"```sql\n{sql}\n```",
                "completion_tokens": 8,
                "mean_logprob": -0.2 - index / 100,
            }
            for index, sql in enumerate(sqls)
        ]


def _config(log_path):
    return {
        "app": {
            "max_concurrent_queries": 2,
            "request_log_path": str(log_path),
        },
        "generation": {
            "accurate": {"sampled_candidates": 7, "seed": 42},
        },
        "execution": {
            "timeout_seconds": 5.0,
            "vote_max_rows": 10_000,
            "display_max_rows": 200,
            "minimum_vote_support": 2,
        },
    }


def _install_fakes(monkeypatch):
    monkeypatch.setattr(service_module, "retrieve_for_query", lambda *args: [])
    monkeypatch.setattr(
        service_module,
        "build_v2_messages",
        lambda *args: [{"role": "user", "content": "question"}],
    )

    def execute_candidate(entry, sql, config):
        value = int(sql.rsplit(" ", 1)[-1])
        return {
            "status": "success",
            "rows": [(value,)],
            "row_count": 1,
            "column_count": 1,
            "empty_result": False,
            "elapsed_ms": 1.0,
            "error_type": None,
            "error_message": None,
            "accessed_tables": [],
            "accessed_columns": [],
        }

    monkeypatch.setattr(service_module, "execute_candidate", execute_candidate)
    monkeypatch.setattr(
        service_module,
        "execute_display",
        lambda *args, **kwargs: {
            "status": "success",
            "columns": ["answer"],
            "rows": [[1]],
            "row_count": 1,
            "displayed_row_count": 1,
            "result_truncated": False,
            "elapsed_ms": 1.0,
            "error_type": None,
            "error_message": None,
        },
    )


def test_fast_query_returns_one_candidate(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    client = FakeModelClient()
    service = QueryService(
        FakeRegistry(), client, _config(tmp_path / "requests.jsonl")
    )
    response = asyncio.run(
        service.query(QueryRequest(db_id="demo", question="Count items"))
    )

    assert response["status"] == "success"
    assert len(response["candidates"]) == 1
    assert client.greedy_calls == 1
    assert client.sampled_calls == 0
    assert response["vote"] is None
    assert response["execution"]["columns"] == ["answer"]
    QueryResponse.model_validate(response)


def test_accurate_query_votes_over_eight_candidates(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    client = FakeModelClient()
    service = QueryService(
        FakeRegistry(), client, _config(tmp_path / "requests.jsonl")
    )
    response = asyncio.run(
        service.query(
            QueryRequest(db_id="demo", question="Count items", mode="accurate")
        )
    )

    assert len(response["candidates"]) == 8
    assert response["candidates"][0]["kind"] == "greedy"
    assert response["vote"]["chosen_index"] == 0
    assert response["vote"]["support"] == 3
    assert response["selected_candidate"]["selected"] is True
    assert response["selected_candidate"]["cluster_support"] == 3
    assert all(value >= 0 for value in response["timings"].values())
    QueryResponse.model_validate(response)


def test_accurate_query_reports_no_executable_candidate(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    client = FakeModelClient(valid=False)
    service = QueryService(
        FakeRegistry(), client, _config(tmp_path / "requests.jsonl")
    )
    response = asyncio.run(
        service.query(
            QueryRequest(db_id="demo", question="Count items", mode="accurate")
        )
    )

    assert response["status"] == "no_executable_candidate"
    assert response["selected_candidate"] is None
    assert response["execution"] is None
    assert response["vote"]["executable_count"] == 0
    assert len(response["candidates"]) == 8
    QueryResponse.model_validate(response)
