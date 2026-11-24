from uuid import uuid4

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from groundmark.analytics import append_result, report
from groundmark.mcp_server import build_server
from groundmark.models import ConflictResult


def test_parquet_append_is_idempotent_and_report_has_real_aggregates(tmp_path, outcome):
    assert report(tmp_path)["run_count"] == 0
    first = outcome.model_copy(deep=True)
    first.trace.total_latency_ms = 100
    second = outcome.model_copy(deep=True)
    second.trace.run_id = str(uuid4())
    second.trace.total_latency_ms = 300
    second.trace.tavily_credits = 2
    second.claims[0].status = "not_enough_evidence"
    second.claims[0].pairs[0].quote_match = "missing"
    second.insufficient_evidence = True
    append_result(tmp_path, first)
    append_result(tmp_path, first)
    append_result(tmp_path, second)
    summary = report(tmp_path)
    assert summary["run_count"] == 2
    assert summary["latency_p50_ms"] == 200
    assert summary["latency_p95_ms"] == pytest.approx(290)
    assert summary["tavily_credits"] == 3
    assert summary["quote_verification_rate"] == 0.5
    assert summary["claim_statuses"] == [
        {"status": "not_enough_evidence", "count": 1},
        {"status": "supported", "count": 1},
    ]
    assert len(summary["per_run"]) == 2
    assert not list(tmp_path.glob(".pending-*"))


def test_mcp_in_memory_tool_listing_and_invocation(outcome):
    outcome.conflicts = [
        ConflictResult(
            topic="Opening date",
            description="The reported schedule differs from the announcement.",
            source_ids=["S1"],
            origin="synthesis",
        )
    ]
    outcome.missing_information = ["The cost of the plant."]

    class Runner:
        def run(self, question, options):
            assert question == "When?"
            assert options.max_rounds == 1
            return outcome

    async def call():
        server = build_server(Runner())
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["research"]
            result = await client.call_tool("research", {"question": "When?", "max_rounds": 1})
            assert not result.isError
            content = result.structuredContent
            assert content["summary"] == outcome.summary
            assert content["claims"][0]["quotes"][0]["source_id"] == "S1"
            assert content["conflicts"] == [outcome.conflicts[0].model_dump(mode="json")]
            assert content["missing_information"] == outcome.missing_information
            assert content["sources"][0]["url"] == "https://example.org/"
            assert content["cost"]["tavily_credits"] == 1

    anyio.run(call)


def test_eval_grade_records_and_receipts_are_private(settings, outcome):
    import hashlib
    import json
    import stat

    import pyarrow.parquet as pq

    path = settings.artifacts_dir / "runs" / outcome.trace.run_id
    (run,) = pq.read_table(path / "run.parquet").to_pylist()
    (claim,) = pq.read_table(path / "claims.parquet").to_pylist()
    assert run["question"] == "When?"
    assert run["question_id"] is None and run["arm"] is None
    assert len(run["config_sha256"]) == 64
    assert json.loads(run["models"])["agent"] == settings.model
    assert (
        run["sources"][0]["content_sha256"]
        == hashlib.sha256(outcome.sources[0].content.encode()).hexdigest()
    )
    assert json.loads(run["action_trajectory"])[0]["query"] == "plant opening"
    assert claim["claim_text"] == outcome.claims[0].claim.text
    assert claim["pair_verdicts"][0]["evidence_quote"] == "opened in 2024"
    receipts = json.loads((path / run["raw_tavily_artifact"]).read_bytes())
    assert receipts[0]["operation"] == "search"
    assert receipts[0]["response"]["results"][0]["url"] == outcome.sources[0].url
    assert {stat.S_IMODE(file.stat().st_mode) for file in path.iterdir()} == {0o600}
