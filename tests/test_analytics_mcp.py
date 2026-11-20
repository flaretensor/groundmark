from uuid import uuid4

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from groundmark.analytics import append_result, report
from groundmark.mcp_server import build_server


def test_parquet_append_is_idempotent_and_report_has_real_aggregates(tmp_path, outcome):
    assert report(tmp_path)["run_count"] == 0
    first = outcome.model_copy(deep=True)
    first.trace.total_latency_ms = 100
    second = outcome.model_copy(deep=True)
    second.trace.run_id = str(uuid4())
    second.trace.total_latency_ms = 300
    second.trace.tavily_credits = 2
    second.claims[0].status = "not_supported"
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
        {"status": "not_supported", "count": 1},
        {"status": "supported", "count": 1},
    ]
    assert len(summary["per_run"]) == 2
    assert not list(tmp_path.glob(".pending-*"))


def test_mcp_in_memory_tool_listing_and_invocation(outcome):
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
            assert content["sources"][0]["url"] == "https://example.org/"
            assert content["cost"]["tavily_credits"] == 1

    anyio.run(call)
