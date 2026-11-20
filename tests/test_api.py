import json
from threading import Event, Thread

import httpx
from fastapi.testclient import TestClient

from groundmark.api import create_app
from groundmark.client import HttpClient, RunError, parse_sse
from groundmark.models import Progress, ResearchResult


class MockRunner:
    def __init__(self, outcome):
        self.outcome = outcome

    def run(self, question, options, progress=None):
        if progress:
            progress(Progress(node="plan", round=0))
        return self.outcome


def test_api_sync_sse_and_validation(outcome):
    with TestClient(create_app(MockRunner(outcome))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        invalid = "private-" * 80
        response = client.post("/research", json={"question": invalid})
        assert response.status_code == 422
        assert invalid not in response.text
        response = client.post("/research", json={"question": "When?"})
        assert ResearchResult.model_validate(response.json()) == outcome
        with client.stream("POST", "/research/stream", json={"question": "When?"}) as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            events = list(parse_sse(stream.iter_lines()))
        assert [name for name, _ in events] == ["progress", "result"]
        assert json.loads(events[0][1]) == {"node": "plan", "round": 0}
        assert ResearchResult.model_validate_json(events[1][1]) == outcome


def test_active_slot_blocks_both_paths_and_recovers(outcome):
    started = Event()
    release = Event()

    class Runner(MockRunner):
        def run(self, question, options, progress=None):
            started.set()
            assert release.wait(5)
            return super().run(question, options, progress)

    with TestClient(create_app(Runner(outcome))) as client:
        responses = []
        thread = Thread(
            target=lambda: responses.append(client.post("/research", json={"question": "First"}))
        )
        thread.start()
        try:
            assert started.wait(3)
            for path in ("/research", "/research/stream"):
                second = client.post(path, json={"question": "Second"})
                assert second.status_code == 503
                assert second.json()["detail"]["kind"] == "busy"
        finally:
            release.set()
            thread.join(6)
        assert responses[0].status_code == 200
        assert client.post("/research/stream", json={"question": "Third"}).status_code == 200


def test_http_client_parses_sse_and_surfaces_busy(outcome):
    def handle(request: httpx.Request):
        assert request.url.path == "/research/stream"
        assert json.loads(request.content)["question"] == "When?"
        return httpx.Response(
            200,
            text=(
                'event: progress\ndata: {"node":"search","round":1}\n\n'
                f"event: result\ndata: {outcome.model_dump_json()}\n\n"
            ),
        )

    client = HttpClient("http://test", transport=httpx.MockTransport(handle))
    assert list(client.stream("When?")) == [Progress(node="search", round=1), outcome]

    busy = HttpClient("http://test", transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    assert list(busy.stream("When?")) == [RunError("busy")]


def test_metrics_cover_sse_results_without_question_labels(outcome):
    with TestClient(create_app(MockRunner(outcome))) as client:
        response = client.post("/research/stream", json={"question": "private demo question"})
        assert response.status_code == 200
        metrics = client.get("/metrics").text
    assert "groundmark_runs_total{" in metrics and 'status="complete"' in metrics
    assert "groundmark_run_latency_seconds_bucket{" in metrics
    assert "groundmark_tavily_credits_total" in metrics
    assert "groundmark_nebius_estimated_usd_total" in metrics
    assert "groundmark_claim_statuses_total" in metrics
    assert "private demo question" not in metrics
