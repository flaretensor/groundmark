import json

import pytest

from groundmark.config import Credentials, Settings
from groundmark.models import TokenUsage
from groundmark.providers import Completion, ExtractionResponse, SearchResponse
from groundmark.research import ResearchRunner

Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    for name in ("NEBIUS_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setattr(Credentials, "load", lambda **_: Credentials("", ""))


def snippet(
    title="Example source", content="The plant opened in 2024.", url="https://example.org/"
):
    return {"url": url, "title": title, "content": content, "score": 0.8}


def decision(action="search", query="", source_ids=(), rationale=""):
    return {
        "action": action,
        "query": query,
        "source_ids": list(source_ids),
        "rationale": rationale,
    }


class ScriptedLLM:
    def __init__(self, **responses):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        response = self.responses[request.schema.__name__].pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, Completion):
            return response
        return Completion(json.dumps(response), "stop", TokenUsage(input=40, output=8, reasoning=2))


class ScriptedSearch:
    def __init__(self, *responses, extraction=None):
        self.responses = list(responses)
        self.requests = []
        self.extraction = extraction or ExtractionResponse([], [])

    def search(self, query, timeout):
        self.requests.append(query)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SearchResponse(value, 1)

    def extract(self, urls, timeout):
        return self.extraction


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        checkpoint_url=str(tmp_path / "checkpoints.sqlite"),
        cache_dir=tmp_path / "cache",
        artifacts_dir=tmp_path / "artifacts",
        otlp_endpoint="",
        mlflow_tracking_uri="",
    )


@pytest.fixture
def outcome(settings):
    llm = ScriptedLLM(
        Action=[decision(query="plant opening"), decision("finish")],
        Synthesis=[
            {
                "summary": "The plant opened in 2024 [S1].",
                "claims": [
                    {"claim_id": "C1", "text": "The plant opened in 2024.", "source_ids": ["S1"]}
                ],
                "insufficient_evidence": False,
                "missing_information": [],
                "conflicts": [],
            }
        ],
        PairJudgments=[
            {
                "pairs": [
                    {
                        "claim_id": "C1",
                        "verdict": "supports",
                        "evidence_quote": "opened in 2024",
                        "reason": "The source gives the year.",
                    }
                ]
            }
        ],
    )
    return ResearchRunner(settings, llm=llm, search=ScriptedSearch([snippet()])).run("When?")
