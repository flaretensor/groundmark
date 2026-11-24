from types import SimpleNamespace

import pytest

from groundmark.models import Plan
from groundmark.pricing import MODEL
from groundmark.providers import CompletionRequest, NebiusClient


@pytest.mark.parametrize("thinking", [False, True])
def test_thinking_and_strict_schema_payload(thinking):
    def create(**payload):
        options = payload["extra_body"]["chat_template_kwargs"]
        assert options["enable_thinking"] == thinking
        assert options.get("medium_effort", False) == thinking
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["max_completion_tokens"] == (8192 if thinking else 1024)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        )

    client = NebiusClient("mock-nebius")
    try:
        client.client.chat.completions.create = create
        result = client.complete(
            CompletionRequest(
                MODEL,
                "System",
                "Prompt",
                Plan,
                thinking,
                8192 if thinking else 1024,
                0.1,
                90,
            )
        )
        assert result.finish_reason == "stop"
    finally:
        client.close()


def test_tavily_search_passes_exclusions_only_when_configured():
    from groundmark.providers import TavilySearchClient

    requests = []
    client = TavilySearchClient("mock-tavily")

    def search(query, **kwargs):
        requests.append((query, kwargs))
        return {"results": [], "usage": {"credits": 1}}

    try:
        client.client.search = search
        assert client.search("query", 1).credits == 1
        assert (
            client.search("query", 1, exclude_domains=("huggingface.co", "kaggle.com")).credits == 1
        )
    finally:
        client.close()
    assert "exclude_domains" not in requests[0][1]
    assert requests[1][1]["exclude_domains"] == ["huggingface.co", "kaggle.com"]
