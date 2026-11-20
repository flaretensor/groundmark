import json
from functools import partial

from conftest import ScriptedLLM, ScriptedSearch, snippet

from groundmark.evidence import chunk_page, extract_evidence
from groundmark.models import TokenUsage
from groundmark.providers import EmbeddingResponse, ExtractionResponse
from groundmark.research import Options, ResearchRunner


def test_source_exact_chunks_with_offsets():
    text = "a" * 1200 + "β" * 1200
    chunks = chunk_page(text)
    assert [(item.start, item.end) for item in chunks] == [(0, 1200), (1000, 2200), (2000, 2400)]
    assert all(item.text == text[item.start : item.end] for item in chunks)


def test_extract_retries_verification_with_claim_specific_ranked_windows(settings):
    content = "A plant was planned." + "x" * 1800 + " The plant opened in 2024."
    beyond_cap = "Text past the page limit."
    content = content.ljust(200_000, "x") + beyond_cap
    llm = ScriptedLLM(
        Plan=[{"queries": ["plant"], "rationale": "Find the date."}],
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
                        "verdict": "not_supported",
                        "reason": "No date.",
                        "evidence_quote": "",
                    }
                ]
            },
            {
                "pairs": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "Gives the date.",
                        "evidence_quote": "The plant opened in 2024.",
                    }
                ]
            },
        ],
    )
    search = ScriptedSearch(
        [snippet(content="A plant was planned.")],
        extraction=ExtractionResponse(
            [{"url": "https://example.org/", "raw_content": content}], [], 1
        ),
    )

    class Embedder:
        def embed(self, texts, timeout):
            assert all(beyond_cap not in text for text in texts)
            return EmbeddingResponse(
                [[1, float("2024" in text)] for text in texts], TokenUsage(input=70)
            )

    class Reranker:
        def score(self, claim, chunks):
            assert claim == "The plant opened in 2024."
            return [float("2024" in chunk.text) for chunk in chunks]

    result = ResearchRunner(
        settings,
        llm=llm,
        search=search,
        embedder=Embedder(),
        extract_node=partial(extract_evidence, reranker=Reranker()),
    ).run("When?", Options(max_rounds=1))
    assert result.claims[0].status == "supported"
    assert result.claims[0].pairs[0].evidence_scope == "page"
    assert result.trace.extracted_source_ids == ["S1"]
    assert [(pair.evidence_scope, pair.verdict) for pair in result.trace.extraction_attempts] == [
        ("snippet", "not_supported"),
        ("page", "supported"),
    ]
    assert len(result.trace.extraction_attempts[1].evidence_preview) <= 600
    assert content not in result.model_dump_json()
    (snapshot,) = (settings.cache_dir / "extract").glob("*.json")
    assert json.loads(snapshot.read_text())["text"] == content[:200_000]
    assert "S1: page truncated to 200000 characters" in result.trace.notes
    assert result.trace.tavily_credits == 2
    date_line = llm.requests[0].prompt.splitlines()[0]
    assert llm.requests[-1].prompt.startswith(f"{date_line}\nQuestion: When?\nClaims:\n")
