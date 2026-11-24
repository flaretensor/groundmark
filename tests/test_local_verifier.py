from types import SimpleNamespace

from groundmark.local_verifier import LocalNLIVerifier
from groundmark.models import Claim, Source
from groundmark.pricing import JUDGE_MODEL, MODEL, PRICES


def test_quote_comes_from_highest_scoring_exact_span(settings):
    class MockScorer:
        def score(self, passages, claims):
            return [
                [0.9, 0.05, 0.05] if "The plant opened in 2024." in text else [0.4, 0.5, 0.1]
                for text in passages
            ], []

    text = "Irrelevant " * 34 + "The plant opened in 2024."
    source = Source(
        source_id="S1",
        url="https://example.org/",
        title="Example",
        content=text,
        retrieved_at="2026-11-24T00:00:00+00:00",
        query="plant opening",
        round_index=1,
    )
    verifier = LocalNLIVerifier(SimpleNamespace(settings=settings), MockScorer())
    claims = [Claim(claim_id="C1", text="The plant opened in 2024.", source_ids=["S1"])]
    (pair,) = verifier.verify(claims, [source], "", "")
    assert pair.verdict == "supports"
    assert pair.quote_match == "exact"
    assert len(pair.evidence_quote) <= 300
    assert pair.evidence_quote in text
    assert pair.evidence_quote.endswith("The plant opened in 2024.")


def test_model_registry_prices(settings):
    assert settings.model == MODEL
    assert settings.resolved_checker_model == MODEL
    assert settings.judge_model == JUDGE_MODEL
    assert PRICES[JUDGE_MODEL].cost(1000, 500, 1000) == 450_000
