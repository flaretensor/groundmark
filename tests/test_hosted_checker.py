import pytest

from groundmark.budget import Amount, BudgetRefused, SpendLedger
from groundmark.config import Credentials
from groundmark.eval.hosted_checker import LIMIT, STAGE, collect, evaluate_pair
from groundmark.models import TokenUsage
from groundmark.providers import Completion, ProviderFailure, TavilySearchClient


class FakeLLM:
    def __init__(self, *, fail_once=False, output_tokens=3000):
        self.fail_once = fail_once
        self.output_tokens = output_tokens
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.fail_once:
            self.fail_once = False
            raise ProviderFailure.http("nebius", 429)
        return Completion(
            '{"pairs":[{"claim_id":"C1","verdict":"supports",'
            '"evidence_quote":"invented quote","reason":"fixture"}]}',
            "stop",
            TokenUsage(input=400, output=self.output_tokens, reasoning=self.output_tokens - 20),
        )


def row(index):
    return {
        "id": str(index),
        "group": str(index),
        "claim": "Fixture",
        "claim_date": "2020-01-02",
        "evidence": "Source text",
    }


def ledger_for(directory, limit=2000):
    return SpendLedger(
        directory / "spend.jsonl", LIMIT, scope=STAGE, scope_limit=LIMIT, scope_call_limit=limit
    )


def test_scope_budget_persists_dollar_and_call_limits(tmp_path):
    ledger = ledger_for(tmp_path, limit=1)
    reservation = ledger.reserve("row", "completion", Amount(nanos=1), LIMIT)
    ledger.settle("row", reservation, Amount())
    reloaded = ledger_for(tmp_path, limit=1)
    assert reloaded.scope_run_ids() == {"row"}
    with pytest.raises(BudgetRefused) as blocked:
        reloaded.reserve("row2", "completion", Amount(nanos=1), LIMIT)
    assert blocked.value.kind == "calls"
    with pytest.raises(BudgetRefused) as blocked:
        ledger_for(tmp_path / "new").reserve(
            "row", "completion", Amount(nanos=LIMIT.nanos + 1), LIMIT
        )
    assert blocked.value.kind == "usd"


def test_hosted_pair_caches_retry_and_raw_and_quote_gated_verdict(tmp_path, monkeypatch, settings):
    monkeypatch.setattr("groundmark.runtime.sleep", lambda _: None)
    provider = FakeLLM(fail_once=True)
    cache = tmp_path / "pair.json"
    ledger = ledger_for(tmp_path)
    credentials = Credentials("", "")
    result = evaluate_pair(row(0), cache, "row", provider, credentials, settings, ledger)
    assert result["pair"]["raw_verdict"] == "supports"
    assert result["pair"]["verdict"] == "not_enough_info"
    assert result["attempts"][0]["http_status"] == 429
    assert result["attempts"][1]["usage"]["reasoning"] == 2980
    assert ledger.scope_reservations() == 2
    assert evaluate_pair(row(0), cache, "row", provider, credentials, settings, ledger) == result
    assert cache.stat().st_mode & 0o777 == 0o600
    assert len(provider.requests) == 2
    assert provider.requests[0].prompt.startswith("Current date: 2020-01-02\nQuestion: \n")


def test_hosted_preflight_stops_on_projected_cost_and_reuses_cache(tmp_path, settings):
    rows = [row(index) for index in range(500)]
    directory = tmp_path / "receipt"
    ledger = ledger_for(tmp_path)
    provider = FakeLLM()
    credentials = Credentials("", "")
    records, preflight, stop = collect(rows, directory, ledger, provider, credentials, settings)
    assert len(records) == 20
    assert preflight["projected_nanos"] > LIMIT.nanos
    assert stop == "preflight_projected_over_budget"
    assert len(provider.requests) == 20
    assert len(list((directory / "pairs").glob("*.json"))) == 20
    replay = collect(rows, directory, ledger_for(tmp_path), provider, credentials, settings)
    assert replay == (records, preflight, stop)
    assert len(provider.requests) == 20


def test_hosted_tavily_guard_rejects_construction(monkeypatch):
    monkeypatch.setenv("GROUNDMARK_DISABLE_TAVILY", "1")
    with pytest.raises(RuntimeError, match="disabled"):
        TavilySearchClient("")


def test_hosted_reservation_without_cache_prevents_duplicate_charge(tmp_path, settings):
    from groundmark.eval.hosted_checker import digest

    rows = [row(0)]
    directory = tmp_path / "receipt"
    ledger = ledger_for(tmp_path)
    ledger.reserve(f"{directory.name}:{digest('0')}", "completion", Amount(nanos=1), LIMIT)
    provider = FakeLLM()
    records, preflight, stop = collect(
        rows, directory, ledger, provider, Credentials("", ""), settings
    )
    assert records == [{"state": "pending", "row_id": "0"}]
    assert preflight == {}
    assert stop == "previous_attempt_unconfirmed"
    assert provider.requests == []


def test_hosted_analysis_matches_on_observed_pairs_and_checks_complete_groups():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from groundmark.eval.checker_analysis import hosted_analysis

    local = [
        {
            "row_id": f"{group}-{index}",
            "group": str(group),
            "native_gold": gold,
            "is_gold": False,
            "support_score": score,
        }
        for group in range(5)
        for index, (gold, score) in enumerate((("supports", 0.9), ("refutes", 0.1)))
    ]
    observed = [
        {**r, "prediction": 0 if r["native_gold"] == "supports" else 1, "raw_prediction": 0}
        for r in local[:3]
    ]
    result = hosted_analysis(observed, {"fixture": local})["open_web"]
    assert result["valid_pairs"] == 3
    assert result["complete_claim_groups"] == 1
    raw = result["variants"]["raw"]
    assert raw["local_oof_on_same_pairs_target_fsr"] == 1
    gated = result["variants"]["quote_gated"]
    assert gated["local_oof_on_same_pairs_target_fsr"] == 0
    assert gated["local_oof_on_same_pairs"]["fixture"]["recall"]["value"] == 1
    assert gated["claim_aggregation_complete_groups_only"]["groups"] == 1
