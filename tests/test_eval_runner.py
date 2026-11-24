import json

import pytest
from conftest import ScriptedLLM, ScriptedSearch, decision, snippet

from groundmark.budget import Amount, BudgetRefused, SpendLedger, usd_amount
from groundmark.config import Credentials
from groundmark.eval.cli import main
from groundmark.eval.judge import AnswerJudge, judge_stage
from groundmark.eval.report import report
from groundmark.eval.runner import (
    ARMS,
    CANARY,
    Question,
    run_stage,
    source_flags,
    stage_ledger,
)
from groundmark.models import PairJudgments, TokenUsage
from groundmark.providers import Completion
from groundmark.research import ResearchRunner


def test_live_and_test_confirmation_fail_before_provider_construction(monkeypatch):
    def forbidden():
        pytest.fail("credentials requested")

    monkeypatch.setattr(Credentials, "load", forbidden)
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--stage", "smoke"])
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--stage", "test", "--live"])


def test_stage_and_combined_caps_persist(tmp_path):
    smoke = stage_ledger(tmp_path, "smoke")
    per_run = usd_amount(1000, 10)
    smoke.reserve("smoke", "extract", usd_amount(29, 0.24), per_run)
    restarted = stage_ledger(tmp_path, "smoke")
    with pytest.raises(BudgetRefused) as last:
        restarted.reserve("smoke-next", "search", Amount(credits=2), per_run)
    assert last.value.kind == "credits"
    assert restarted.scope_blocked

    dev = stage_ledger(tmp_path, "dev")
    dev.reserve("dev", "search", usd_amount(140, 0.75), per_run)
    test = stage_ledger(tmp_path, "test")
    test.reserve("test", "search", usd_amount(550, 3), per_run)
    assert test.total() == usd_amount(719, 3.99)
    with pytest.raises(BudgetRefused):
        dev.reserve("dev", "judge", usd_amount(0, 0.000001), per_run)

    combined = SpendLedger(smoke.path, usd_amount(720, 4), scope="other", scope_limit=per_run)
    combined.reserve("other", "search", Amount(credits=1), per_run)
    with pytest.raises(BudgetRefused, match="budget"):
        combined.reserve("other", "search", Amount(credits=1), per_run)


def test_test_guard_rejects_second_start_even_without_results(tmp_path):
    ledger = stage_ledger(tmp_path, "test")
    with pytest.raises(ValueError, match="--confirm-test"):
        run_stage("test", [], None, ledger, root=tmp_path)
    assert run_stage("test", [], None, ledger, root=tmp_path, confirm_test=True) == 0
    with pytest.raises(ValueError, match="already started"):
        run_stage("test", [], None, ledger, root=tmp_path, confirm_test=True)


def _synthesis(group):
    return {
        "summary": f"{group} answer",
        "claims": [{"claim_id": "C1", "text": "Year 2024", "source_ids": ["S2"]}],
        "insufficient_evidence": False,
        "missing_information": [],
        "conflicts": [],
    }


class SourceAwareLLM(ScriptedLLM):
    def complete(self, request):
        if request.schema is not PairJudgments:
            return super().complete(request)
        self.requests.append(request)
        return Completion(
            json.dumps(
                {
                    "pairs": [
                        {
                            "claim_id": "C1",
                            "verdict": "supports",
                            "evidence_quote": "Year 2024",
                            "reason": "The source gives a year.",
                        }
                    ]
                }
            ),
            "stop",
            TokenUsage(),
        )


def _read_one(directory):
    (path,) = directory.glob("*.json")
    return json.loads(path.read_bytes())


def test_one_trajectory_four_arms_flagging_judge_and_report(settings, tmp_path):
    root = tmp_path / "eval"
    ledger = stage_ledger(root, "smoke")
    llm = SourceAwareLLM(
        Action=[decision(query="first"), decision(query="more"), decision("finish")],
        Synthesis=[_synthesis("full"), _synthesis("first")],
    )
    question = Question("sealqa", "1", "What year?", "2024", True)
    search = ScriptedSearch(
        [snippet(content=f"SealQA {CANARY}")],
        [snippet(url="https://another.example/", content="What year? Year 2024")],
    )
    runner = ResearchRunner(settings, llm=llm, search=search, ledger=ledger)
    assert run_stage("smoke", [question], runner, ledger, root=root) == 1
    assert search.requests == ["first", "more"]
    row = _read_one(root / "smoke")
    assert tuple(row["arms"]) == ARMS
    assert [row["arms"][name]["answer"] for name in ARMS] == [
        "first answer",
        "",
        "full answer",
        "Year 2024 [S2]",
    ]
    assert row["arms"]["B"]["abstain"]
    assert [len(row["arms"][name]["flagged_sources"]) for name in ARMS] == [1, 1, 2, 2]
    assert row["arms"]["D"]["flagged_sources"][1]["flags"] == ["question_and_gold"]
    assert row["latency_ms"] >= row["processing_ms"] > 0

    pseudo_judge = ScriptedLLM(Grade=[{"grade": "B"}, {"grade": "C"}, {"grade": "A"}])
    judge = AnswerJudge(settings, Credentials("", ""), ledger, pseudo_judge)
    assert judge_stage(root / "smoke", judge) == 3
    assert len(pseudo_judge.requests) == 3
    assert all(request.model == settings.judge_model for request in pseudo_judge.requests)
    assert judge_stage(root / "smoke", judge) == 0
    result = report("smoke", root)
    assert result["paired"]["D-A"] == {"n": 1, "delta": 1.0, "ci95": [1.0, 1.0]}
    assert result["paired"]["D-C"]["delta"] == 1.0
    assert result["paired"]["B-A"]["delta"] == 0.0
    assert result["arms"]["D"]["accuracy_among_answered"] == 1
    assert result["arms"]["C"]["counts"]["judge_not_attempted"] == 1
    assert result["arms"]["B"]["rates"]["abstain"] == 1
    assert result["seal_0"]["questions"] == 1
    assert result["tavily_credits_charged_or_reserved"] == 2


def test_one_search_reuses_synthesis_and_flags_without_dropping(settings, tmp_path):
    root = tmp_path / "eval"
    ledger = stage_ledger(root, "smoke")
    llm = ScriptedLLM(
        Action=[decision(query="year"), decision("finish")],
        Synthesis=[_synthesis("only") | {"claims": []}],
    )
    source = snippet(content="SimpleQA")
    runner = ResearchRunner(settings, llm=llm, search=ScriptedSearch([source]), ledger=ledger)
    run_stage("smoke", [Question("simpleqa", "1", "When?", "2024")], runner, ledger, root=root)
    row = _read_one(root / "smoke")
    assert row["arms"]["A"] == row["arms"]["C"]
    assert row["arms"]["B"] == row["arms"]["D"]
    assert row["arms"]["A"]["answer"] == "only answer"
    assert row["arms"]["A"]["flagged_sources"][0]["flags"] == ["SimpleQA"]
    assert [r.schema.__name__ for r in llm.requests] == ["Action", "Action", "Synthesis"]
    assert source_flags("What year? no gold", "What year?", "2024") == set()
