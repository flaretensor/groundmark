import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import ScriptedLLM, ScriptedSearch, decision, snippet

from groundmark import config
from groundmark.budget import Amount, BudgetRefused, SpendLedger, usd_amount
from groundmark.config import Credentials
from groundmark.eval import judge as judge_module
from groundmark.eval import report as report_module
from groundmark.eval import runner as runner_module
from groundmark.eval.cli import main
from groundmark.eval.graders import dataset_graders
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
from groundmark.providers import Completion, EmbeddingResponse, ExtractionResponse
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
    assert row["arms"]["A"] == row["arms"]["C"] | {"evidence_scope": "first_search"}
    assert row["arms"]["B"] == row["arms"]["D"] | {"evidence_scope": "first_search"}
    assert row["arms"]["A"]["answer"] == "only answer"
    assert row["arms"]["A"]["flagged_sources"][0]["flags"] == ["SimpleQA"]
    assert [r.schema.__name__ for r in llm.requests] == ["Action", "Action", "Synthesis"]
    assert source_flags("What year? no gold", "What year?", "2024") == set()


@pytest.mark.parametrize("later_search", [False, True])
def test_first_search_arms_never_read_extracted_or_later_evidence(settings, tmp_path, later_search):
    actions = [decision(query="year"), decision("extract", source_ids=["S1"])]
    responses = [snippet(title="First title", content="The opening is planned.")]
    if later_search:
        actions.append(decision(query="more"))
        responses.append(snippet(url="https://later.example/", content="A later source."))
    actions.append(decision("finish"))
    synthesis = {"claim_id": "C1", "text": "Year 2024", "source_ids": ["S1"]}
    llm = SourceAwareLLM(
        Action=actions,
        Synthesis=[_synthesis(group) | {"claims": [synthesis]} for group in ("full", "first")],
    )
    search = ScriptedSearch(
        *[[response] for response in responses],
        extraction=ExtractionResponse(
            [{"url": "https://example.org/", "raw_content": "SealQA says Year 2024"}], [], 1
        ),
    )

    class Embedder:
        def embed(self, texts, timeout):
            return EmbeddingResponse([[1.0, 0.0] for _ in texts], TokenUsage())

    ledger = stage_ledger(tmp_path, "smoke")
    runner = ResearchRunner(settings, llm=llm, search=search, ledger=ledger, embedder=Embedder())
    run_stage("smoke", [Question("simpleqa", "1", "When?", "2024")], runner, ledger, root=tmp_path)
    arms = _read_one(tmp_path / "smoke")["arms"]
    assert [arms[name]["evidence_scope"] for name in ARMS] == [
        "first_search",
        "first_search",
        "full_trajectory",
        "full_trajectory",
    ]
    assert [arms[name]["answer"] for name in ARMS] == [
        "first answer",
        "",
        "full answer",
        "Year 2024 [S1]",
    ]
    assert [bool(arms[name]["flagged_sources"]) for name in ARMS] == [False, False, True, True]
    full, first = [
        request.prompt for request in llm.requests if request.schema.__name__ == "Synthesis"
    ]
    assert "The opening is planned." in first and "First title" in first
    assert "https://example.org/" in first
    assert "SealQA says Year 2024" in full and "SealQA says Year 2024" not in first
    assert ("A later source." in full) == later_search
    assert "A later source." not in first


def test_live_smoke_command_grades_each_dataset_and_passes_template_hashes_to_mlflow(
    settings, tmp_path, monkeypatch
):
    questions = [Question(dataset, "1", "What year?", "2024") for dataset in ("simpleqa", "sealqa")]
    smoke_settings = settings.model_copy(update={"mlflow_tracking_uri": "http://127.0.0.1:51433"})
    model = ScriptedLLM(Grade=[{"grade": "A"}, {"grade": "B"}])
    recorded = []

    @contextmanager
    def fake_judge(settings, ledger):
        yield AnswerJudge(settings, Credentials("", ""), ledger, model)

    def fake_runner(settings, ledger):
        llm = ScriptedLLM(
            Action=[decision(query="year"), decision("finish")] * 2,
            Synthesis=[_synthesis("only") | {"claims": []}] * 2,
        )
        return ResearchRunner(
            settings, llm=llm, search=ScriptedSearch([snippet()], [snippet()]), ledger=ledger
        )

    monkeypatch.setattr(config, "Settings", lambda: smoke_settings)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "stage_questions", lambda stage: questions)
    monkeypatch.setattr(runner_module, "live_runner", fake_runner)
    monkeypatch.setattr(judge_module, "live_judge", fake_judge)
    monkeypatch.setattr(
        report_module, "log_mlflow", lambda summary, root, uri: recorded.append(summary)
    )

    main(["run", "--stage", "smoke", "--live"])

    expected = {}
    rows = [json.loads(path.read_bytes()) for path in (tmp_path / "smoke").glob("*.json")]
    for row in rows:
        (grader,) = dataset_graders(row["dataset"])
        expected[grader.name] = grader.sha256
        assert 0 < row["total_latency_ms"] <= row["processing_ms"]
        assert row["arms"]["A"]["grader_template"] == grader.name
        assert row["arms"]["A"]["grader_sha256"] == grader.sha256
        assert row["judge_usage"][0]["tokens"]["input"] == 40
    assert len(rows) == len(model.requests) == 2
    assert recorded[0]["grader_templates"] == expected


def test_eval_report_latency_clocks_and_legacy_product_receipt(tmp_path, monkeypatch, capsys):
    root = tmp_path / "eval"
    (root / "smoke").mkdir(parents=True)
    run_id = "4f59296b-3b4b-4727-bad8-f474a9b3a233"
    legacy = root.parent / "runs" / run_id
    legacy.mkdir(parents=True)
    pq.write_table(pa.table({"latency_ms": [55.0]}), legacy / "run.parquet")
    arm = {"abstain": True, "grade": None, "stop_reason": "completed", "flagged_sources": []}
    common = {
        "arms": dict.fromkeys(ARMS, arm),
        "stop_reason": "completed",
        "tavily_credits_charged_or_reserved": 0,
        "nebius_nanos_charged_or_reserved": 0,
        "dataset": "sealqa",
        "seal_0": False,
    }
    cases = [
        {"dataset": "simpleqa", "latency_ms": 100, "processing_ms": 60, "total_latency_ms": 35},
        {"latency_ms": 250, "processing_ms": 90, "run_id": run_id},
        {"latency_ms": 400, "seal_0": True},
    ]
    for index, case in enumerate(cases):
        (root / "smoke" / f"{index}.json").write_text(json.dumps(common | case), encoding="utf-8")
    client = Mock()
    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(MlflowClient=Mock(return_value=client))
    )
    summary = report("smoke", root=root, mlflow_uri="mock")
    assert summary["cumulative_batch_ms"] == {"p50": 250, "p95": 385}
    assert summary["processing_ms"] == {"p50": 75, "p95": 88.5}
    assert summary["total_latency_ms"] == {"p50": 45, "p95": 54}
    assert summary["seal_0"]["total_latency_ms"] == {"p50": None, "p95": None}
    assert "queue_inclusive_latency_ms" not in summary
    recorded = {call.args[1]: call.args[2] for call in client.log_metric.call_args_list}
    assert recorded["cumulative_batch_p95_ms"] == 385
    assert recorded["processing_p95_ms"] == 88.5
    assert recorded["total_latency_p95_ms"] == 54

    monkeypatch.setattr(report_module, "report", lambda *args, **kwargs: summary)
    main(["report", "--stage", "smoke", "--mlflow-uri", "", "--latency-markdown"])
    rendered = capsys.readouterr().out
    assert "| All questions | Time since batch start | 250.00 | 385.00 |" in rendered
    assert "| All questions | Per-question eval processing | 75.00 | 88.50 |" in rendered
    assert "| All questions | Product trace latency | 45.00 | 54.00 |" in rendered
    assert "| Seal-0 | Product trace latency | not recorded | not recorded |" in rendered
