from collections import Counter

import pytest

from groundmark.budget import Amount, BudgetRefused
from groundmark.config import Credentials
from groundmark.eval import evouna
from groundmark.eval.cli import main
from groundmark.eval.datasets import EVOUNA_JUDGES
from groundmark.eval.evouna import LIMIT, Pair, validation_ledger
from groundmark.eval.evouna_metrics import disagreements, judge_metrics
from groundmark.eval.graders import dataset_graders
from groundmark.eval.judge import SYSTEM, AnswerJudge
from groundmark.models import TokenUsage
from groundmark.providers import Completion, ProviderFailure


def row(question, *, improper=False):
    return {
        "improper": improper,
        "question": question,
        "golden_answer": "a / b",
        **{f"answer_{system}": "a" for system in EVOUNA_JUDGES},
        **{f"judge_{system}": True for system in EVOUNA_JUDGES},
        "answer_gpt35": "",
        "judge_chatgpt": "nan",
        "judge_newbing": False,
    }


def test_evouna_eligibility_and_distinct_question_sampling():
    nq = evouna.eligible_pairs("NQ", [row("Same question"), row("ineligible", improper=True)])
    tq = evouna.eligible_pairs("TQ", [row(" SAME   question "), row("Another")])
    assert [(pair.system, pair.human_correct) for pair in nq] == [
        ("fid", True),
        ("gpt4", True),
        ("newbing", False),
    ]
    first = evouna.sample_groups(nq + tq)
    assert first == evouna.sample_groups(nq + tq)
    sample = evouna.selected_pairs(first, 150)
    assert len(sample) == 2
    assert {"another", "same question"} == {" ".join(x.question.casefold().split()) for x in sample}


def test_validation_cli_requires_live_and_explicit_bounded_cap(monkeypatch):
    monkeypatch.setattr(Credentials, "load", lambda **_: pytest.fail("credentials loaded"))
    prefix = ["judge-validate", "--dataset", "evouna"]
    for flags in ([], ["--live"], ["--stage-cap-usd", ".50"]):
        with pytest.raises(SystemExit, match="2"):
            main(prefix + flags)
    with pytest.raises(SystemExit, match="1"):
        main(prefix + ["--live", "--stage-cap-usd", ".50001"])


def test_evouna_binary_mapping_metrics_and_paired_disagreement():
    graders = tuple(grader.name for grader in dataset_graders("evouna"))
    outcomes = [
        {"grader_template": graders[0], "pair_id": "1", "human_correct": False, "grade": "CORRECT"},
        {
            "grader_template": graders[0],
            "pair_id": "2",
            "human_correct": False,
            "grade": "NOT_ATTEMPTED",
        },
        {"grader_template": graders[0], "pair_id": "3", "human_correct": True, "grade": "CORRECT"},
        {
            "grader_template": graders[0],
            "pair_id": "4",
            "human_correct": True,
            "grade": "PARSE_FAILURE",
        },
        {
            "grader_template": graders[1],
            "pair_id": "1",
            "human_correct": False,
            "grade": "INCORRECT",
        },
        {
            "grader_template": graders[1],
            "pair_id": "2",
            "human_correct": False,
            "grade": "INCORRECT",
        },
    ]
    records = [item | {"dataset": "NQ", "attempts": []} for item in outcomes]
    report = judge_metrics(
        records[:4], Counter({("NQ", False): 70, ("NQ", True): 30}), evouna.PRICE, resamples=10
    )
    for metric in ("false_correct_rate", "miss_rate", "reweighted_agreement_accuracy"):
        assert report[metric]["value"] == 0.5
    assert report["kappa"]["value"] == 0
    assert report["usage"]["parse_failures"] == 1
    assert report["human_incorrect_labels"]["NOT_ATTEMPTED"]["count"] == 1
    paired = disagreements(records, graders, resamples=10)
    assert paired["human_incorrect"]["different"] == 1
    assert paired["human_incorrect"]["rate"]["value"] == 0.5
    assert paired["human_correct"]["n"] == 0


def test_evouna_preflight_budget_plan_preserves_prefixes():
    strata = {
        (dataset, correct): [
            Pair(dataset, i, "fid", correct, f"{dataset} {correct} {i}", "a", "a")
            for i in range(150)
        ]
        for dataset, correct in evouna.STRATA
    }
    all_pairs = evouna.selected_pairs(strata, 150)
    assert len(all_pairs) == 600
    assert evouna.plan_after_preflight(strata, 6_000_000, LIMIT) == 150
    assert evouna.plan_after_preflight(strata, 9_000_000, LIMIT) == 125
    assert evouna.plan_after_preflight(strata, 24_000_000, LIMIT) == 0
    assert evouna.selected_pairs(strata, 125) == all_pairs[:500]


def test_evouna_cached_grades_and_retries_use_persistent_ledger(settings, tmp_path):
    class Model:
        calls = 0

        def complete(self, request):
            assert request.system == SYSTEM
            assert "Gold target: a / b" in request.prompt
            self.calls += 1
            if self.calls == 1:
                raise ProviderFailure.http("nebius", 429)
            return Completion('{"grade":"B"}', "stop", TokenUsage(input=5, output=2, reasoning=1))

    ledger = validation_ledger(tmp_path, LIMIT)
    model = Model()
    judge = AnswerJudge(settings, Credentials("", ""), ledger, model)
    pair = Pair("NQ", 1, "fid", False, "Question?", "a / b", "result")
    first, second = dataset_graders("evouna")
    directory = tmp_path / "run"
    record = evouna.grade_pair(pair, first, directory, judge, set())
    assert record["grade"] == "INCORRECT"
    assert record["grader_sha256"] == first.sha256
    assert ledger.scope_reservations() == 2 and ledger.scope_unsettled() == 1
    restored = validation_ledger(tmp_path, LIMIT)
    assert evouna.grade_pair(pair, first, directory, judge, restored.scope_run_ids()) == record
    key = evouna.digest(f"{pair.id}:{second.name}")
    reserved = {f"{directory.name}:{key}"}
    assert evouna.grade_pair(pair, second, directory, judge, reserved)["state"] == "pending"
    assert model.calls == 2
    with pytest.raises(BudgetRefused, match="budget"):
        restored.reserve("other", "completion", LIMIT, LIMIT)
    with pytest.raises(ValueError, match="positive stage cap"):
        validation_ledger(tmp_path, Amount())
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in directory.rglob("*.json"))
