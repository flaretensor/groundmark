import numpy as np
import pytest

from groundmark.eval.cascade import analyze_subset


def test_cascade_uses_highest_training_cutoff_that_retains_hosted_refutes_and_ties():
    references, hosted, costs = [], [], {}
    for group in range(5):
        for index, (gold, label, score) in enumerate(
            [
                ("supports", 0, 0.8),
                ("refutes", 1, 0.4),
                ("insufficient-neutral", 2, 0.4),
                ("insufficient-neutral", 2, 0.2),
                ("supports", 0, 0.6),
            ]
        ):
            key = f"{group}-{index}"
            row = {"group": group, "row_id": key, "native_gold": gold}
            references.append({**row, "support_score": score})
            if key != "0-4":
                hosted.append({**row, "prediction": label})
                costs[key] = 0.01
    result = analyze_subset(hosted, {"local": references}, costs, resamples=20)
    assert result["valid_pairs"] == 24
    assert result["complete_claim_groups"] == 4
    cascade = next(row for row in result["variants"] if row["setting"] == "retain_0.95")
    assert cascade["fold_thresholds"] == [0.4] * 5
    assert cascade["hosted_calls_saved"] == 5
    assert cascade["metrics"]["refutes_recall"]["value"] == 1
    assert cascade["metrics"]["hosted_refutes_lost"]["value"] == 0
    assert cascade["metrics"]["usd_per_pair_mean"]["value"] == pytest.approx(0.01 * 19 / 24)
    assert cascade["metrics"]["usd_per_25_pairs_p95"]["value"] == pytest.approx(0.25)
    assert cascade["paired_minus_v1"]["supported_recall"]["value"] is None


def test_local_only_loses_hosted_refutes_and_turns_conflict_into_false_supported():
    references, hosted, costs = [], [], {}
    for group in range(5):
        for index, (gold, label, score) in enumerate([("supports", 0, 0.9), ("refutes", 1, 0.1)]):
            key = f"{group}-{index}"
            row = {"group": group, "row_id": key, "native_gold": gold}
            references.append({**row, "support_score": score})
            hosted.append({**row, "prediction": label})
            costs[key] = 0.02
    local = analyze_subset(hosted, {"local": references}, costs, resamples=20)["variants"][1]
    assert local["fold_thresholds"] == [0.9] * 5
    assert local["metrics"]["skipped_fraction"]["value"] == 1
    assert local["metrics"]["usd_per_pair_p95"]["value"] == 0
    assert local["metrics"]["refutes_recall"]["value"] == 0
    assert local["metrics"]["hosted_refutes_lost"]["value"] == 5
    assert local["paired_minus_v1"]["false_supported_rate"]["value"] == 1
    np.testing.assert_allclose(local["paired_minus_v1"]["false_supported_rate"]["ci95"], [1, 1])
