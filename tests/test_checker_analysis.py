import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from groundmark.eval.checker_analysis import ClaimGroups, fit_cutoff  # noqa: E402
from groundmark.eval.checker_metrics import checker_metrics  # noqa: E402


def row(group, native_gold, prediction):
    return {"group": group, "native_gold": native_gold, "prediction": prediction}


def test_cutoff_counts_ties_and_optimizes_recall():
    scores = np.array([0.9, 0.8, 0.8, 0.6, 0.2])
    positives = np.array([True, False, True, True, False])
    assert fit_cutoff(scores, positives, 0) == 0.9
    assert fit_cutoff(scores, positives, 0.5) == 0.6
    assert np.isinf(fit_cutoff(np.array([0.9, 0.9]), np.array([True, False]), 0))


def test_group_cv_produces_one_held_out_prediction_and_deterministic_paired_rates():
    rows = [
        row(str(group), label, 2)
        for group in range(10)
        for label in ("supports", "insufficient-neutral")
    ]
    groups = ClaimGroups(rows, resamples=20)
    good_scores = np.tile([0.9, 0.1], 10)
    weak_scores = np.tile([0.1, 0.9], 10)
    good = groups.out_of_fold(good_scores, (0.1,))[0.1]
    weak = groups.out_of_fold(weak_scores, (0.1,))[0.1]
    assert good["thresholds"] == [0.9] * 5
    assert weak["thresholds"] == [None] * 5
    good_rates = groups.operating_rates(good["supports"])
    weak_rates = groups.operating_rates(weak["supports"])
    np.testing.assert_equal(good_rates["recall"] - weak_rates["recall"], np.ones(21))
    np.testing.assert_equal(good_rates["fsr"], np.zeros(21))


def test_aggregation_handles_conflict_and_binary_without_refutation():
    rows = [
        row("a", "supports", 0),
        row("a", "refutes", 1),
        row("b", "refutes", 0),
        row("c", "supports", 0),
        row("c", "insufficient-supports", 2),
    ]
    groups = ClaimGroups(rows, resamples=20)
    predicted = np.array([entry["prediction"] for entry in rows])
    ternary = groups.aggregate(predicted, ternary=True)
    assert ternary["confusion_gold_rows"] == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    assert ternary["passages_per_group_histogram"] == {1: 1, 2: 2}
    assert ternary["false_supported_rate"]["value"] == 0.5
    assert ternary["supported_recall"]["value"] == 1
    assert groups.aggregate(np.where(predicted == 1, 2, predicted), False)[
        "confusion_gold_rows"
    ] == [[0, 2], [0, 1]]


def test_native_breakdown_and_refutes_false_positive_rate():
    rows = [
        row("a", "supports", 0),
        row("b", "insufficient-supports", 0),
        row("c", "insufficient-neutral", 1),
        row("d", "refutes", 1),
    ]
    result = checker_metrics(rows, ternary=True, resamples=20)
    assert result["native_prediction_matrix"][1] == [1, 0, 0]
    assert result["metrics"]["false_support_rate.insufficient-supports"]["value"] == 1
    assert result["metrics"]["false_support_rate.not_applicable"]["value"] is None
    assert result["metrics"]["refutes_fpr"]["value"] == pytest.approx(1 / 3)
