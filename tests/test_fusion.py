import numpy as np
from sklearn.model_selection import GroupKFold

from groundmark.eval.checker_analysis import ClaimGroups
from groundmark.eval.fusion import fuse, recall_cutoff


def test_fusion_vetoes_only_hosted_support():
    hosted = np.array([0, 0, 1, 1, 2, 2])
    local = np.array([True, False, True, False, True, False])
    assert fuse(hosted, local).tolist() == [0, 2, 1, 1, 2, 2]


def test_recall_cutoff_uses_original_group_folds_and_preserves_ties():
    rows = [
        {"group": str(group), "native_gold": label}
        for group in range(7)
        for label in ("supports", "insufficient-neutral")
    ]
    groups = ClaimGroups(rows, resamples=5)
    scores = np.array([0.3, 0.9, 0.4, 0.5, 0.6, 0.5, 0.4, 0.7, 0.8, 0.2, 0.7, 0.3, 0.9, 0.6])
    result = groups.out_of_fold(scores, (0.80,), fit=recall_cutoff)[0.80]
    for (train, held), cutoff in zip(
        GroupKFold(n_splits=5, shuffle=True, random_state=42).split(scores, groups=groups.groups),
        result["thresholds"],
        strict=True,
    ):
        train_support = scores[train][groups.gold[train] == 0]
        assert np.mean(train_support >= cutoff) >= 0.80
        assert np.mean(train_support > cutoff) < 0.80
        assert np.array_equal(result["supports"][held], scores[held] >= cutoff)


def test_fusion_can_turn_hosted_conflict_into_refuted():
    groups = ClaimGroups(
        [{"group": "x", "native_gold": "refutes"}, {"group": "x", "native_gold": "supports"}],
        resamples=5,
    )
    predictions = fuse(np.array([1, 0]), np.array([True, False]))
    result = groups.aggregate(predictions, ternary=True)
    assert result["confusion_gold_rows"][3] == [0, 0, 1, 0]
