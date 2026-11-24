import numpy as np

from groundmark.nli import NLI_LABELS

NATIVE_LABELS = (
    "supports",
    "insufficient-supports",
    "insufficient-neutral",
    "insufficient-contradictory",
    "insufficient-refutes",
    "refutes",
    "not_applicable",
)
GOLD = {
    status: 0 if status == "supports" else 1 if status == "refutes" else 2
    for status in NATIVE_LABELS
}


def ratio(numerator, denominator, *, zero=0.0):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, zero, dtype=float),
        where=denominator != 0,
    )


def checker_metrics(rows: list[dict], ternary: bool, resamples: int = 2000) -> dict:
    native = np.array([NATIVE_LABELS.index(row["native_gold"]) for row in rows])
    gold = np.array([GOLD[row["native_gold"]] for row in rows])
    prediction = np.array([row["prediction"] for row in rows])
    scores = np.array([row["support_score"] for row in rows])
    is_support = gold == 0
    support_predicted = prediction == 0
    group_keys = np.array([row["group"] for row in rows])
    groups, group_index = np.unique(group_keys, return_inverse=True)
    boot = np.random.default_rng(42).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=resamples
    )
    weights = np.vstack((np.ones(len(groups), dtype=np.int64), boot))
    result = {"rows": len(rows), "groups": len(groups), "metrics": {}}

    def aggregate(flags):
        return weights @ np.bincount(group_index, weights=flags, minlength=len(groups))

    def record(key, values):
        observed = float(values[0])
        comparable = values[1:][np.isfinite(values[1:])]
        result["metrics"][key] = {
            "value": observed if np.isfinite(observed) else None,
            "ci95": np.quantile(comparable, [0.025, 0.975]).tolist() if len(comparable) else None,
            "bootstrap_valid": len(comparable),
        }

    tp = aggregate(is_support & support_predicted)
    pos_gold = aggregate(is_support)
    pos_pred = aggregate(support_predicted)
    neg_gold = aggregate(~is_support)
    recall = ratio(tp, pos_gold)
    false_support = ratio(pos_pred - tp, neg_gold)
    record("support_precision", ratio(tp, pos_pred))
    record("support_recall", recall)
    record("support_f1", ratio(2 * tp, pos_pred + pos_gold))
    record(
        "balanced_accuracy",
        np.where((pos_gold > 0) & (neg_gold > 0), (recall + 1 - false_support) / 2, np.nan),
    )
    record("false_support_rate", ratio(pos_pred - tp, neg_gold, zero=np.nan))

    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(rows) - 1]
    ranked_weights = weights[:, group_index[order]]
    true_counts = (ranked_weights * is_support[order]).cumsum(axis=1)[:, ends]
    predicted_counts = ranked_weights.cumsum(axis=1)[:, ends]
    tpr = ratio(true_counts, pos_gold[:, None], zero=np.nan)
    fpr = ratio(predicted_counts - true_counts, neg_gold[:, None], zero=np.nan)
    precision = ratio(true_counts, predicted_counts, zero=1.0)
    tpr = np.column_stack((np.zeros(len(weights)), tpr))
    fpr = np.column_stack((np.zeros(len(weights)), fpr))
    precision = np.column_stack((np.ones(len(weights)), precision))
    record("roc_auc", np.trapezoid(tpr, fpr, axis=1))
    record("pr_auc", np.trapezoid(precision, tpr, axis=1))

    if ternary:
        confusion = np.bincount(3 * gold + prediction, minlength=9).reshape(3, 3)
        native_matrix = np.bincount(3 * native + prediction, minlength=21).reshape(7, 3)
        result.update(
            class_order=NLI_LABELS,
            native_label_order=NATIVE_LABELS,
            confusion_matrix=confusion.tolist(),
            native_prediction_matrix=native_matrix.tolist(),
        )
        f1s = []
        for index, label in enumerate(NLI_LABELS):
            gold_class, predicted_class = gold == index, prediction == index
            correct = aggregate(gold_class & predicted_class)
            count_pred, count_gold = aggregate(predicted_class), aggregate(gold_class)
            f1 = ratio(2 * correct, count_pred + count_gold)
            record(f"{label}_precision", ratio(correct, count_pred))
            record(f"{label}_recall", ratio(correct, count_gold))
            record(f"{label}_f1", f1)
            f1s.append(f1)
        record("macro_f1", np.mean(f1s, axis=0))
    return result
