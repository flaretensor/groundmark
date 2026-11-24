import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from groundmark.eval.checker_metrics import GOLD, checker_metrics, group_weights, interval, ratio
from groundmark.sources import private_json
from groundmark.verification import CLAIM_STANCE_ORDER

TARGETS = (0.05, 0.10, 0.20)
SUBSETS = {"open_web": False, "fact_checker": True}
FOLDS = 5
RESAMPLES = 2000


def fit_cutoff(scores: np.ndarray, gold_support: np.ndarray, target: float) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(scores) - 1]
    tp = gold_support[order].cumsum()[ends]
    fp = ends + 1 - tp
    eligible = np.flatnonzero((fp <= target * (~gold_support).sum()) & (tp > 0))
    return float(ranked[ends[eligible[np.argmax(tp[eligible])]]]) if len(eligible) else np.inf


class ClaimGroups:
    def __init__(self, rows: list[dict], resamples: int = RESAMPLES):
        self.rows = rows
        self.gold = np.array([GOLD[row["native_gold"]] for row in rows])
        self.groups = np.array([row["group"] for row in rows])
        unique, self.group_index, self.sizes = np.unique(
            self.groups, return_inverse=True, return_counts=True
        )
        self.weights = group_weights(len(unique), resamples)

    def sum(self, flags: np.ndarray) -> np.ndarray:
        per_group = np.bincount(self.group_index, weights=flags, minlength=len(self.sizes))
        return self.weights @ per_group

    def operating_rates(self, support: np.ndarray, mask=True) -> dict[str, np.ndarray]:
        positives = (self.gold == 0) & mask
        negatives = (self.gold != 0) & mask
        return {
            "recall": ratio(self.sum(support & positives), self.sum(positives), zero=np.nan),
            "fsr": ratio(self.sum(support & negatives), self.sum(negatives), zero=np.nan),
        }

    def out_of_fold(self, scores: np.ndarray, targets: tuple) -> dict:
        split = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        gold_support = self.gold == 0
        result = {
            target: {"supports": np.zeros(len(scores), dtype=bool), "thresholds": []}
            for target in targets
        }
        for train, held in split.split(scores, groups=self.groups):
            for target, output in result.items():
                cutoff = fit_cutoff(scores[train], gold_support[train], target)
                output["supports"][held] = scores[held] >= cutoff
                output["thresholds"].append(cutoff if np.isfinite(cutoff) else None)
        return result

    def aggregate(self, predictions: np.ndarray, ternary: bool) -> dict:
        def claim_labels(labels):
            support = np.bincount(self.group_index[labels == 0], minlength=len(self.sizes)) > 0
            refutes = np.bincount(self.group_index[labels == 1], minlength=len(self.sizes)) > 0
            return support + 2 * refutes

        gold = claim_labels(self.gold)
        predicted = claim_labels(predictions)
        gold_supported, pred_supported = gold == 1, predicted == 1
        tp = self.weights @ (gold_supported & pred_supported)
        fp = self.weights @ (~gold_supported & pred_supported)
        if ternary:
            labels = CLAIM_STANCE_ORDER
        else:
            gold, predicted = gold_supported.astype(int), pred_supported.astype(int)
            labels = ("not_supported", "supported")
        counts = np.bincount(len(labels) * gold + predicted, minlength=len(labels) ** 2)
        sizes, groups = np.unique(self.sizes, return_counts=True)
        return {
            "gold_source": "Derived from passage labels, not a human claim verdict.",
            "groups": len(self.sizes),
            "passages_per_group_histogram": dict(
                zip(map(int, sizes), map(int, groups), strict=True)
            ),
            "label_order": labels,
            "confusion_gold_rows": counts.reshape(len(labels), len(labels)).tolist(),
            "false_supported_rate": interval(
                ratio(fp, self.weights @ ~gold_supported, zero=np.nan)
            ),
            "supported_recall": interval(ratio(tp, self.weights @ gold_supported, zero=np.nan)),
        }


def read_local_receipts(paths: list[Path]) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    receipts, predictions = {}, {}
    for path in paths:
        receipt = json.loads((path / "metrics.json").read_bytes())
        model = receipt["model"]
        if model in receipts or receipt["split"] != "dev":
            raise ValueError("Expected unique DRUID DEV checker receipts")
        rows = pq.read_table(path / "predictions.parquet").to_pylist()
        rows.sort(key=lambda row: row["row_id"])
        if not rows or len({row["row_id"] for row in rows}) != len(rows):
            raise ValueError("Missing or duplicate prediction IDs")
        if any(
            row.get("support_score") is None
            or not 0 <= row["support_score"] <= 1
            or "contradiction_score" not in row
            for row in rows
        ):
            raise ValueError("Expected finite scores from current checker receipts")
        if receipts:
            first = next(iter(receipts.values()))
            if any(receipt[key] != first[key] for key in ("dataset", "split_sha256", "commit")):
                raise ValueError("Checker receipts differ in dataset, split, or commit")
            keys = ("row_id", "group", "native_gold", "is_gold")
            for row, ref in zip(rows, next(iter(predictions.values())), strict=True):
                if any(row[key] != ref[key] for key in keys):
                    raise ValueError("Checker predictions differ in row identities or gold labels")
        receipts[model], predictions[model] = receipt, rows
    if not receipts:
        raise ValueError("Provide at least one checker receipt directory")
    return receipts, predictions


def analyze(paths: list[Path], output: Path) -> None:
    receipts, predictions = read_local_receipts(paths)
    result = {
        "selection_subset": "open_web",
        "folds": FOLDS,
        "seed": 42,
        "resamples": RESAMPLES,
        "interval_design": "Group percentiles on fixed predictions and held-out-fold cutoffs.",
        "cv_refute_rule": "Below the support cutoff, retain only default argmax refutes.",
        "receipts": {
            receipt["model"]: str(path)
            for receipt, path in zip(receipts.values(), paths, strict=True)
        },
        "commit": next(iter(receipts.values()))["commit"],
        "split_sha256": next(iter(receipts.values()))["split_sha256"],
    }
    for subset, is_gold in SUBSETS.items():
        models = {
            model: [row for row in rows if row["is_gold"] == is_gold]
            for model, rows in predictions.items()
        }
        groups = ClaimGroups(next(iter(models.values())))
        by_model, rates = {}, {}
        for model, rows in models.items():
            default = np.array([row["prediction"] for row in rows])
            ternary = all(row["contradiction_score"] is not None for row in rows)
            operating = groups.out_of_fold(
                np.array([row["support_score"] for row in rows]), TARGETS
            )
            rates[model] = {
                target: groups.operating_rates(point["supports"])
                for target, point in operating.items()
            }
            cv = np.where(operating[0.10]["supports"], 0, np.where(default == 1, 1, 2))
            by_model[model] = {
                "default": checker_metrics(rows, ternary),
                "operating_points": {
                    target: {
                        "thresholds": point["thresholds"],
                        **{name: interval(values) for name, values in rates[model][target].items()},
                    }
                    for target, point in operating.items()
                },
                "claim_aggregation": {
                    "default": groups.aggregate(default, ternary),
                    "cv_fsr_0.10": groups.aggregate(cv, ternary),
                },
            }
        paired = [
            {
                "left": left,
                "right": right,
                "left_minus_right": {
                    target: {
                        name: interval(values - rates[right][target][name])
                        for name, values in rates[left][target].items()
                    }
                    for target in TARGETS
                },
            }
            for left, right in combinations(models, 2)
        ]
        result[subset] = {"models": by_model, "paired": paired}
    private_json(output, result)
