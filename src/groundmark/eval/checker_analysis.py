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

    def out_of_fold(
        self, scores: np.ndarray, targets: tuple, *, fit=fit_cutoff, positive=None, valid=None
    ) -> dict:
        split = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        if positive is None:
            positive = self.gold == 0
        result = {
            target: {"supports": np.zeros(len(scores), dtype=bool), "thresholds": []}
            for target in targets
        }
        for train, held in split.split(scores, groups=self.groups):
            if valid is not None:
                train = train[valid[train]]
            for target, output in result.items():
                cutoff = fit(scores[train], positive[train], target)
                output["supports"][held] = scores[held] >= cutoff
                output["thresholds"].append(cutoff if np.isfinite(cutoff) else None)
        return result

    def claim_labels(self, labels: np.ndarray) -> np.ndarray:
        support = np.bincount(self.group_index[labels == 0], minlength=len(self.sizes)) > 0
        refutes = np.bincount(self.group_index[labels == 1], minlength=len(self.sizes)) > 0
        return support + 2 * refutes

    def supported_rates(self, predictions: np.ndarray) -> dict[str, np.ndarray]:
        gold = self.claim_labels(self.gold) == 1
        predicted = self.claim_labels(predictions) == 1
        return {
            "false_supported_rate": ratio(
                self.weights @ (~gold & predicted), self.weights @ ~gold, zero=np.nan
            ),
            "supported_recall": ratio(
                self.weights @ (gold & predicted), self.weights @ gold, zero=np.nan
            ),
        }

    def aggregate(self, predictions: np.ndarray, ternary: bool) -> dict:
        gold = self.claim_labels(self.gold)
        predicted = self.claim_labels(predictions)
        if ternary:
            labels = CLAIM_STANCE_ORDER
        else:
            gold, predicted = (gold == 1).astype(int), (predicted == 1).astype(int)
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
            **{
                name: interval(values) for name, values in self.supported_rates(predictions).items()
            },
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


def hosted_analysis(predictions: list[dict], local: dict[str, list[dict]]) -> dict:
    from collections import Counter

    output = {}
    first_local = next(iter(local.values()))
    for subset, is_gold in SUBSETS.items():
        selected = [row for row in predictions if row["is_gold"] == is_gold]
        eligible = [row for row in first_local if row["is_gold"] == is_gold]
        selected_ids = {row["row_id"] for row in selected}
        full_group_sizes = Counter(row["group"] for row in eligible)
        observed_sizes = Counter(row["group"] for row in selected)
        complete_groups = {
            group for group, size in observed_sizes.items() if full_group_sizes[group] == size
        }
        result = {
            "valid_pairs": len(selected),
            "eligible_pairs": len(eligible),
            "complete_claim_groups": len(complete_groups),
            "eligible_groups": len(full_group_sizes),
            "variants": {},
        }
        output[subset] = result
        if not selected:
            continue
        groups = ClaimGroups(eligible)
        mask = np.array([row["row_id"] in selected_ids for row in eligible])
        for variant, column in (("raw", "raw_prediction"), ("quote_gated", "prediction")):
            rows = [{**row, "prediction": row[column]} for row in selected]
            metrics = checker_metrics(rows, ternary=True)
            target = metrics["metrics"]["false_support_rate"]["value"]
            complete = [row for row in rows if row["group"] in complete_groups]
            aggregation = (
                ClaimGroups(complete).aggregate(
                    np.array([row["prediction"] for row in complete]), ternary=True
                )
                if complete
                else None
            )
            matched = {}
            if target is not None:
                for model, local_rows in local.items():
                    scores = np.array(
                        [row["support_score"] for row in local_rows if row["is_gold"] == is_gold]
                    )
                    oof = groups.out_of_fold(scores, (target,))[target]
                    matched[model] = {
                        "thresholds": oof["thresholds"],
                        **{
                            name: interval(values)
                            for name, values in groups.operating_rates(
                                oof["supports"], mask
                            ).items()
                        },
                    }
            result["variants"][variant] = {
                "passages": metrics,
                "claim_aggregation_complete_groups_only": aggregation,
                "local_oof_on_same_pairs_target_fsr": target,
                "local_oof_on_same_pairs": matched,
            }
    return output
