import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq

from groundmark.eval.checker_analysis import (
    RESAMPLES,
    SUBSETS,
    ClaimGroups,
    fit_cutoff,
    read_local_receipts,
)
from groundmark.eval.checker_metrics import checker_metrics, interval, ratio
from groundmark.eval.runner import ROOT
from groundmark.pricing import NANOS_PER_DOLLAR, PRICES
from groundmark.sources import private_json

EXPERIMENT = "groundmark-fusion-druid-dev"
LOCAL_COMMIT = "9679d3ddcaaf610ccaf751e225c19e80413d27eb"
MODELS = {
    "yaxili96/FactCG-DeBERTa-v3-Large": False,
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": True,
}
VARIANTS = {"raw": "raw_prediction", "quote_gated": "prediction"}


def recall_cutoff(scores: np.ndarray, gold_support: np.ndarray, target: float) -> float:
    positives = np.sort(scores[gold_support])[::-1]
    if not len(positives):
        return np.inf
    return float(positives[int(np.ceil(target * len(positives))) - 1])


def fuse(hosted: np.ndarray, local_supports: np.ndarray) -> np.ndarray:
    return np.where((hosted == 0) & ~local_supports, 2, hosted)


def pair_rates(groups: ClaimGroups, predictions: np.ndarray) -> dict[str, np.ndarray]:
    gold, support = groups.gold == 0, predictions == 0
    tp, fp = groups.sum(gold & support), groups.sum(~gold & support)
    pos, neg = groups.sum(gold), groups.sum(~gold)
    recall, fsr = ratio(tp, pos), ratio(fp, neg, zero=np.nan)
    return {
        "precision": ratio(tp, tp + fp),
        "recall": recall,
        "f1": ratio(2 * tp, pos + tp + fp),
        "balanced_accuracy": np.where(pos > 0, (recall + 1 - fsr) / 2, np.nan),
        "fsr": fsr,
    }


def scores(groups, aggregate_groups, predictions, complete) -> dict[str, np.ndarray]:
    return {
        **pair_rates(groups, predictions),
        **aggregate_groups.supported_rates(predictions[complete]),
    }


def summary(values, aggregate_groups, predictions, complete) -> dict:
    result = {name: interval(rates) for name, rates in values.items()}
    result["confusion_gold_rows"] = aggregate_groups.aggregate(predictions[complete], ternary=True)[
        "confusion_gold_rows"
    ]
    return result


def analyze_subset(hosted, local, resamples=RESAMPLES) -> dict:
    reference = next(iter(local.values()))
    observed = {row["row_id"]: row for row in hosted}
    ids = [row["row_id"] for row in reference]
    valid = np.array([key in observed for key in ids])
    rows = [observed[key] for key in ids if key in observed]
    sizes = Counter(row["group"] for row in reference)
    valid_sizes = Counter(row["group"] for row in rows)
    complete = np.array([sizes[row["group"]] == valid_sizes[row["group"]] for row in rows])
    eligible = ClaimGroups(reference, resamples=0)
    groups = ClaimGroups(rows, resamples=resamples)
    claim_groups = ClaimGroups(
        [row for row, kept in zip(rows, complete, strict=True) if kept], resamples
    )
    result = {
        "valid_pairs": len(rows),
        "pair_groups": len(groups.sizes),
        "complete_claim_groups": len(claim_groups.sizes),
        "local_detail": {},
        "hosted": {},
    }
    operating = {}
    for model, candidates in local.items():
        probabilities = np.array([row["support_score"] for row in candidates])
        operating[model] = {}
        for mode, targets, fit in [
            ("fsr", (0.10, 0.20, 0.30), fit_cutoff),
            ("recall", (0.90, 0.95), recall_cutoff),
        ]:
            for target, point in eligible.out_of_fold(probabilities, targets, fit=fit).items():
                operating[model][f"{mode}_{target:.2f}"] = point
        detail = [
            {**row, "support_score": None}
            for row, keep in zip(candidates, valid, strict=True)
            if keep
        ]
        result["local_detail"][model] = checker_metrics(detail, MODELS[model], resamples)
    for variant, column in VARIANTS.items():
        hosted_labels = np.array([row[column] for row in rows])
        baseline = scores(groups, claim_groups, hosted_labels, complete)
        result["hosted"][variant] = {
            "baseline": summary(baseline, claim_groups, hosted_labels, complete),
            "detail": checker_metrics(
                [{**row, "prediction": row[column]} for row in rows], True, resamples
            ),
            "fusion": {},
        }
        for model, settings in operating.items():
            result["hosted"][variant]["fusion"][model] = model_results = {}
            for setting, point in settings.items():
                fused = fuse(hosted_labels, point["supports"][valid])
                values = scores(groups, claim_groups, fused, complete)
                model_results[setting] = {
                    **summary(values, claim_groups, fused, complete),
                    "cutoffs": point["thresholds"],
                    "paired_fused_minus_hosted": {
                        name: interval(samples - baseline[name]) for name, samples in values.items()
                    },
                }
    return result


def cost_summary(directory: Path, receipt: dict) -> dict:
    records = [
        json.loads(path.read_bytes()) for path in sorted((directory / "pairs").glob("*.json"))
    ]
    usage = [
        attempt["usage"] for row in records for attempt in row["attempts"] if attempt.get("usage")
    ]
    attempts = sum(len(row["attempts"]) for row in records)
    if len(records) != receipt["rows_processed"] or len(usage) != attempts:
        return {"aggregate_usage": receipt["usage"], "per_pair_usage_complete": False}
    price = PRICES[receipt["model"]]
    values = {key: [] for key in ("input", "output", "reasoning", "usd")}
    for row in records:
        by_attempt = [attempt["usage"] for attempt in row["attempts"]]
        for key in ("input", "output", "reasoning"):
            values[key].append(sum(item[key] for item in by_attempt))
        values["usd"].append(
            sum(price.cost(u["input"], u["output"], u["cached_input"]) for u in by_attempt)
            / NANOS_PER_DOLLAR
        )
    distribution = {
        key: {
            "mean": float(np.mean(sample)),
            "p50": float(np.quantile(sample, 0.5)),
            "p95": float(np.quantile(sample, 0.95)),
            "max": max(sample),
        }
        for key, sample in values.items()
    }
    return {
        "attempted_pairs": len(records),
        "per_pair_usage_complete": True,
        "per_pair": distribution,
        "usd_per_question": {
            size: {key: size * distribution["usd"][key] for key in ("mean", "p95")}
            for size in (10, 25, 60)
        },
    }


def run_fusion(local_receipts: list[Path], hosted_receipt: Path, mlflow_uri: str) -> Path:
    receipts, local = read_local_receipts(local_receipts)
    hosted_meta = json.loads((hosted_receipt / "metrics.json").read_bytes())
    if set(local) != set(MODELS) or any(
        row["commit"] != LOCAL_COMMIT or row["split_sha256"] != hosted_meta["split_sha256"]
        for row in receipts.values()
    ):
        raise ValueError("Expected v2 DRUID DEV receipts for FactCG and NLI-Large")
    hosted = pq.read_table(hosted_receipt / "predictions.parquet").to_pylist()
    local_index = {row["row_id"]: row for row in next(iter(local.values()))}
    if (
        len(local_index) != 1631
        or len({row["row_id"] for row in hosted}) != 1630
        or len(hosted) != 1630
    ):
        raise ValueError("Expected 1,630 valid hosted pairs and 1,631 local DEV pairs")
    for row in hosted:
        ref = local_index.get(row["row_id"], {})
        if any(row[key] != ref.get(key) for key in ("group", "native_gold", "is_gold")):
            raise ValueError("Hosted and local DEV labels do not align")
    result = {
        "exploratory": True,
        "local_commit": LOCAL_COMMIT,
        "hosted_commit": hosted_meta["commit"],
        "split_sha256": hosted_meta["split_sha256"],
        "bootstrap_replicates": RESAMPLES,
        "selection": "Fixed checker-analyze group folds; valid pairs scored; incomplete groups excluded for aggregate metrics.",
    }
    for subset, is_gold in SUBSETS.items():
        result[subset] = analyze_subset(
            [row for row in hosted if row["is_gold"] == is_gold],
            {
                model: [row for row in rows if row["is_gold"] == is_gold]
                for model, rows in local.items()
            },
        )
    result["cost"] = cost_summary(hosted_receipt, hosted_meta)
    path = ROOT / "druid-dev-fusion" / uuid4().hex / "metrics.json"
    private_json(path, result)
    if mlflow_uri:
        from groundmark.eval.mlflow_log import log_metrics_artifact

        log_metrics_artifact(
            mlflow_uri,
            EXPERIMENT,
            path,
            {
                "exploratory": True,
                "local_commit": LOCAL_COMMIT,
                "split_sha256": result["split_sha256"],
            },
            {f"{subset}.valid_pairs": result[subset]["valid_pairs"] for subset in SUBSETS},
        )
    return path
