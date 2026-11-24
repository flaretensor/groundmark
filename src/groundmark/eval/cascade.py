import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from groundmark.config import Credentials
from groundmark.eval.checker_analysis import RESAMPLES, SUBSETS, ClaimGroups
from groundmark.eval.checker_metrics import interval, ratio
from groundmark.eval.fusion import (
    LOCAL_COMMIT,
    align_subset,
    fuse,
    read_fusion_inputs,
    recall_cutoff,
    scores,
)
from groundmark.eval.mlflow_log import log_metrics_artifact
from groundmark.eval.runner import ROOT
from groundmark.pricing import NANOS_PER_DOLLAR, PRICES
from groundmark.providers import NebiusClient, TavilySearchClient
from groundmark.sources import private_json

EXPERIMENT = "groundmark-cascade-druid-dev"


def cached_usage(directory: Path, hosted: list[dict], model: str) -> tuple[dict, dict | str]:
    remaining = {row["row_id"] for row in hosted}
    costs, latencies = {}, []
    price = PRICES[model]
    for path in (directory / "pairs").glob("*.json"):
        record = json.loads(path.read_bytes())
        key = record["row_id"]
        if key in costs:
            raise ValueError("Duplicate cached hosted pair")
        if key not in remaining:
            continue
        attempts = record["attempts"]
        if not attempts or any(not attempt.get("usage") for attempt in attempts):
            raise ValueError("Incomplete usage for a hosted-valid pair")
        costs[key] = (
            sum(
                price.cost(usage["input"], usage["output"], usage["cached_input"])
                for usage in (attempt["usage"] for attempt in attempts)
            )
            / NANOS_PER_DOLLAR
        )
        latencies.extend(attempt.get("latency_ms") for attempt in attempts)
    if costs.keys() != remaining:
        raise ValueError("Missing usage for a hosted-valid pair")
    latency = "not recorded"
    if latencies and all(value is not None for value in latencies):
        latency = dict(
            zip(("p50", "p95"), np.quantile(latencies, (0.5, 0.95)).tolist(), strict=True)
        )
    return costs, latency


def group_p95(groups: ClaimGroups, values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = groups.weights[:, groups.group_index[order]].cumsum(axis=1)
    # Interpolate between the two ranks in each bootstrap sample's expanded pair list.
    rank = 0.95 * (cumulative[:, -1] - 1)
    low = np.floor(rank)
    left = sorted_values[(cumulative > low[:, None]).argmax(axis=1)]
    right = sorted_values[(cumulative > np.ceil(rank)[:, None]).argmax(axis=1)]
    return left + (right - left) * (rank - low)


def variant_samples(groups, claim_groups, complete, hosted, labels, called, cached_cost) -> dict:
    pair_count = groups.weights @ groups.sizes
    costs = np.where(called, cached_cost, 0.0)
    usd_mean = groups.sum(costs) / pair_count
    usd_p95 = group_p95(groups, costs)
    return {
        **scores(groups, claim_groups, labels, complete),
        "refutes_recall": ratio(
            groups.sum((groups.gold == 1) & (labels == 1)),
            groups.sum(groups.gold == 1),
            zero=np.nan,
        ),
        "hosted_refutes_lost": groups.sum((hosted == 1) & (labels != 1)),
        "skipped_fraction": groups.sum(~called) / pair_count,
        "usd_per_pair_mean": usd_mean,
        "usd_per_pair_p95": usd_p95,
        "usd_per_25_pairs_mean": 25 * usd_mean,
        "usd_per_25_pairs_p95": 25 * usd_p95,
    }


def analyze_subset(hosted, local, costs, resamples=RESAMPLES) -> dict:
    reference, observed, valid, complete = align_subset(hosted, local)
    eligible = ClaimGroups(reference, resamples=0)
    groups = ClaimGroups(observed, resamples)
    claim_groups = ClaimGroups(
        [row for row, keep in zip(observed, complete, strict=True) if keep], resamples
    )
    labels = np.array([row["prediction"] for row in observed])
    cached_cost = np.array([costs[row["row_id"]] for row in observed])
    all_called = np.ones(len(observed), dtype=bool)
    hosted_non_nei = np.zeros(len(reference), dtype=bool)
    hosted_non_nei[valid] = labels != 2
    baseline = variant_samples(
        groups, claim_groups, complete, labels, labels, all_called, cached_cost
    )

    def record(variant, model, setting, predictions, called, thresholds):
        values = variant_samples(
            groups, claim_groups, complete, labels, predictions, called, cached_cost
        )
        return {
            "variant": variant,
            "local_model": model,
            "setting": setting,
            "fold_thresholds": thresholds,
            "hosted_calls_saved": int((~called).sum()),
            "metrics": {name: interval(sample) for name, sample in values.items()},
            "paired_minus_v1": {
                name: interval(sample - baseline[name]) for name, sample in values.items()
            },
        }

    variants = [record("V1", None, None, labels, all_called, [])]
    for model, candidates in local.items():
        probabilities = np.array([row["support_score"] for row in candidates])
        for target, point in eligible.out_of_fold(probabilities, (0.10, 0.20), valid=valid).items():
            variants.append(
                record(
                    "V2",
                    model,
                    f"fsr_{target:.2f}",
                    np.where(point["supports"][valid], 0, 2),
                    ~all_called,
                    point["thresholds"],
                )
            )
        for target, point in eligible.out_of_fold(
            probabilities,
            (0.95, 0.98, 0.99),
            fit=recall_cutoff,
            positive=hosted_non_nei,
            valid=valid,
        ).items():
            called = point["supports"][valid]
            variants.append(
                record(
                    "V3",
                    model,
                    f"retain_{target:.2f}",
                    np.where(called, labels, 2),
                    called,
                    point["thresholds"],
                )
            )
        point = eligible.out_of_fold(probabilities, (0.95,), fit=recall_cutoff)[0.95]
        variants.append(
            record(
                "V4",
                model,
                "recall_0.95",
                fuse(labels, point["supports"][valid]),
                all_called,
                point["thresholds"],
            )
        )
    return {
        "valid_pairs": len(observed),
        "pair_groups": len(groups.sizes),
        "complete_claim_groups": len(claim_groups.sizes),
        "variants": variants,
    }


def run_cascade(local_receipts: list[Path], hosted_receipt: Path, mlflow_uri: str) -> Path:
    forbidden = RuntimeError("Provider access is forbidden during stored cascade analysis")
    with (
        patch.object(Credentials, "load", side_effect=forbidden),
        patch.object(NebiusClient, "__init__", side_effect=forbidden),
        patch.object(TavilySearchClient, "__init__", side_effect=forbidden),
    ):
        receipts, local, hosted_meta, hosted = read_fusion_inputs(local_receipts, hosted_receipt)
        costs, latency = cached_usage(hosted_receipt, hosted, hosted_meta["model"])
        result = {
            "local_commit": LOCAL_COMMIT,
            "hosted_commit": hosted_meta["commit"],
            "hosted_run_id": hosted_receipt.name,
            "split_sha256": hosted_meta["split_sha256"],
            "bootstrap_replicates": RESAMPLES,
            "seed": 42,
            "estimate": "Cached quote-gated replay. Skipped pairs cost zero. Per-25 values scale pair statistics.",
            "hosted_latency_ms": latency,
            "local_performance": {
                model: {
                    key: receipt[key]
                    for key in (
                        "batch_size",
                        "dtype",
                        "pairs_per_s",
                        "batch_latency_ms",
                        "peak_allocated_bytes",
                        "nvidia_smi_peak_mib",
                    )
                }
                for model, receipt in receipts.items()
            },
        }
        for subset, is_gold in SUBSETS.items():
            result[subset] = analyze_subset(
                [row for row in hosted if row["is_gold"] == is_gold],
                {
                    model: [row for row in rows if row["is_gold"] == is_gold]
                    for model, rows in local.items()
                },
                costs,
            )
        path = ROOT / "druid-dev-cascade" / uuid4().hex / "metrics.json"
        private_json(path, result)
        if mlflow_uri:
            log_metrics_artifact(
                mlflow_uri,
                EXPERIMENT,
                path,
                {"local_commit": LOCAL_COMMIT, "split_sha256": result["split_sha256"], "seed": 42},
                {f"{subset}.valid_pairs": result[subset]["valid_pairs"] for subset in SUBSETS},
            )
    return path
