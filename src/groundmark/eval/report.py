import json
from collections import Counter
from pathlib import Path
from uuid import UUID

import numpy as np
from pyarrow.parquet import ParquetFile

from groundmark.eval.runner import ARMS, ROOT, stage_ledger
from groundmark.sources import private_json

EXPECTED = {"smoke": 10, "dev": 50, "test": 210}
PAIRS = (("D", "A"), ("D", "C"), ("B", "A"))
LATENCY_LABELS = {
    "cumulative_batch_ms": "Time since batch start",
    "processing_ms": "Per-question eval processing",
    "total_latency_ms": "Product trace latency",
}


def status(arm: dict) -> str:
    if arm["abstain"]:
        return "abstain"
    return {
        "CORRECT": "correct",
        "INCORRECT": "incorrect",
        "NOT_ATTEMPTED": "judge_not_attempted",
    }.get(arm["grade"], "ungraded")


def _paired(rows, left: str, right: str) -> dict:
    differences = []
    for row in rows:
        a, b = (status(row["arms"][key]) for key in (left, right))
        if "ungraded" not in (a, b):
            differences.append(int(a == "correct") - int(b == "correct"))
    if not differences:
        return {"n": 0, "delta": None, "ci95": None}
    values = np.asarray(differences)
    rng = np.random.default_rng(42)
    indices = rng.integers(len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": len(values),
        "delta": float(values.mean()),
        "ci95": np.quantile(means, (0.025, 0.975)).tolist(),
    }


def _latency_quantiles(rows: list[dict], field: str) -> dict:
    values = [value for row in rows if (value := row.get(field)) is not None]
    quantiles = np.quantile(values, (0.5, 0.95)).tolist() if values else (None, None)
    return dict(zip(("p50", "p95"), quantiles, strict=True))


def _recorded_product_latency(root: Path, run_id: str | None) -> float | None:
    if run_id is None:
        return None
    try:
        identifier = UUID(run_id)
    except ValueError:
        return None
    path = root.parent / "runs" / str(identifier) / "run.parquet"
    if not path.is_file():
        return None
    receipt = ParquetFile(path)
    if "latency_ms" in receipt.schema_arrow.names and receipt.metadata.num_rows:
        # The product run receipt stores trace.total_latency_ms as latency_ms.
        return receipt.read(columns=["latency_ms"])[0][0].as_py()
    return None


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    by_arm = {}
    for name in ARMS:
        arms = [row["arms"][name] for row in rows]
        counts = Counter(status(arm) for arm in arms)
        counts = {
            key: counts[key]
            for key in (
                "correct",
                "incorrect",
                "abstain",
                "judge_not_attempted",
                "ungraded",
            )
        }
        answered = total - counts["abstain"]
        graded = answered - counts["ungraded"]
        accuracy = counts["correct"] / answered if answered and answered == graded else None
        by_arm[name] = {
            "counts": counts,
            "rates": {key: count / total if total else None for key, count in counts.items()},
            "accuracy_among_answered": accuracy,
            "accuracy_among_graded_answers": counts["correct"] / graded if graded else None,
            "risk_coverage": {
                "risk": 1 - accuracy if accuracy is not None else None,
                "coverage": answered / total if total else None,
            },
            "flagged_sources": sum(len(arm["flagged_sources"]) for arm in arms),
            "flagged_questions": sum(bool(arm["flagged_sources"]) for arm in arms),
            "stop_reasons": dict(Counter(arm["stop_reason"] for arm in arms)),
        }
    return {
        "questions": total,
        "arms": by_arm,
        "paired": {f"{a}-{b}": _paired(rows, a, b) for a, b in PAIRS},
        # Eval rows store time since batch start in latency_ms.
        "cumulative_batch_ms": _latency_quantiles(rows, "latency_ms"),
        # Per-question wall time includes extra A/B processing and grading when run inline.
        "processing_ms": _latency_quantiles(rows, "processing_ms"),
        "total_latency_ms": _latency_quantiles(rows, "total_latency_ms"),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
    }


def report(stage: str, root: Path = ROOT, mlflow_uri: str = "") -> dict:
    paths = sorted((root / stage).glob("*.json"))
    rows = [json.loads(path.read_bytes()) for path in paths]
    for row in rows:
        if row.get("total_latency_ms") is None:
            row["total_latency_ms"] = _recorded_product_latency(root, row.get("run_id"))
    ledger_path = root / "spend_ledger.jsonl"
    used = stage_ledger(root, stage).scope_total() if ledger_path.is_file() else None
    summary = {
        **summarize(rows),
        "stage": stage,
        "expected_questions": EXPECTED[stage],
        "grader_templates": {
            arm["grader_template"]: arm["grader_sha256"]
            for row in rows
            for arm in row["arms"].values()
            if "grader_template" in arm
        },
        "seal_0": summarize([row for row in rows if row["seal_0"]]),
        "datasets": {
            dataset: summarize([row for row in rows if row["dataset"] == dataset])
            for dataset in ("simpleqa", "sealqa")
        },
        "tavily_credits_charged_or_reserved": used.credits if used else 0,
        "nebius_usd_estimated_or_reserved": used.nanos / 1e9 if used else 0,
        "agent_tavily_credits_charged_or_reserved": sum(
            row["tavily_credits_charged_or_reserved"] for row in rows
        ),
        "agent_nebius_usd_estimated_or_reserved": sum(
            row["nebius_nanos_charged_or_reserved"] for row in rows
        )
        / 1e9,
    }
    if mlflow_uri:
        log_mlflow(summary, root, mlflow_uri)
    return summary


def latency_markdown(summary: dict) -> str:
    lines = [
        "| Scope | Latency measure | p50 (ms) | p95 (ms) |",
        "|---|---|---:|---:|",
    ]
    scopes = {
        "All questions": summary,
        "SimpleQA Verified": summary["datasets"]["simpleqa"],
        "Seal-Hard": summary["datasets"]["sealqa"],
        "Seal-0": summary["seal_0"],
    }
    for scope, results in scopes.items():
        for field, label in LATENCY_LABELS.items():
            p50, p95 = (
                f"{value:.2f}" if value is not None else "not recorded"
                for value in (results[field]["p50"], results[field]["p95"])
            )
            lines.append(f"| {scope} | {label} | {p50} | {p95} |")
    return "\n".join(lines)


def log_mlflow(summary: dict, root: Path, uri: str) -> None:
    from mlflow import MlflowClient

    path = root / "reports" / f"{summary['stage']}.json"
    private_json(path, summary)
    client = MlflowClient(tracking_uri=uri)
    name = f"groundmark-eval-{summary['stage']}"
    experiment = client.get_experiment_by_name(name)
    identifier = experiment.experiment_id if experiment else client.create_experiment(name)
    run_id = client.create_run(identifier).info.run_id
    try:
        for template, sha256 in summary["grader_templates"].items():
            client.log_param(run_id, f"grader.{template}.sha256", sha256)
        for key in (
            "questions",
            "tavily_credits_charged_or_reserved",
            "nebius_usd_estimated_or_reserved",
        ):
            client.log_metric(run_id, key, summary[key])
        for arm, result in summary["arms"].items():
            for key, numerator in result["rates"].items():
                if numerator is not None:
                    client.log_metric(run_id, f"{arm}.{key}_rate", numerator)
        for field in LATENCY_LABELS:
            for quantile, latency in summary[field].items():
                if latency is not None:
                    client.log_metric(run_id, f"{field.removesuffix('_ms')}_{quantile}_ms", latency)
        for comparison, result in summary["paired"].items():
            if result["delta"] is not None:
                client.log_metric(run_id, f"{comparison}.delta", result["delta"])
        client.log_artifact(run_id, str(path))
        client.set_terminated(run_id)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
