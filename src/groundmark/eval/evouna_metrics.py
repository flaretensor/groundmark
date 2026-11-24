from collections import Counter

import numpy as np

from groundmark.eval.checker_metrics import interval, ratio
from groundmark.pricing import NANOS_PER_DOLLAR, Price

LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED", "PARSE_FAILURE")
DATASETS = ("NQ", "TQ")
STRATA = tuple((dataset, correct) for dataset in DATASETS for correct in (False, True))
RESAMPLES = 2_000


def usage_summary(records: list[dict], price: Price) -> dict:
    attempts = [attempt for row in records for attempt in row["attempts"]]
    reported = [attempt["usage"] for attempt in attempts if attempt.get("usage") is not None]
    tokens = {
        key: sum(usage[key] for usage in reported)
        for key in ("input", "cached_input", "output", "reasoning")
    }
    return {
        "calls": len(attempts),
        "retries": sum(max(0, len(row["attempts"]) - 1) for row in records),
        "tokens": tokens,
        "attempts_without_usage": len(attempts) - len(reported),
        "estimated_usd": price.cost(tokens["input"], tokens["output"], tokens["cached_input"])
        / NANOS_PER_DOLLAR,
        "http_errors": dict(
            Counter(attempt["http_status"] for attempt in attempts if attempt.get("http_status"))
        ),
        "attempt_errors": dict(
            Counter(attempt["error"] for attempt in attempts if "error" in attempt)
        ),
        "parse_failures": sum(row["grade"] == "PARSE_FAILURE" for row in records),
        "errors": dict(Counter(row["error"] for row in records if row.get("error"))),
    }


def bootstrap_counts(counts: np.ndarray, rng, resamples: int) -> np.ndarray:
    count = int(counts.sum())
    samples = (
        rng.multinomial(count, counts / count, size=resamples)
        if count
        else np.zeros((resamples, len(counts)), dtype=int)
    )
    return np.vstack((counts, samples))


def judge_metrics(
    records: list[dict], population: Counter, price: Price, resamples=RESAMPLES
) -> dict:
    graded = [row for row in records if row["grade"] in LABELS]
    counts = Counter((row["dataset"], row["human_correct"], row["grade"]) for row in graded)
    rng = np.random.default_rng(42)
    draws = {
        (dataset, human): bootstrap_counts(
            np.array([counts[dataset, human, grade] for grade in LABELS]), rng, resamples
        )
        for dataset, human in STRATA
    }
    observed = [sum(draws[dataset, human] for dataset in DATASETS) for human in (False, True)]
    incorrect, correct = observed
    neg, pos = incorrect.sum(axis=1), correct.sum(axis=1)
    fp, tp = incorrect[:, 0], correct[:, 0]
    n = pos + neg
    agreements = tp + neg - fp
    agreement = ratio(agreements, n, zero=np.nan)
    expected = ratio(pos * (tp + fp) + neg * (n - tp - fp), n**2, zero=np.nan)
    kappa = ratio(agreement - expected, 1 - expected, zero=np.nan)
    weighted = np.zeros(resamples + 1)
    for (dataset, human), cell in draws.items():
        size = population[dataset, human]
        if size:
            success = cell[:, 0] if human else cell[:, 1:].sum(axis=1)
            weighted += size * ratio(success, cell.sum(axis=1), zero=np.nan)
    if population.total():
        weighted /= population.total()
    else:
        weighted[:] = np.nan
    return {
        "n": len(graded),
        "human_incorrect": int(neg[0]),
        "human_correct": int(pos[0]),
        "eligible_incorrect": sum(
            count for (_, correct), count in population.items() if not correct
        ),
        "eligible_correct": sum(count for (_, correct), count in population.items() if correct),
        "false_correct_rate": interval(ratio(fp, neg, zero=np.nan)),
        "miss_rate": interval(ratio(pos - tp, pos, zero=np.nan)),
        "kappa": interval(kappa),
        "unweighted_agreement": interval(agreement),
        "reweighted_agreement_accuracy": interval(weighted),
        "human_incorrect_labels": {
            label: {
                "count": int(incorrect[0, i]),
                **interval(ratio(incorrect[:, i], neg, zero=np.nan)),
            }
            for i, label in enumerate(LABELS)
        },
        "usage": usage_summary(records, price),
    }


def disagreements(records: list[dict], templates: tuple[str, str], resamples=RESAMPLES) -> dict:
    by_pair = {}
    for row in records:
        if row["grade"] in LABELS:
            key = (row["pair_id"], row["dataset"], row["human_correct"])
            by_pair.setdefault(key, {})[row["grader_template"]] = row["grade"] == "CORRECT"
    counts = Counter()
    for (_, dataset, human), grades in by_pair.items():
        if all(template in grades for template in templates):
            counts[dataset, human, grades[templates[0]], grades[templates[1]]] += 1
    rng = np.random.default_rng(42)
    out = {}
    for human in (False, True):
        draws = sum(
            bootstrap_counts(
                np.array(
                    [counts[dataset, human, x, y] for x in (False, True) for y in (False, True)]
                ),
                rng,
                resamples,
            )
            for dataset in DATASETS
        )
        n = draws.sum(axis=1)
        different = draws[:, 1] + draws[:, 2]
        out["human_correct" if human else "human_incorrect"] = {
            "n": int(n[0]),
            "different": int(different[0]),
            "first_only_correct": int(draws[0, 2]),
            "second_only_correct": int(draws[0, 1]),
            "rate": interval(ratio(different, n, zero=np.nan)),
        }
    return out
