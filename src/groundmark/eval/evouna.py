import json
import os
import random
import subprocess
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path

from groundmark.budget import Amount, BudgetRefused, SpendLedger
from groundmark.config import Settings
from groundmark.eval.datasets import EVOUNA, EVOUNA_JUDGES, evouna_counts
from groundmark.eval.evouna_metrics import (
    DATASETS,
    STRATA,
    disagreements,
    judge_metrics,
    usage_summary,
)
from groundmark.eval.graders import Grader, dataset_graders
from groundmark.eval.hosted_checker import FATAL, CachedPair, digest, projected_nanos
from groundmark.eval.judge import SYSTEM, AnswerJudge, live_judge
from groundmark.eval.mlflow_log import log_metrics_artifact
from groundmark.eval.runner import COMBINED, ROOT, stage_lock
from groundmark.pricing import JUDGE_MODEL, NANOS_PER_DOLLAR, PRICES
from groundmark.providers import ProviderFailure
from groundmark.sources import private_json

STAGE = "judge-validation-evouna"
EXPERIMENT = "groundmark-judge-validation-evouna"
LIMIT = Amount(nanos=500_000_000)
PROJECTION_LIMIT = 450_000_000
CALL_LIMIT = 1500
SEED = 20261124
PREFLIGHT = 20
N_PER_STRATUM = 150
WORKERS = 4
PRICE = PRICES[JUDGE_MODEL]


@dataclass(frozen=True)
class Pair:
    dataset: str
    index: int
    system: str
    human_correct: bool
    question: str
    gold: str
    answer: str

    @property
    def id(self) -> str:
        return f"{self.dataset}:{self.index}:{self.system}"


def eligible_pairs(dataset: str, rows: list[dict]) -> list[Pair]:
    evouna_counts(rows)
    return [
        Pair(
            dataset,
            index,
            system,
            row[f"judge_{system}"],
            row["question"],
            row["golden_answer"],
            answer,
        )
        for index, row in enumerate(rows)
        if row["improper"] is False
        for system in EVOUNA_JUDGES
        if isinstance(answer := row[f"answer_{system}"], str)
        and answer.strip()
        and type(row[f"judge_{system}"]) is bool
    ]


def sample_groups(pairs: list[Pair]) -> dict[tuple[str, bool], list[Pair]]:
    questions = defaultdict(list)
    for pair in pairs:
        key = " ".join(unicodedata.normalize("NFKC", pair.question).casefold().split())
        questions[key].append(pair)
    rng = random.Random(SEED)
    strata = {key: [] for key in STRATA}
    for key in sorted(questions):
        pair = rng.choice(questions[key])
        strata[pair.dataset, pair.human_correct].append(pair)
    for group in strata.values():
        rng.shuffle(group)
    return strata


def selected_pairs(strata: dict, n: int) -> list[Pair]:
    # Round-robin across human-label and dataset strata keeps preflight inside every retained prefix.
    return [strata[key][i] for i in range(n) for key in STRATA if i < len(strata[key])]


def validation_ledger(root: Path, cap: Amount) -> SpendLedger:
    if not 0 < cap.nanos <= LIMIT.nanos or cap.credits:
        raise ValueError("EVOUNA requires a positive stage cap of at most $0.50 and zero credits")
    return SpendLedger(
        root / "spend_ledger.jsonl",
        COMBINED,
        scope=STAGE,
        scope_limit=cap,
        scope_call_limit=CALL_LIMIT,
    )


def grade_pair(
    pair: Pair,
    grader: Grader,
    directory: Path,
    judge: AnswerJudge,
    already_reserved: set[str],
) -> dict:
    key = digest(f"{pair.id}:{grader.name}")
    path = directory / "responses" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_bytes())
    run_id = f"{directory.name}:{key}"
    identity = {
        "pair_id": pair.id,
        "dataset": pair.dataset,
        "system": pair.system,
        "human_correct": pair.human_correct,
        "grader_template": grader.name,
        "grader_sha256": grader.sha256,
        "grade": None,
    }
    if run_id in already_reserved:
        return {**identity, "state": "pending", "attempts": []}
    cache = CachedPair(path, judge.llm, judge.credentials)
    cache.response.update(identity)
    private_json(path, cache.response)
    try:
        bounded = replace(judge, llm=cache, run_id=run_id, retry_statuses=frozenset({429, 529}))
        cache.response["grade"] = bounded.estimate(
            pair.question, pair.gold, pair.answer, grader=grader
        )
    except BudgetRefused as error:
        cache.response.update(error=f"budget_{error.kind}")
    except ProviderFailure as error:
        cache.response["error"] = error.kind
        if error.kind in {"invalid_output", "truncation"}:
            cache.response["grade"] = "PARSE_FAILURE"
    cache.response["state"] = "complete"
    private_json(path, cache.response)
    return cache.response


def plan_after_preflight(strata: dict, nanos: int, cap: Amount) -> int:
    def projection(n):
        return projected_nanos(nanos, PREFLIGHT, 2 * len(selected_pairs(strata, n)))

    ceiling = min(cap.nanos, PROJECTION_LIMIT)
    if projection(N_PER_STRATUM) <= ceiling:
        return N_PER_STRATUM
    return next((n for n in range(125, 49, -25) if projection(n) <= ceiling), 0)


def collect(
    strata: dict, graders: tuple[Grader, ...], directory: Path, judge: AnswerJudge
) -> tuple[list[dict], dict, int, str]:
    ledger = judge.ledger
    reserved = ledger.scope_run_ids()
    recorded = {}

    def batches(tasks):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while chunk := list(islice(tasks, WORKERS)):
                results = list(
                    pool.map(lambda task: grade_pair(*task, directory, judge, reserved), chunk)
                )
                for result in results:
                    recorded[result["pair_id"], result["grader_template"]] = result
                if any(result["state"] == "pending" for result in results):
                    return "previous_attempt_unconfirmed"
                fatal = next(
                    (result["error"] for result in results if result.get("error") in FATAL), None
                )
                if fatal:
                    return f"provider_{fatal}"
                if ledger.scope_blocked or any(
                    result.get("error", "").startswith("budget_") for result in results
                ):
                    return "budget"
        return "finished"

    full_sample = selected_pairs(strata, N_PER_STRATUM)
    if len(full_sample) < PREFLIGHT:
        return [], {}, 0, "too_few_preflight_pairs"
    stop = batches(iter((pair, graders[0]) for pair in full_sample[:PREFLIGHT]))
    preflight_path = directory / "preflight.json"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_bytes())
    else:
        usage = usage_summary(list(recorded.values()), PRICE)
        nanos = ledger.scope_total().nanos
        sampled = len(recorded)
        preflight = {
            "pairs": sampled,
            "template": graders[0].name,
            "usage": usage,
            "estimated_or_reserved_usd": nanos / NANOS_PER_DOLLAR,
            "projected_full_usd": projected_nanos(nanos, sampled, 2 * len(full_sample))
            / NANOS_PER_DOLLAR,
            "n_per_stratum": plan_after_preflight(strata, nanos, ledger.scope_limit)
            if stop == "finished"
            else 0,
        }
        private_json(preflight_path, preflight)
    n = preflight["n_per_stratum"]
    if stop != "finished" or n == 0:
        return (
            list(recorded.values()),
            preflight,
            n,
            stop if stop != "finished" else "preflight_projected_over_budget",
        )
    if not any(
        result["grade"] in {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"} for result in recorded.values()
    ):
        return list(recorded.values()), preflight, 0, "preflight_no_valid_grades"

    tasks = (
        (pair, grader)
        for pair in selected_pairs(strata, n)
        for grader in graders
        if (pair.id, grader.name) not in recorded
    )
    stop = batches(tasks)
    return list(recorded.values()), preflight, n, stop


def summarize(records, eligible, graders) -> dict:
    def stats(rows, universe):
        population = Counter((pair.dataset, pair.human_correct) for pair in universe)
        return judge_metrics(rows, population, PRICE)

    names = tuple(grader.name for grader in graders)
    report = {}
    for name in names:
        rows = [row for row in records if row["grader_template"] == name]
        report[name] = {
            "overall": stats(rows, eligible),
            "by_dataset": {
                dataset: stats(
                    [row for row in rows if row["dataset"] == dataset],
                    [pair for pair in eligible if pair.dataset == dataset],
                )
                for dataset in DATASETS
            },
            "by_system": {
                system: stats(
                    [row for row in rows if row["system"] == system],
                    [pair for pair in eligible if pair.system == system],
                )
                for system in EVOUNA_JUDGES
            },
        }
    return {"templates": report, "paired_disagreements": disagreements(records, names)}


def run_validation(cap: Amount, mlflow_uri: str, root: Path = ROOT) -> dict:
    ledger = validation_ledger(root, cap)
    eligible = [
        pair for spec in EVOUNA for pair in eligible_pairs(Path(spec.filename).stem, spec.read())
    ]
    strata = sample_groups(eligible)
    graders = dataset_graders("evouna")
    settings = Settings(_env_file=None, judge_model=JUDGE_MODEL, thinking_off_timeout=45)
    config = {
        "dataset_files": [spec.manifest() for spec in EVOUNA],
        "templates": {grader.name: grader.sha256 for grader in graders},
        "system_sha256": digest(SYSTEM),
        "judge_model": settings.judge_model,
        "seed": SEED,
        "stage_cap_nanos": cap.nanos,
        "call_cap": CALL_LIMIT,
        "preflight_pairs": PREFLIGHT,
        "initial_n": N_PER_STRATUM,
    }
    run_id = digest(json.dumps(config, sort_keys=True))[:32]
    base = root / STAGE
    directory = base / run_id
    with stage_lock(base):
        if any(path.is_dir() and path != directory for path in base.iterdir()):
            raise ValueError("An EVOUNA validation run already exists with different settings")
        private_json(directory / "config.json", config)
        os.environ["TAVILY_API_KEY"] = ""
        os.environ["GROUNDMARK_DISABLE_TAVILY"] = "1"
        with live_judge(settings, ledger) as judge:
            records, preflight, n, stop = collect(strata, graders, directory, judge)
        usage = {
            **usage_summary(records, PRICE),
            "ledger_reservations": ledger.scope_reservations(),
            "unsettled_reservations": ledger.scope_unsettled(),
            "estimated_or_reserved_usd": ledger.scope_total().nanos / NANOS_PER_DOLLAR,
            "tavily_credits": ledger.scope_total().credits,
        }
        result = {
            **config,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "preflight": preflight,
            "n_per_stratum": n,
            "strata": {
                f"{dataset}:{human}": {
                    "available": len(strata[dataset, human]),
                    "selected": min(n, len(strata[dataset, human])),
                    "shortfall": max(0, n - len(strata[dataset, human])),
                }
                for dataset, human in STRATA
            },
            "stop_reason": stop,
            "usage": usage,
            "metrics": summarize(records, eligible, graders),
        }
        path = directory / "metrics.json"
        private_json(path, result)
        if mlflow_uri:
            log_metrics_artifact(
                mlflow_uri,
                EXPERIMENT,
                path,
                {
                    **{f"grader.{grader.name}.sha256": grader.sha256 for grader in graders},
                    "model": settings.judge_model,
                    "n_per_stratum": n,
                    "commit": result["commit"],
                },
                {"calls": usage["calls"], "estimated_usd": usage["estimated_or_reserved_usd"]},
            )
    return {
        "path": str(path),
        "preflight": preflight,
        "n_per_stratum": n,
        "stop_reason": stop,
        "usage": usage,
    }
