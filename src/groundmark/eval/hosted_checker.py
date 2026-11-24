import hashlib
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from groundmark.budget import Amount, SpendLedger
from groundmark.config import Credentials, Settings
from groundmark.eval.checkers import druid_dev, write_predictions
from groundmark.eval.runner import COMBINED, ROOT, stage_lock
from groundmark.models import Claim, PairJudgments, Source, TokenUsage
from groundmark.nli import NLI_LABELS
from groundmark.pricing import MODEL, NANOS_PER_DOLLAR, PRICES
from groundmark.providers import (
    Completion,
    CompletionRequest,
    LLMClient,
    NebiusClient,
    ProviderFailure,
)
from groundmark.runtime import Calls
from groundmark.sources import private_json
from groundmark.telemetry import RunTelemetry
from groundmark.verification import SYSTEM, NebiusVerifier

STAGE = "druid-hosted-dev"
EXPERIMENT = "groundmark-hosted-verifier-druid-dev"
LIMIT = Amount(nanos=NANOS_PER_DOLLAR)
CALL_LIMIT = 2_000
PREFLIGHT = 20
WORKERS = 4
FATAL = {"auth", "forbidden", "paid_balance", "credits"}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def claim_date(value: str) -> str:
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return ""


class CachedPair:
    def __init__(self, path: Path, llm: LLMClient, credentials: Credentials):
        self.path = path
        self.llm = llm
        self.credentials = credentials
        self.response: dict = {"state": "pending", "attempts": []}

    def complete(self, request: CompletionRequest) -> Completion:
        attempt = {"estimated_input": request.estimated_input(), "max_output": request.max_tokens}
        self.response["attempts"].append(attempt)
        private_json(self.path, self.response)
        try:
            completion = self.llm.complete(request)
        except ProviderFailure as error:
            attempt.update(error=error.kind, http_status=error.status)
            raise
        else:
            usage = completion.usage
            attempt.update(
                content=self.credentials.redact(completion.content),
                finish_reason=completion.finish_reason,
                usage=usage.model_dump() if usage is not None else None,
            )
            return completion
        finally:
            private_json(self.path, self.response)


def evaluate_pair(
    row: dict,
    path: Path,
    row_run: str,
    llm: LLMClient,
    credentials: Credentials,
    settings: Settings,
    ledger: SpendLedger,
) -> dict:
    cache = CachedPair(path, llm, credentials)
    if path.exists():
        return json.loads(path.read_bytes())
    today = claim_date(row["claim_date"])
    cache.response.update(row_id=row["id"], group=row["group"], question="", today=today)
    private_json(path, cache.response)

    text = credentials.redact(row["evidence"])
    claim = Claim(claim_id="C1", text=credentials.redact(row["claim"]), source_ids=["S1"])
    source = Source(
        source_id="S1", url="", title="", content=text, retrieved_at="", query="", round_index=0
    )
    telemetry = RunTelemetry("")
    try:
        calls = Calls(
            settings=settings,
            credentials=credentials,
            ledger=ledger,
            run_id=row_run,
            limit=LIMIT,
            telemetry=telemetry,
            llm=cache,
            search_client=None,
        )
        pair = NebiusVerifier(calls).verify_source([claim], source, text, question="", today=today)[
            0
        ]
        errors = [error.model_dump(exclude_none=True) for error in calls.errors]
        attempts = cache.response["attempts"]
        valid = False
        if (
            attempts
            and "content" in attempts[-1]
            and not any(
                error["kind"] in {"budget", "invalid_output", "truncation"} for error in errors
            )
        ):
            answer = PairJudgments.model_validate_json(attempts[-1]["content"])
            valid = sum(judgment.claim_id == "C1" for judgment in answer.pairs) == 1
            if not valid:
                errors.append({"kind": "missing_or_duplicate_claim_id"})
        cache.response.update(
            state="completed", pair=pair.model_dump() if valid else None, errors=errors
        )
        private_json(path, cache.response)
        return cache.response
    finally:
        telemetry.close()


def projected_nanos(observed_nanos: int, sampled: int, total: int) -> int:
    return (observed_nanos * total + sampled - 1) // sampled


def _fatal(records: list[dict]) -> str | None:
    if any(record.get("state") == "pending" for record in records):
        return "previous_attempt_unconfirmed"
    for record in records:
        for error in record.get("errors", []):
            if error["kind"] in FATAL:
                return "provider_" + error["kind"]
    return None


def collect(
    rows: list[dict],
    directory: Path,
    ledger: SpendLedger,
    llm: LLMClient,
    credentials: Credentials,
    settings: Settings,
) -> tuple[list[dict], dict, str]:
    ordered = list(rows)
    random.Random(42).shuffle(ordered)
    pairs = directory / "pairs"
    previously_reserved = ledger.scope_run_ids()
    projection_file = directory / "preflight.json"
    projection = {}

    def run(row: dict) -> dict:
        key = digest(row["id"])
        path = pairs / f"{key}.json"
        row_run = f"{directory.name}:{key}"
        if row_run in previously_reserved and not path.exists():
            return {"state": "pending", "row_id": row["id"]}
        return evaluate_pair(row, path, row_run, llm, credentials, settings, ledger)

    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index in range(0, len(ordered), WORKERS):
            records.extend(pool.map(run, ordered[index : index + WORKERS]))
            reason = _fatal(records[-WORKERS:])
            if reason:
                return records, projection, reason
            if ledger.scope_blocked:
                return records, projection, "budget"
            if len(records) == PREFLIGHT:
                if projection_file.exists():
                    projection = json.loads(projection_file.read_bytes())
                else:
                    used = ledger.scope_total().nanos
                    projection = {
                        "sampled": PREFLIGHT,
                        "valid": sum(record["pair"] is not None for record in records),
                        "attempts": ledger.scope_reservations(),
                        "charged_or_reserved_nanos": used,
                        "projected_nanos": projected_nanos(used, PREFLIGHT, len(ordered)),
                    }
                    private_json(projection_file, projection)
                if projection["projected_nanos"] > LIMIT.nanos:
                    return records, projection, "preflight_projected_over_budget"
                if not projection["valid"]:
                    return records, projection, "preflight_no_valid_verdicts"
    return records, projection, "finished"


def _usage(records: list[dict], ledger: SpendLedger) -> dict:
    from collections import Counter

    price = PRICES[MODEL]
    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    counts = TokenUsage()
    charged = 0
    for attempt in attempts:
        if attempt.get("usage") is None:
            continue
        usage = TokenUsage.model_validate(attempt["usage"])
        counts.input += usage.input
        counts.cached_input += usage.cached_input
        counts.output += usage.output
        counts.reasoning += usage.reasoning
        charged += price.cost(usage.input, usage.output, usage.cached_input)
    return {
        "calls": ledger.scope_reservations(),
        "recorded_attempts": len(attempts),
        "retries": sum(max(0, len(record.get("attempts", [])) - 1) for record in records),
        "reported_usage": counts.model_dump(),
        "attempts_without_reported_usage": sum(a.get("usage") is None for a in attempts),
        "estimated_usd_from_reported_usage": charged / NANOS_PER_DOLLAR,
        "estimated_usd_including_unsettled_reservations": ledger.scope_total().nanos
        / NANOS_PER_DOLLAR,
        "errors": dict(
            Counter(error["kind"] for record in records for error in record.get("errors", []))
        ),
        "http_error_codes": dict(
            Counter(
                str(attempt["http_status"]) for attempt in attempts if attempt.get("http_status")
            )
        ),
    }


def _prediction_rows(rows: list[dict], records: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in rows}
    predictions = []
    for record in records:
        if (pair := record.get("pair")) is None:
            continue
        row = by_id[record["row_id"]]
        predictions.append(
            {
                "row_id": row["id"],
                "group": row["group"],
                "native_gold": row["evidence_stance"],
                "is_gold": row["is_gold"] == "True",
                "prediction": NLI_LABELS.index(pair["verdict"]),
                "raw_prediction": NLI_LABELS.index(pair["raw_verdict"]),
                "quote": pair["evidence_quote"],
                "quote_match": pair["quote_match"],
                "question": record["question"],
                "date": record["today"],
            }
        )
    return predictions


def _mlflow(uri: str, result: dict, path: Path) -> None:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    run_id = client.create_run(experiment_id, run_name=path.parent.name).info.run_id
    try:
        for key in ("model", "commit", "split_sha256", "prompt_sha256", "stop_reason"):
            client.log_param(run_id, key, result[key])
        client.log_metric(run_id, "calls", result["usage"]["calls"])
        client.log_metric(run_id, "valid_pairs", result["valid_pairs"])
        client.log_metric(
            run_id,
            "estimated_usd",
            result["usage"]["estimated_usd_including_unsettled_reservations"],
        )
        client.log_artifact(run_id, str(path))
        client.set_terminated(run_id)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise


def run_hosted(local_receipts: list[Path], mlflow_uri: str) -> dict:
    from groundmark.eval.checker_analysis import hosted_analysis, read_local_receipts

    os.environ["TAVILY_API_KEY"] = ""
    os.environ["GROUNDMARK_DISABLE_TAVILY"] = "1"
    rows, manifest, split_sha = druid_dev()
    receipts, local = read_local_receipts(local_receipts)
    if any(receipt["split_sha256"] != split_sha for receipt in receipts.values()):
        raise ValueError("Local receipts must use the same DRUID DEV split")
    expected = {
        (row["id"], row["group"], row["evidence_stance"], row["is_gold"] == "True") for row in rows
    }
    if len(rows) != 1_631 or any(
        {(row["row_id"], row["group"], row["native_gold"], row["is_gold"]) for row in batch}
        != expected
        for batch in local.values()
    ):
        raise ValueError("Expected aligned local predictions for all 1,631 DRUID DEV pairs")

    settings = Settings(
        _env_file=None,
        NEBIUS_MODEL=MODEL,
        checker_model=MODEL,
        checker="nebius",
        require_quote=True,
        thinking_on_timeout=90,
    )
    ledger = SpendLedger(
        ROOT / "spend_ledger.jsonl",
        COMBINED,
        scope=STAGE,
        scope_limit=LIMIT,
        scope_call_limit=CALL_LIMIT,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {
        "dataset": manifest["dataset"],
        "split_sha256": split_sha,
        "model": MODEL,
        "prompt_sha256": digest(SYSTEM),
        "question": "Empty string for every row",
        "current_date": "ISO DRUID claim_date (fact-check publication date), or empty if invalid",
        "temperature": 0.2,
        "thinking": True,
        "max_completion_tokens": 8192,
        "timeout_seconds": settings.thinking_on_timeout,
        "preflight_seed": 42,
    }
    run_id = digest(json.dumps(config, sort_keys=True))[:32]
    base = ROOT / "druid-dev-hosted"
    directory = base / run_id
    with stage_lock(base):
        if any(path.is_dir() and path != directory for path in base.iterdir()):
            raise ValueError("Hosted DEV already has a run under a different configuration")
        private_json(directory / "config.json", config)
        credentials = Credentials.load(nebius_only=True)
        key = credentials.nebius.get_secret_value()
        if not key:
            raise ValueError("Missing Nebius credentials")
        client = NebiusClient(key)
        try:
            records, projection, stop = collect(
                rows, directory, ledger, client, credentials, settings
            )
        finally:
            client.close()
        predictions = _prediction_rows(rows, records)
        if predictions:
            write_predictions(directory / "predictions.parquet", predictions)
        result = {
            **config,
            "commit": commit,
            "stop_reason": stop,
            "preflight": projection,
            "eligible": len(rows),
            "valid_pairs": len(predictions),
            "rows_processed": len(records),
            "usage": _usage(records, ledger),
            "metrics": hosted_analysis(predictions, local),
        }
        private_json(directory / "metrics.json", result)
        if mlflow_uri:
            _mlflow(mlflow_uri, result, directory / "metrics.json")
        return {
            "path": str(directory),
            "stop_reason": stop,
            "preflight": projection,
            "valid_pairs": len(predictions),
            "usage": result["usage"],
        }
