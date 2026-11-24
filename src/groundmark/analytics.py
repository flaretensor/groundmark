import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from groundmark.config import Settings
from groundmark.models import ResearchResult
from groundmark.nli import NLI_MODELS
from groundmark.pricing import EMBEDDING_MODEL
from groundmark.sources import private_json

RUN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("finished_at", pa.string()),
        ("status", pa.string()),
        ("rounds", pa.int32()),
        ("latency_ms", pa.float64()),
        ("tavily_credits", pa.int32()),
        ("estimated_usd", pa.float64()),
        ("question", pa.string()),
        ("question_id", pa.string()),
        ("arm", pa.string()),
        ("git_commit_sha", pa.string()),
        ("config_sha256", pa.string()),
        ("models", pa.string()),
        (
            "sources",
            pa.list_(
                pa.struct(
                    [
                        ("source_id", pa.string()),
                        ("url", pa.string()),
                        ("content_sha256", pa.string()),
                        ("retrieved_at", pa.string()),
                        ("page_part_sha256s", pa.list_(pa.string())),
                        ("page_retrieved_at", pa.string()),
                    ]
                )
            ),
        ),
        ("action_trajectory", pa.string()),
        ("raw_tavily_artifact", pa.string()),
    ]
)
CLAIM_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("claim_id", pa.string()),
        ("status", pa.string()),
        ("quote_verified", pa.int32()),
        ("quote_attempted", pa.int32()),
        ("claim_text", pa.string()),
        (
            "pair_verdicts",
            pa.list_(
                pa.struct(
                    [
                        ("source_id", pa.string()),
                        ("verdict", pa.string()),
                        ("raw_verdict", pa.string()),
                        ("quote_match", pa.string()),
                        ("evidence_quote", pa.string()),
                        ("evidence_scope", pa.string()),
                        ("cited", pa.bool_()),
                        ("error_code", pa.string()),
                    ]
                )
            ),
        ),
    ]
)


def run_status(result: ResearchResult) -> str:
    if result.trace.stop_reason == "budget":
        return "budget"
    if result.trace.stop_reason == "provider_error" and not result.summary:
        return "provider_error"
    if result.insufficient_evidence:
        return "partial" if result.summary else "insufficient"
    return "complete"


@dataclass(frozen=True)
class RunProvenance:
    question_id: str | None = None
    arm: str | None = None
    git_commit_sha: str | None = None
    config_sha256: str | None = None
    models: dict = field(default_factory=dict)
    tavily_receipts: list[dict] = field(default_factory=list)


def provenance(
    settings: Settings,
    receipts: list[dict],
    question_id: str | None,
    arm: str | None,
    options: dict,
) -> RunProvenance:
    commit = settings.git_commit_sha
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3
            ).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
    if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    configuration = settings.model_dump(
        mode="json", exclude={"checkpoint_url", "api_url", "otlp_endpoint", "mlflow_tracking_uri"}
    )
    digest = hashlib.sha256(
        json.dumps({"settings": configuration, "options": options}, sort_keys=True).encode()
    ).hexdigest()
    return RunProvenance(
        question_id,
        arm,
        commit,
        digest,
        {
            "agent": settings.model,
            "checker": settings.resolved_checker_model,
            "checker_type": settings.checker,
            "checker_revision": NLI_MODELS.get(settings.resolved_checker_model),
            "embedding": EMBEDDING_MODEL,
            "judge": settings.judge_model,
            "reranker": settings.reranker,
        },
        receipts,
    )


def append_result(
    directory: Path, result: ResearchResult, metadata: RunProvenance | None = None
) -> None:
    metadata = metadata or RunProvenance()
    trace = result.trace
    run_id = str(UUID(trace.run_id))
    destination = directory / run_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_dir():
        return

    run = {
        "run_id": run_id,
        "finished_at": datetime.now(UTC).isoformat(),
        "status": run_status(result),
        "rounds": len(trace.rounds),
        "latency_ms": trace.total_latency_ms,
        "tavily_credits": trace.tavily_credits,
        "estimated_usd": trace.estimated_usd,
        "question": result.question,
        "question_id": metadata.question_id,
        "arm": metadata.arm,
        "git_commit_sha": metadata.git_commit_sha,
        "config_sha256": metadata.config_sha256,
        "models": json.dumps(metadata.models, sort_keys=True),
        "sources": [
            {
                "source_id": source.source_id,
                "url": source.url,
                "content_sha256": hashlib.sha256(source.content.encode()).hexdigest(),
                "retrieved_at": source.retrieved_at,
                "page_part_sha256s": [
                    hashlib.sha256(part.encode()).hexdigest() for part in source.page_parts
                ],
                "page_retrieved_at": source.page_retrieved_at,
            }
            for source in result.sources
        ],
        "action_trajectory": json.dumps(
            [step.model_dump(mode="json") for step in trace.trajectory]
        ),
        "raw_tavily_artifact": "raw_tavily.json",
    }
    claims = [
        {
            "run_id": run_id,
            "claim_id": row.claim.claim_id,
            "status": row.status,
            "quote_verified": sum(
                pair.quote_match in {"exact", "normalized"} for pair in row.pairs
            ),
            "quote_attempted": sum(pair.quote_match != "not_applicable" for pair in row.pairs),
            "claim_text": row.claim.text,
            "pair_verdicts": [pair.model_dump(mode="json") for pair in row.pairs],
        }
        for row in result.claims
    ]
    with TemporaryDirectory(prefix=".pending-", dir=directory) as temp:
        staged = Path(temp)
        pq.write_table(pa.Table.from_pylist([run], schema=RUN_SCHEMA), staged / "run.parquet")
        pq.write_table(pa.Table.from_pylist(claims, schema=CLAIM_SCHEMA), staged / "claims.parquet")
        private_json(staged / "raw_tavily.json", metadata.tavily_receipts)
        for path in staged.iterdir():
            path.chmod(0o600)
        try:
            os.rename(staged, destination)
        except OSError:
            if not destination.is_dir():
                raise


def _rows(connection, query: str) -> list[dict]:
    response = connection.execute(query)
    columns = [column[0] for column in response.description]
    return [dict(zip(columns, row, strict=True)) for row in response.fetchall()]


def report(directory: Path) -> dict:
    runs = sorted(directory.glob("*/run.parquet"))
    claims = sorted(directory.glob("*/claims.parquet"))
    if not runs:
        return {"run_count": 0, "run_statuses": [], "claim_statuses": [], "per_run": []}
    with duckdb.connect(":memory:") as connection:
        connection.read_parquet([str(path) for path in runs], union_by_name=True).create_view(
            "runs"
        )
        connection.read_parquet([str(path) for path in claims], union_by_name=True).create_view(
            "claims"
        )
        headline = _rows(
            connection,
            """
            SELECT COUNT(*) AS run_count,
                   AVG(rounds) AS rounds_mean,
                   quantile_cont(latency_ms, 0.5) AS latency_p50_ms,
                   quantile_cont(latency_ms, 0.95) AS latency_p95_ms,
                   SUM(tavily_credits) AS tavily_credits,
                   SUM(estimated_usd) AS estimated_usd
            FROM runs
            """,
        )[0]
        return {
            **headline,
            "run_statuses": _rows(
                connection,
                "SELECT status, COUNT(*) AS count FROM runs GROUP BY status ORDER BY status",
            ),
            "claim_statuses": _rows(
                connection,
                """
                SELECT CASE claims.status
                         WHEN 'contradicted' THEN 'refuted'
                         WHEN 'not_supported' THEN 'not_enough_evidence'
                         ELSE status
                       END AS status, COUNT(*) AS count
                FROM claims GROUP BY 1 ORDER BY 1
                """,
            ),
            "quote_verification_rate": connection.execute(
                "SELECT SUM(quote_verified)::DOUBLE / NULLIF(SUM(quote_attempted), 0) FROM claims"
            ).fetchone()[0],
            "per_run": _rows(
                connection,
                """SELECT run_id, status, rounds, latency_ms, tavily_credits, estimated_usd
                   FROM runs ORDER BY finished_at, run_id""",
            ),
        }
