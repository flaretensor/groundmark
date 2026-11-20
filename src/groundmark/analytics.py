import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from groundmark.models import ResearchResult

RUN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("finished_at", pa.string()),
        ("status", pa.string()),
        ("rounds", pa.int32()),
        ("latency_ms", pa.float64()),
        ("tavily_credits", pa.int32()),
        ("estimated_usd", pa.float64()),
    ]
)
CLAIM_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("claim_id", pa.string()),
        ("status", pa.string()),
        ("quote_verified", pa.int32()),
        ("quote_attempted", pa.int32()),
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


def append_result(directory: Path, result: ResearchResult) -> None:
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
        }
        for row in result.claims
    ]
    with TemporaryDirectory(prefix=".pending-", dir=directory) as temp:
        staged = Path(temp)
        pq.write_table(pa.Table.from_pylist([run], schema=RUN_SCHEMA), staged / "run.parquet")
        pq.write_table(pa.Table.from_pylist(claims, schema=CLAIM_SCHEMA), staged / "claims.parquet")
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
        connection.read_parquet([str(path) for path in runs]).create_view("runs")
        connection.read_parquet([str(path) for path in claims]).create_view("claims")
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
                "SELECT status, COUNT(*) AS count FROM claims GROUP BY status ORDER BY status",
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
