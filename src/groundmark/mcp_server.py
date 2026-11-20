from typing import Annotated, TypedDict

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from groundmark.models import ClaimStatus, ConflictResult, PairVerdict, QuoteMatch
from groundmark.research import Options, ResearchFailure, ResearchRunner


class ToolQuote(TypedDict):
    source_id: str
    verdict: PairVerdict
    quote: str
    quote_match: QuoteMatch


class ToolClaim(TypedDict):
    id: str
    text: str
    status: ClaimStatus
    quotes: list[ToolQuote]


class ToolSource(TypedDict):
    id: str
    title: str
    url: str


class ToolCost(TypedDict):
    tavily_credits: int
    estimated_usd: float


class ToolResult(TypedDict):
    summary: str
    insufficient_evidence: bool
    missing_information: list[str]
    claims: list[ToolClaim]
    conflicts: list[ConflictResult]
    sources: list[ToolSource]
    cost: ToolCost


def build_server(runner: ResearchRunner | None = None) -> FastMCP:
    server = FastMCP("Groundmark")
    runner = runner or ResearchRunner()

    @server.tool()
    async def research(
        question: Annotated[str, Field(min_length=1, max_length=500)],
        max_rounds: Annotated[int, Field(ge=1, le=2)] = 2,
    ) -> ToolResult:
        try:
            result = await to_thread.run_sync(runner.run, question, Options(max_rounds=max_rounds))
        except ResearchFailure as error:
            raise ToolError(str(error)) from None
        except Exception:
            raise ToolError("internal") from None
        return {
            "summary": result.summary,
            "insufficient_evidence": result.insufficient_evidence,
            "missing_information": result.missing_information,
            "claims": [
                {
                    "id": row.claim.claim_id,
                    "text": row.claim.text,
                    "status": row.status,
                    "quotes": [
                        {
                            "source_id": pair.source_id,
                            "verdict": pair.verdict,
                            "quote": pair.evidence_quote,
                            "quote_match": pair.quote_match,
                        }
                        for pair in row.pairs
                        if pair.evidence_quote
                    ],
                }
                for row in result.claims
            ],
            "conflicts": result.conflicts,
            "sources": [
                {"id": source.source_id, "title": source.title, "url": source.url}
                for source in result.sources
            ],
            "cost": {
                "tavily_credits": result.trace.tavily_credits,
                "estimated_usd": result.trace.estimated_usd,
            },
        }

    return server


def main() -> None:
    build_server().run(transport="stdio")
