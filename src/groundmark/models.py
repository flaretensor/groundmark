from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PairVerdict = Literal["supported", "not_supported", "contradicted"]
ClaimStatus = Literal["supported", "not_supported", "contradicted", "conflicting"]
QuoteMatch = Literal["exact", "normalized", "missing", "not_applicable"]
EvidenceScope = Literal["snippet", "page"]


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Plan(StrictOutput):
    queries: list[str]
    rationale: str


class Assessment(StrictOutput):
    decision: Literal["sufficient", "refine"]
    followup_query: str
    stop_reason: str


class Source(BaseModel):
    source_id: str
    url: str
    title: str
    content: str
    score: float | None = None
    published_date: str | None = None
    retrieved_at: str
    query: str
    round_index: int
    evidence_scope: EvidenceScope = "snippet"


class SearchRound(BaseModel):
    round_index: int
    queries: list[str]
    source_ids: list[str]
    cache_hits: int = 0


class Claim(StrictOutput):
    claim_id: str
    text: str
    source_ids: list[str]


class Conflict(StrictOutput):
    topic: str
    description: str
    source_ids: list[str]


class Synthesis(StrictOutput):
    summary: str
    claims: list[Claim]
    insufficient_evidence: bool
    missing_information: list[str]
    conflicts: list[Conflict]


class PairJudgment(StrictOutput):
    claim_id: str
    verdict: PairVerdict
    evidence_quote: str
    reason: str


class PairJudgments(StrictOutput):
    pairs: list[PairJudgment]


class PairVerification(BaseModel):
    claim_id: str
    source_id: str
    verdict: PairVerdict
    raw_verdict: PairVerdict
    quote_match: QuoteMatch
    evidence_quote: str
    reason: str
    evidence_scope: EvidenceScope = "snippet"
    evidence_preview: str = ""


class ClaimResult(BaseModel):
    claim: Claim
    status: ClaimStatus
    pairs: list[PairVerification] = Field(default_factory=list)
    reason: str = ""


class TokenUsage(BaseModel):
    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0


class ResearchError(BaseModel):
    kind: str
    stage: str
    provider: str | None = None
    http_status: int | None = None
    retried: bool = False


class SpanRecord(BaseModel):
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    duration_ms: float
    attributes: dict


class ResearchTrace(BaseModel):
    run_id: str
    model: str
    plan_rationale: str = ""
    rounds: list[SearchRound] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    stop_reason: str
    claim_status_counts: dict[str, int] = Field(default_factory=dict)
    pair_verdict_counts: dict[str, int] = Field(default_factory=dict)
    quote_match_counts: dict[str, int] = Field(default_factory=dict)
    stripped_source_ids: list[str] = Field(default_factory=list)
    synthesis_source_ids: list[str] = Field(default_factory=list)
    extraction_attempts: list[PairVerification] = Field(default_factory=list)
    extracted_source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float
    tokens: TokenUsage
    tavily_credits: int
    estimated_usd: float
    charged_or_reserved_tavily_credits: int
    charged_or_reserved_usd: float
    errors: list[ResearchError] = Field(default_factory=list)
    spans: list[SpanRecord] = Field(default_factory=list)


class ResearchResult(BaseModel):
    question: str
    summary: str
    insufficient_evidence: bool
    missing_information: list[str]
    claims: list[ClaimResult]
    conflicts: list[Conflict]
    sources: list[Source]
    trace: ResearchTrace


class Progress(BaseModel):
    node: str
    round: int
