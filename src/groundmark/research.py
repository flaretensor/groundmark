from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

import httpx
from langsmith import tracing_context
from pydantic import BaseModel, Field

from groundmark.analytics import append_result, provenance
from groundmark.budget import BudgetRefused, SpendLedger, usd_amount
from groundmark.checkpoints import build_checkpointer
from groundmark.config import Credentials, Settings
from groundmark.evidence import extract_evidence
from groundmark.graph import GraphContext, State, build_graph, finalize
from groundmark.local_verifier import LocalNLIVerifier
from groundmark.models import Progress, ResearchResult
from groundmark.providers import Embedder, LLMClient, NebiusClient, SearchClient, TavilySearchClient
from groundmark.runtime import Calls
from groundmark.sources import clean_text
from groundmark.telemetry import RunTelemetry, mlflow_headers
from groundmark.verification import NebiusVerifier, Verifier

EXAMPLES = (
    "Who won the 2026 Nobel Prize in Physics, and for what?",
    "What is the latest stable release of Python?",
    "What revenue did Apple report for its fiscal 2027 first quarter?",
)


class Options(BaseModel):
    max_rounds: Annotated[int | None, Field(ge=1)] = None
    max_credits: Annotated[int | None, Field(ge=0)] = None
    max_cost: Annotated[float | None, Field(ge=0)] = None
    use_cache: bool | None = None


class ResearchFailure(Exception):
    pass


class ResearchRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        credentials: Credentials | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        embedder: Embedder | None = None,
        verifier_factory: Callable[[Calls], Verifier] | None = None,
        extract_node=extract_evidence,
        ledger: SpendLedger | None = None,
    ):
        self.settings = settings or Settings()
        self.credentials = credentials
        self.llm = llm
        self.search = search
        self.embedder = embedder
        self.verifier_factory = verifier_factory or (
            LocalNLIVerifier if self.settings.checker == "local-nli" else NebiusVerifier
        )
        self.extract_node = extract_node
        self.ledger = ledger

    def run(
        self,
        question: str,
        options: Options | None = None,
        progress: Callable[[Progress], None] | None = None,
        *,
        eval_question_id: str | None = None,
        eval_arm: str | None = None,
        eval_hook: Callable[[State, GraphContext], None] | None = None,
    ) -> ResearchResult:
        if len(question) > 500 or not question.strip():
            raise ResearchFailure("invalid_question")

        settings = self.settings
        options = options or Options()
        live = self.llm is None or self.search is None
        credentials = self.credentials or (Credentials.load() if live else Credentials("", ""))
        question = clean_text(credentials.redact(question.strip())).strip()
        if not question:
            raise ResearchFailure("invalid_question")
        if self.llm is None and not credentials.nebius.get_secret_value():
            raise ResearchFailure("configuration")
        if self.search is None and not credentials.tavily.get_secret_value():
            raise ResearchFailure("configuration")

        credits = settings.max_tavily_credits
        if options.max_credits is not None:
            credits = min(credits, options.max_credits)
        usd = (
            settings.max_usd
            if options.max_cost is None
            else min(settings.max_usd, options.max_cost)
        )
        limit = usd_amount(credits, usd)
        ledger = self.ledger or SpendLedger(
            settings.ledger_path,
            usd_amount(settings.max_total_tavily_credits, settings.max_total_usd),
        )
        run_id = str(uuid4())
        nebius = NebiusClient(credentials.nebius.get_secret_value()) if self.llm is None else None
        tavily = (
            TavilySearchClient(credentials.tavily.get_secret_value())
            if self.search is None
            else None
        )
        endpoint = settings.otlp_endpoint
        headers = None
        otlp_note = ""
        if endpoint and settings.mlflow_tracking_uri:
            try:
                headers = mlflow_headers(settings.mlflow_tracking_uri)
            except (httpx.HTTPError, KeyError, ValueError):
                endpoint = ""
                otlp_note = (
                    "The MLflow experiment could not be resolved. The local trace was retained."
                )
        telemetry = RunTelemetry(endpoint, headers)
        calls = Calls(
            settings=settings,
            credentials=credentials,
            ledger=ledger,
            run_id=run_id,
            limit=limit,
            telemetry=telemetry,
            llm=self.llm or nebius,
            search_client=self.search or tavily,
            embedder=self.embedder or nebius,
            progress=progress,
        )
        context = GraphContext(
            calls,
            self.verifier_factory(calls),
            min(options.max_rounds or settings.max_searches, settings.max_searches),
            settings.use_cache if options.use_cache is None else options.use_cache,
            today=datetime.now(UTC).date().isoformat(),
        )
        try:
            result = self._run_graph(question, context, live, eval_hook)
            if otlp_note:
                result.trace.notes.append(otlp_note)
            try:
                metadata = provenance(
                    settings,
                    calls.tavily_receipts,
                    eval_question_id,
                    eval_arm,
                    options.model_dump(mode="json"),
                )
                append_result(settings.artifacts_dir / "runs", result, metadata)
            except OSError:
                result.trace.notes.append("Run analytics could not be written.")
            return result
        finally:
            telemetry.close()
            if nebius is not None:
                nebius.close()
            if tavily is not None:
                tavily.close()

    def _run_graph(
        self,
        question: str,
        context: GraphContext,
        live: bool,
        eval_hook: Callable[[State, GraphContext], None] | None = None,
    ) -> ResearchResult:
        calls = context.calls
        telemetry = calls.telemetry
        config = {
            "configurable": {"thread_id": calls.run_id},
            "recursion_limit": 2 * self.settings.max_steps + 10,
        }
        with (
            tracing_context(enabled=False),
            build_checkpointer(self.settings.checkpoint_url) as saver,
        ):
            graph = build_graph(saver, self.extract_node)
            with telemetry.span(
                "groundmark.research", **{"groundmark.run_id": calls.run_id}
            ) as root:
                try:
                    if live:
                        calls.ledger.start(calls.run_id)
                except BudgetRefused as error:
                    calls.error("plan", error)
                    result = finalize({"question": question}, context)["result"]
                    persisted = False
                else:
                    result = None
                    for update in graph.stream(
                        {"question": question}, config, context=context, stream_mode="updates"
                    ):
                        if "finalize" in update:
                            result = update["finalize"]["result"]
                    if result is None:
                        raise ResearchFailure("incomplete")
                    persisted = True
                root.set_attribute("groundmark.stop_reason", result["trace"]["stop_reason"])
                if eval_hook:
                    state = graph.get_state(config).values if persisted else {"question": question}
                    eval_hook({**state, "result": result}, context)
            elapsed, spans = telemetry.collect()
            result["trace"].update(
                total_latency_ms=elapsed,
                stage_latency_ms=context.stage_latency_ms,
                spans=[span.model_dump(mode="json") for span in spans],
            )
            if persisted:
                graph.update_state(config, {"result": result}, as_node="finalize")
        return ResearchResult.model_validate(result)
