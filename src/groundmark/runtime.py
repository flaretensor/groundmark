import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from time import sleep
from typing import TypeVar

from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ValidationError

from groundmark.budget import Amount, BudgetRefused, SpendLedger
from groundmark.config import Credentials, Settings
from groundmark.models import Progress, ResearchError, TokenUsage
from groundmark.pricing import EMBEDDING_MODEL, PRICES
from groundmark.providers import (
    CompletionRequest,
    Embedder,
    ExtractionResponse,
    LLMClient,
    ProviderFailure,
    SearchClient,
    SearchResponse,
)
from groundmark.telemetry import RunTelemetry

T = TypeVar("T")
Output = TypeVar("Output", bound=BaseModel)


@dataclass
class Calls:
    settings: Settings
    credentials: Credentials
    ledger: SpendLedger
    run_id: str
    limit: Amount
    telemetry: RunTelemetry
    llm: LLMClient
    search_client: SearchClient
    embedder: Embedder | None = None
    progress: Callable[[Progress], None] | None = None
    errors: list[ResearchError] = field(default_factory=list)
    tavily_receipts: list[dict] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    spent: Amount = field(default_factory=Amount)
    credits_stopped: bool = False
    usd_stopped: bool = False
    lock: Lock = field(default_factory=Lock)

    @property
    def budget_stopped(self) -> bool:
        return self.credits_stopped or self.usd_stopped

    def error(self, stage: str, error: BudgetRefused | ProviderFailure, retried: bool = False):
        with self.lock:
            if isinstance(error, BudgetRefused):
                if error.kind == "credits":
                    self.credits_stopped = True
                else:
                    self.usd_stopped = True
                entry = ResearchError(kind="budget", stage=stage, budget_kind=error.kind)
            else:
                entry = ResearchError(
                    kind=error.kind,
                    stage=stage,
                    provider=error.provider,
                    http_status=error.status,
                    retried=retried,
                )
            self.errors.append(entry)

    def _add_usage(self, amount: Amount, usage: TokenUsage | None):
        with self.lock:
            self.spent += amount
            if usage is not None:
                self.tokens.input += usage.input
                self.tokens.cached_input += usage.cached_input
                self.tokens.output += usage.output
                self.tokens.reasoning += usage.reasoning

    def _request(
        self,
        operation: str,
        stage: str,
        provider: str,
        worst_case: Amount,
        execute: Callable[[], T],
        usage_of: Callable[[T], tuple[Amount, TokenUsage | None, str | None]],
        model: str | None = None,
    ) -> T:
        for attempt in (1, 2):
            with self.telemetry.span(
                f"{provider}.{operation}",
                **{
                    "gen_ai.system": provider,
                    "groundmark.node": stage,
                    "groundmark.run_id": self.run_id,
                    "groundmark.attempt": attempt,
                },
            ) as span:
                if model:
                    span.set_attribute("gen_ai.request.model", model)
                try:
                    if self.usd_stopped or (provider == "tavily" and self.credits_stopped):
                        raise BudgetRefused(kind="usd" if self.usd_stopped else "credits")
                    reservation = self.ledger.reserve(
                        self.run_id, operation, worst_case, self.limit
                    )
                    response = execute()
                    actual, usage, finish = usage_of(response)
                    if actual.exceeds(worst_case):
                        span.set_attribute("groundmark.reservation_underestimated", True)
                    self.ledger.settle(self.run_id, reservation, actual)
                    self._add_usage(actual, usage)
                    if finish:
                        span.set_attribute("gen_ai.response.finish_reasons", (finish,))
                    if usage is not None:
                        span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": usage.input,
                                "gen_ai.usage.output_tokens": usage.output,
                                "groundmark.reasoning_tokens": usage.reasoning,
                                "groundmark.cached_tokens": usage.cached_input,
                            }
                        )
                    span.set_attributes(
                        {
                            "groundmark.tavily_credits": actual.credits,
                            "groundmark.estimated_usd": actual.nanos / 1e9,
                        }
                    )
                    return response
                except (ProviderFailure, BudgetRefused) as error:
                    retry = isinstance(error, ProviderFailure) and error.retryable and attempt == 1
                    self.error(stage, error, retried=retry)
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute(
                        "error.type", error.kind if isinstance(error, ProviderFailure) else "budget"
                    )
                    if not retry:
                        raise
            sleep(0.5)
        raise AssertionError("Unreachable request retry state")

    def structured(
        self,
        stage: str,
        schema: type[Output],
        system: str,
        prompt: str,
        *,
        thinking: bool = False,
        temperature: float = 0.1,
        model: str | None = None,
    ) -> Output:
        request = CompletionRequest(
            model=model or self.settings.model,
            system=system,
            prompt=prompt,
            schema=schema,
            thinking=thinking,
            max_tokens=8192 if thinking else 1024,
            temperature=temperature,
            timeout=(
                self.settings.thinking_on_timeout
                if thinking
                else self.settings.thinking_off_timeout
            ),
        )
        price = PRICES[request.model]
        worst_case = Amount(nanos=price.cost(request.estimated_input(), request.max_tokens))

        def usage_of(response):
            usage = response.usage
            actual = (
                Amount(nanos=price.cost(usage.input, usage.output, usage.cached_input))
                if usage is not None
                else worst_case
            )
            return actual, usage, response.finish_reason

        response = self._request(
            "completion",
            stage,
            "nebius",
            worst_case,
            lambda: self.llm.complete(request),
            usage_of,
            request.model,
        )
        if response.finish_reason == "length":
            error = ProviderFailure("nebius", "truncation")
            self.error(stage, error)
            raise error
        try:
            return schema.model_validate_json(self.credentials.redact(response.content))
        except ValidationError:
            error = ProviderFailure("nebius", "invalid_output")
            self.error(stage, error)
            raise error from None

    def _tavily_receipt(self, operation: str, request: dict, response: dict) -> None:
        def redact(value):
            if isinstance(value, str):
                return self.credentials.redact(value)
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        self.tavily_receipts.append(
            redact(
                {
                    "operation": operation,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "request": request,
                    "response": response,
                }
            )
        )

    def search(self, query: str) -> SearchResponse:
        filters = (
            {"exclude_domains": self.settings.exclude_domains}
            if self.settings.exclude_domains
            else {}
        )
        response = self._request(
            "search",
            "search",
            "tavily",
            Amount(credits=1),
            lambda: self.search_client.search(query, self.settings.tavily_timeout, **filters),
            lambda response: (
                Amount(credits=1 if response.credits is None else response.credits),
                None,
                None,
            ),
        )

        self._tavily_receipt(
            "search",
            {"query": query, **filters},
            response.raw
            or {
                "results": response.results,
                "credits": response.credits,
            },
        )
        return response

    def extract(self, urls: list[str]) -> ExtractionResponse:
        def usage_of(response):
            credits = response.credits
            if credits is None:
                credits = (len(response.pages) + 4) // 5
            return Amount(credits=credits), None, None

        response = self._request(
            "extract",
            "extract",
            "tavily",
            Amount(credits=(len(urls) + 4) // 5),
            lambda: self.search_client.extract(urls, self.settings.extract_timeout),
            usage_of,
        )

        self._tavily_receipt(
            "extract",
            {"urls": urls},
            response.raw
            or {
                "results": response.pages,
                "failed_results": response.failures,
                "credits": response.credits,
            },
        )
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is None:
            raise ProviderFailure("nebius", "embedding_unavailable")
        price = PRICES[EMBEDDING_MODEL]
        estimated = 1024 + len(json.dumps(texts, ensure_ascii=False).encode())
        worst_case = Amount(nanos=price.cost(estimated, 0))

        def usage_of(response):
            usage = response.usage
            actual = Amount(nanos=price.cost(usage.input, 0)) if usage else worst_case
            return actual, usage, None

        response = self._request(
            "embedding",
            "extract",
            "nebius",
            worst_case,
            lambda: self.embedder.embed(texts, self.settings.thinking_off_timeout),
            usage_of,
            EMBEDDING_MODEL,
        )
        return response.vectors
