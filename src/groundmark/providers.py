import json
from dataclasses import dataclass
from typing import Protocol

import openai
import requests
from pydantic import BaseModel
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeout

from groundmark.models import TokenUsage
from groundmark.pricing import EMBEDDING_MODEL, JUDGE_MODEL

NEBIUS_URL = "https://api.tokenfactory.nebius.com/v1/"


class ProviderFailure(Exception):
    def __init__(self, provider: str, kind: str, status: int | None = None):
        super().__init__(kind)
        self.provider = provider
        self.kind = kind
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.kind in {"rate_limit", "overloaded", "timeout"}

    @classmethod
    def http(cls, provider: str, status: int) -> "ProviderFailure":
        kinds = {
            401: "auth",
            403: "forbidden",
            429: "rate_limit",
            432: "credits",
            433: "paid_balance",
            434: "unsupported_fetch",
            529: "overloaded",
        }
        return cls(provider, kinds.get(status, "server" if status >= 500 else "request"), status)


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    system: str
    prompt: str
    schema: type[BaseModel]
    thinking: bool
    max_tokens: int
    temperature: float
    timeout: float

    def response_format(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": self.schema.__name__,
                "schema": self.schema.model_json_schema(),
            },
        }

    def estimated_input(self) -> int:
        # UTF-8 byte count plus room for the provider's message and schema templates.
        return 1024 + len(
            json.dumps(
                [self.system, self.prompt, self.response_format()], ensure_ascii=False
            ).encode()
        )


@dataclass(frozen=True)
class Completion:
    content: str
    finish_reason: str
    usage: TokenUsage | None


@dataclass(frozen=True)
class SearchResponse:
    results: list[dict]
    credits: int | None = None
    raw: dict | None = None


@dataclass(frozen=True)
class ExtractionResponse:
    pages: list[dict]
    failures: list[dict]
    credits: int | None = None
    raw: dict | None = None


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    usage: TokenUsage | None


class LLMClient(Protocol):
    def complete(self, request: CompletionRequest) -> Completion: ...


class SearchClient(Protocol):
    def search(self, query: str, timeout: float) -> SearchResponse: ...

    def extract(self, urls: list[str], timeout: float) -> ExtractionResponse: ...


class Embedder(Protocol):
    def embed(self, texts: list[str], timeout: float) -> EmbeddingResponse: ...


class NebiusClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=NEBIUS_URL, max_retries=0)

    def complete(self, request: CompletionRequest) -> Completion:
        thinking = {"enable_thinking": request.thinking}
        if request.thinking:
            # Keep reasoning enabled while reserving room for the structured response.
            thinking["medium_effort"] = True
        extra = None if request.model == JUDGE_MODEL else {"chat_template_kwargs": thinking}
        try:
            response = self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.prompt},
                ],
                response_format=request.response_format(),
                max_completion_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout,
                extra_body=extra,
            )
        except openai.APITimeoutError:
            raise ProviderFailure("nebius", "timeout") from None
        except openai.APIStatusError as error:
            raise ProviderFailure.http("nebius", error.status_code) from None
        except openai.APIError:
            raise ProviderFailure("nebius", "connection") from None

        usage = None
        if response.usage is not None:
            details = response.usage
            usage = TokenUsage(
                input=details.prompt_tokens,
                cached_input=getattr(details.prompt_tokens_details, "cached_tokens", None) or 0,
                output=details.completion_tokens,
                reasoning=getattr(details.completion_tokens_details, "reasoning_tokens", None) or 0,
            )
        if not response.choices:
            return Completion("", "empty", usage)
        choice = response.choices[0]
        return Completion(choice.message.content or "", choice.finish_reason, usage)

    def embed(self, texts: list[str], timeout: float) -> EmbeddingResponse:
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL, input=texts, encoding_format="float", timeout=timeout
            )
        except openai.APITimeoutError:
            raise ProviderFailure("nebius", "timeout") from None
        except openai.APIStatusError as error:
            raise ProviderFailure.http("nebius", error.status_code) from None
        except openai.APIError:
            raise ProviderFailure("nebius", "connection") from None
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ProviderFailure("nebius", "invalid_embeddings")
        return EmbeddingResponse(
            [item.embedding for item in ordered], TokenUsage(input=response.usage.prompt_tokens)
        )

    def close(self) -> None:
        self.client.close()


class TavilySearchClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.hooks["response"].append(self._check_status)
        self.client = TavilyClient(api_key=api_key, session=self.session)

    @staticmethod
    def _check_status(response, **_):
        if response.status_code >= 400:
            response.close()
            raise ProviderFailure.http("tavily", response.status_code)
        return response

    @staticmethod
    def _credits(response: dict) -> int | None:
        usage = response.get("usage")
        credits = usage.get("credits") if isinstance(usage, dict) else None
        return credits if type(credits) is int and credits >= 0 else None

    def search(self, query: str, timeout: float) -> SearchResponse:
        try:
            data = self.client.search(
                query,
                search_depth="basic",
                max_results=5,
                chunks_per_source=3,
                include_usage=True,
                include_published_date=True,
                auto_parameters=False,
                timeout=timeout,
            )
        except (TavilyTimeout, requests.Timeout):
            raise ProviderFailure("tavily", "timeout") from None
        except requests.RequestException:
            raise ProviderFailure("tavily", "connection") from None
        return SearchResponse(data["results"], self._credits(data), raw=data)

    def extract(self, urls: list[str], timeout: float) -> ExtractionResponse:
        try:
            data = self.client.extract(
                urls, extract_depth="basic", include_usage=True, format="text", timeout=timeout
            )
        except (TavilyTimeout, requests.Timeout):
            raise ProviderFailure("tavily", "timeout") from None
        except requests.RequestException:
            raise ProviderFailure("tavily", "connection") from None
        return ExtractionResponse(
            data["results"], data["failed_results"], self._credits(data), raw=data
        )

    def close(self) -> None:
        self.session.close()
