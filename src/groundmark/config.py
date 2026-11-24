import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from groundmark.nli import NLI_MODELS
from groundmark.pricing import COMPLETION_MODELS, JUDGE_MODEL, MODEL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROUNDMARK_",
        env_file=".env",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    model: str = Field(MODEL, validation_alias="NEBIUS_MODEL")
    judge_model: str = JUDGE_MODEL
    checker: Literal["nebius", "local-nli"] = "nebius"
    checker_model: str | None = None
    max_steps: Annotated[int, Field(ge=1)] = 6
    max_searches: Annotated[int, Field(ge=1)] = Field(
        3, validation_alias=AliasChoices("GROUNDMARK_MAX_SEARCHES", "GROUNDMARK_MAX_ROUNDS")
    )
    max_extract_calls: Annotated[int, Field(ge=0)] = 1
    max_tavily_credits: Annotated[int, Field(ge=0)] = 6
    max_usd: Annotated[float, Field(ge=0)] = 0.15
    max_total_tavily_credits: Annotated[int, Field(ge=0)] = 1000
    max_total_usd: Annotated[float, Field(ge=0)] = 5.0
    thinking_on_timeout: Annotated[float, Field(gt=0)] = 90
    thinking_off_timeout: Annotated[float, Field(gt=0)] = 45
    tavily_timeout: Annotated[float, Field(gt=0, le=120)] = 30
    extract_timeout: Annotated[float, Field(gt=0, le=120)] = 30
    checkpoint_url: str = Field(".cache/checkpoints.sqlite", repr=False)
    cache_dir: Path = Path(".cache")
    artifacts_dir: Path = Path(".artifacts")
    use_cache: bool = False
    require_quote: bool = True
    reranker: Literal["cross-encoder", "none"] = "none"
    api_url: str = ""
    otlp_endpoint: str = Field("", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    mlflow_tracking_uri: str = Field("", validation_alias="MLFLOW_TRACKING_URI")

    @model_validator(mode="after")
    def known_models(self) -> "Settings":
        if self.model not in COMPLETION_MODELS or self.judge_model not in COMPLETION_MODELS:
            raise ValueError("Unregistered completion model")
        candidates = NLI_MODELS if self.checker == "local-nli" else COMPLETION_MODELS
        if self.resolved_checker_model not in candidates:
            raise ValueError("Unregistered checker model")
        return self

    @property
    def resolved_checker_model(self) -> str:
        default = next(reversed(NLI_MODELS)) if self.checker == "local-nli" else self.model
        return self.checker_model or default

    @property
    def ledger_path(self) -> Path:
        return self.artifacts_dir / "spend_ledger.jsonl"


class Credentials:
    def __init__(self, nebius: str, tavily: str):
        self.nebius = SecretStr(nebius)
        self.tavily = SecretStr(tavily)

    @classmethod
    def load(cls) -> "Credentials":
        values = (
            os.environ
            if os.environ.get("GROUNDMARK_CONTAINER") == "1"
            else dotenv_values(".env", interpolate=False)
        )
        return cls(values.get("NEBIUS_API_KEY") or "", values.get("TAVILY_API_KEY") or "")

    def redact(self, text: str) -> str:
        for secret in (self.nebius, self.tavily):
            value = secret.get_secret_value()
            if value:
                text = text.replace(value, "[redacted]")
        return text
