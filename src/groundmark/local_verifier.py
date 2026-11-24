import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from groundmark.models import Claim, EvidenceScope, PairJudgment, PairVerification, Source
from groundmark.nli import NLI_LABELS, NLI_MODELS, NLIModel
from groundmark.providers import ProviderFailure
from groundmark.runtime import Calls
from groundmark.verification import checked_pair

_MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def _load(model: str, cache: Path) -> NLIModel:
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model,
        revision=NLI_MODELS[model],
        cache_dir=str(cache),
        token=False,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer*",
            "spm.model",
            "*tokens*.json",
        ],
    )
    gpu = os.environ.get("GROUNDMARK_CONTAINER") != "1" and torch.cuda.is_available()
    return NLIModel(
        Path(path), device="cuda" if gpu else "cpu", dtype="float16" if gpu else "float32"
    )


def exact_spans(text: str) -> list[str]:
    spans = []
    start = 0
    while start < len(text):
        end = min(start + 300, len(text))
        span = text[start:end].strip()
        if span:
            spans.append(span)
        if end == len(text):
            break
        start += 240
    return spans


class LocalNLIVerifier:
    def __init__(self, calls: Calls, scorer: NLIModel | None = None):
        self.calls = calls
        self.scorer = scorer

    def verify(
        self, claims: list[Claim], sources: list[Source], question: str, today: str
    ) -> list[PairVerification]:
        return [
            pair
            for source in sources
            for pair in self.verify_source(claims, source, source.content)
            if pair.cited or pair.verdict != "not_enough_info"
        ]

    def verify_source(
        self,
        claims: list[Claim],
        source: Source,
        text: str,
        *,
        scope: EvidenceScope = "snippet",
        full_text: str | None = None,
        preview: str = "",
        question: str = "",
        today: str = "",
    ) -> list[PairVerification]:
        if not claims:
            return []
        evidence = full_text if full_text is not None else text
        spans = [span for span in exact_spans(text) if span in evidence]
        judgments = []
        try:
            with _MODEL_LOCK:
                scorer = self.scorer or _load(
                    self.calls.settings.resolved_checker_model,
                    Path(os.environ.get("HF_HOME", self.calls.settings.cache_dir / "huggingface"))
                    / "hub",
                )
                probabilities, _ = scorer.score([text] * len(claims), [c.text for c in claims])
                for claim, row in zip(claims, probabilities, strict=True):
                    label = max(range(len(NLI_LABELS)), key=row.__getitem__)
                    quote = ""
                    if label != 2 and spans:
                        local, _ = scorer.score(spans, [claim.text] * len(spans))
                        index = max(range(len(local)), key=lambda index: local[index][label])
                        quote = spans[index]
                    judgments.append(
                        PairJudgment(
                            claim_id=claim.claim_id,
                            verdict=NLI_LABELS[label],
                            evidence_quote=quote,
                            reason="",
                        )
                    )
        except (ImportError, LookupError, OSError, RuntimeError, ValueError):
            self.calls.error("verify", ProviderFailure("local-nli", "model_unavailable"))
            judgments = [
                PairJudgment(
                    claim_id=claim.claim_id,
                    verdict="not_enough_info",
                    evidence_quote="",
                    reason="Verification could not complete for this pair.",
                )
                for claim in claims
            ]
        return [
            checked_pair(
                judgment,
                source,
                evidence,
                scope,
                preview,
                supplied_text=text,
                require_quote=self.calls.settings.require_quote,
            ).model_copy(update={"cited": source.source_id in claim.source_ids})
            for judgment, claim in zip(judgments, claims, strict=True)
        ]
