import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from math import fsum, sqrt
from pathlib import Path
from typing import Protocol

from groundmark.budget import BudgetRefused
from groundmark.graph import GraphContext, State
from groundmark.models import Claim, ClaimResult, PairVerification, Source
from groundmark.providers import ProviderFailure
from groundmark.sources import canonical_url, private_json, text_digest
from groundmark.verification import aggregate

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str


class Reranker(Protocol):
    def score(self, claim: str, chunks: list[Chunk]) -> list[float]: ...


def chunk_page(text: str, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks = []
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        chunks.append(Chunk(start, end, text[start:end]))
        if end == len(text):
            break
    return chunks


def cosine(left: list[float], right: list[float]) -> float:
    scale = sqrt(fsum(x * x for x in left) * fsum(y * y for y in right))
    return fsum(x * y for x, y in zip(left, right, strict=True)) / scale if scale else 0.0


@lru_cache(maxsize=1)
def _cross_encoder():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION, device="cpu")


class CrossEncoderReranker:
    def score(self, claim: str, chunks: list[Chunk]) -> list[float]:
        scores = _cross_encoder().predict(
            [(claim, chunk.text) for chunk in chunks], batch_size=16, show_progress_bar=False
        )
        return list(scores)


def needs_extraction(state: State) -> bool:
    return any(
        row["status"] == "not_supported" and row["claim"]["source_ids"]
        for row in state.get("claims", [])
    )


def _pages(sources: list[Source], context: GraphContext, notes: list[str]) -> dict[str, str]:
    pages = {}
    wanted = {}
    cache_dir = context.calls.settings.cache_dir / "extract"
    for source in sources:
        url = canonical_url(source.url)
        path = cache_dir / f"{text_digest(url)}.json"
        if context.use_cache and path.is_file():
            try:
                text = json.loads(path.read_text(encoding="utf-8"))["text"]
                if isinstance(text, str) and text.strip():
                    pages[source.source_id] = text
                    notes.append(f"{source.source_id}: reused a private extraction snapshot")
                    continue
            except (ValueError, KeyError, TypeError):
                pass
        wanted[url] = (source, path)
    if not wanted:
        return pages
    try:
        response = context.calls.extract([source.url for source, _ in wanted.values()])
    except (ProviderFailure, BudgetRefused):
        return pages
    for item in response.pages:
        canonical = canonical_url(str(item.get("url", "")))
        raw = item.get("raw_content") or item.get("content")
        if canonical not in wanted or not isinstance(raw, str) or not raw.strip():
            continue
        source, path = wanted[canonical]
        text = context.safe(raw)
        pages[source.source_id] = text
        private_json(
            path,
            {"url": source.url, "retrieved_at": datetime.now(UTC).isoformat(), "text": text},
        )
    for source, _ in wanted.values():
        if source.source_id not in pages:
            notes.append(f"{source.source_id}: extraction returned no usable source text")
    return pages


def _vectorize(context: GraphContext, texts: list[str]) -> list[list[float]]:
    vectors = []
    for offset in range(0, len(texts), 32):
        batch = texts[offset : offset + 32]
        response = context.calls.embed(batch)
        if len(response) != len(batch) or not response[0]:
            raise ProviderFailure("nebius", "invalid_embeddings")
        vectors.extend(response)
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ProviderFailure("nebius", "invalid_embeddings")
    return vectors


def _attempt_page(
    context: GraphContext,
    source: Source,
    page: str,
    claims: list[Claim],
    reranker: Reranker | None,
    notes: list[str],
) -> list[PairVerification]:
    chunks = chunk_page(page)
    try:
        vectors = _vectorize(
            context, [chunk.text for chunk in chunks] + [claim.text for claim in claims]
        )
    except BudgetRefused:
        return []
    except ProviderFailure as error:
        notes.append(f"{source.source_id}: evidence embeddings unavailable ({error.kind})")
        return []
    attempts = []
    for claim, vector in zip(claims, vectors[len(chunks) :], strict=True):
        order = sorted(range(len(chunks)), key=lambda index: -cosine(vectors[index], vector))
        candidates = [chunks[index] for index in order[:20]]
        if reranker is not None:
            try:
                scores = reranker.score(claim.text, candidates)
                candidates = [
                    item
                    for _, item in sorted(
                        zip(scores, candidates, strict=True), key=lambda row: -row[0]
                    )
                ]
            except (ImportError, RuntimeError, OSError, ValueError):
                notes.append(f"{source.source_id}: CPU reranker unavailable; used embedding order")
                reranker = None
        selected = candidates[:3]
        supplied = "\n\n".join(chunk.text for chunk in selected)
        attempts.extend(
            context.verifier.verify_source(
                [claim],
                source,
                supplied,
                scope="page",
                full_text=page,
                preview=selected[0].text,
            )
        )
        if context.calls.budget_stopped:
            break
    return attempts


def extract_evidence(
    state: State, context: GraphContext, reranker: Reranker | None = None
) -> State:
    if not hasattr(context.verifier, "verify_source") or context.calls.budget_stopped:
        return {}
    claims = [ClaimResult.model_validate(row) for row in state.get("claims", [])]
    pending: dict[str, list[Claim]] = {}
    for row in claims:
        if row.status == "not_supported":
            for source_id in row.claim.source_ids:
                pending.setdefault(source_id, []).append(row.claim)
    sources = [
        Source.model_validate(row)
        for row in state.get("sources", [])
        if row["source_id"] in pending
    ][:5]
    notes = list(state.get("notes", []))
    pages = _pages(sources, context, notes)
    if reranker is None and context.calls.settings.reranker == "cross-encoder":
        reranker = CrossEncoderReranker()

    attempts = []
    revisions = {}
    for source in sources:
        if source.source_id not in pages:
            continue
        attempts.extend(
            _attempt_page(
                context, source, pages[source.source_id], pending[source.source_id], reranker, notes
            )
        )
        if context.calls.budget_stopped:
            break
    pairs = {(row.claim_id, row.source_id): row for row in attempts}
    originals = []
    for row in claims:
        for pair in row.pairs:
            key = (pair.claim_id, pair.source_id)
            if key in pairs:
                originals.append(pair)
                if pairs[key].verdict != "not_supported":
                    revisions[key] = pairs[key]
    revised = [
        aggregate(
            row.claim,
            [revisions.get((pair.claim_id, pair.source_id), pair) for pair in row.pairs],
        ).model_dump(mode="json")
        for row in claims
    ]
    return {
        "claims": revised,
        "extraction_attempts": [pair.model_dump(mode="json") for pair in originals + attempts],
        "extracted_source_ids": list(pages),
        "notes": notes,
    }
