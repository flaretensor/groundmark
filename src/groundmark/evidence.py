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
from groundmark.models import Source
from groundmark.providers import ProviderFailure
from groundmark.sources import canonical_url, private_json, text_digest

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
PAGE_CHAR_LIMIT = 200_000


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


def _cap_page(text: str, source_id: str, notes: list[str]) -> str:
    if len(text) > PAGE_CHAR_LIMIT:
        notes.append(f"{source_id}: page truncated to {PAGE_CHAR_LIMIT} characters")
    return text[:PAGE_CHAR_LIMIT]


def _pages(
    sources: list[Source], context: GraphContext, notes: list[str]
) -> dict[str, tuple[str | None, str]]:
    pages = {}
    wanted = {}
    cache_dir = context.calls.settings.cache_dir / "extract"
    for source in sources:
        url = canonical_url(source.url)
        path = cache_dir / f"{text_digest(url)}.json"
        if context.use_cache and path.is_file():
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                text = snapshot["text"]
                if isinstance(text, str):
                    text = _cap_page(text, source.source_id, notes)
                    if text.strip():
                        pages[source.source_id] = (snapshot.get("retrieved_at"), text)
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
    retrieved_at = datetime.now(UTC).isoformat()
    for item in response.pages:
        canonical = canonical_url(str(item.get("url", "")))
        raw = item.get("raw_content") or item.get("content")
        if canonical not in wanted or not isinstance(raw, str) or not raw.strip():
            continue
        source, path = wanted[canonical]
        text = _cap_page(context.safe(raw), source.source_id, notes)
        if not text.strip():
            continue
        pages[source.source_id] = (retrieved_at, text)
        private_json(
            path,
            {"url": source.url, "retrieved_at": retrieved_at, "text": text},
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


def _select_parts(
    context: GraphContext,
    source: Source,
    page: str,
    question: str,
    reranker: Reranker | None,
    notes: list[str],
) -> list[str]:
    chunks = chunk_page(page)
    try:
        vectors = _vectorize(context, [chunk.text for chunk in chunks] + [question])
    except BudgetRefused:
        return []
    except ProviderFailure as error:
        notes.append(f"{source.source_id}: evidence embeddings unavailable ({error.kind})")
        return []
    order = sorted(range(len(chunks)), key=lambda index: -cosine(vectors[index], vectors[-1]))
    candidates = [chunks[index] for index in order[:20]]
    if reranker is not None:
        try:
            scores = reranker.score(question, candidates)
            candidates = [
                item
                for _, item in sorted(zip(scores, candidates, strict=True), key=lambda row: -row[0])
            ]
        except (ImportError, RuntimeError, OSError, ValueError):
            notes.append(f"{source.source_id}: CPU reranker unavailable; used embedding order")
    return [chunk.text for chunk in candidates[:3]]


def extract_evidence(
    state: State, context: GraphContext, reranker: Reranker | None = None
) -> State:
    wanted = set(state["action"]["source_ids"])
    sources = [Source.model_validate(row) for row in state.get("sources", [])]
    selected = [source for source in sources if source.source_id in wanted][:5]
    notes = list(state.get("notes", []))
    pages = _pages(selected, context, notes)
    if reranker is None and context.calls.settings.reranker == "cross-encoder":
        reranker = CrossEncoderReranker()

    received = []
    for source in selected:
        page = pages.get(source.source_id)
        if page is None or context.calls.budget_stopped:
            continue
        retrieved_at, content = page
        parts = _select_parts(context, source, content, state["question"], reranker, notes)
        if parts:
            source.page_parts = parts
            source.page_retrieved_at = retrieved_at
            source.evidence_scope = "page"
            received.append(source.source_id)
    return {
        "sources": [source.model_dump(mode="json") for source in sources],
        "extracted_source_ids": list(
            dict.fromkeys([*state.get("extracted_source_ids", []), *received])
        ),
        "last_extracted_ids": received,
        "notes": notes,
    }
