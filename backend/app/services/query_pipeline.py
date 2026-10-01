"""Orchestrates one question through the caches and the RAG stages.

    L1 exact cache ──hit──▶ answer                         (~ms, no upstream calls)
         │ miss
         ▼
    detect language ─▶ translation (bsb | cuv)
    moderation ║ embed (embedding cache)                   two independent network calls, in parallel
         │
         ▼
    L2 semantic lookup ║ retrieve                          both need only the vector, in parallel
         │
         ├─ flagged ──────────────▶ refusal (not cached)
         ├─ semantic hit ─────────▶ cached answer          (skips the LLM)
         ├─ no verses ≥ threshold ▶ "insufficient context" (not cached)
         └─ generate ─────────────▶ answer, stored in L1 + L2

Every stage is timed so the API can report where a request spent its time.
"""

import logging
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.language import detect_language
from app.services.cache import RedisCache, SemanticCache
from app.services.messages import INSUFFICIENT_CONTEXT, MODERATION_REFUSED
from app.services.rag_service import RAGService

logger = logging.getLogger("bible-backend.pipeline")


@dataclass
class QueryResult:
    answer: str
    verses: List[Dict[str, Any]]
    cache: str  # "exact" | "semantic" | "miss"
    language: str = "en"
    translation: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    embedding_cached: bool = False
    semantic_similarity: Optional[float] = None
    flagged: bool = False


def _timed(fn: Callable[..., Any], *args: Any) -> Tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000


class QueryPipeline:
    def __init__(
        self,
        rag: RAGService,
        cache: RedisCache,
        semantic_cache: SemanticCache,
        moderation_fn: Callable[[str], bool],
        executor: Executor,
    ):
        self.rag = rag
        self.cache = cache
        self.semantic_cache = semantic_cache
        self.moderation_fn = moderation_fn
        self.executor = executor

    def run(self, question: str) -> QueryResult:
        start = time.perf_counter()
        timings: Dict[str, float] = {}
        lang = detect_language(question)
        translation = self.rag.cfg.translation_for(lang)

        def finish(result: QueryResult) -> QueryResult:
            timings["total"] = round((time.perf_counter() - start) * 1000, 1)
            result.timings_ms = {k: round(v, 1) for k, v in timings.items()}
            result.language = lang
            result.translation = translation
            return result

        # L1: exact match. The same normalized question already passed
        # moderation when it was first answered, so nothing else is needed.
        cached = self.cache.get_response(question)
        if cached is not None:
            self.cache.record("exact_hits")
            return finish(QueryResult(answer=cached["answer"], verses=cached["verses"], cache="exact"))

        # Moderation and embedding are independent network calls.
        moderation_future: Future = self.executor.submit(_timed, self.moderation_fn, question)
        vector, embedding_cached, timings["embed"] = self._embed(question)

        # Both consumers of the vector are Pinecone queries; overlap them too.
        # A semantic hit is only useful if Redis can hand back the answer.
        semantic_future: Optional[Future] = (
            self.executor.submit(_timed, self.semantic_cache.lookup, vector, lang)
            if self.semantic_cache.enabled and self.cache.available
            else None
        )
        docs, timings["retrieve"] = _timed(self.rag.retrieve, vector, translation)

        safe, timings["moderation"] = moderation_future.result()
        if not safe:
            if semantic_future is not None:
                semantic_future.result()  # keep the executor tidy; result unused
            return finish(QueryResult(answer=MODERATION_REFUSED, verses=[], cache="miss", flagged=True))

        if semantic_future is not None:
            hit, timings["semantic_lookup"] = semantic_future.result()
            if hit is not None:
                payload = self.cache.get_response_by_key(hit.response_key)
                if payload is not None:
                    self.cache.record("semantic_hits")
                    return finish(
                        QueryResult(
                            answer=payload["answer"],
                            verses=payload["verses"],
                            cache="semantic",
                            embedding_cached=embedding_cached,
                            semantic_similarity=round(hit.score, 4),
                        )
                    )
                # The vector is still in Pinecone but the answer expired in
                # Redis; fall through and regenerate (which refreshes both).

        self.cache.record("misses")
        if not docs:
            return finish(
                QueryResult(answer=INSUFFICIENT_CONTEXT, verses=[], cache="miss", embedding_cached=embedding_cached)
            )

        answer, timings["generate"] = _timed(self.rag.generate, question, docs, lang)
        verses = self.rag.to_verses(docs, lang)

        payload = {"answer": answer, "verses": verses}
        self.cache.set_response(question, payload)
        if self.semantic_cache.enabled:
            # Pinecone upsert is ~100ms of pure bookkeeping; don't make the user wait for it.
            self.executor.submit(
                self.semantic_cache.store, question, vector, self.cache.response_key(question), lang
            )

        return finish(QueryResult(answer=answer, verses=verses, cache="miss", embedding_cached=embedding_cached))

    def _embed(self, question: str) -> Tuple[List[float], bool, float]:
        start = time.perf_counter()
        vector = self.cache.get_embedding(question)
        if vector is not None:
            return vector, True, (time.perf_counter() - start) * 1000
        vector = self.rag.embed(question)
        self.cache.set_embedding(question, vector)
        return vector, False, (time.perf_counter() - start) * 1000
