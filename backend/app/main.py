import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.services.cache import RedisCache, SemanticCache
from app.services.logger import log_query
from app.services.messages import RATE_LIMITED
from app.services.openai_client import is_safe_content
from app.services.pinecone_client import get_index
from app.services.query_pipeline import QueryPipeline
from app.services.rag_service import get_rag_service
from app.services.rate_limiter import RateLimiter, client_ip

app = FastAPI()
logger = logging.getLogger("bible-backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://bible.local",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Moderation, semantic-cache lookups and cache writes run here so they can
# overlap with the embedding and retrieval calls made on the request thread.
# Each request submits at most three short tasks.
executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="rag")

cache = RedisCache(
    settings.redis_url,
    namespace=settings.cache_namespace,
    embedding_model=settings.embedding_model,
    response_ttl=settings.response_ttl_seconds,
    embedding_ttl=settings.embedding_ttl_seconds,
    enabled=settings.cache_enabled,
)
semantic_cache = SemanticCache(
    get_index,
    namespace=settings.semantic_cache_namespace,
    threshold=settings.semantic_cache_threshold,
    enabled=settings.cache_enabled and settings.semantic_cache_enabled,
)
limiter = RateLimiter(cache)

_pipeline: Optional[QueryPipeline] = None


def get_pipeline() -> QueryPipeline:
    """Built lazily: RAGService opens the OpenAI / Pinecone clients."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline(get_rag_service(), cache, semantic_cache, is_safe_content, executor)
    return _pipeline


def enforce_rate_limit(request: Request, response: Response) -> None:
    """Per-client, then global, sliding-window limits on the paid endpoint."""
    ip = client_ip(
        request.client.host if request.client else "unknown",
        request.headers.get("x-forwarded-for"),
        settings.client_ip_xff_index,
    )
    per_ip = limiter.check(f"ip:{ip}", settings.rate_limit_per_minute)
    overall = limiter.check("global", settings.rate_limit_global_per_minute) if per_ip.allowed else per_ip
    for scope, decision in (("ip", per_ip), ("global", overall)):
        if not decision.allowed:
            logger.warning("rate limited scope=%s ip=%s limit=%d/min", scope, ip, decision.limit)
            raise HTTPException(
                status_code=429,
                detail=RATE_LIMITED,
                headers={"Retry-After": str(decision.retry_after)},
            )
    response.headers["X-RateLimit-Limit"] = str(per_ip.limit)
    response.headers["X-RateLimit-Remaining"] = str(per_ip.remaining)


class Verse(BaseModel):
    id: str                        # "John.3.16", the same in every translation
    usfm: Optional[str] = None     # "JHN" — lets the UI build reader links
    book: str                      # English book name
    book_zh: Optional[str] = None  # 和合本 book name
    chapter: int
    verse: int
    text: str                      # in the language of the question
    text_en: Optional[str] = None  # Berean Standard Bible
    text_zh: Optional[str] = None  # 和合本（神版）
    translation: Optional[str] = None  # which translation was searched ("bsb" | "cuv")
    score: Optional[float] = None  # cosine similarity between question and verse


class QueryRequest(BaseModel):
    question: str


class QueryMeta(BaseModel):
    cache: Literal["exact", "semantic", "miss"]
    language: Literal["en", "zh"] = "en"
    translation: Optional[str] = None
    embedding_cached: bool = False
    semantic_similarity: Optional[float] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    verses: List[Verse]
    meta: Optional[QueryMeta] = None


_CACHE_HEADER = {"exact": "HIT-EXACT", "semantic": "HIT-SEMANTIC", "miss": "MISS"}


@app.get("/health")
def health():
    # Redis is optional: the service stays healthy without it, just slower.
    return {"status": "ok", "redis": "up" if cache.ping() else "down"}


@app.get("/api/cache/stats")
def cache_stats():
    return {
        "redis": "up" if cache.ping() else "down",
        "namespace": cache.namespace,
        "semantic_cache": {
            "enabled": semantic_cache.enabled,
            "namespace": semantic_cache.namespace,
            "threshold": semantic_cache.threshold,
        },
        **cache.stats(),
        "config": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "translations": {"en": settings.en_translation, "zh": settings.zh_translation},
            "top_k": settings.top_k,
            "score_threshold": settings.score_threshold,
            "score_threshold_zh": settings.score_threshold_zh,
            "response_ttl_seconds": settings.response_ttl_seconds,
            "embedding_ttl_seconds": settings.embedding_ttl_seconds,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "rate_limit_global_per_minute": settings.rate_limit_global_per_minute,
        },
    }


@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)])
def query(req: QueryRequest, response: Response):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_pipeline().run(question)
    except Exception as e:
        # 让你在本地调试时能直接看到具体报错原因（比如 OpenAI/Pinecone 配置问题）
        logger.exception("Failed to handle /api/query")
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {type(e).__name__}: {e}",
        )

    response.headers["X-Cache"] = _CACHE_HEADER[result.cache]
    response.headers["X-Response-Time-Ms"] = str(result.timings_ms.get("total", 0))
    logger.info(
        "query cache=%s total=%.0fms verses=%d",
        result.cache, result.timings_ms.get("total", 0), len(result.verses),
    )

    if not result.flagged:
        log_query(question, result.answer, result.verses, cache=result.cache)

    return QueryResponse(
        answer=result.answer,
        verses=[Verse(**v) for v in result.verses],
        meta=QueryMeta(
            cache=result.cache,
            language=result.language,
            translation=result.translation,
            embedding_cached=result.embedding_cached,
            semantic_similarity=result.semantic_similarity,
            timings_ms=result.timings_ms,
        ),
    )
