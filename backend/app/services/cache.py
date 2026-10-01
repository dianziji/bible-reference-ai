"""Caching for the query pipeline.

Two tiers, both keyed by the *normalized* question (case, whitespace and
trailing punctuation folded away):

  L1  RedisCache     exact-match answers, query embeddings, hit/miss counters
  L2  SemanticCache  question vectors in a dedicated Pinecone namespace; a new
                     question whose nearest cached question scores above the
                     similarity threshold reuses that answer (skips the LLM)

Both tiers are optional at runtime. Redis being unreachable turns L1 into a
no-op behind a short circuit-breaker window, so requests never pay a
connection timeout on every call while Redis is down. Pinecone errors inside
L2 are logged and treated as misses.
"""

import array
import hashlib
import json
import logging
import time
import unicodedata
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import redis

logger = logging.getLogger("bible-backend.cache")

_TRAILING_PUNCT = "?？!！.。,，;；:：、"


def normalize_question(question: str) -> str:
    """Fold away differences that never change the answer.

    "  What is LOVE?? " and "what is love" hash to the same key.
    """
    text = unicodedata.normalize("NFKC", question).casefold()
    text = " ".join(text.split())
    return text.strip().rstrip(_TRAILING_PUNCT).strip()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def _pack_vector(vector: List[float]) -> bytes:
    return array.array("f", vector).tobytes()


def _unpack_vector(raw: bytes) -> List[float]:
    arr = array.array("f")
    arr.frombytes(raw)
    return arr.tolist()


class RedisCache:
    """L1: exact-match response cache + embedding cache + counters."""

    OUTCOMES = ("exact_hits", "semantic_hits", "misses")

    def __init__(
        self,
        url: str,
        *,
        namespace: str,
        embedding_model: str,
        response_ttl: int,
        embedding_ttl: int,
        enabled: bool = True,
        breaker_seconds: float = 30.0,
        client: Optional["redis.Redis"] = None,
    ):
        self.enabled = enabled
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.response_ttl = response_ttl
        self.embedding_ttl = embedding_ttl
        self.breaker_seconds = breaker_seconds
        self._down_until = 0.0
        # Short timeouts: a dead Redis must cost milliseconds, not seconds.
        self._client = client or redis.Redis.from_url(
            url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=False,
        )

    # ---- keys ---------------------------------------------------------------

    def response_key(self, question: str) -> str:
        return f"{self.namespace}:resp:{question_hash(question)}"

    def embedding_key(self, question: str) -> str:
        # Embeddings depend only on the embedding model, not on the chat model
        # or prompt, so they live outside the answer namespace and survive a
        # prompt change.
        return f"bible:v1:emb:{self.embedding_model}:{question_hash(question)}"

    @property
    def stats_key(self) -> str:
        return f"{self.namespace}:stats"

    # ---- guarded access -----------------------------------------------------

    def _call(self, fn: Callable[..., Any], *args: Any, default: Any = None) -> Any:
        if not self.enabled or time.monotonic() < self._down_until:
            return default
        try:
            return fn(*args)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            self._down_until = time.monotonic() + self.breaker_seconds
            logger.warning(
                "Redis unavailable (%s: %s); cache disabled for %.0fs",
                type(e).__name__, e, self.breaker_seconds,
            )
            return default

    @property
    def available(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    def ping(self) -> bool:
        """Probe Redis directly, bypassing (and on success clearing) the breaker."""
        if not self.enabled:
            return False
        try:
            ok = bool(self._client.ping())
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            return False
        if ok:
            self._down_until = 0.0
        return ok

    # ---- responses ----------------------------------------------------------

    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        return self.get_response_by_key(self.response_key(question))

    def get_response_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._call(self._client.get, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("Dropping unreadable cache entry %s", key)
            self._call(self._client.delete, key)
            return None

    def set_response(self, question: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._call(self._client.set, self.response_key(question), data, self.response_ttl)

    # ---- embeddings ---------------------------------------------------------

    def get_embedding(self, question: str) -> Optional[List[float]]:
        raw = self._call(self._client.get, self.embedding_key(question))
        return _unpack_vector(raw) if raw else None

    def set_embedding(self, question: str, vector: List[float]) -> None:
        self._call(
            self._client.set, self.embedding_key(question), _pack_vector(vector), self.embedding_ttl
        )

    # ---- rate-limit windows -------------------------------------------------

    def window_counts(self, key: str, window_seconds: int, now: float) -> Optional[Tuple[int, int]]:
        """Count one hit for ``key`` in the current fixed window.

        Returns (previous_window_count, current_window_count), or None when
        Redis is unavailable. Keys live outside the answer namespace: a config
        change must not reset anyone's quota.
        """
        window = int(now // window_seconds)
        current_key = f"bible:v1:rl:{key}:{window}"
        previous_key = f"bible:v1:rl:{key}:{window - 1}"

        def run() -> Tuple[int, int]:
            pipe = self._client.pipeline()
            pipe.incr(current_key)
            pipe.expire(current_key, 2 * window_seconds)
            pipe.get(previous_key)
            current, _, previous = pipe.execute()
            return int(previous or 0), int(current)

        return self._call(run)

    # ---- counters -----------------------------------------------------------

    def record(self, outcome: str) -> None:
        if outcome not in self.OUTCOMES:
            raise ValueError(f"unknown cache outcome {outcome!r}")
        self._call(self._client.hincrby, self.stats_key, outcome, 1)

    def stats(self) -> Dict[str, Any]:
        raw = self._call(self._client.hgetall, self.stats_key, default={}) or {}
        counts = {name: 0 for name in self.OUTCOMES}
        for k, v in raw.items():
            name = k.decode() if isinstance(k, bytes) else k
            if name in counts:
                counts[name] = int(v)
        hits = counts["exact_hits"] + counts["semantic_hits"]
        total = hits + counts["misses"]
        return {
            **counts,
            "requests": total,
            "hit_rate": round(hits / total, 4) if total else None,
        }

    def flush(self) -> int:
        """Delete every key this cache owns (answers, stats and embeddings)."""
        deleted = 0
        for pattern in (f"{self.namespace}:*", f"bible:v1:emb:{self.embedding_model}:*"):
            keys = self._call(lambda p: list(self._client.scan_iter(match=p, count=500)), pattern, default=[]) or []
            for i in range(0, len(keys), 500):
                deleted += int(self._call(self._client.delete, *keys[i : i + 500], default=0) or 0)
        return deleted


class SemanticHit(NamedTuple):
    response_key: str
    question: str
    score: float


class SemanticCache:
    """L2: reuse an answer when a *similar* question has been answered before.

    The stored vector is the cached question's embedding, which the pipeline
    already has to compute for retrieval, so a lookup costs one extra Pinecone
    query and no extra OpenAI calls. The answer itself still lives in Redis
    (``response_key``); if it has expired there the lookup is treated as a miss
    and the entry is refreshed on the next full answer.
    """

    def __init__(
        self,
        index_getter: Callable[[], Any],
        *,
        namespace: str,
        threshold: float,
        enabled: bool = True,
    ):
        self._index_getter = index_getter
        self._index = None
        self.namespace = namespace
        self.threshold = threshold
        self.enabled = enabled

    @property
    def index(self):
        if self._index is None:
            self._index = self._index_getter()
        return self._index

    def lookup(self, vector: List[float], lang: str) -> Optional[SemanticHit]:
        """Nearest cached question *in the same language*: a paraphrase must not
        be answered in the other language, however close the vectors are."""
        if not self.enabled:
            return None
        try:
            res = self.index.query(
                vector=vector, top_k=1, include_metadata=True, namespace=self.namespace,
                filter={"lang": {"$eq": lang}},
            )
            matches = res["matches"] if "matches" in res else []
        except Exception as e:  # never let the cache break a request
            logger.warning("Semantic cache lookup failed: %s: %s", type(e).__name__, e)
            return None
        if not matches:
            return None
        best = matches[0]
        score = float(best.get("score") or 0.0)
        meta = best.get("metadata") or {}
        if score < self.threshold or "response_key" not in meta:
            return None
        return SemanticHit(
            response_key=str(meta["response_key"]),
            question=str(meta.get("question", "")),
            score=score,
        )

    def store(self, question: str, vector: List[float], response_key: str, lang: str) -> None:
        if not self.enabled:
            return
        try:
            self.index.upsert(
                vectors=[
                    {
                        "id": question_hash(question),
                        "values": vector,
                        "metadata": {
                            "question": normalize_question(question),
                            "response_key": response_key,
                            "lang": lang,
                        },
                    }
                ],
                namespace=self.namespace,
            )
        except Exception as e:
            logger.warning("Semantic cache store failed: %s: %s", type(e).__name__, e)

    def flush(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception as e:
            # Pinecone reports a namespace that was never written as not found.
            logger.debug("Semantic cache flush skipped: %s: %s", type(e).__name__, e)
