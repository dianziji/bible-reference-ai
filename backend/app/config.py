"""Runtime configuration, read once from the environment.

Everything that changes the *content* of an answer (models, retrieval
parameters, prompt wording) is folded into ``Settings.fingerprint``. Cache
keys are namespaced by that fingerprint, so changing any of it rolls the
cache over automatically instead of serving answers produced by an old
configuration.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Bump this whenever the QA prompt in rag_service.py changes wording.
PROMPT_VERSION = "3"

# Chosen by scripts/compare_models.py + Ragas on the 32-question golden set
# (2026-09-30): gpt-5-nano matched gpt-4o-mini on faithfulness (0.90 vs 0.89)
# and relevancy (0.88 vs 0.86) at 35% lower cost with no refusals; gpt-6-luna
# and gpt-5.4-nano hedged far more (relevancy 0.71 / 0.41) and luna was slowest.
DEFAULT_CHAT_MODEL = "gpt-5-nano"


def supports_temperature(model: str) -> bool:
    """Only the pre-reasoning chat families accept an explicit temperature.

    gpt-5 / gpt-6 / o-series models reject it, so we never send one to them.
    """
    return model.startswith(("gpt-4o", "gpt-4.1", "gpt-4-", "gpt-3.5", "chatgpt-"))


def default_reasoning_effort(model: str) -> Optional[str]:
    """Cheapest reasoning setting each family accepts; None for non-reasoning models.

    Grounded QA over five short verses needs no deliberation, and reasoning
    tokens are billed as output, so the floor of each family is the default.
    """
    if model.startswith("gpt-6") or re.match(r"gpt-5\.[1-9]", model):
        return "none"
    if model.startswith("gpt-5"):
        return "minimal"  # the original gpt-5 family has no "none"
    if model.startswith(("o1", "o3", "o4")):
        return "low"
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _redis_url() -> str:
    # REDIS_URL wins; otherwise assemble one from the REDIS_HOST / REDIS_PORT
    # pair that the k8s ConfigMap provides.
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = os.getenv("REDIS_PORT", "6379")
    return f"redis://{host}:{port}/0"


@dataclass(frozen=True)
class Settings:
    # Models
    chat_model: str
    chat_reasoning_effort: Optional[str]
    embedding_model: str
    moderation_model: str

    # Retrieval. Each translation lives in its own Pinecone namespace; the
    # question's language picks which one is searched (see app.language).
    pinecone_index_name: str
    en_translation: str
    zh_translation: str
    top_k: int
    # Minimum cosine similarity for a verse to count as context. Calibrated on
    # text-embedding-3-small: off-topic English questions score ≤ 0.38, on-topic
    # ones ≥ 0.47; Chinese scores run ~0.05 higher across the board.
    score_threshold: float
    score_threshold_zh: float

    # Caching
    cache_enabled: bool
    redis_url: str
    response_ttl_seconds: int
    embedding_ttl_seconds: int
    semantic_cache_enabled: bool
    semantic_cache_threshold: float

    # Abuse protection. Not part of the fingerprint: it never changes an answer.
    rate_limit_per_minute: int         # per client IP on POST /api/query; 0 disables
    rate_limit_global_per_minute: int  # all clients together; caps the OpenAI bill
    client_ip_xff_index: int           # which X-Forwarded-For entry is the client (see rate_limiter.client_ip)

    @property
    def fingerprint(self) -> str:
        """Short hash of everything that determines an answer's content."""
        parts = [
            self.chat_model,
            self.chat_reasoning_effort or "",
            self.embedding_model,
            self.pinecone_index_name,
            self.en_translation,
            self.zh_translation,
            str(self.top_k),
            str(self.score_threshold),
            str(self.score_threshold_zh),
            PROMPT_VERSION,
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]

    def translation_for(self, lang: str) -> str:
        return self.zh_translation if lang == "zh" else self.en_translation

    def threshold_for(self, lang: str) -> float:
        return self.score_threshold_zh if lang == "zh" else self.score_threshold

    @property
    def cache_namespace(self) -> str:
        return f"bible:v1:{self.fingerprint}"

    @property
    def semantic_cache_namespace(self) -> str:
        """Pinecone namespace holding cached question vectors."""
        return f"semantic-cache-{self.fingerprint}"


def load_settings() -> Settings:
    chat_model = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return Settings(
        chat_model=chat_model,
        chat_reasoning_effort=os.getenv("CHAT_REASONING_EFFORT") or default_reasoning_effort(chat_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        moderation_model=os.getenv("MODERATION_MODEL", "omni-moderation-latest"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "bible-verses"),
        en_translation=os.getenv("EN_TRANSLATION", "bsb"),
        zh_translation=os.getenv("ZH_TRANSLATION", "cuv"),
        top_k=_env_int("TOP_K", 5),
        score_threshold=_env_float("SCORE_THRESHOLD", 0.40),
        score_threshold_zh=_env_float("SCORE_THRESHOLD_ZH", 0.45),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        redis_url=_redis_url(),
        response_ttl_seconds=_env_int("CACHE_RESPONSE_TTL_SECONDS", 24 * 3600),
        embedding_ttl_seconds=_env_int("CACHE_EMBEDDING_TTL_SECONDS", 7 * 24 * 3600),
        semantic_cache_enabled=_env_bool("SEMANTIC_CACHE_ENABLED", True),
        # Calibrated on 16 paraphrase pairs vs 496 cross-question pairs: paraphrases
        # score 0.71–0.97 (median 0.88), different questions ≤ 0.76. 0.85 reuses an
        # answer for 9/16 paraphrases with zero false matches; 0.82 would be 14/16.
        semantic_cache_threshold=_env_float("SEMANTIC_CACHE_THRESHOLD", 0.85),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 20),
        rate_limit_global_per_minute=_env_int("RATE_LIMIT_GLOBAL_PER_MINUTE", 60),
        client_ip_xff_index=_env_int("CLIENT_IP_XFF_INDEX", -1),
    )


settings = load_settings()
