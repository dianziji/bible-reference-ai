"""Shared fixtures.

The app modules validate API keys at import time and open (but do not use)
OpenAI / Pinecone clients, so placeholder keys are enough. Redis is replaced
with fakeredis, the RAG stages with in-memory fakes, and the executor with a
synchronous one so background work has finished by the time a test asserts.
"""

import os
from concurrent.futures import Future
from typing import Any, Callable, List, Optional

# Must run before any `app.*` import.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:1/0")  # never connected; fakeredis is injected

import fakeredis
import pytest
from langchain_core.documents import Document

from app.config import settings
from app.services.cache import RedisCache, SemanticHit
from app.services.rag_service import RAGService


class SyncExecutor:
    """Runs submitted work immediately on the calling thread."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:  # pragma: no cover - surfaced by the test
            fut.set_exception(e)
        return fut


class FakeRAG:
    cfg = settings

    def __init__(self, docs: Optional[List] = None, answer: str = "ANSWER"):
        self.docs = docs if docs is not None else sample_docs()
        self.answer = answer
        self.embed_calls = 0
        self.retrieve_calls: List[str] = []   # translations searched
        self.generate_calls: List[str] = []   # languages answered in

    def embed(self, question: str) -> List[float]:
        self.embed_calls += 1
        return [0.25, -0.5, 1.0, 0.0]

    def retrieve(self, vector: List[float], translation: str):
        self.retrieve_calls.append(translation)
        return self.docs

    def generate(self, question: str, docs, lang: str = "en") -> str:
        self.generate_calls.append(lang)
        return self.answer

    to_verses = staticmethod(RAGService.to_verses)


class FakeSemantic:
    namespace = "semantic-cache-test"
    threshold = 0.95

    def __init__(self, enabled: bool = True, hit: Optional[SemanticHit] = None):
        self.enabled = enabled
        self.hit = hit
        self.lookups: List[str] = []   # languages looked up
        self.stored: List = []

    def lookup(self, vector: List[float], lang: str) -> Optional[SemanticHit]:
        self.lookups.append(lang)
        return self.hit

    def store(self, question: str, vector: List[float], response_key: str, lang: str) -> None:
        self.stored.append((question, response_key, lang))

    def flush(self) -> None:
        self.stored.clear()


def sample_docs():
    return [
        (
            Document(
                id="Genesis.1.1",
                page_content="In the beginning God created the heavens and the earth.",
                metadata={"book": "Genesis", "chapter": 1, "verse": 1, "usfm": "GEN", "translation": "bsb",
                          "text_en": "In the beginning God created the heavens and the earth.",
                          "text_zh": "起初，　神创造天地。"},
            ),
            0.71,
        ),
        (
            Document(
                id="John.1.1",
                page_content="In the beginning was the Word, and the Word was with God, and the Word was God.",
                metadata={"book": "John", "chapter": 1, "verse": 1, "usfm": "JHN", "translation": "bsb",
                          "text_en": "In the beginning was the Word, and the Word was with God, and the Word was God.",
                          "text_zh": "太初有道，道与　神同在，道就是　神。"},
            ),
            0.64,
        ),
    ]


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def cache(fake_redis):
    return RedisCache(
        "redis://unused",
        namespace="bible:v1:test",
        embedding_model="text-embedding-test",
        response_ttl=60,
        embedding_ttl=120,
        client=fake_redis,
    )


@pytest.fixture
def executor():
    return SyncExecutor()
