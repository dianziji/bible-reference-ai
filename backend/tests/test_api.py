import dataclasses

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.query_pipeline import QueryPipeline
from app.services.rate_limiter import RateLimiter
from tests.conftest import FakeRAG, FakeSemantic


@pytest.fixture
def client(cache, executor, monkeypatch):
    rag = FakeRAG()
    pipeline = QueryPipeline(rag, cache, FakeSemantic(), lambda q: True, executor)
    monkeypatch.setattr(main, "_pipeline", pipeline)
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "limiter", RateLimiter(cache))
    logged = []
    monkeypatch.setattr(main, "log_query", lambda *a, **kw: logged.append((a, kw)))
    c = TestClient(main.app)
    c.rag = rag
    c.logged = logged
    return c


def test_health_reports_redis(client):
    assert client.get("/health").json() == {"status": "ok", "redis": "up"}


def test_empty_question_is_rejected(client):
    assert client.post("/api/query", json={"question": "   "}).status_code == 400


def test_query_miss_then_exact_hit(client):
    first = client.post("/api/query", json={"question": "What is love?"})
    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    body = first.json()
    assert body["answer"] == "ANSWER"
    assert body["verses"][0] == {
        "id": "Genesis.1.1", "usfm": "GEN", "book": "Genesis", "book_zh": "创世记", "chapter": 1, "verse": 1,
        "text": "In the beginning God created the heavens and the earth.",
        "text_en": "In the beginning God created the heavens and the earth.",
        "text_zh": "起初，　神创造天地。",
        "translation": "bsb", "score": 0.71,
    }
    assert body["meta"]["cache"] == "miss"
    assert body["meta"]["language"] == "en"
    assert body["meta"]["translation"] == "bsb"
    assert body["meta"]["embedding_cached"] is False
    assert {"embed", "retrieve", "moderation", "generate", "total"} <= set(body["meta"]["timings_ms"])

    second = client.post("/api/query", json={"question": "what is love"})
    assert second.headers["X-Cache"] == "HIT-EXACT"
    assert float(second.headers["X-Response-Time-Ms"]) >= 0
    assert second.json()["meta"]["cache"] == "exact"
    assert second.json()["verses"] == body["verses"]
    assert client.rag.generate_calls == ["en"]

    # Both requests are logged, tagged with how they were served.
    assert [kw["cache"] for _, kw in client.logged] == ["miss", "exact"]


def test_cache_stats_endpoint(client):
    client.post("/api/query", json={"question": "What is love?"})
    client.post("/api/query", json={"question": "What is love?"})
    stats = client.get("/api/cache/stats").json()
    assert stats["redis"] == "up"
    assert stats["requests"] == 2
    assert stats["exact_hits"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["semantic_cache"]["enabled"] is True
    assert "chat_model" in stats["config"]


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, rate_limit_per_minute=2))

    first = client.post("/api/query", json={"question": "What is love?"})
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert first.headers["X-RateLimit-Remaining"] == "1"
    assert client.post("/api/query", json={"question": "What is love?"}).status_code == 200

    third = client.post("/api/query", json={"question": "What is love?"})
    assert third.status_code == 429
    assert int(third.headers["Retry-After"]) >= 1
    assert "请求太频繁" in third.json()["detail"]
    assert client.rag.generate_calls == ["en"]  # the blocked request never reached the pipeline

    # A different client (per X-Forwarded-For, as behind nginx) has its own quota.
    other = client.post("/api/query", json={"question": "What is love?"}, headers={"X-Forwarded-For": "203.0.113.9"})
    assert other.status_code == 200


def test_global_limit_caps_everyone(client, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, rate_limit_global_per_minute=1))
    assert client.post("/api/query", json={"question": "q1"}, headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200
    assert client.post("/api/query", json={"question": "q2"}, headers={"X-Forwarded-For": "203.0.113.2"}).status_code == 429


def test_pipeline_error_becomes_500(client, monkeypatch):
    def boom(question):
        raise RuntimeError("pinecone exploded")

    monkeypatch.setattr(main._pipeline, "run", boom)
    res = client.post("/api/query", json={"question": "What is love?"})
    assert res.status_code == 500
    assert "RuntimeError: pinecone exploded" in res.json()["detail"]
