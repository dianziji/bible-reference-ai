import pytest

from app.services.cache import RedisCache, normalize_question, question_hash


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("What is love?", "what is love"),
        ("  what   IS  love ?? ", "what is love"),
        ("What is love", "what is love"),
        ("神是爱吗？", "神是爱吗"),
        ("Ｗhat is love", "what is love"),  # full-width letters fold via NFKC
        ("who is Jesus, and why?", "who is jesus, and why"),  # interior punctuation kept
    ],
)
def test_normalize_question(raw, expected):
    assert normalize_question(raw) == expected


def test_same_question_different_surface_hashes_equal():
    assert question_hash("What is love?") == question_hash("what is LOVE")
    assert question_hash("What is love?") != question_hash("What is faith?")


def test_keys_are_namespaced_by_config(fake_redis):
    a = RedisCache("redis://unused", namespace="bible:v1:aaaa", embedding_model="m",
                   response_ttl=1, embedding_ttl=1, client=fake_redis)
    b = RedisCache("redis://unused", namespace="bible:v1:bbbb", embedding_model="m",
                   response_ttl=1, embedding_ttl=1, client=fake_redis)
    assert a.response_key("q") != b.response_key("q")
    # Embeddings depend only on the embedding model, so they are shared.
    assert a.embedding_key("q") == b.embedding_key("q")


def test_response_roundtrip_and_ttl(cache, fake_redis):
    payload = {"answer": "爱是恒久忍耐 / love is patient", "verses": [{"id": "1 Corinthians.13.4", "score": 0.7}]}
    cache.set_response("What is love?", payload)
    assert cache.get_response("what is love") == payload
    assert 0 < fake_redis.ttl(cache.response_key("what is love")) <= 60
    assert cache.get_response("what is faith") is None


def test_embedding_roundtrip_float32(cache, fake_redis):
    vector = [0.123456789, -1.0, 2.5, 1e-7]
    cache.set_embedding("q", vector)
    got = cache.get_embedding("q")
    assert got is not None
    assert got == pytest.approx(vector, abs=1e-6)
    assert 0 < fake_redis.ttl(cache.embedding_key("q")) <= 120


def test_unreadable_entry_is_dropped(cache, fake_redis):
    fake_redis.set(cache.response_key("q"), b"{not json")
    assert cache.get_response("q") is None
    assert fake_redis.get(cache.response_key("q")) is None


def test_stats(cache):
    assert cache.stats() == {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "requests": 0, "hit_rate": None}
    cache.record("misses")
    cache.record("exact_hits")
    cache.record("semantic_hits")
    cache.record("exact_hits")
    assert cache.stats() == {"exact_hits": 2, "semantic_hits": 1, "misses": 1, "requests": 4, "hit_rate": 0.75}
    with pytest.raises(ValueError):
        cache.record("bogus")


def test_flush_only_touches_own_keys(cache, fake_redis):
    cache.set_response("q1", {"answer": "a", "verses": []})
    cache.set_embedding("q1", [1.0])
    cache.record("misses")
    fake_redis.set("unrelated", b"keep me")
    assert cache.flush() == 3
    assert cache.get_response("q1") is None
    assert cache.get_embedding("q1") is None
    assert cache.stats()["requests"] == 0
    assert fake_redis.get("unrelated") == b"keep me"


def test_disabled_cache_is_a_noop(fake_redis):
    cache = RedisCache("redis://unused", namespace="ns", embedding_model="m",
                       response_ttl=1, embedding_ttl=1, enabled=False, client=fake_redis)
    cache.set_response("q", {"answer": "a", "verses": []})
    assert cache.get_response("q") is None
    assert cache.ping() is False
    assert fake_redis.keys() == []


def test_unreachable_redis_trips_breaker_and_degrades():
    # Port 1 is never listening; the client must fail fast and then stop trying.
    cache = RedisCache("redis://127.0.0.1:1/0", namespace="ns", embedding_model="m",
                       response_ttl=1, embedding_ttl=1, breaker_seconds=60)
    assert cache.available is True
    assert cache.get_response("q") is None
    assert cache.available is False  # breaker open: subsequent calls skip the network
    cache.set_response("q", {"answer": "a", "verses": []})  # no exception
    assert cache.stats()["requests"] == 0
    assert cache.ping() is False
