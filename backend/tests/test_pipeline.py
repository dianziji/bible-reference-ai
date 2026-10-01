import pytest

from app.services.cache import RedisCache, SemanticHit
from app.services.messages import INSUFFICIENT_CONTEXT, MODERATION_REFUSED
from app.services.query_pipeline import QueryPipeline
from tests.conftest import FakeRAG, FakeSemantic


def make_pipeline(cache, executor, rag=None, semantic=None, safe=True):
    rag = rag or FakeRAG()
    semantic = semantic or FakeSemantic()
    return QueryPipeline(rag, cache, semantic, lambda q: safe, executor), rag, semantic


def test_miss_generates_and_populates_both_tiers(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor)

    result = pipeline.run("What does the Bible say about creation?")

    assert result.cache == "miss"
    assert (result.language, result.translation) == ("en", "bsb")
    assert result.answer == "ANSWER"
    assert [v["id"] for v in result.verses] == ["Genesis.1.1", "John.1.1"]
    assert result.verses[0]["score"] == 0.71
    assert result.verses[0]["text_zh"] == "起初，　神创造天地。"
    assert result.verses[0]["book_zh"] == "创世记"
    assert result.embedding_cached is False
    assert (rag.embed_calls, rag.retrieve_calls, rag.generate_calls) == (1, ["bsb"], ["en"])
    assert set(result.timings_ms) == {"embed", "retrieve", "moderation", "semantic_lookup", "generate", "total"}

    # L1 holds the answer, the embedding is cached, L2 got the vector tagged with the language.
    assert cache.get_response("what does the bible say about creation") == {"answer": "ANSWER", "verses": result.verses}
    assert cache.get_embedding("What does the Bible say about creation?") is not None
    key = cache.response_key("what does the bible say about creation")
    assert semantic.stored == [("What does the Bible say about creation?", key, "en")]
    assert semantic.lookups == ["en"]
    assert cache.stats()["misses"] == 1


def test_chinese_question_uses_the_chinese_translation(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor)

    result = pipeline.run("耶稣是谁？")

    assert (result.language, result.translation) == ("zh", "cuv")
    assert rag.retrieve_calls == ["cuv"]
    assert rag.generate_calls == ["zh"]
    assert semantic.lookups == ["zh"]
    assert semantic.stored[0][2] == "zh"


def test_exact_hit_skips_every_upstream_call(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor)
    first = pipeline.run("What is love?")
    assert first.cache == "miss"

    # Different casing / punctuation. Moderation is swapped for one that would
    # refuse, proving an exact hit never reaches it (or any other upstream call).
    pipeline.moderation_fn = lambda q: False
    second = pipeline.run("  what IS love ")

    assert second.cache == "exact"
    assert second.answer == first.answer
    assert second.verses == first.verses
    assert (rag.embed_calls, rag.retrieve_calls, rag.generate_calls) == (1, ["bsb"], ["en"])
    assert semantic.lookups == ["en"]
    assert set(second.timings_ms) == {"total"}
    assert cache.stats() == {"exact_hits": 1, "semantic_hits": 0, "misses": 1, "requests": 2, "hit_rate": 0.5}


def test_semantic_hit_reuses_answer_without_llm(cache, executor):
    cache.set_response("who is jesus", {"answer": "He is the Christ.", "verses": [{"id": "x", "book": "John", "chapter": 1, "verse": 1, "text": "t", "score": 0.6}]})
    hit = SemanticHit(response_key=cache.response_key("who is jesus"), question="who is jesus", score=0.971)
    pipeline, rag, semantic = make_pipeline(cache, executor, semantic=FakeSemantic(hit=hit))

    result = pipeline.run("Who was Jesus?")

    assert result.cache == "semantic"
    assert result.answer == "He is the Christ."
    assert result.semantic_similarity == 0.971
    assert rag.embed_calls == 1  # the vector is needed for the lookup
    assert rag.generate_calls == []
    assert cache.get_response("Who was Jesus?") is None  # only the original question is in L1
    assert cache.stats()["semantic_hits"] == 1


def test_semantic_hit_with_expired_answer_falls_through(cache, executor):
    hit = SemanticHit(response_key="bible:v1:test:resp:expired", question="q", score=0.99)
    pipeline, rag, semantic = make_pipeline(cache, executor, semantic=FakeSemantic(hit=hit))

    result = pipeline.run("Who was Jesus?")

    assert result.cache == "miss"
    assert rag.generate_calls == ["en"]
    assert len(semantic.stored) == 1  # refreshed
    assert cache.stats() == {"exact_hits": 0, "semantic_hits": 0, "misses": 1, "requests": 1, "hit_rate": 0.0}


def test_moderation_refusal_is_never_cached(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor, safe=False)

    result = pipeline.run("something unsafe")

    assert result.flagged is True
    assert result.answer == MODERATION_REFUSED
    assert result.verses == []
    assert rag.generate_calls == []
    assert cache.get_response("something unsafe") is None
    assert semantic.stored == []
    assert cache.stats()["requests"] == 0


def test_no_verses_above_threshold_is_not_cached(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor, rag=FakeRAG(docs=[]))

    result = pipeline.run("What is the airspeed of a swallow?")

    assert result.cache == "miss"
    assert result.answer == INSUFFICIENT_CONTEXT
    assert rag.generate_calls == []
    assert cache.get_response("What is the airspeed of a swallow?") is None
    assert semantic.stored == []
    assert cache.stats()["misses"] == 1


def test_embedding_cache_survives_answer_expiry(cache, executor, fake_redis):
    pipeline, rag, semantic = make_pipeline(cache, executor)
    pipeline.run("What is grace?")
    fake_redis.delete(cache.response_key("What is grace?"))  # answer expired, embedding still there

    result = pipeline.run("What is grace?")

    assert result.cache == "miss"
    assert result.embedding_cached is True
    assert rag.embed_calls == 1
    assert rag.generate_calls == ["en", "en"]


def test_redis_down_still_answers(executor):
    down = RedisCache("redis://127.0.0.1:1/0", namespace="ns", embedding_model="m",
                      response_ttl=1, embedding_ttl=1, breaker_seconds=60)
    pipeline, rag, semantic = make_pipeline(down, executor)

    result = pipeline.run("What is faith?")

    assert result.cache == "miss"
    assert result.answer == "ANSWER"
    assert rag.generate_calls == ["en"]
    assert semantic.lookups == []  # pointless without Redis to hold the answer
    assert pipeline.run("What is faith?").cache == "miss"  # no L1 without Redis


def test_semantic_cache_disabled(cache, executor):
    pipeline, rag, semantic = make_pipeline(cache, executor, semantic=FakeSemantic(enabled=False))
    result = pipeline.run("What is hope?")
    assert result.cache == "miss"
    assert semantic.lookups == []
    assert semantic.stored == []
    assert "semantic_lookup" not in result.timings_ms
