import dataclasses

import pytest

from app.config import default_reasoning_effort, load_settings, settings, supports_temperature


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4o-mini", True),
        ("gpt-4.1-nano", True),
        ("gpt-5-mini", False),
        ("gpt-5.4-nano", False),
        ("gpt-6-luna", False),
        ("o4-mini", False),
    ],
)
def test_supports_temperature(model, expected):
    assert supports_temperature(model) is expected


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4o-mini", None),
        ("gpt-4.1-mini", None),
        ("gpt-5-mini", "minimal"),
        ("gpt-5-nano", "minimal"),
        ("gpt-5.1", "none"),
        ("gpt-5.4-nano", "none"),
        ("gpt-6-luna", "none"),
        ("gpt-6.1-sol", "none"),
        ("o3", "low"),
    ],
)
def test_default_reasoning_effort(model, expected):
    assert default_reasoning_effort(model) == expected


def test_env_overrides_and_defaults(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "gpt-5-mini")
    monkeypatch.delenv("CHAT_REASONING_EFFORT", raising=False)
    assert load_settings().chat_reasoning_effort == "minimal"  # derived from the model

    monkeypatch.setenv("CHAT_REASONING_EFFORT", "low")
    assert load_settings().chat_reasoning_effort == "low"  # explicit wins

    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.42")
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "off")
    s = load_settings()
    assert (s.top_k, s.score_threshold, s.semantic_cache_enabled) == (8, 0.42, False)


def test_redis_url_assembled_from_host_port(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "bible-redis-svc")
    monkeypatch.setenv("REDIS_PORT", "6380")
    assert load_settings().redis_url == "redis://bible-redis-svc:6380/0"
    monkeypatch.setenv("REDIS_URL", "redis://elsewhere:1/2")
    assert load_settings().redis_url == "redis://elsewhere:1/2"


def test_fingerprint_rolls_over_when_answer_inputs_change():
    base = settings.fingerprint
    assert len(base) == 8
    assert dataclasses.replace(settings, chat_model="other-model").fingerprint != base
    assert dataclasses.replace(settings, chat_reasoning_effort="high").fingerprint != base
    assert dataclasses.replace(settings, top_k=settings.top_k + 1).fingerprint != base
    assert dataclasses.replace(settings, score_threshold=0.99).fingerprint != base
    # Cache plumbing does not change answers, so it must not change the namespace.
    assert dataclasses.replace(settings, response_ttl_seconds=1, redis_url="redis://x").fingerprint == base
