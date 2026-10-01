import dataclasses

import pytest

from app.services.cache import RedisCache
from app.services.rate_limiter import RateLimiter, client_ip


@pytest.mark.parametrize(
    "peer, xff, index, expected",
    [
        ("10.0.0.5", None, -1, "10.0.0.5"),                              # no proxy header: TCP peer
        ("10.0.0.5", "203.0.113.9", -1, "203.0.113.9"),                  # nginx: client is last
        ("10.0.0.5", "1.2.3.4, 203.0.113.9", -1, "203.0.113.9"),         # client-supplied junk ignored
        ("10.0.0.5", "203.0.113.9, 35.190.0.1", -2, "203.0.113.9"),      # Google L7 LB appends client, lb
        ("10.0.0.5", "1.2.3.4, 203.0.113.9, 35.190.0.1", -2, "203.0.113.9"),
        ("10.0.0.5", "203.0.113.9", -2, "10.0.0.5"),                     # too short for the LB shape: not via LB
        ("10.0.0.5", " 203.0.113.9 ,  35.190.0.1 ", -2, "203.0.113.9"),  # whitespace tolerated
        ("10.0.0.5", "", -1, "10.0.0.5"),
    ],
)
def test_client_ip(peer, xff, index, expected):
    assert client_ip(peer, xff, index) == expected


def test_allows_up_to_limit_then_denies(cache):
    limiter = RateLimiter(cache, window_seconds=60)
    t = 1_000_000.0  # start of a window
    decisions = [limiter.check("ip:a", 3, now=t + i) for i in range(5)]
    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert [d.remaining for d in decisions[:3]] == [2, 1, 0]
    assert decisions[3].retry_after >= 1
    # Other keys are independent.
    assert limiter.check("ip:b", 3, now=t + 5).allowed is True


def test_previous_window_is_weighted_by_elapsed_time(cache):
    limiter = RateLimiter(cache, window_seconds=60)
    t0 = 1_000_020.0  # some point inside window W
    for _ in range(6):
        limiter.check("ip:a", 6, now=t0)  # fill window W exactly to the limit

    early = 1_000_020.0 + 40 + 1  # 1s into window W+1: previous still weighs ~98%
    assert limiter.check("ip:a", 6, now=early).allowed is False

    late = 1_000_020.0 + 40 + 59  # 59s into W+1: previous weighs ~2%
    assert limiter.check("ip:a", 6, now=late).allowed is True


def test_disabled_when_limit_is_zero(cache):
    limiter = RateLimiter(cache)
    assert all(limiter.check("ip:a", 0).allowed for _ in range(50))


def test_fails_open_without_redis():
    down = RedisCache("redis://127.0.0.1:1/0", namespace="ns", embedding_model="m",
                      response_ttl=1, embedding_ttl=1, breaker_seconds=60)
    limiter = RateLimiter(down)
    assert all(limiter.check("ip:a", 1).allowed for _ in range(5))


def test_windows_do_not_share_the_answer_namespace(cache, fake_redis):
    RateLimiter(cache).check("ip:a", 5, now=1_000_000.0)
    keys = [k.decode() for k in fake_redis.keys()]
    assert keys and all(k.startswith("bible:v1:rl:ip:a:") for k in keys)
    assert cache.flush() == 0  # flushing the cache must not reset quotas
