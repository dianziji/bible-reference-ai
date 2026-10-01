"""Sliding-window rate limiting on Redis.

Two fixed windows are kept per key, the current one and the previous one, and
the previous count is weighted by how much of the current window is still
ahead. That smooths out the double-sized burst a plain fixed window allows at
each boundary, for the price of one INCR and one GET per check.

If Redis is unavailable the limiter fails open: an outage of the cache should
cost latency, not availability.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from app.services.cache import RedisCache


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds; only meaningful when not allowed


class RateLimiter:
    def __init__(self, cache: RedisCache, window_seconds: int = 60):
        self.cache = cache
        self.window = window_seconds

    def check(self, key: str, limit: int, now: Optional[float] = None) -> Decision:
        """Count one hit for ``key`` and say whether it stays within ``limit`` per window."""
        if limit <= 0:  # disabled
            return Decision(True, limit, limit, 0)
        now = time.time() if now is None else now
        counts = self.cache.window_counts(key, self.window, now)
        if counts is None:  # Redis down: fail open
            return Decision(True, limit, limit, 0)

        previous, current = counts
        elapsed = now % self.window
        estimate = previous * (1 - elapsed / self.window) + current
        if estimate > limit:
            return Decision(False, limit, 0, max(1, math.ceil(self.window - elapsed)))
        return Decision(True, limit, max(0, int(limit - estimate)), 0)


def client_ip(peer: str, forwarded_for: Optional[str], xff_index: int) -> str:
    """Pick the client address out of X-Forwarded-For.

    Proxies append to the header, so the trustworthy entries are at the end:
    nginx puts the client last (index -1); Google's L7 load balancer appends
    ``<client>, <lb-ip>`` (client at -2). Anything a client sent itself sits
    further left and is ignored. Without enough entries the header did not
    come through the expected proxy, so the TCP peer is used instead.
    """
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if xff_index < 0 and len(parts) >= -xff_index:
            return parts[xff_index]
        if 0 <= xff_index < len(parts):
            return parts[xff_index]
    return peer
