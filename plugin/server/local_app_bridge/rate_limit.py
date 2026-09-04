from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from plugin.server.local_app_bridge.contracts import (
    MAX_RATE_LIMIT_BUCKETS,
    PAIR_RATE_LIMIT,
    PAIR_RATE_WINDOW_SECONDS,
    SESSION_RATE_LIMIT,
    SESSION_RATE_WINDOW_SECONDS,
)
from plugin.server.local_app_bridge.errors import LocalAppBridgeError


@dataclass(slots=True)
class _Bucket:
    timestamps: deque[float] = field(default_factory=deque)
    last_seen: float = 0.0


class LocalAppRateLimiter:
    """Bounded monotonic-clock sliding windows for pair and session traffic."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_buckets: int = MAX_RATE_LIMIT_BUCKETS,
    ) -> None:
        if max_buckets < 2:
            raise ValueError("max_buckets must be at least 2")
        self._clock = clock
        self._max_buckets = max_buckets
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def check_pair(self, *, app_id: str, client_id: str) -> None:
        # The global bucket prevents bypass by churning forged client ids; the
        # bound identity bucket keeps one app launch from consuming all attempts.
        await self._check(
            "pair:global", limit=PAIR_RATE_LIMIT * 4, window=PAIR_RATE_WINDOW_SECONDS
        )
        await self._check(
            f"pair:{app_id}:{client_id}",
            limit=PAIR_RATE_LIMIT,
            window=PAIR_RATE_WINDOW_SECONDS,
        )

    async def check_session(self, session_id: str) -> None:
        await self._check(
            f"session:{session_id}",
            limit=SESSION_RATE_LIMIT,
            window=SESSION_RATE_WINDOW_SECONDS,
        )

    async def cleanup(self) -> None:
        async with self._lock:
            self._cleanup_locked(
                self._clock(),
                max(PAIR_RATE_WINDOW_SECONDS, SESSION_RATE_WINDOW_SECONDS),
            )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._buckets.clear()

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    async def _check(self, key: str, *, limit: int, window: float) -> None:
        now = self._clock()
        async with self._lock:
            if self._closed:
                raise LocalAppBridgeError("bridge_closed", 503, "Bridge is unavailable")
            self._cleanup_locked(
                now, max(PAIR_RATE_WINDOW_SECONDS, SESSION_RATE_WINDOW_SECONDS)
            )
            bucket = self._buckets.get(key)
            if bucket is None:
                self._ensure_capacity_locked()
                bucket = _Bucket(last_seen=now)
                self._buckets[key] = bucket
            cutoff = now - window
            while bucket.timestamps and bucket.timestamps[0] <= cutoff:
                bucket.timestamps.popleft()
            bucket.last_seen = now
            if len(bucket.timestamps) >= limit:
                retry_after = max(0.001, bucket.timestamps[0] + window - now)
                raise LocalAppBridgeError(
                    "rate_limited",
                    429,
                    "Too many requests",
                    retry_after=retry_after,
                )
            bucket.timestamps.append(now)

    def _cleanup_locked(self, now: float, retention: float) -> None:
        cutoff = now - retention
        for key, bucket in tuple(self._buckets.items()):
            if bucket.last_seen <= cutoff:
                self._buckets.pop(key, None)

    def _ensure_capacity_locked(self) -> None:
        if len(self._buckets) < self._max_buckets:
            return
        oldest_key = min(self._buckets, key=lambda key: self._buckets[key].last_seen)
        self._buckets.pop(oldest_key, None)
