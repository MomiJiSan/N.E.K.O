from __future__ import annotations

from dataclasses import dataclass

import pytest

from plugin.server.local_app_bridge.errors import LocalAppBridgeError
from plugin.server.local_app_bridge.rate_limit import LocalAppRateLimiter

pytestmark = pytest.mark.plugin_unit


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_session_rate_limit_is_thirty_requests_per_ten_seconds() -> None:
    clock = _Clock()
    limiter = LocalAppRateLimiter(clock=clock)
    for _ in range(30):
        await limiter.check_session("session-1")
    with pytest.raises(LocalAppBridgeError) as limited:
        await limiter.check_session("session-1")
    assert limited.value.status_code == 429
    assert limited.value.retry_after == 10.0
    clock.now = 10.0
    await limiter.check_session("session-1")


@pytest.mark.asyncio
async def test_pair_limit_is_stricter_and_scoped_to_bound_identity() -> None:
    clock = _Clock()
    limiter = LocalAppRateLimiter(clock=clock)
    for _ in range(5):
        await limiter.check_pair(app_id="demo.app", client_id="client-1")
    with pytest.raises(LocalAppBridgeError) as limited:
        await limiter.check_pair(app_id="demo.app", client_id="client-1")
    assert limited.value.code == "rate_limited"
    await limiter.check_pair(app_id="demo.app", client_id="client-2")


@pytest.mark.asyncio
async def test_rate_limit_buckets_are_bounded_evicted_and_cleaned() -> None:
    clock = _Clock()
    limiter = LocalAppRateLimiter(clock=clock, max_buckets=3)
    await limiter.check_session("one")
    clock.now += 1
    await limiter.check_session("two")
    clock.now += 1
    await limiter.check_session("three")
    clock.now += 1
    await limiter.check_session("four")
    assert limiter.bucket_count == 3
    clock.now += 61
    await limiter.cleanup()
    assert limiter.bucket_count == 0
    await limiter.close()
    with pytest.raises(LocalAppBridgeError) as closed:
        await limiter.check_session("five")
    assert closed.value.code == "bridge_closed"
