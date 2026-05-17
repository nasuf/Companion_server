from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.services import auth_security


class FakeRequest:
    def __init__(self, ip: str = "1.2.3.4"):
        self.headers = {}
        self.client = SimpleNamespace(host=ip)


@pytest.mark.asyncio
async def test_login_rate_limit_blocks_after_failure_threshold():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=str(auth_security._LOGIN_MAX_FAILURES))

    with patch.object(auth_security, "get_redis", AsyncMock(return_value=redis)):
        with pytest.raises(HTTPException) as exc:
            await auth_security.enforce_login_rate_limit(FakeRequest(), "alice")

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_login_failure_increments_and_sets_ttl_on_first_failure():
    redis = AsyncMock()
    redis.incr = AsyncMock(return_value=1)
    redis.expire = AsyncMock()

    with patch.object(auth_security, "get_redis", AsyncMock(return_value=redis)):
        await auth_security.record_login_failure(FakeRequest(), "alice")

    redis.incr.assert_awaited_once()
    redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_rate_limit_blocks_after_threshold():
    redis = AsyncMock()
    redis.incr = AsyncMock(return_value=auth_security._REGISTER_MAX_ATTEMPTS + 1)
    redis.expire = AsyncMock()

    with patch.object(auth_security, "get_redis", AsyncMock(return_value=redis)):
        with pytest.raises(HTTPException) as exc:
            await auth_security.enforce_register_rate_limit(FakeRequest())

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_fails_open_on_redis_error():
    with patch.object(
        auth_security,
        "get_redis",
        AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        await auth_security.enforce_login_rate_limit(FakeRequest(), "alice")
        await auth_security.enforce_register_rate_limit(FakeRequest())
