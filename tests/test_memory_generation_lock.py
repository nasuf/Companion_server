"""Unit tests for memory_generation_lock (Redis SET NX + Lua atomic release)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.generation_lock import (
    MemoryGenerationLocked,
    _RELEASE_LUA,
    memory_generation_lock,
)


@pytest.mark.asyncio
async def test_lock_acquire_and_release(fake_redis):
    with patch("app.services.memory.generation_lock.get_redis",
               AsyncMock(return_value=fake_redis)):
        async with memory_generation_lock("agent-a"):
            # While held, key exists.
            assert "memgen:lock:agent-a" in fake_redis._store
        # After release, key is gone.
        assert "memgen:lock:agent-a" not in fake_redis._store


@pytest.mark.asyncio
async def test_reacquire_while_held_raises(fake_redis):
    with patch("app.services.memory.generation_lock.get_redis",
               AsyncMock(return_value=fake_redis)):
        async with memory_generation_lock("agent-b"):
            with pytest.raises(MemoryGenerationLocked):
                async with memory_generation_lock("agent-b"):
                    pytest.fail("second acquire should have raised")


@pytest.mark.asyncio
async def test_release_does_not_delete_foreign_lock(fake_redis):
    """If our lock expired and another worker grabbed it with a different
    token, our release MUST NOT delete theirs."""
    with patch("app.services.memory.generation_lock.get_redis",
               AsyncMock(return_value=fake_redis)):
        async with memory_generation_lock("agent-c"):
            # Simulate TTL expiry + another worker re-acquiring with new token.
            fake_redis._store["memgen:lock:agent-c"] = "foreign-token"
        # Foreign token must still be intact — our release was a no-op.
        assert fake_redis._store.get("memgen:lock:agent-c") == "foreign-token"


@pytest.mark.asyncio
async def test_release_survives_redis_error(fake_redis):
    """Release failures should not propagate."""
    fake_redis.eval = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("app.services.memory.generation_lock.get_redis",
               AsyncMock(return_value=fake_redis)):
        # Should not raise.
        async with memory_generation_lock("agent-d"):
            pass


def test_release_lua_shape():
    """Sanity: script contains CAS pattern."""
    assert "redis.call('get'" in _RELEASE_LUA
    assert "redis.call('del'" in _RELEASE_LUA
    assert "ARGV[1]" in _RELEASE_LUA
