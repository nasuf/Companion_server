"""Test configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db():
    """Mock Prisma DB client."""
    mock = MagicMock()
    mock.memory = MagicMock()
    mock.memory.find_many = AsyncMock(return_value=[])
    mock.memory.find_unique = AsyncMock(return_value=None)
    mock.memory.create = AsyncMock()
    mock.memory.update = AsyncMock()
    mock.memory.count = AsyncMock(return_value=0)
    mock.message = MagicMock()
    mock.message.find_many = AsyncMock(return_value=[])
    mock.message.create = AsyncMock()
    mock.message.count = AsyncMock(return_value=0)
    mock.conversation = MagicMock()
    mock.conversation.find_unique = AsyncMock(return_value=None)
    mock.aiagent = MagicMock()
    mock.aiagent.find_unique = AsyncMock(return_value=None)
    mock.aiemotionstate = MagicMock()
    mock.aiemotionstate.find_unique = AsyncMock(return_value=None)
    mock.aiemotionstate.upsert = AsyncMock()
    mock.execute_raw = AsyncMock()
    mock.query_raw = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.hgetall = AsyncMock(return_value={})
    mock.hset = AsyncMock()
    mock.expire = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.incr = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def patch_boundary_redis(mock_redis):
    """Auto-patch get_redis for boundary service tests."""
    with patch("app.services.relationship.boundary.get_redis", return_value=mock_redis):
        yield mock_redis


@pytest.fixture
def patch_intimacy_redis(mock_redis):
    """Auto-patch get_redis for intimacy service tests."""
    with patch("app.services.relationship.intimacy.get_redis", return_value=mock_redis):
        yield mock_redis


class FakeRedis:
    """In-memory Redis stub for lock + SET/GET/EVAL assertions.

    Supports `set(key, value, nx=?, ex=?)`, `get`, `delete`, and `eval` with
    a Lua-compatible compare-and-delete emulation. Tests that need richer
    behavior (pubsub, hashes) should mock those methods on top.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value, *, nx: bool = False, ex: int | None = None):
        if nx and key in self._store:
            return False
        self._store[key] = value
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def delete(self, key: str):
        return self._store.pop(key, None) is not None

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        # Emulate the only Lua pattern we use: CAS compare-and-delete.
        keys = keys_and_args[:numkeys]
        args = keys_and_args[numkeys:]
        key = keys[0]
        token = args[0]
        if self._store.get(key) == token:
            self._store.pop(key, None)
            return 1
        return 0


@pytest.fixture
def fake_redis() -> FakeRedis:
    """Fresh FakeRedis per test."""
    return FakeRedis()
