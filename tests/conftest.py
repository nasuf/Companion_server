"""Test configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
    return mock
