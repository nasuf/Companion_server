"""Test configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── 共享 JWT / TestClient helpers ────────────────────────────────────────

def make_auth_header(user_id: str = "user-id", role: str = "user") -> dict:
    """Build Authorization header with a valid JWT for the given user."""
    from app.services.auth import create_jwt
    return {"Authorization": f"Bearer {create_jwt(user_id, role=role)}"}


@pytest.fixture
def auth_header():
    """Convenience fixture returning the helper itself.

    Usage: `auth_header("user-id")` or `auth_header("admin", role="admin")`.
    """
    return make_auth_header


@pytest.fixture
def api_client():
    """TestClient(app) with all external deps mocked. Replaces the inline
    `mock_deps` fixture in test_api_integration.py / test_public_endpoints_ownership.py.
    """
    with (
        patch("app.db.connect_db", new_callable=AsyncMock),
        patch("app.db.disconnect_db", new_callable=AsyncMock),
        patch("app.redis_client.get_redis", new_callable=AsyncMock),
        patch("app.redis_client.close_redis", new_callable=AsyncMock),
        patch("jobs.scheduler.setup_scheduler"),
        patch("jobs.scheduler.shutdown_scheduler"),
        patch("app.middleware.configure_logging"),
        patch("app.middleware.configure_langsmith"),
    ):
        from app.main import app
        yield TestClient(app)


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
    with patch("app.services.interaction.boundary.get_redis", return_value=mock_redis):
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


class FakeAggregationRedis:
    """In-memory Redis stub for aggregation queue tests.

    Supports rpush / lrange / set / get / del / zadd / zrangebyscore / zrem
    / expire / pipeline / eval — the minimum subset needed by
    push_pending / flush_pending / scan_expired.
    Extend with lock CAS via FakeRedis if ever both contracts are needed
    in the same test.
    """

    def __init__(self) -> None:
        from collections import defaultdict
        self.lists: dict[str, list[str]] = defaultdict(list)
        self.strings: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = defaultdict(dict)

    def pipeline(self):
        return FakeAggregationPipeline(self)

    async def rpush(self, key, *values):
        self.lists[key].extend(values)
        return len(self.lists[key])

    async def expire(self, key, ttl):  # noqa: ARG002 — interface shim
        return True

    async def set(self, key, value, *, ex=None, nx=False):  # noqa: ARG002 — interface shim
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        return True

    async def get(self, key):
        return self.strings.get(key)

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.lists:
                del self.lists[k]; n += 1
            if k in self.strings:
                del self.strings[k]; n += 1
        return n

    async def zadd(self, key, mapping):
        self.zsets[key].update(mapping)
        return len(mapping)

    async def zrangebyscore(self, key, min_, max_):
        return [m for m, s in sorted(self.zsets.get(key, {}).items(), key=lambda x: x[1])
                if min_ <= s <= max_]

    async def zrem(self, key, *members):
        zs = self.zsets.get(key, {})
        n = 0
        for m in members:
            if m in zs:
                del zs[m]; n += 1
        return n

    async def eval(self, script, numkeys, *args):  # noqa: ARG002 — aggregation Lua is hard-coded below
        """Emulate the aggregation Lua script: LRANGE + GET + DEL + ZREM."""
        keys = args[:numkeys]
        argv = args[numkeys:]
        msgs = self.lists.get(keys[0], [])
        if not msgs:
            return None
        conv_id = self.strings.get(keys[2])
        ctx = self.strings.get(keys[3])
        msgs = list(msgs)
        if keys[0] in self.lists:
            del self.lists[keys[0]]
        self.zsets.get(keys[1], {}).pop(argv[0], None)
        for k in (keys[2], keys[3]):
            self.strings.pop(k, None)
        return [conv_id, ctx, *msgs]


class FakeAggregationPipeline:
    def __init__(self, parent: FakeAggregationRedis) -> None:
        self.parent = parent
        self.ops: list[tuple] = []

    def rpush(self, key, value): self.ops.append(("rpush", key, value))
    def expire(self, key, ttl): self.ops.append(("expire", key, ttl))
    def set(self, key, value, ex=None): self.ops.append(("set", key, value, ex))
    def zadd(self, key, mapping): self.ops.append(("zadd", key, mapping))
    def delete(self, *keys): self.ops.append(("delete", *keys))

    async def execute(self):
        for op in self.ops:
            if op[0] == "rpush":
                await self.parent.rpush(op[1], op[2])
            elif op[0] == "expire":
                await self.parent.expire(op[1], op[2])
            elif op[0] == "set":
                await self.parent.set(op[1], op[2], ex=op[3])
            elif op[0] == "zadd":
                await self.parent.zadd(op[1], op[2])
            elif op[0] == "delete":
                await self.parent.delete(*op[1:])


@pytest.fixture
def fake_aggregation_redis() -> FakeAggregationRedis:
    """Fresh FakeAggregationRedis per test."""
    return FakeAggregationRedis()
