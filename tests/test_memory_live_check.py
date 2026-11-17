"""组装 prompt 前的记忆存活校验: 已删/已归档/跨作用域的召回不得进入 prompt。"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.retrieval import live_check
from app.services.memory.retrieval.context_selector import ClassifiedMemory


def _m(mid, source="user"):
    return ClassifiedMemory(text="t-" + mid, relevance="strong", score=0.9, id=mid, source=source)


@pytest.mark.asyncio
async def test_drops_rows_no_longer_live_per_side():
    async def _query(sql, ids, user_id, workspace_id):
        assert user_id == "u1" and workspace_id == "w1"
        live = {"memories_user": {"u-live"}, "memories_ai": {"a-live"}}
        table = "memories_ai" if "memories_ai" in sql else "memories_user"
        return [{"id": i} for i in ids if i in live[table]]

    memories = [_m("u-live"), _m("u-gone"), _m("a-live", "ai"), _m("u-live", "ai"),
                ClassifiedMemory(text="legacy", relevance="medium", score=0.5)]
    diag: dict = {}
    with patch.object(live_check, "db") as db:
        db.query_raw = AsyncMock(side_effect=_query)
        kept = await live_check.keep_live_memories(memories, user_id="u1", workspace_id="w1",
                                                   diagnostics=diag)
    assert [(m.id, m.source) for m in kept] == [("u-live", "user"), ("a-live", "ai"), ("", "user")]
    assert diag["memory_live_check"] == {"status": "checked", "checked": 4, "dropped": 2}


@pytest.mark.asyncio
async def test_check_failure_drops_every_checkable_row():
    diag: dict = {}
    with patch.object(live_check, "db") as db:
        db.query_raw = AsyncMock(side_effect=RuntimeError("db down"))
        kept = await live_check.keep_live_memories(
            [_m("u1"), ClassifiedMemory(text="legacy", relevance="medium", score=0.5)],
            user_id="u1", workspace_id="w1", diagnostics=diag)
    assert [m.text for m in kept] == ["legacy"]
    assert diag["memory_live_check"]["status"] == "failed_closed"


@pytest.mark.asyncio
async def test_empty_or_unidentified_input_skips_the_query():
    with patch.object(live_check, "db") as db:
        db.query_raw = AsyncMock()
        assert await live_check.keep_live_memories(None, user_id="u", workspace_id="w") is None
        legacy = [ClassifiedMemory(text="x", relevance="medium", score=0.5)]
        assert await live_check.keep_live_memories(legacy, user_id="u", workspace_id="w") == legacy
    db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_written_after_a_bump_lands_under_the_old_version(monkeypatch):
    from app.services.runtime import cache

    store: dict = {}
    versions = {"v": 3}

    async def _ver(*_a):
        return versions["v"]

    async def _get(ns, ident):
        return store.get((ns, ident))

    async def _set(ns, ident, value, ttl=0):
        store[(ns, ident)] = value

    monkeypatch.setattr(cache, "get_cache_version", _ver)
    monkeypatch.setattr(cache, "cache_get", _get)
    monkeypatch.setattr(cache, "cache_set", _set)

    @asynccontextmanager
    async def _scope(_uid):
        yield None

    from app.services.runtime import user_erasure
    monkeypatch.setattr(user_erasure, "owner_write_scope", _scope)
    pinned = await cache.get_cache_version("ret", "u1", "w1")
    versions["v"] = 4  # 召回进行中用户遗忘了一条记忆
    await cache.cache_set_retrieval("q", "u1", {"memories": ["stale"]}, "w1", version=pinned)
    assert await cache.cache_retrieval("q", "u1", "w1") is None
    assert await cache.cache_retrieval("q", "u1", "w1", version=3) == {"memories": ["stale"]}
