"""O01.03.V01: 提示词实际生效证据 — worker 读到的 revision 对比 DB 当前 revision。"""
import os
from unittest.mock import AsyncMock

import pytest

from app.services.prompting import propagation as prop


# ── propagation_status: 纯合成逻辑 ──────────────────────────────────────
def test_verified_when_a_worker_serves_the_current_revision():
    status = prop.propagation_status(3, {3: {"workers": 2, "last_seen": "2026-11-08T22:00:00+00:00"}})
    assert status["propagation"] == "verified"
    assert status["served_revisions"] == [{"revision": 3, "workers": 2, "last_seen": "2026-11-08T22:00:00+00:00"}]
    assert status["sampled_at"] == "2026-11-08T22:00:00+00:00"


def test_stale_when_any_worker_still_serves_an_older_revision():
    status = prop.propagation_status(3, {2: {"workers": 1, "last_seen": "a"}, 3: {"workers": 1, "last_seen": "b"}})
    assert status["propagation"] == "stale"
    # newest revision first, newest last_seen reported
    assert [r["revision"] for r in status["served_revisions"]] == [3, 2]
    assert status["sampled_at"] == "b"


def test_unverified_without_any_worker_evidence():
    status = prop.propagation_status(3, {})
    assert status == {"propagation": "unverified", "served_revisions": [], "sampled_at": None}
    assert prop.propagation_status(3, None)["propagation"] == "unverified"


def test_a_worker_ahead_of_db_is_not_stale():
    # A worker on a newer revision than this read's row (its own read raced a save) is not "older".
    assert prop.propagation_status(2, {3: {"workers": 1, "last_seen": "t"}})["propagation"] == "verified"


# ── note_served: 去重, 不阻塞, 失败可重试 ───────────────────────────────
@pytest.fixture(autouse=True)
def _clear_reported(monkeypatch):
    prop._reported.clear()
    monkeypatch.setattr(prop, "_last_sent", 0.0)
    yield
    prop._reported.clear()


def test_note_served_only_sends_revisions_that_changed(monkeypatch):
    sent = []

    async def record(revisions):
        sent.append(revisions)

    monkeypatch.setattr(prop, "_record", record)
    scheduled = []
    monkeypatch.setattr(prop, "fire_background", lambda coro: scheduled.append(coro))
    prop.note_served({"a": 1, "b": 2})
    prop.note_served({"a": 1, "b": 2})  # 都没变: 不写
    prop.note_served({"a": 1, "b": 3, "c": 0})  # 只发变化的
    assert len(scheduled) == 2
    for coro in scheduled:
        try:
            coro.send(None)
        except StopIteration:
            pass
    assert sent == [{"a": 1, "b": 2}, {"b": 3, "c": 0}]


@pytest.mark.asyncio
async def test_record_failure_clears_the_dedup_so_next_read_retries(monkeypatch):
    monkeypatch.setattr(prop, "get_redis", AsyncMock(side_effect=RuntimeError("redis down")))
    prop._reported.update({"k": 3, "other": 1})
    await prop._record({"k": 3})
    assert "k" not in prop._reported and prop._reported == {"other": 1}  # 失败的那批下次重试


@pytest.mark.asyncio
async def test_observed_revisions_returns_empty_when_redis_is_unavailable(monkeypatch):
    monkeypatch.setattr(prop, "get_redis", AsyncMock(side_effect=RuntimeError("redis down")))
    assert await prop.observed_revisions() == {}


# ── Lua record + read roundtrip: 真 Redis ───────────────────────────────
@pytest.fixture
async def redis_client():
    url = os.environ.get("PROACTIVE_E2E_REDIS_URL") or os.environ.get("REDIS_URL", "")
    if not url or "127.0.0.1" not in url and "localhost" not in url:
        pytest.skip("Requires isolated Redis")
    import redis.asyncio as aioredis
    client = aioredis.from_url(url, decode_responses=True)
    await client.delete(prop._PREFIX)
    try:
        yield client
    finally:
        await client.delete(prop._PREFIX)
        await client.aclose()


@pytest.mark.asyncio
async def test_record_and_read_roundtrip_counts_workers_per_revision(redis_client, monkeypatch):
    monkeypatch.setattr(prop, "get_redis", AsyncMock(return_value=redis_client))
    await prop._record({"chat.main": 5, "memory.relevance": 2})
    monkeypatch.setattr(prop, "_INCARNATION", "second-worker")
    await prop._record({"chat.main": 5})
    await prop._record({"chat.main": 4})  # 该 worker 的快照退回旧版本(刷新失败沿用上一份)
    observed = await prop.observed_revisions()
    assert observed["chat.main"][5]["workers"] == 1 and observed["chat.main"][4]["workers"] == 1
    assert observed["memory.relevance"] == {2: {"workers": 1, "last_seen": observed["memory.relevance"][2]["last_seen"]}}
    assert observed["memory.relevance"][2]["last_seen"]
    status = prop.propagation_status(5, observed["chat.main"])
    assert status["propagation"] == "stale"


@pytest.mark.asyncio
async def test_registry_is_bounded_and_refuses_new_workers_past_the_limit(redis_client, monkeypatch):
    monkeypatch.setattr(prop, "get_redis", AsyncMock(return_value=redis_client))
    for i in range(prop._LIMIT):
        monkeypatch.setattr(prop, "_INCARNATION", f"w{i}")
        await prop._record({"k": 1})
    monkeypatch.setattr(prop, "_INCARNATION", "overflow")
    await prop._record({"k": 1})  # 满了, 放弃; 不抛
    observed = await prop.observed_revisions()
    assert observed["k"][1]["workers"] == prop._LIMIT


# ── 接入点: 快照装载与直接读取都上报 ──────────────────────────────────
@pytest.mark.asyncio
async def test_direct_prompt_read_reports_the_row_revision(monkeypatch):
    from types import SimpleNamespace as Row

    from app.services.prompting import store

    noted = []
    monkeypatch.setattr(prop, "note_served", noted.append)
    row = Row(isEnabled=True, content="hi", revision=7, id="p", key="memory.relevance")
    monkeypatch.setattr(store, "db", Row(prompttemplate=Row(find_unique=AsyncMock(return_value=row))))
    monkeypatch.setattr(store, "_sync_cache", AsyncMock(return_value=True))
    await store.get_prompt_text("memory.relevance")
    assert noted == [{"memory.relevance": 7}]


@pytest.mark.asyncio
async def test_list_prompts_carries_propagation_per_key(monkeypatch):
    from types import SimpleNamespace as Row

    from app.services.prompting import store
    from app.services.prompting.registry import PROMPT_DEFINITIONS

    key = PROMPT_DEFINITIONS[0].key
    row = Row(key=key, content="x", isEnabled=True, revision=4, updatedAt=None, webManaged=False, id="p")

    class Tx:
        async def execute_raw(self, *_):
            return None
        prompttemplate = Row(find_many=AsyncMock(return_value=[row]))

    class Ctx:
        async def __aenter__(self):
            return Tx()

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(store, "db", Row(tx=lambda: Ctx()))
    monkeypatch.setattr(store, "_publication_data", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(store, "_response", lambda d, r, publication=None: {"key": d.key})
    monkeypatch.setattr(store, "_publication_metadata", lambda *_: None)
    monkeypatch.setattr(prop, "observed_revisions", AsyncMock(return_value={key: {3: {"workers": 1, "last_seen": "t"}}}))
    out = {item["key"]: item for item in await store.list_prompts()}
    assert out[key]["propagation"]["propagation"] == "stale"
    other = next(d.key for d in PROMPT_DEFINITIONS if d.key != key)
    assert out[other]["propagation"]["propagation"] == "unverified"


def test_unchanged_revisions_are_refreshed_only_after_the_interval(monkeypatch):
    scheduled = []
    monkeypatch.setattr(prop, "fire_background", lambda coro: (scheduled.append(coro), coro.close()))
    clock = [1000.0]
    monkeypatch.setattr(prop.time, "monotonic", lambda: clock[0])
    prop.note_served({"a": 1})
    clock[0] += prop._REFRESH_SECONDS - 1
    prop.note_served({"a": 1})  # 未到续报时间
    assert len(scheduled) == 1
    clock[0] += 2
    prop.note_served({"a": 1})  # 到了: 续报, worker 不会因空闲被移除后误判
    prop.note_served({})  # 空快照不写
    assert len(scheduled) == 2


@pytest.mark.asyncio
async def test_a_failed_config_refresh_keeps_reporting_the_old_snapshot(monkeypatch):
    from app.services import runtime_config as rc

    noted = []
    monkeypatch.setattr(prop, "note_served", noted.append)
    monkeypatch.setattr(rc, "_CACHE_LOADED", True)
    monkeypatch.setattr(rc, "_PROMPT_REVISIONS", {"chat.main": 4})

    class Broken:
        def tx(self, **_):
            raise RuntimeError("db down")

    monkeypatch.setattr(rc, "db", Broken())
    await rc._load_caches_unlocked()
    assert noted == [{"chat.main": 4}]  # 仍按旧快照提供: 照实续报


@pytest.mark.asyncio
async def test_a_successful_config_load_reports_the_new_snapshot(monkeypatch):
    from types import SimpleNamespace as Row

    from app.services import runtime_config as rc

    noted = []
    monkeypatch.setattr(prop, "note_served", noted.append)
    for name in ("_GLOBAL_CACHE", "_AGENT_CACHE", "_PRICING_CACHE", "_TTS_PRICING_CACHE", "_PROMPTS_CACHE",
                 "_PROMPT_REVISIONS", "_CACHE_REVISION", "_CACHE_LOADED"):
        monkeypatch.setattr(rc, name, getattr(rc, name))
    prompts = [Row(key="chat.main", content="c", isEnabled=True, revision=6),
               Row(key="memory.relevance", content="m", isEnabled=True, revision=2)]

    class Tx:
        systemconfig = Row(find_unique=AsyncMock(return_value=Row()))
        agentconfigoverride = Row(find_many=AsyncMock(return_value=[]))
        modelregistry = Row(find_many=AsyncMock(return_value=[]))
        prompttemplate = Row(find_many=AsyncMock(return_value=prompts))

        async def execute_raw(self, *_):
            return None

        async def query_raw(self, *_):
            return [{"revision": 3}]

    class Ctx:
        async def __aenter__(self):
            return Tx()

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(rc, "db", Row(systemconfig=Row(find_unique=AsyncMock(return_value=Row())), tx=lambda **_: Ctx()))
    monkeypatch.setattr(rc, "_verify_prisma_client_fields", lambda _row: None)
    monkeypatch.setattr(rc, "_row_to_dict", lambda _row: {})
    await rc._load_caches_unlocked()
    assert noted == [{"chat.main": 6, "memory.relevance": 2}]
    assert rc._PROMPT_REVISIONS == {"chat.main": 6, "memory.relevance": 2}


@pytest.mark.asyncio
async def test_malformed_registry_rows_are_skipped_and_newest_sample_wins(monkeypatch):
    import json

    raw = {
        "bad": "not json",
        "w1": json.dumps({"served": {"k": 2}, "sampled_at": 1_800_000_100}),
        "w2": json.dumps({"served": {"k": 2}, "sampled_at": 1_800_000_000}),  # older sample, same revision
        "w3": json.dumps({"served": {"k": 2}}),  # no timestamp
    }
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value=raw)
    monkeypatch.setattr(prop, "get_redis", AsyncMock(return_value=redis))
    observed = await prop.observed_revisions()
    assert observed["k"][2]["workers"] == 3
    assert observed["k"][2]["last_seen"].startswith("2027-01-15T08:")


@pytest.mark.asyncio
async def test_admin_prompt_list_returns_propagation(monkeypatch):
    from app.api.admin import prompts as admin
    from app.services.prompting.registry import PROMPT_DEFINITIONS

    d = PROMPT_DEFINITIONS[0]
    item = {"key": d.key, "title": d.title, "stage": d.stage, "category": d.category,
            "description": d.description, "default_text": d.default_text, "content": "x", "source": "db",
            "revision": 4, "propagation": {"propagation": "stale", "sampled_at": "t",
                                           "served_revisions": [{"revision": 3, "workers": 1, "last_seen": "t"}]}}
    bare = {key: value for key, value in item.items() if key != "propagation"}
    monkeypatch.setattr(admin, "list_prompts", AsyncMock(return_value=[item, bare]))
    stale, missing = await admin.get_prompts(_="admin")
    assert stale.propagation.propagation == "stale"
    assert [r.model_dump() for r in stale.propagation.served_revisions] == [
        {"revision": 3, "workers": 1, "last_seen": "t"}]
    # 老的响应没有这个字段: 默认显示未验证, 而不是报错
    assert missing.propagation.propagation == "unverified" and missing.propagation.served_revisions == []


def test_evidence_failure_never_breaks_serving_a_prompt(monkeypatch):
    def boom(_revisions):
        raise RuntimeError("no event loop")

    monkeypatch.setattr(prop, "note_served", boom)
    prop.note_served_safely({"k": 1})  # swallowed


@pytest.mark.asyncio
async def test_rows_without_a_revision_still_serve(monkeypatch):
    # Older fakes and partial rows carry no revision; the read must not fail on evidence.
    from types import SimpleNamespace as Row

    from app.services.prompting import store

    noted = []
    monkeypatch.setattr(prop, "note_served", noted.append)
    row = Row(isEnabled=True, content="hi", id="p", key="memory.relevance")
    monkeypatch.setattr(store, "db", Row(prompttemplate=Row(find_unique=AsyncMock(return_value=row))))
    monkeypatch.setattr(store, "_sync_cache", AsyncMock(return_value=True))
    assert "hi" in await store.get_prompt_text("memory.relevance")
    assert noted == [{"memory.relevance": 0}]
