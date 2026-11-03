"""Admin "真人判定": offline IP region, signal collection, scoring and the admin API.

Database cases run on the isolated PostgreSQL used by the other *_postgres tests.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma import Prisma

from app.services.authenticity import ip_region, scoring, signals
from scripts import backfill_authenticity as backfill
from tests.test_chat_endpoint_security import secured_chat  # noqa: F401

NOW = datetime(2027, 4, 2, 4, 0, tzinfo=timezone.utc)


# ── offline region ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("ip,expected,kind", [
    ("120.230.1.1", ("中国", "广东省", "汕头市", "移动"), "carrier"),
    ("106.52.115.80", ("中国", "广东省", "广州市", "腾讯"), "datacenter"),
    ("47.96.0.1", ("中国", "浙江省", "杭州市", "阿里"), "datacenter"),
    ("202.112.1.1", ("中国", "北京市", "北京市", "中国教育网"), "campus"),
    ("8.8.8.8", ("United States", "California", None, "Google LLC"), "datacenter"),
    ("10.0.0.1", (None, None, None, None), "unknown"),
    ("::1", (None, None, None, None), "unknown"),
    ("not-an-ip", (None, None, None, None), "unknown"),
])
def test_offline_lookup_and_network_kind(ip, expected, kind):
    region = ip_region.lookup(ip)
    assert (region.country, region.province, region.city, region.isp) == expected
    assert ip_region.network_kind(region) == kind


def test_network_kind_overseas_and_unknown_carrier():
    assert ip_region.network_kind(ip_region.IpRegion("Japan", None, None, "KDDI CORPORATION")) == "overseas"
    assert ip_region.network_kind(ip_region.IpRegion("中国", "江苏省", None, None)) == "unknown"
    assert ip_region.network_kind(ip_region.IpRegion("United States", None, None, "Google Fiber Inc.")) == "overseas"


def test_lookup_without_segment_returns_empty(monkeypatch):
    # A zero vector-index entry means the upstream data has no block for it.
    data = bytearray(ip_region._buffer())
    offset = 256 + (1 * 256 + 0) * 8  # vector entry for 1.0.x.x
    data[offset:offset + 8] = b"\x00" * 8
    monkeypatch.setattr(ip_region, "_buffer", lambda: bytes(data))
    assert ip_region.lookup("1.0.0.1") == ip_region._EMPTY


def test_unsupported_data_is_rejected(monkeypatch, tmp_path):
    import gzip
    path = tmp_path / "x.xdb.gz"
    path.write_bytes(gzip.compress(b"\x02\x00" + b"\x00" * 300))
    monkeypatch.setattr(ip_region, "DATA_PATH", path)
    ip_region._buffer.cache_clear()
    try:
        with pytest.raises(ValueError):
            ip_region._buffer()
    finally:
        ip_region._buffer.cache_clear()


# ── pure signal helpers ─────────────────────────────────────────────────────

def test_client_ip_prefers_proxy_overwritten_header_and_ignores_forwarded_for():
    assert signals.client_ip({"x-real-ip": "120.230.1.1", "x-forwarded-for": "6.6.6.6"}, "172.18.0.1") == "120.230.1.1"
    assert signals.client_ip({"x-forwarded-for": "6.6.6.6"}, "172.18.0.1") == "172.18.0.1"
    assert signals.client_ip({"x-real-ip": "garbage"}, None) is None


def test_hash_prefix_and_client_kind_never_keep_the_address():
    hashed = signals.ip_hash("120.230.1.1")
    assert len(hashed) == 32 and "120.230" not in hashed
    assert hashed == signals.ip_hash("120.230.1.1") != signals.ip_hash("120.230.1.2")
    assert signals.ip_prefix("120.230.1.77") == "120.230.1.0/24"
    assert signals.ip_prefix("2408:8000:1:2::5") == "2408:8000:1::/48"
    assert [signals.client_kind(a) for a in ("Dart/3.12 (dart:io)", "Mozilla/5.0", None)] == ["app", "browser", "other"]


def test_client_metrics_keep_only_bounded_integers():
    assert signals.client_input_metrics({"compose_ms": 1200, "max_insert_chars": 30, "edit_count": 4,
                                         "text": "draft"}) == {"compose_ms": 1200, "max_insert_chars": 30, "edit_count": 4}
    assert signals.client_input_metrics({"compose_ms": -1, "max_insert_chars": True, "edit_count": "3"}) == {}
    assert signals.client_input_metrics({"compose_ms": 10**9}) == {}
    assert signals.client_input_metrics("x") == {}


def test_observers_without_identity_or_loop_do_nothing():
    signals.observe_network(None, {"x-real-ip": "1.1.1.1"}, None, source="login")
    signals.observe_network("u", {}, None, source="login")
    signals.observe_input(None, "u", "c", "hi", NOW)
    signals.observe_input("m", "u", "c", "  ", NOW)
    # No running event loop: scheduling is skipped instead of raising.
    signals.observe_network("u", {"x-real-ip": "1.1.1.1"}, None, source="login")
    signals.observe_input("m", "u", "c", "hello", NOW)


async def test_observers_schedule_background_work(monkeypatch):
    scheduled = []
    monkeypatch.setattr("app.services.runtime.tasks.fire_background",
                        lambda coro: (scheduled.append(coro), coro.close()))
    signals.observe_network("u", {"x-real-ip": "120.230.1.1", "user-agent": "Dart/3"}, None, source="login")
    signals.observe_input("m", "u", "c", "hello", NOW, {"compose_ms": 5})
    assert len(scheduled) == 2


async def test_observe_failure_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr(signals, "record_network", AsyncMock(side_effect=RuntimeError("db down")))
    await signals._observe("u", "120.230.1.1", None, "login")
    assert "network observation failed" in caplog.text


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)


async def test_compose_clock_starts_once_and_is_taken_by_the_next_send(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(return_value=redis))
    clock = iter([1000.0, 1004.0])
    monkeypatch.setattr(signals.time, "time", lambda: next(clock))
    conv = str(uuid4())
    await signals.mark_compose_started("u", conv)
    await signals.mark_compose_started("u", conv)  # heartbeat keeps the first start
    received = datetime.fromtimestamp(1006.5, tz=timezone.utc)
    assert await signals._take_compose_ms("u", conv, received) == 6500
    assert await signals._take_compose_ms("u", conv, received) is None
    for bogus in ("x" * 65, "conv", "../../etc"):  # not a conversation id
        await signals.mark_compose_started("u", bogus)
    assert redis.values == {}


async def test_compose_clock_ignores_future_start_and_stale_clock(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(return_value=redis))
    redis.values[signals._compose_key("u", "c")] = "5000000"
    assert await signals._take_compose_ms("u", "c", datetime.fromtimestamp(4000, tz=timezone.utc)) is None
    assert redis.values  # the next message's clock is kept
    redis.values[signals._compose_key("u", "c")] = "0"
    assert await signals._take_compose_ms("u", "c", datetime.fromtimestamp(4000, tz=timezone.utc)) is None


async def test_compose_clock_redis_outage_is_logged(monkeypatch, caplog):
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(side_effect=ConnectionError()))
    await signals.mark_compose_started("u", str(uuid4()))
    assert "compose clock unavailable" in caplog.text


# ── scoring ─────────────────────────────────────────────────────────────────

RELAY = {"messages": 77, "short_share": 0.0, "punct_share": 1.0, "median_chars": 34, "replies": 71,
         "fast_reply_share": 0.0, "burst_share": 0.0, "timed": 77, "compose_cps": 12.75,
         "client_samples": 0, "paste_share": None, "network_hits": 840, "datacenter_share": 0.0,
         "shared_accounts": 0}
HUMAN = {"messages": 338, "short_share": 0.15, "punct_share": 0.29, "median_chars": 7, "replies": 266,
         "fast_reply_share": 0.14, "burst_share": 0.06, "timed": 18, "compose_cps": 1.57,
         "client_samples": 0, "paste_share": None, "network_hits": 120, "datacenter_share": 0.0,
         "shared_accounts": 0}


def test_relay_profile_scores_as_likely_ai_and_human_as_human():
    relay, human = scoring.assess(RELAY), scoring.assess(HUMAN)
    assert relay["probability"] >= 0.9 and relay["verdict"] == "likely_ai"
    assert human["probability"] <= 0.05 and human["verdict"] == "human"
    terms = {t["key"]: t for t in relay["contributions"]}
    assert terms["compose_cps"]["logit"] > 1.5 and terms["paste_share"]["applied"] is False


def test_thin_evidence_is_not_applied_and_marked_insufficient():
    assessed = scoring.assess({"messages": 3, "punct_share": 1.0, "median_chars": 40, "replies": 2,
                               "network_hits": 0, "shared_accounts": 0})
    assert assessed["verdict"] == "insufficient"
    assert not any(t["applied"] for t in assessed["contributions"] if t["key"] != "shared_accounts")


@pytest.mark.parametrize("p,verdict", [(0.85, "likely_ai"), (0.6, "suspicious"), (0.3, "likely_human"), (0.1, "human")])
def test_verdict_bands(p, verdict):
    applied = [scoring.Contribution("punct_share", "", 1.0, 30, 0.1, True, "")]
    assert scoring.verdict(p, applied) == verdict


def test_paste_datacenter_and_shared_accounts_push_up():
    row = dict(HUMAN, client_samples=20, paste_share=0.9, datacenter_share=1.0, shared_accounts=4)
    # Network and paste alone cannot condemn a human-looking chat style.
    assert 0.2 < scoring.assess(row)["probability"] < 0.5


def test_reference_ranges_use_external_users_with_enough_messages():
    rows = [dict(HUMAN, internal=False, compose_cps=v) for v in (0.5, 1.0, 2.0)] + [
        dict(RELAY, internal=True), dict(HUMAN, internal=False, messages=3)]
    ranges = scoring.reference_ranges(rows)
    assert ranges["compose_cps"] == {"p10": 0.6, "p50": 1.0, "p90": 1.8, "users": 3}
    assert scoring.reference_ranges(rows[:2]) == {}
    rows[0]["user_id"] = "self"
    assert scoring.reference_ranges(rows, exclude="self") == {}  # only 2 others left


# ── PostgreSQL ──────────────────────────────────────────────────────────────

@pytest.fixture
async def pg(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Authenticity storage requires isolated PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    monkeypatch.setattr("app.db.db", database)
    from app.api.admin import authenticity
    monkeypatch.setattr(authenticity, "db", database)
    created = SimpleNamespace(users=[], conversations=[], agents=[])
    try:
        yield SimpleNamespace(db=database, created=created)
    finally:
        if created.conversations:
            await database.execute_raw("DELETE FROM messages WHERE conversation_id = ANY($1::text[])", created.conversations)
            await database.execute_raw("DELETE FROM conversations WHERE id = ANY($1::text[])", created.conversations)
        if created.agents:
            await database.execute_raw("DELETE FROM ai_agents WHERE id = ANY($1::text[])", created.agents)
        if created.users:
            await database.execute_raw("DELETE FROM users WHERE id = ANY($1::text[])", created.users)
        await database.disconnect()


async def _user(pg, *, role="user", status="active") -> str:
    ident = uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO users (id, username, role, status, updated_at) VALUES ($1, $2, $3, $4, NOW())",
        ident, f"authn-{ident}", role, status)
    pg.created.users.append(ident)
    return ident


async def _conversation(pg, user_id: str) -> str:
    # Production conversation IDs are dashed UUIDs; the nginx backfill matches that form.
    agent, conversation = uuid4().hex, str(uuid4())
    await pg.db.execute_raw(
        "INSERT INTO ai_agents (id, name, user_id, updated_at) VALUES ($1, '判定', $2, NOW())", agent, user_id)
    await pg.db.execute_raw(
        "INSERT INTO conversations (id, user_id, agent_id, updated_at) VALUES ($1, $2, $3, NOW())",
        conversation, user_id, agent)
    pg.created.agents.append(agent)
    pg.created.conversations.append(conversation)
    return conversation


async def _msg(pg, conversation, role, content, at, metadata=None) -> str:
    ident = uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)"
        " VALUES ($1, $2, $3, $4, $5::jsonb, $6::timestamp)",
        ident, conversation, role, content, json.dumps(metadata) if metadata else None,
        at.replace(tzinfo=None).isoformat())
    return ident


async def _seed_relay(pg, *, start=NOW - timedelta(days=2)):
    user = await _user(pg)
    conversation = await _conversation(pg, user)
    at = start
    for i in range(24):
        await _msg(pg, conversation, "assistant", "你最近在追什么番呀", at)
        at += timedelta(seconds=70)
        text = f"我最近在追《飙马野郎》第{i}集，乔尼的成长太戳我了，你看到哪了？"
        mid = await _msg(pg, conversation, "user", text, at)
        await signals.record_input(mid, user, conversation, text, at, {"compose_ms": 2000, "max_insert_chars": len(text)})
        at += timedelta(seconds=5)
    await _msg(pg, conversation, "user", "", at, {"component_card": {"type": "gift"}})
    return user, conversation


async def _seed_human(pg, *, start=NOW - timedelta(days=2)):
    user = await _user(pg)
    conversation = await _conversation(pg, user)
    at = start
    for i in range(24):
        await _msg(pg, conversation, "assistant", "在干嘛", at)
        at += timedelta(seconds=6)
        text = ["嗯", "哈哈", "刚下班好累啊", "在吃饭"][i % 4]
        mid = await _msg(pg, conversation, "user", text, at)
        await signals.record_input(mid, user, conversation, text, at, {"compose_ms": 4000 + i * 100})
        at += timedelta(seconds=20)
    return user, conversation


async def test_network_rows_merge_hits_and_skip_deleting_accounts(pg):
    user = await _user(pg)
    gone = await _user(pg, status="deleting")
    await signals.record_network(user, "120.230.1.1", "Dart/3.12", source="ws_ticket", seen_at=NOW - timedelta(hours=2))
    await signals.record_network(user, "120.230.1.1", "Dart/3.12", source="login", seen_at=NOW,
                                 first_seen_at=NOW - timedelta(days=1))
    await signals.record_network(user, "120.230.1.1", None, source="nginx_backfill", seen_at=NOW, hits=50, merge=False)
    await signals.record_network(gone, "47.96.0.1", None, source="login", seen_at=NOW)
    await signals.record_network(user, "172.18.0.1", None, source="login", seen_at=NOW)  # proxy/gateway
    rows = await pg.db.query_raw(
        "SELECT user_id, ip_prefix, isp, network_kind, client_kind, source, hits, first_seen_at, last_seen_at"
        " FROM user_network_observations WHERE user_id = ANY($1::text[])", [user, gone])
    assert len(rows) == 1
    row = rows[0]
    assert (row["ip_prefix"], row["isp"], row["network_kind"], row["client_kind"], row["source"], row["hits"]) == (
        "120.230.1.0/24", "移动", "carrier", "app", "login", 2)
    assert row["first_seen_at"].startswith((NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H"))
    columns = await pg.db.query_raw(
        "SELECT column_name FROM information_schema.columns WHERE table_name='user_network_observations'")
    assert "ip" not in {c["column_name"] for c in columns}


async def test_input_rows_take_server_clock_and_survive_redis_outage(pg, monkeypatch):
    user = await _user(pg)
    conversation = await _conversation(pg, user)
    redis = FakeRedis()
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(return_value=redis))
    redis.values[signals._compose_key(user, conversation)] = str(int((NOW - timedelta(seconds=3)).timestamp() * 1000))
    first = await _msg(pg, conversation, "user", "  你好呀  ", NOW)
    await signals.record_input(first, user, conversation, "  你好呀  ", NOW)
    await signals.record_input(first, user, conversation, "  你好呀  ", NOW)  # replay: no second row
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(side_effect=ConnectionError()))
    second = await _msg(pg, conversation, "user", "在吗", NOW + timedelta(seconds=9))
    await signals.record_input(second, user, conversation, "在吗", NOW + timedelta(seconds=9), {"edit_count": 2})
    rows = await pg.db.query_raw(
        "SELECT message_id, chars, server_compose_ms, client_compose_ms, edit_count, source"
        " FROM user_input_signals WHERE user_id=$1 ORDER BY created_at", user)
    assert [(r["message_id"], r["chars"], r["server_compose_ms"], r["edit_count"], r["source"]) for r in rows] == [
        (first, 3, 3000, None, "live"), (second, 2, None, 2, "live")]


async def test_input_write_failure_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(signals, "_take_compose_ms", AsyncMock(return_value=None))
    monkeypatch.setattr("app.db.db", SimpleNamespace(execute_raw=AsyncMock(side_effect=RuntimeError())))
    await signals.record_input("m", "u", "c", "hi", NOW)
    assert "input signal failed" in caplog.text


async def test_rows_cascade_with_the_account_and_expire_after_retention(pg):
    user = await _user(pg)
    conversation = await _conversation(pg, user)
    old = await _msg(pg, conversation, "user", "旧消息", NOW - timedelta(days=200))
    await signals.record_input(old, user, conversation, "旧消息", NOW - timedelta(days=200))
    await signals.record_network(user, "120.230.1.1", None, source="login", seen_at=NOW - timedelta(days=200))
    await signals.record_network(user, "120.230.1.2", None, source="login", seen_at=NOW)
    assert await signals.purge_expired(NOW) >= 2
    left = await pg.db.query_raw("SELECT count(*)::int AS n FROM user_network_observations WHERE user_id=$1", user)
    assert left[0]["n"] == 1
    await pg.db.execute_raw("DELETE FROM messages WHERE conversation_id=$1", conversation)
    await pg.db.execute_raw("DELETE FROM conversations WHERE id=$1", conversation)
    await pg.db.execute_raw("DELETE FROM ai_agents WHERE user_id=$1", user)
    await pg.db.execute_raw("DELETE FROM users WHERE id=$1", user)
    left = await pg.db.query_raw("SELECT count(*)::int AS n FROM user_network_observations WHERE user_id=$1", user)
    assert left[0]["n"] == 0


async def test_features_from_sql_separate_relay_and_human(pg):
    relay, _ = await _seed_relay(pg)
    human, _ = await _seed_human(pg)
    shared_ip = "47.96.7.7"
    for user in (relay, human):
        await signals.record_network(user, shared_ip, "Dart/3", source="login", seen_at=NOW)
    await signals.record_network(human, "120.230.9.9", "Dart/3", source="login", seen_at=NOW, hits=9)
    rows = {r["user_id"]: r for r in await scoring.feature_rows(30, now=NOW)}
    r, h = rows[relay], rows[human]
    assert r["messages"] == 24  # the card-only message is excluded
    assert r["punct_share"] == 1.0 and r["short_share"] == 0.0 and r["fast_reply_share"] == 0.0
    assert r["compose_cps"] > 8 and r["paste_share"] == 1.0 and r["client_samples"] == 24
    assert r["datacenter_share"] == 1.0 and r["shared_accounts"] == 1
    assert h["fast_reply_share"] == 1.0 and not h["timed"] and h["datacenter_share"] == 0.1
    assert scoring.assess(r)["verdict"] == "likely_ai"
    assert scoring.assess(h)["probability"] < 0.2
    only = await scoring.feature_rows(30, user_id=relay, now=NOW)
    assert [row["user_id"] for row in only] == [relay]


def _admin_app():
    from app.api.admin import authenticity
    from app.api.jwt_auth import require_admin_jwt

    app = FastAPI()
    app.include_router(authenticity.router)
    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app


async def test_admin_endpoints_rank_and_explain(pg, monkeypatch):
    relay, _ = await _seed_relay(pg, start=datetime.now(timezone.utc) - timedelta(days=1))
    human, _ = await _seed_human(pg, start=datetime.now(timezone.utc) - timedelta(days=1))
    admin = await _user(pg, role="admin")
    admin_conv = await _conversation(pg, admin)
    await _msg(pg, admin_conv, "user", "测试", datetime.now(timezone.utc))
    await signals.record_network(relay, "47.96.7.7", "Dart/3", source="login")
    await signals.record_network(human, "47.96.7.7", "Dart/3", source="login")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_admin_app()), base_url="http://t") as client:
        ranking = (await client.get("/admin-api/users/authenticity", params={"days": 7, "limit": 200})).json()
        ids = [item["user_id"] for item in ranking["items"]]
        assert ids.index(relay) < ids.index(human) and admin not in ids
        with_internal = (await client.get("/admin-api/users/authenticity",
                                          params={"days": 7, "limit": 200, "include_internal": True})).json()
        assert admin in [item["user_id"] for item in with_internal["items"]]
        detail = (await client.get(f"/admin-api/users/{relay}/authenticity")).json()
        assert detail["verdict"] == "likely_ai" and detail["model_version"] == scoring.MODEL_VERSION
        assert detail["networks"][0]["shared_account_count"] == 1
        assert detail["networks"][0]["shared_user_ids"] == [human]
        assert "ip_hash" not in detail["networks"][0]
        assert detail["inputs"][0]["compose_source"] == "app" and detail["inputs"][0]["compose_ms"] == 2000
        assert "compose_cps" in detail["reference"] or detail["reference"] == {}
        empty = await _user(pg)
        none = (await client.get(f"/admin-api/users/{empty}/authenticity")).json()
        assert none["verdict"] == "insufficient" and none["probability"] is None and none["networks"] == []
        assert (await client.get(f"/admin-api/users/{uuid4().hex}/authenticity")).status_code == 404


async def test_admin_detail_reports_server_and_missing_compose_sources(pg):
    user = await _user(pg)
    conversation = await _conversation(pg, user)
    at = datetime.now(timezone.utc)
    timed = await _msg(pg, conversation, "user", "服务端计时", at)
    plain = await _msg(pg, conversation, "user", "没有计时", at + timedelta(seconds=1))
    await pg.db.execute_raw(
        "INSERT INTO user_input_signals (message_id, user_id, created_at, chars, server_compose_ms, source)"
        " VALUES ($1, $2, $3::timestamp, 5, 4000, 'nginx_backfill'), ($4, $2, $5::timestamp, 4, NULL, 'live')",
        timed, user, at.replace(tzinfo=None).isoformat(), plain, (at + timedelta(seconds=1)).replace(tzinfo=None).isoformat())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_admin_app()), base_url="http://t") as client:
        detail = (await client.get(f"/admin-api/users/{user}/authenticity")).json()
    assert [(i["compose_source"], i["compose_ms"]) for i in detail["inputs"]] == [(None, None), ("server", 4000)]


# ── wiring ──────────────────────────────────────────────────────────────────

async def test_typing_ping_starts_the_compose_clock_only_when_active(monkeypatch):
    from app.api.jwt_auth import require_user
    from app.api.public import conversations

    started = AsyncMock()
    monkeypatch.setattr("app.services.authenticity.signals.mark_compose_started", started)
    monkeypatch.setattr("app.services.admin_takeover.service.publish_user_typing", AsyncMock(return_value=False))
    app = FastAPI()
    app.include_router(conversations.router)
    app.dependency_overrides[require_user] = lambda: {"sub": "actual-user", "role": "user"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        for active in (True, False):
            response = await client.post("/conversations/conv/typing",
                                         json={"active": active, "revision": 1, "session_id": "s"})
            assert response.status_code == 200
    started.assert_awaited_once_with("actual-user", "conv")


def test_auth_success_observes_network_and_failure_does_not(monkeypatch):
    from app.services import auth_security

    seen = []
    monkeypatch.setattr("app.services.authenticity.signals.observe_network",
                        lambda *args, **kwargs: seen.append((args, kwargs)))
    request = SimpleNamespace(headers={"x-real-ip": "120.230.1.1"}, client=SimpleNamespace(host="172.18.0.1"))
    auth_security.audit_auth_request_event("login_success", request, user_id="u1", outcome="success")
    auth_security.audit_auth_request_event("login_failed", request, username="x", outcome="failed")
    assert seen == [(("u1", request.headers, "172.18.0.1"), {"source": "login"})]


async def test_retention_job_purges_and_reports_failure(monkeypatch):
    from jobs import scheduler

    async def run(name, ttl, body, **_):
        await body()

    monkeypatch.setattr(scheduler, "_run_distributed_job", run)
    purge = AsyncMock(return_value=3)
    monkeypatch.setattr("app.services.authenticity.signals.purge_expired", purge)
    await scheduler._run_authenticity_retention()
    purge.assert_awaited_once()
    failed = []
    monkeypatch.setattr(scheduler, "_job_failed", lambda label, exc: failed.append(label))
    purge.side_effect = RuntimeError()
    await scheduler._run_authenticity_retention()
    assert failed == ["Authenticity signal purge"]


# ── nginx backfill ──────────────────────────────────────────────────────────

CONV = "451d2e2b-400e-453a-9131-3fcbddd48740"


def _line(ip, at, method, path, referer="-", agent="Dart/3.12 (dart:io)"):
    return f'{ip} - - [{at} +0800] "{method} {path} HTTP/1.1" 200 12 "{referer}" "{agent}"\n'


def test_backfill_parses_only_user_side_chat_requests():
    lines = [
        _line("120.230.1.1", "30/Oct/2026:21:29:32", "POST", f"/api/conversations/{CONV}/typing"),
        _line("120.230.1.1", "30/Oct/2026:21:29:40", "POST", f"/api/chat/{CONV}/ws-ticket"),
        _line("1.85.1.1", "02/Nov/2026:19:43:01", "POST", f"/api/conversations/{CONV}/typing",
              referer="https://www.banshengcomp.com/admin?section=users"),
        _line("1.85.1.1", "02/Nov/2026:19:43:01", "GET", f"/api/conversations/{CONV}/messages"),
        _line("1.85.1.1", "99/Xyz/2026:19:43:01", "POST", f"/api/conversations/{CONV}/typing"),
        _line("120.230.1.2", "30/Oct/2026:21:29:41", "POST", f"/chat/{CONV}/ws-ticket"),
        "garbage\n",
    ]
    parsed = list(backfill.parse(lines))
    assert [(c, k, ip) for c, k, ip, _, _ in parsed] == [
        (CONV, "typing", "120.230.1.1"), (CONV, "ticket", "120.230.1.1"), (CONV, "ticket", "120.230.1.2")]
    assert parsed[0][4] == datetime(2026, 10, 30, 13, 29, 32, tzinfo=timezone.utc)


def test_backfill_compose_times_use_first_ping_after_previous_send():
    t = datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc)
    s = lambda sec: t + timedelta(seconds=sec)  # noqa: E731
    messages = [("m1", s(10)), ("m2", s(60)), ("m3", s(2000))]
    pings = [s(3), s(5), s(55), s(58), s(1300)]
    assert backfill.compose_times(messages, pings) == {"m1": 7000, "m2": 5000}


async def test_backfill_writes_hashed_networks_and_missing_inputs(pg, monkeypatch):
    user = await _user(pg)
    admin = await _user(pg, role="admin")
    conversation = await _conversation(pg, user)
    admin_conv = await _conversation(pg, admin)
    base = datetime(2026, 10, 30, 13, 0)
    message = await _msg(pg, conversation, "user", "嗨嗨，我是小麦，19岁", base + timedelta(seconds=10))
    stamp = lambda sec: (base + timedelta(hours=8, seconds=sec)).strftime("%d/%b/%Y:%H:%M:%S")  # noqa: E731
    lines = [
        _line("120.230.1.1", stamp(3), "POST", f"/api/conversations/{conversation}/typing"),
        _line("120.230.1.1", stamp(1), "POST", f"/api/chat/{conversation}/ws-ticket"),
        _line("120.230.1.1", stamp(4), "POST", f"/api/chat/{conversation}/ws-ticket", agent="Dart/3.13"),
        _line("223.68.1.1", stamp(30), "POST", f"/api/chat/{conversation}/ws-ticket"),
        _line("47.96.0.1", stamp(5), "POST", f"/api/chat/{admin_conv}/ws-ticket"),
        _line("unknown", stamp(5), "POST", f"/api/chat/{conversation}/ws-ticket"),
    ]
    dry = await backfill.main(lines, dry_run=True)
    assert dry == {"lines_users": 1, "networks": 2, "inputs": 1, "dry_run": True}
    await backfill.main(lines, dry_run=False)
    await backfill.main(lines, dry_run=False)  # re-run leaves rows untouched
    nets = await pg.db.query_raw(
        "SELECT ip_prefix, hits, source, user_agent FROM user_network_observations WHERE user_id=$1 ORDER BY ip_prefix",
        user)
    assert [(n["ip_prefix"], n["hits"], n["source"]) for n in nets] == [
        ("120.230.1.0/24", 3, "nginx_backfill"), ("223.68.1.0/24", 1, "nginx_backfill")]
    assert nets[0]["user_agent"] == "Dart/3.13"
    inputs = await pg.db.query_raw(
        "SELECT message_id, server_compose_ms, source FROM user_input_signals WHERE user_id=$1", user)
    assert [(i["message_id"], i["server_compose_ms"], i["source"]) for i in inputs] == [(message, 7000, "nginx_backfill")]
    assert not await pg.db.query_raw("SELECT 1 FROM user_network_observations WHERE user_id=$1", admin)


async def test_ws_message_records_input_signal_after_persisting(monkeypatch):
    from unittest.mock import MagicMock, patch

    from app.api.realtime import ws as ws_mod
    from tests.test_ws_ack import _aggregation_plan

    socket = MagicMock()
    socket.send_json = AsyncMock()
    observed = []
    monkeypatch.setattr("app.services.authenticity.signals.observe_input",
                        lambda *args: observed.append(args))
    plan = _aggregation_plan("fragment_window", text="嗯", metadata={"fragment": True})
    with (
        patch.object(ws_mod.wallet, "is_vip", new=AsyncMock(return_value=False)),
        patch.object(ws_mod.chat_quota, "consume_one", new=AsyncMock(
            return_value={"allowed": True, "mode": "free", "used": 1, "limit": 20, "charged": 0})),
        patch.object(ws_mod, "_persist_user_message", new=AsyncMock(return_value="db-msg-1")),
        patch.object(ws_mod, "plan_user_message_aggregation", new=AsyncMock(return_value=plan)),
        patch.object(ws_mod, "enqueue_planned_user_message", new=AsyncMock(return_value=True)),
        patch.object(ws_mod, "get_cached_schedule", new=AsyncMock(
            return_value=[{"activity": "自由时间", "type": "leisure"}])),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context", new=AsyncMock(return_value={})),
    ):
        await ws_mod._handle_message(socket, "conv-1", "user-1", SimpleNamespace(id="a1", name="A"), "嗯",
                                     client_id="c1", input_metrics={"compose_ms": 900})
    assert len(observed) == 1
    message_id, user_id, conversation_id, text, received_at, metrics = observed[0]
    assert (message_id, user_id, conversation_id, text, metrics) == (
        "db-msg-1", "user-1", "conv-1", "嗯", {"compose_ms": 900})
    assert received_at.tzinfo is not None


def test_http_chat_records_input_signal_for_the_saved_message(secured_chat, auth_header, monkeypatch):
    observed = []
    monkeypatch.setattr("app.services.authenticity.signals.observe_input",
                        lambda *args: observed.append(args))
    response = secured_chat.client.post("/chat/conv-1", json={"message": "hello"}, headers=auth_header("owner"))
    assert response.status_code == 200
    assert [args[:4] for args in observed] == [("message-1", "owner", "conv-1", "hello")]
