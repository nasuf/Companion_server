"""UXM01.01 人群与单一口径: 真实迁移后的 PostgreSQL, 原始 SQL 与接口结果对照.

用一个远离其他测试的固定本地日 (2001-05-14) 造数, 回复效果的日聚合只看到这里的
行; 数据监控按"现在"取窗口, 所以比较造数前后的增量。外部用户 1 个, 内部账号 3 种
(admin / 后台勾选 / 模板系统账号) 各 1 个。
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma import Prisma

from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME
from app.services.effect import definitions, signals, store
from app.services.effect.definitions import local_day_bounds

DAY = date(2001, 5, 14)


def _local(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return local_day_bounds(day)[0] + timedelta(hours=hour, minutes=minute, seconds=second)


@pytest.fixture
async def pg(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Cohort statistics require isolated PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"}
    assert parsed.path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    from app.api.admin import stats, users
    monkeypatch.setattr("app.db.db", database)
    monkeypatch.setattr(stats, "db", database)
    monkeypatch.setattr(users, "db", database)
    created = SimpleNamespace(users=[], agents=[], conversations=[], messages=[])
    try:
        yield SimpleNamespace(db=database, created=created)
    finally:
        if created.messages:
            await database.execute_raw("DELETE FROM messages WHERE id = ANY($1::text[])", created.messages)
        if created.conversations:
            await database.execute_raw(
                "DELETE FROM conversations WHERE id = ANY($1::text[])", created.conversations)
        if created.agents:
            await database.execute_raw("DELETE FROM ai_agents WHERE id = ANY($1::text[])", created.agents)
        if created.users:
            await database.execute_raw("DELETE FROM users WHERE id = ANY($1::text[])", created.users)
        await database.disconnect()


async def _user(pg, *, role="user", internal=False, template=False) -> str:
    if template:
        rows = await pg.db.query_raw("SELECT id FROM users WHERE username = $1", TEMPLATE_SYSTEM_USERNAME)
        if rows:
            return rows[0]["id"]
    ident = uuid4().hex
    username = TEMPLATE_SYSTEM_USERNAME if template else f"cohort-{ident}"
    await pg.db.execute_raw(
        "INSERT INTO users (id, username, role, is_internal, updated_at) VALUES ($1,$2,$3,$4,NOW())",
        ident, username, role, internal,
    )
    pg.created.users.append(ident)
    return ident


async def _conversation(pg, user_id: str) -> str:
    agent, conversation = uuid4().hex, uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO ai_agents (id, name, user_id, updated_at) VALUES ($1,'口径',$2,NOW())", agent, user_id)
    await pg.db.execute_raw(
        "INSERT INTO conversations (id, user_id, agent_id, updated_at) VALUES ($1,$2,$3,NOW())",
        conversation, user_id, agent)
    pg.created.agents.append(agent)
    pg.created.conversations.append(conversation)
    return conversation


async def _message(pg, conversation: str, role: str, at: datetime, metadata: dict | None = None):
    import json
    ident = uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)"
        " VALUES ($1,$2,$3,'合成',$4::jsonb,$5::timestamp)",
        ident, conversation, role, json.dumps(metadata) if metadata is not None else None,
        at.isoformat(),
    )
    pg.created.messages.append(ident)


async def _seed_turn(pg, conversation: str, day: date):
    """一个被接住的回合: 用户 → 10 秒后回复 → 1 分钟后用户又开口 (这条没有回复)."""
    await _message(pg, conversation, "user", _local(day, 10))
    await _message(pg, conversation, "assistant", _local(day, 10, 0, 10))
    await _message(pg, conversation, "user", _local(day, 10, 1))


async def _seed(pg):
    external = await _user(pg)
    admin = await _user(pg, role="admin")
    flagged = await _user(pg, internal=True)
    template = await _user(pg, template=True)
    ext_main, ext_side = await _conversation(pg, external), await _conversation(pg, external)
    others = [await _conversation(pg, uid) for uid in (admin, flagged, template)]
    await _seed_turn(pg, ext_main, DAY)
    for conversation in others:
        await _seed_turn(pg, conversation, DAY)
    # 次日回访: 外部用户和 admin 都回来了
    await _message(pg, ext_main, "user", _local(DAY + timedelta(days=1), 9))
    await _message(pg, others[0], "user", _local(DAY + timedelta(days=1), 9))
    # 外部用户的主动发送: 卡片 + 文字共用一个事件 ID, 1 小时后用户回应
    event = {"proactive": True, "proactive_event_id": "evt-" + uuid4().hex}
    await _message(pg, ext_side, "assistant", _local(DAY, 15), {**event, "component_card": {"t": 1}})
    await _message(pg, ext_side, "assistant", _local(DAY, 15, 0, 1), event)
    await _message(pg, ext_side, "user", _local(DAY, 16))
    # 事件 ID 上线前的旧主动消息: 只能按条计, 无人回应; 键存在但值为 null 也算旧消息
    await _message(pg, ext_side, "assistant", _local(DAY, 18), {"proactive": True})
    await _message(pg, ext_side, "assistant", _local(DAY, 19),
                   {"proactive": True, "proactive_event_id": None})
    # admin 的主动发送 (外部口径下不计)
    await _message(pg, others[0], "assistant", _local(DAY, 12),
                   {"proactive": True, "proactive_event_id": "evt-" + uuid4().hex})
    return SimpleNamespace(external=external, admin=admin, flagged=flagged, template=template)


async def test_effect_metrics_exclude_every_internal_kind_and_dedupe_events(pg):
    await _seed(pg)
    now = datetime.combine(DAY + timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc)
    external = (await signals.collect(DAY, now=now)).as_dict()
    everyone = (await signals.collect(DAY, now=now, cohort="all")).as_dict()

    assert external["cohort"] == "external" and external["failed"] == []
    assert (external["active_users"], external["returned_next_day"]) == (1, 1)
    assert (everyone["active_users"], everyone["returned_next_day"]) == (4, 2)
    # 外部: 主会话 1 个被接住的回合 + 侧会话 1 个回应主动消息后未被接住的回合
    assert (external["turns"], external["continued"]) == (2, 1)
    # 全量多出 admin 的 10:01 那句: 现有回合定义把之后任何一条 AI 消息 (这里是 12:00
    # 的主动消息) 当作它的回复, 所以 4 个内外会话共 6 个回合、4 个被接住。
    assert (everyone["turns"], everyone["continued"]) == (6, 4)
    assert external["sessions"] == 2
    # 一次发送 = 一个事件; 旧消息单独计数, 不和事件混在一起看不出口径
    assert (external["proactive_sent"], external["proactive_answered"]) == (3, 1)
    assert (external["proactive_legacy_messages"], external["proactive_bubbles"]) == (2, 4)
    assert (everyone["proactive_sent"], everyone["proactive_bubbles"]) == (4, 5)

    # 原始 SQL 核对: 不经过统计代码, 直接数外部用户当天的用户消息发送人
    start, end = local_day_bounds(DAY)
    raw = await pg.db.query_raw(
        """SELECT COUNT(DISTINCT c.user_id)::int AS n FROM messages m
           JOIN conversations c ON c.id = m.conversation_id JOIN users u ON u.id = c.user_id
           WHERE m.role = 'user' AND m.created_at >= $1::timestamp AND m.created_at < $2::timestamp
             AND u.role <> 'admin' AND NOT u.is_internal AND u.username <> $3""",
        start.isoformat(), end.isoformat(), TEMPLATE_SYSTEM_USERNAME,
    )
    assert raw[0]["n"] == external["active_users"]


async def test_return_rate_stays_unsettled_until_the_next_local_day_ends(pg):
    await _seed(pg)
    during_next_day = _local(DAY + timedelta(days=1), 12).replace(tzinfo=timezone.utc)
    day = (await signals.collect(DAY, now=during_next_day)).as_dict()
    assert day["active_users"] == 1 and day["returned_next_day"] is None
    summary = store.summarise([day])
    assert summary["ratios"]["next_day_return"]["state"] == "unsettled"
    assert summary["ratios"]["next_day_return"]["value"] is None


def _app():
    from app.api.admin import stats, users
    from app.api.jwt_auth import require_admin_jwt
    app = FastAPI()
    app.include_router(stats.router)
    app.include_router(users.router)
    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "cohort-admin", "role": "admin"}
    return app


async def test_monitoring_counts_only_external_users_by_default(pg, monkeypatch):
    from app.api.admin import stats
    monkeypatch.setattr(stats, "get_redis", AsyncMock(side_effect=RuntimeError("no cache in test")))
    monkeypatch.setattr(stats, "count_online_users", AsyncMock(return_value=(0, True)))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://test") as client:
        async def snapshot(cohort):
            r = await client.get("/admin-api/stats/monitoring",
                                 params={"days": 1, "refresh": "true", "cohort": cohort})
            assert r.status_code == 200, r.text
            return r.json()

        before = {c: await snapshot(c) for c in ("external", "all")}
        await _seed_now(pg)
        after = {c: await snapshot(c) for c in ("external", "all")}

    def delta(cohort, key):
        return after[cohort]["overview"][key] - before[cohort]["overview"][key]

    assert delta("external", "dau") == 1 and delta("all", "dau") == 4
    assert delta("external", "user_messages_window") == 2
    assert delta("all", "user_messages_window") == 8
    assert delta("external", "total_users") == 1
    assert after["external"]["cohort"] == "external"
    assert after["external"]["definitions"]["definition_version"] == definitions.DEFINITION_VERSION


async def _seed_now(pg):
    """当前本地日的数据: 外部 1 人, 内部 3 种各 1 人, 每人 2 条用户消息."""
    users = SimpleNamespace(
        external=await _user(pg), admin=await _user(pg, role="admin"),
        flagged=await _user(pg, internal=True), template=await _user(pg, template=True),
    )
    # DAU 窗口只有下界 (本地今天 0 点); 不往前减时间, 免得在午夜附近落到昨天。
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for uid in (users.external, users.admin, users.flagged, users.template):
        conversation = await _conversation(pg, uid)
        await _message(pg, conversation, "user", now)
        await _message(pg, conversation, "user", now + timedelta(seconds=1))
    return users


async def test_internal_flag_toggle_changes_cohort_and_cache_identity(pg):
    ids = await _seed(pg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://test") as client:
        before = await definitions.population_fingerprint()
        on = await client.patch(f"/admin-api/users/{ids.external}/internal", json={"is_internal": True})
        assert on.status_code == 200
        assert on.json() == {"ok": True, "id": ids.external, "is_internal": True,
                             "counts_as_internal": True, "changed": True}
        again = await client.patch(f"/admin-api/users/{ids.external}/internal", json={"is_internal": True})
        assert again.json()["changed"] is False
        flagged_fingerprint = await definitions.population_fingerprint()
        assert flagged_fingerprint != before  # 已定型日子的旧缓存键不会再被读到
        now = datetime.combine(DAY + timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc)
        assert (await signals.collect(DAY, now=now)).active_users == 0

        admin = await client.patch(f"/admin-api/users/{ids.admin}/internal", json={"is_internal": False})
        assert admin.json()["counts_as_internal"] is True  # admin 不论勾选都算内部
        off = await client.patch(f"/admin-api/users/{ids.external}/internal", json={"is_internal": False})
        assert off.json()["is_internal"] is False
        assert await definitions.population_fingerprint() == before

        template = await client.patch(f"/admin-api/users/{ids.template}/internal", json={"is_internal": True})
        assert template.status_code == 404
        missing = await client.patch(f"/admin-api/users/{uuid4().hex}/internal", json={"is_internal": True})
        assert missing.status_code == 404
        listed = await client.get("/admin-api/users", params={"search": "cohort-"})
        row = next(u for u in listed.json()["users"] if u["id"] == ids.flagged)
        assert row["is_internal"] is True and row["counts_as_internal"] is True
