"""UXM01.02 派生表: 真实迁移后的 PostgreSQL, 用造好的一天核对每个计数.

造数放在 2027-03-12 (未来的固定本地日), 所有计算显式传 now, 不受其他测试数据和
真实时钟影响。外部用户 E 一天里的消息 (北京时间):
    10:00 / 10:00:30  一串两条, 10:01 收到回复 (带耗时埋点, 两个气泡)  -> 回合
    14:00             10 分钟内没回复 (14:30 才回)                    -> 无回复
    16:00             人工接管中                                       -> 不计
    18:10             游戏进行中 (18:00-18:30)                         -> 不计
    20:00             主动消息 (卡片+文字共用一个事件 ID)
    22:00             旧主动消息 (无事件 ID)
    23:58 -> 次日 00:03 一串跨过午夜, 00:04 回复                       -> 算在次日
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Prisma

from app.services.effect import daily_facts, definitions, signals
from app.services.effect.definitions import local_day_bounds

DAY = date(2027, 3, 12)
NEXT = DAY + timedelta(days=1)


def _at(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return local_day_bounds(day)[0] + timedelta(hours=hour, minutes=minute, seconds=second)


def _now(day: date, hour: int = 0, minute: int = 0) -> datetime:
    return _at(day, hour, minute).replace(tzinfo=timezone.utc)


@pytest.fixture
async def pg(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Daily facts require isolated PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    monkeypatch.setattr("app.db.db", database)
    created = SimpleNamespace(users=[], conversations=[], agents=[])
    days = [DAY + timedelta(days=i) for i in range(-1, 8)]
    try:
        yield SimpleNamespace(db=database, created=created)
    finally:
        if created.conversations:
            for table in ("messages", "game_sessions"):
                await database.execute_raw(
                    f"DELETE FROM {table} WHERE conversation_id = ANY($1::text[])", created.conversations)
            await database.execute_raw("DELETE FROM conversations WHERE id = ANY($1::text[])", created.conversations)
        if created.agents:
            await database.execute_raw("DELETE FROM ai_agents WHERE id = ANY($1::text[])", created.agents)
        if created.users:
            await database.execute_raw("DELETE FROM users WHERE id = ANY($1::text[])", created.users)
        await database.execute_raw(
            "DELETE FROM metrics_daily WHERE local_date::text = ANY($1::text[])", [d.isoformat() for d in days])
        await database.execute_raw(
            "DELETE FROM user_daily_facts WHERE local_date::text = ANY($1::text[])", [d.isoformat() for d in days])
        await database.disconnect()


async def _user(pg, *, role="user", internal=False, created_at: datetime | None = None) -> str:
    ident = uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO users (id, username, role, is_internal, created_at, updated_at)"
        " VALUES ($1, $2, $3, $4, $5::timestamp, NOW())",
        ident, f"facts-{ident}", role, internal, (created_at or _at(DAY - timedelta(days=30), 9)).isoformat(),
    )
    pg.created.users.append(ident)
    return ident


async def _conversation(pg, user_id: str) -> tuple[str, str]:
    agent, conversation = uuid4().hex, uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO ai_agents (id, name, user_id, updated_at) VALUES ($1, '统计', $2, NOW())", agent, user_id)
    await pg.db.execute_raw(
        "INSERT INTO conversations (id, user_id, agent_id, updated_at) VALUES ($1, $2, $3, NOW())",
        conversation, user_id, agent)
    pg.created.agents.append(agent)
    pg.created.conversations.append(conversation)
    return conversation, agent


async def _msg(pg, conversation: str, role: str, at: datetime, metadata: dict | None = None) -> None:
    await pg.db.execute_raw(
        "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)"
        " VALUES ($1, $2, $3, '合成', $4::jsonb, $5::timestamp)",
        uuid4().hex, conversation, role, json.dumps(metadata) if metadata is not None else None, at.isoformat(),
    )


def _timing(received: datetime, bubble: datetime) -> dict:
    return {"response_diagnostics": {
        "timing_definition_version": "server-turn-timing-v1", "first_bubble_clock": "server_bubble_ready",
        "reply_path": "main_llm",
        "turn_received_at": received.replace(tzinfo=timezone.utc).isoformat(),
        "first_bubble_at": bubble.replace(tzinfo=timezone.utc).isoformat(),
        "generation_ms": 5000.0, "in_graph_delay_ms": 1200.0,
    }}


async def _seed(pg) -> SimpleNamespace:
    external = await _user(pg, created_at=_at(DAY, 8))
    admin = await _user(pg, role="admin")
    flagged = await _user(pg, internal=True)
    conv, agent = await _conversation(pg, external)
    timing = _timing(_at(DAY, 10, 0, 30), _at(DAY, 10, 0, 58))
    await _msg(pg, conv, "user", _at(DAY, 10))
    await _msg(pg, conv, "user", _at(DAY, 10, 0, 30))
    await _msg(pg, conv, "assistant", _at(DAY, 10, 1), timing)
    await _msg(pg, conv, "assistant", _at(DAY, 10, 1, 2), timing)  # 同一回合第二个气泡
    await _msg(pg, conv, "user", _at(DAY, 14))
    await _msg(pg, conv, "assistant", _at(DAY, 14, 30))
    await _msg(pg, conv, "user", _at(DAY, 16), {"human_takeover": True})
    await pg.db.execute_raw(
        "INSERT INTO game_sessions (id, user_id, agent_id, conversation_id, room_id, user_player, ai_player,"
        " started_at, ended_at) VALUES ($1, $2, $3, $4, 'room', '{}'::jsonb, '{}'::jsonb,"
        " $5::timestamp, $6::timestamp)",
        uuid4().hex, external, agent, conv, _at(DAY, 18).isoformat(), _at(DAY, 18, 30).isoformat(),
    )
    await _msg(pg, conv, "user", _at(DAY, 18, 10))
    event = "evt-" + uuid4().hex
    await _msg(pg, conv, "assistant", _at(DAY, 20),
               {"proactive": True, "proactive_event_id": event, "trigger_type": "silence_wakeup_card",
                "component_card": {"type": "x"}})
    await _msg(pg, conv, "assistant", _at(DAY, 20, 0, 1),
               {"proactive": True, "proactive_event_id": event, "trigger_type": "silence_wakeup"})
    await _msg(pg, conv, "assistant", _at(DAY, 22), {"proactive": True, "trigger_type": "memory_proactive"})
    await _msg(pg, conv, "user", _at(DAY, 23, 58))
    await _msg(pg, conv, "user", _at(NEXT, 0, 3))
    await _msg(pg, conv, "assistant", _at(NEXT, 0, 4))
    await pg.db.execute_raw(
        "INSERT INTO user_daily_activity (id, user_id, local_date, source) VALUES"
        " ($1, $2, $3::date, 'auth_me'), ($4, $5, $3::date, 'auth_me')",
        uuid4().hex, external, DAY.isoformat(), uuid4().hex, admin,
    )
    await pg.db.execute_raw(
        "INSERT INTO user_feedback (id, user_id, content, contact, created_at, updated_at)"
        " VALUES ($1, $2, '合成反馈', '', $3::timestamp, NOW())",
        uuid4().hex, external, _at(DAY, 12).isoformat(),
    )
    await pg.db.execute_raw(
        "INSERT INTO llm_usage (id, user_id, input_tokens, output_tokens, tokens_by_model, cost_cny, call_count,"
        " created_at) VALUES ($1, $2, 10, 10, '{}'::jsonb, 0.5, 1, $3::timestamp)",
        uuid4().hex, external, _at(DAY, 10, 1).isoformat(),
    )
    admin_conv, _ = await _conversation(pg, admin)
    await _msg(pg, admin_conv, "user", _at(DAY, 11))
    await _msg(pg, admin_conv, "assistant", _at(DAY, 11, 0, 10))
    await _msg(pg, admin_conv, "assistant", _at(DAY, 12), {"proactive": True, "trigger_type": "memory_proactive"})
    flagged_conv, _ = await _conversation(pg, flagged)
    await _msg(pg, flagged_conv, "user", _at(DAY, 9))
    return SimpleNamespace(external=external, admin=admin, flagged=flagged)


async def _metrics(pg, day: date, cohort: str, dimension: str = "") -> dict:
    rows = await pg.db.query_raw(
        "SELECT metric, dimension_value, numerator, denominator, value, sample_size, settled"
        " FROM metrics_daily WHERE local_date = $1::date AND cohort = $2 AND dimension = $3",
        day.isoformat(), cohort, dimension,
    )
    return {(r["metric"], r["dimension_value"]) if dimension else r["metric"]: r for r in rows}


async def test_one_day_matches_hand_counted_facts_and_metrics(pg):
    ids = await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    result = await daily_facts.compute_day(DAY, now=now)
    assert result["facts"] == 3

    facts = (await pg.db.query_raw(
        "SELECT * FROM user_daily_facts WHERE user_id = $1 AND local_date = $2::date",
        ids.external, DAY.isoformat()))[0]
    assert (facts["opened"], facts["chatted"], facts["user_messages"]) == (True, True, 6)
    # 两串计数: 10:00 那串有回复, 14:00 那串没有; 接管、游戏、跨午夜那串都不算在这一天
    assert (facts["bursts"], facts["turns"], facts["unreplied_bursts"]) == (2, 1, 1)
    assert facts["sessions"] == 3  # 10:00, 14:00 (隔 4 小时), 23:58 (隔 5 小时多)
    assert facts["spontaneous"] is True
    assert (facts["proactive_received"], facts["proactive_answered"]) == (2, 2)
    assert (facts["feedback_count"], facts["cost_cny"], facts["settled"]) == (1, 0.5, True)
    assert facts["definition_version"] == definitions.DEFINITION_VERSION

    ext = await _metrics(pg, DAY, "external")
    assert {m: ext[m]["value"] for m in ("chat_active_users", "opened_users", "new_users", "user_messages",
                                         "turns", "proactive_events")} == {
        "chat_active_users": 1, "opened_users": 1, "new_users": 1, "user_messages": 6,
        "turns": 1, "proactive_events": 2}
    assert (ext["no_reply_10m"]["numerator"], ext["no_reply_10m"]["denominator"]) == (1, 2)
    assert (ext["proactive_response"]["numerator"], ext["proactive_response"]["denominator"]) == (2, 2)
    assert (ext["proactive_response_3h"]["numerator"], ext["proactive_response_3h"]["denominator"]) == (1, 2)
    assert (ext["next_day_return"]["numerator"], ext["next_day_return"]["denominator"]) == (1, 1)
    assert (ext["cost_per_active_user"]["numerator"], ext["cost_per_active_user"]["value"]) == (0.5, 0.5)
    # 两个气泡同一个首气泡时刻, 只算一个回合: 28 秒, 处理 5000-1200 毫秒
    assert (ext["reply_first_bubble_ms_p50"]["value"], ext["reply_first_bubble_ms_p50"]["sample_size"]) == (28000, 1)
    assert ext["reply_processing_ms_p90"]["value"] == 3800
    assert all(r["settled"] for r in ext.values())
    # 延续率直接复用回复效果的回合定义, 不另写一份
    core = (await signals.collect(DAY, now=now)).as_dict()
    assert (ext["continuation"]["numerator"], ext["continuation"]["denominator"]) == (
        core["continued"], core["turns"])

    everyone = await _metrics(pg, DAY, "all")
    assert everyone["chat_active_users"]["value"] == 3 and everyone["opened_users"]["value"] == 2
    assert everyone["turns"]["value"] == 2 and everyone["proactive_events"]["value"] == 3
    assert (everyone["no_reply_10m"]["numerator"], everyone["no_reply_10m"]["denominator"]) == (2, 4)
    by_trigger = await _metrics(pg, DAY, "external", "trigger_type")
    assert by_trigger[("proactive_events", "silence_wakeup")]["value"] == 1
    assert by_trigger[("proactive_events", "memory_proactive")]["value"] == 1
    by_path = await _metrics(pg, DAY, "external", "reply_path")
    assert by_path[("reply_first_bubble_ms_p50", "main_llm")]["sample_size"] == 1
    rows = await pg.db.query_raw("SELECT * FROM metrics_daily WHERE local_date = $1::date", DAY.isoformat())
    assert all("user_id" not in r for r in rows)  # 聚合表不含用户标识

    # 跨午夜那串算在次日, 且只算一次
    await daily_facts.compute_day(NEXT, now=now)
    nxt = (await pg.db.query_raw(
        "SELECT bursts, turns, user_messages FROM user_daily_facts WHERE user_id = $1 AND local_date = $2::date",
        ids.external, NEXT.isoformat()))[0]
    assert (nxt["bursts"], nxt["turns"], nxt["user_messages"]) == (1, 1, 1)


async def test_unsettled_windows_and_recompute_are_idempotent(pg):
    await _seed(pg)
    # 当天结束后 2 小时: 回合 (10 分钟) 已定型, 主动回应 (6 小时) 与次日回访 (24 小时) 未定型
    early = _now(NEXT, 2)
    await daily_facts.compute_day(DAY, now=early)
    ext = await _metrics(pg, DAY, "external")
    assert ext["turns"]["settled"] and ext["proactive_response_3h"]["settled"] is False
    assert ext["proactive_response"]["settled"] is False and ext["next_day_return"]["settled"] is False
    facts = await pg.db.query_raw("SELECT settled FROM user_daily_facts WHERE local_date = $1::date", DAY.isoformat())
    assert facts and not any(r["settled"] for r in facts)
    assert DAY in await daily_facts.days_needing_compute(DAY, DAY, now=_now(DAY + timedelta(days=3)))
    # 重算不产生重复行
    await daily_facts.compute_day(DAY, now=_now(DAY + timedelta(days=3)))
    counts = await pg.db.query_raw(
        "SELECT COUNT(*)::int AS n, COUNT(*) FILTER (WHERE NOT settled)::int AS open FROM metrics_daily"
        " WHERE local_date = $1::date", DAY.isoformat())
    assert counts[0]["open"] == 0 and counts[0]["n"] == len(await pg.db.query_raw(
        "SELECT DISTINCT metric, cohort, dimension, dimension_value FROM metrics_daily WHERE local_date = $1::date",
        DAY.isoformat()))
    assert await daily_facts.days_needing_compute(DAY, DAY, now=_now(DAY + timedelta(days=3))) == []


async def test_concurrent_computations_of_one_day_serialize(pg):
    await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    results = await asyncio.gather(*(daily_facts.compute_day(DAY, now=now) for _ in range(3)))
    assert [r["facts"] for r in results] == [3, 3, 3]
    dupes = await pg.db.query_raw(
        "SELECT COUNT(*)::int AS n FROM user_daily_facts WHERE local_date = $1::date", DAY.isoformat())
    assert dupes[0]["n"] == 3
    # 页面刷新: 有新结果就不重算; 正在被别处重算时不排队
    assert (await daily_facts.compute_day(DAY, now=now, skip_if_fresher_than=600))["skipped"] is True
    async with pg.db.tx() as holder:
        await holder.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"beta-daily-facts:{DAY.isoformat()}")
        skipped = await daily_facts.compute_day(DAY, now=now, wait_for_lock=False)
    assert skipped["skipped"] == "locked"


async def test_population_change_during_computation_rolls_back_and_retries(pg, monkeypatch):
    await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    calls = iter(["before", "after", "after", "after"])
    monkeypatch.setattr(daily_facts, "population_fingerprint", AsyncMock(side_effect=lambda *_: next(calls)))
    result = await daily_facts.compute_days([DAY], now=now)
    assert result == {"computed": [DAY.isoformat()], "failed": []}
    marks = await pg.db.query_raw(
        "SELECT DISTINCT population_fingerprint AS f FROM metrics_daily WHERE local_date = $1::date AND cohort = 'external'",
        DAY.isoformat())
    assert [r["f"] for r in marks] == ["after"]  # 第一次的结果整天回滚, 没有混入


async def test_accounts_being_erased_get_no_new_facts_and_facts_cascade(pg):
    ids = await _seed(pg)
    await pg.db.execute_raw("UPDATE users SET status = 'deleting' WHERE id = $1", ids.flagged)
    now = _now(DAY + timedelta(days=5))
    assert (await daily_facts.compute_day(DAY, now=now))["facts"] == 2
    users = {r["user_id"] for r in await pg.db.query_raw(
        "SELECT user_id FROM user_daily_facts WHERE local_date = $1::date", DAY.isoformat())}
    assert users == {ids.external, ids.admin}
    for table in ("messages", "game_sessions"):
        await pg.db.execute_raw(f"DELETE FROM {table} WHERE conversation_id = ANY($1::text[])", pg.created.conversations)
    await pg.db.execute_raw("DELETE FROM conversations WHERE user_id = $1", ids.external)
    await pg.db.execute_raw("DELETE FROM ai_agents WHERE user_id = $1", ids.external)
    await pg.db.execute_raw("DELETE FROM users WHERE id = $1", ids.external)
    left = await pg.db.query_raw("SELECT COUNT(*)::int AS n FROM user_daily_facts WHERE user_id = $1", ids.external)
    assert left[0]["n"] == 0


async def test_computation_does_not_block_profile_role_or_internal_updates(pg):
    """事实行写入只对账号行加外键的 KEY SHARE 锁: 普通 UPDATE 照常进行, 不用等计算结束."""
    ids = await _seed(pg)
    url = os.environ["PROACTIVE_E2E_DATABASE_URL"]
    other = Prisma(datasource={"url": url}, http={"trust_env": False})
    await other.connect()
    try:
        async with pg.db.tx(timeout=timedelta(seconds=30)) as tx:
            await tx.execute_raw(
                "INSERT INTO user_daily_facts (user_id, local_date, definition_version, computed_at)"
                " VALUES ($1, $2::date, 'x', NOW())", ids.external, (DAY + timedelta(days=6)).isoformat())
            async with other.tx(timeout=timedelta(seconds=10)) as writer:
                await writer.execute_raw("SET LOCAL lock_timeout = '1s'")
                await writer.execute_raw(
                    "UPDATE users SET display_name = 'x', role = 'user', is_internal = TRUE,"
                    " status = 'deleting' WHERE id = $1", ids.external)
            await tx.execute_raw("DELETE FROM user_daily_facts WHERE local_date = $1::date",
                                 (DAY + timedelta(days=6)).isoformat())
    finally:
        await other.disconnect()


async def test_events_and_turns_crossing_midnight_count_once_on_their_first_bubble(pg):
    """卡片在 23:59:59, 文字在 00:00:01; 同一回合的两个气泡也跨午夜. 都只算在前一天."""
    user = await _user(pg)
    conv, _ = await _conversation(pg, user)
    event = "evt-" + uuid4().hex
    meta = {"proactive": True, "proactive_event_id": event, "trigger_type": "special_date"}
    await _msg(pg, conv, "assistant", _at(DAY, 23, 59, 59), {**meta, "component_card": {"t": 1}})
    await _msg(pg, conv, "assistant", _at(NEXT, 0, 0, 1), meta)
    # 次日 09:00 同一事件重试一次 (同一 ID): 不算新事件
    await _msg(pg, conv, "assistant", _at(NEXT, 9), meta)
    timing = _timing(_at(DAY, 23, 59, 30), _at(DAY, 23, 59, 58))
    await _msg(pg, conv, "user", _at(DAY, 23, 59, 30))
    await _msg(pg, conv, "assistant", _at(DAY, 23, 59, 58), timing)
    await _msg(pg, conv, "assistant", _at(NEXT, 0, 0, 2), timing)
    now = datetime.combine(NEXT + timedelta(days=3), datetime.min.time())  # 不带时区: 按 UTC 处理
    for day in (DAY, NEXT):
        await daily_facts.compute_day(day, now=now)
    first = await _metrics(pg, DAY, "all")
    second = await _metrics(pg, NEXT, "all")
    assert first["proactive_events"]["value"] == 1 and second["proactive_events"]["value"] == 0
    assert first["reply_first_bubble_ms_p50"]["sample_size"] == 1
    assert second["reply_first_bubble_ms_p50"]["sample_size"] == 0
    received = await pg.db.query_raw(
        "SELECT local_date::text AS d, proactive_received FROM user_daily_facts WHERE user_id = $1 ORDER BY 1", user)
    assert [(r["d"], r["proactive_received"]) for r in received] == [(DAY.isoformat(), 1)]


async def test_read_series_states_dimensions_and_background_heal(pg, monkeypatch):
    ids = await _seed(pg)
    now = _now(DAY + timedelta(days=5), 12)
    await daily_facts.compute_day(DAY, now=now)
    healed = []

    def capture(coro):
        healed.append(coro)
        coro.close()

    monkeypatch.setattr("app.services.runtime.tasks.fire_background", capture)
    metrics = ["chat_active_users", "turns_per_active_user", "no_reply_10m", "reply_first_bubble_ms_p50",
               "proactive_response", "user_messages"]
    out = await daily_facts.read_series(DAY, DAY + timedelta(days=5), cohort="external", metrics=metrics, now=now)
    point = {m: {p["date"]: p for p in out["series"][m]} for m in metrics}
    d = DAY.isoformat()
    assert (point["chat_active_users"][d]["state"], point["chat_active_users"][d]["value"]) == ("ok", 1)
    assert (point["turns_per_active_user"][d]["state"], point["turns_per_active_user"][d]["value"]) == ("ok", 1)
    # 分母 2 < 20: 只给分子分母
    nr = point["no_reply_10m"][d]
    assert (nr["state"], nr["value"], nr["numerator"], nr["denominator"]) == ("insufficient_sample", None, 1, 2)
    assert point["reply_first_bubble_ms_p50"][d]["state"] == "insufficient_sample"
    # 中间几天还没算: 未运行, 交给后台; 今天和昨天按需算了
    gap = (DAY + timedelta(days=2)).isoformat()
    assert point["chat_active_users"][gap]["state"] == "not_computed"
    assert gap in out["recomputing"] and healed
    today = (DAY + timedelta(days=5)).isoformat()
    assert point["user_messages"][today]["state"] == "unsettled" and point["user_messages"][today]["value"] == 0
    assert point["proactive_response"][today]["state"] == "no_sample"
    assert out["computed_at"] is not None

    # 维度: 有事件的日子按触发类型拆; 算过但没事件的日子是真实为零
    dim = await daily_facts.read_series(DAY, DAY + timedelta(days=5), cohort="external",
                                        metrics=["proactive_events"], dimension="trigger_type", now=now)
    cells = {(p["date"], p["dimension_value"]): p for p in dim["series"]["proactive_events"]}
    assert cells[(d, "silence_wakeup")]["value"] == 1
    yesterday = (DAY + timedelta(days=4)).isoformat()
    # 昨天算过但没有这类事件: 真实为零; 是否定型跟昨天的总计行一致 (昨天在 now 时已定型)
    assert (cells[(yesterday, "silence_wakeup")]["state"], cells[(yesterday, "silence_wakeup")]["value"]) == ("ok", 0)
    today_cell = cells[(today, "silence_wakeup")]
    assert (today_cell["state"], today_cell["value"]) == ("unsettled", 0)
    assert cells[(gap, "silence_wakeup")]["state"] == "not_computed"
    # 整个窗口都没有任何分类: 逐日占位, 分得清没算和没有数据
    empty = await daily_facts.read_series(DAY + timedelta(days=4), DAY + timedelta(days=5), cohort="external",
                                          metrics=["reply_first_bubble_ms_p50"], dimension="reply_path", now=now)
    assert [(p["dimension_value"], p["state"]) for p in empty["series"]["reply_first_bubble_ms_p50"]] == [
        ("", "no_sample"), ("", "no_sample")]

    # 勾选内部账号后, 外部口径的旧结果显示为过期, 交给后台重算
    await pg.db.execute_raw("UPDATE users SET is_internal = TRUE WHERE id = $1", ids.external)
    later = await daily_facts.read_series(DAY, DAY, cohort="external", metrics=["chat_active_users"], now=now)
    assert later["series"]["chat_active_users"][0]["state"] == "stale"
    assert later["series"]["chat_active_users"][0]["value"] is None and d in later["recomputing"]
    everyone = await daily_facts.read_series(DAY, DAY, cohort="all", metrics=["chat_active_users"], now=now)
    assert everyone["series"]["chat_active_users"][0]["state"] == "ok"

    # 采集起点之前: 未采集, 不是 0
    before = await daily_facts.read_series(date(2026, 5, 27), date(2026, 5, 28), cohort="external",
                                           metrics=["opened_users"], now=now)
    assert {p["state"] for p in before["series"]["opened_users"]} == {"not_collected"}


async def test_invariant_flags_missing_unsettled_and_outdated_days(pg, monkeypatch):
    from app.services.ops import invariants as inv

    ids = await _seed(pg)
    nightly = _now(NEXT + timedelta(days=1), 4, 40)
    for day in (DAY, NEXT):
        await daily_facts.compute_day(day, now=nightly)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            moment = _now(NEXT + timedelta(days=1), 6)
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)

    monkeypatch.setattr(inv, "datetime", Clock)
    ok = await inv._check_daily_metrics_settled()
    assert ok.status == "ok", ok.detail

    class BeforeNightly(datetime):
        @classmethod
        def now(cls, tz=None):  # 04:40 之前: 以前一天的夜间那一轮为准
            moment = _now(NEXT + timedelta(days=1), 3)
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)

    monkeypatch.setattr(inv, "datetime", BeforeNightly)
    early = await inv._check_daily_metrics_settled()
    assert early.status in {"ok", "warn"} and early.observed["unsettled"] == 0
    monkeypatch.setattr(inv, "datetime", Clock)
    await pg.db.execute_raw(
        "UPDATE metrics_daily SET settled = FALSE WHERE local_date = $1::date AND metric = 'turns'", NEXT.isoformat())
    unsettled = await inv._check_daily_metrics_settled()
    assert unsettled.status == "violated" and unsettled.observed["unsettled"] == 2
    await daily_facts.compute_day(NEXT, now=nightly)
    await pg.db.execute_raw(
        "DELETE FROM metrics_daily WHERE local_date = $1::date AND metric = 'turns' AND cohort = 'all'",
        DAY.isoformat())
    missing = await inv._check_daily_metrics_settled()
    assert missing.status == "violated" and missing.observed["missing"] == 1
    await daily_facts.compute_day(DAY, now=nightly)
    await pg.db.execute_raw("UPDATE users SET is_internal = TRUE WHERE id = $1", ids.external)
    outdated = await inv._check_daily_metrics_settled()
    assert outdated.status == "warn" and outdated.observed["outdated"] > 0


async def test_backfill_script_plans_then_applies_only_needed_days(pg, monkeypatch, capsys):
    import scripts.backfill_daily_facts as backfill

    await _seed(pg)
    earliest = await daily_facts.earliest_data_day()
    assert earliest is not None and earliest <= DAY  # 回填起点取原始数据最早的本地日
    monkeypatch.setattr(backfill, "db", SimpleNamespace(connect=AsyncMock(), disconnect=AsyncMock()))
    with pytest.raises(SystemExit):
        await backfill.main(["--start", "2001-02-02", "--end", "2001-02-01"])
    capsys.readouterr()
    span = ["--start", "2001-02-01", "--end", "2001-02-02"]
    try:
        assert await backfill.main(span) == 0
        planned = json.loads(capsys.readouterr().out)
        assert planned == {"days": 2, "first": "2001-02-01", "last": "2001-02-02", "applied": False}
        assert await backfill.main([*span, "--apply"]) == 0
        assert json.loads(capsys.readouterr().out)["computed"] == 2
        assert await backfill.main(span) == 0
        assert json.loads(capsys.readouterr().out)["days"] == 0  # 已定型, 无需再算
        assert await backfill.main([*span, "--all"]) == 0
        assert json.loads(capsys.readouterr().out)["days"] == 2
    finally:
        await pg.db.execute_raw("DELETE FROM metrics_daily WHERE local_date BETWEEN '2001-02-01' AND '2001-02-02'")
        await pg.db.execute_raw("DELETE FROM user_daily_facts WHERE local_date BETWEEN '2001-02-01' AND '2001-02-02'")
