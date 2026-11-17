"""O01.02.V01: 后台 Trace 采集覆盖 —— 真实 JWT → 管理员 → 只读聚合 → 隔离 PG + Redis。

库是全量回归共用的, 只断言本用例插入前后的增量。
"""
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
from fastapi import FastAPI

from app.api.admin import stats
from app.services.auth import create_jwt
from app.services.chat.local_tracer import TRACE_WRITE_FAILURE_KEY
from tests.test_beta_daily_facts_postgres import pg, facts_refresh_redis, _seed  # noqa: F401


def _delta(after: dict, before: dict, *path: str) -> int:
    def pick(d):
        for key in path:
            d = d.get(key, {}) if isinstance(d, dict) else {}
        return d if isinstance(d, int) else 0
    return pick(after) - pick(before)


async def test_trace_coverage_separates_collected_missing_and_stale(pg, facts_refresh_redis, monkeypatch):
    ids = await _seed(pg)
    monkeypatch.setattr(stats, "db", pg.db)

    async def redis():
        return facts_refresh_redis

    monkeypatch.setattr(stats, "get_redis", redis)
    monkeypatch.setattr(stats.settings, "trace_backend", "local")
    conv = (await pg.db.query_raw("SELECT id FROM conversations ORDER BY id LIMIT 1"))[0]["id"]
    app = FastAPI()
    app.include_router(stats.router)
    traces = [str(uuid4()) for _ in range(4)]
    day_key = TRACE_WRITE_FAILURE_KEY.format(day=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated") as client:
        assert (await client.get("/admin-api/stats/trace-coverage")).status_code == 401
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.external, "user")
        assert (await client.get("/admin-api/stats/trace-coverage")).status_code == 403
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.admin, "admin")
        assert (await client.get("/admin-api/stats/trace-coverage", params={"days": 31})).status_code == 422
        before = (await client.get("/admin-api/stats/trace-coverage")).json()

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        roots = [(traces[0], "success", now), (traces[2], "running", now - timedelta(minutes=30))]
        for run_id, status, started in roots:
            await pg.db.execute_raw(
                "INSERT INTO trace_runs(run_id, trace_id, name, run_type, status, started_at) "
                "VALUES ($1, $1, 'chat_request', 'chain', $2, $3::timestamp)", run_id, status, started.isoformat())
        multi = f"user-turn:{uuid4()}"
        turns = [
            ({"achievement_turn_id": f"user-turn:{uuid4()}", "trace_id": traces[0]}, 5),  # 已采集
            ({"achievement_turn_id": f"user-turn:{uuid4()}", "trace_id": traces[1]}, 5),  # 有 id 无行
            ({"achievement_turn_id": f"user-turn:{uuid4()}", "trace_id": traces[3]}, 0),  # 刚产生, 根行可能还在写
            ({"achievement_turn_id": f"user-turn:{uuid4()}"}, 5),                         # 没有 id
            ({"achievement_turn_id": f"user-turn:{uuid4()}", "trace_id": traces[2]}, 5),  # 卡在 running
            ({"achievement_turn_id": multi, "trace_id": traces[1]}, 5),                   # 同一回合两条气泡,
            ({"achievement_turn_id": multi, "trace_id": traces[0]}, 5),                   # 任一 id 有根行即算采集
            ({"achievement_turn_id": f"proactive:{uuid4()}", "trace_id": traces[1]}, 5),  # 主动消息不在范围
        ]
        for meta, minutes_ago in turns:
            await pg.db.execute_raw(
                "INSERT INTO messages(id, conversation_id, role, content, metadata, created_at) "
                "VALUES ($1, $2, 'assistant', '合成回复', $3::jsonb, NOW() - make_interval(mins => $4::int))",
                str(uuid4()), conv, json.dumps(meta), minutes_ago)
        await facts_refresh_redis.hincrby(day_key, "root_end", 2)
        await facts_refresh_redis.hincrby(day_key, "orphaned", 3)
        try:
            response = await client.get("/admin-api/stats/trace-coverage")
        finally:
            await pg.db.execute_raw("DELETE FROM trace_runs WHERE run_id = ANY($1::text[])", traces)
            await facts_refresh_redis.hincrby(day_key, "root_end", -2)
            await facts_refresh_redis.hincrby(day_key, "orphaned", -3)
    assert response.status_code == 200, response.text[:300]
    after = response.json()
    assert "合成" not in response.text
    assert _delta(after, before, "coverage", "turns") == 6
    assert _delta(after, before, "coverage", "collected") == 3
    assert _delta(after, before, "coverage", "multiple_trace_ids") == 1
    assert _delta(after, before, "coverage", "in_flight") == 1
    assert _delta(after, before, "coverage", "rows_missing") == 1
    assert _delta(after, before, "coverage", "no_trace_id") == 1
    assert _delta(after, before, "root_status", "success") == 2  # 含多气泡回合
    assert _delta(after, before, "root_status", "running_stale") == 1
    assert _delta(after, before, "write_failures", "by_kind", "root_end") == 2
    assert _delta(after, before, "write_failures", "total") == 2
    assert _delta(after, before, "write_failures", "orphaned") == 3
    assert "orphaned" not in after["write_failures"]["by_kind"]
    assert after["write_failures"]["status"] == "ok"
    assert after["truncation"] == {"status": "not_collected"}
    assert after["replay"]["status"] == "not_enabled"
    assert after["collection"]["retention_days"] > 0
    assert after["collection"]["status"] == "enabled"
    assert after["missing_hours"]["timezone"] == "Asia/Shanghai"


async def test_trace_coverage_expiry_truncation_and_missing_hours(pg, facts_refresh_redis, monkeypatch):
    """过了保留期的不算缺失; 截断时给出实际覆盖起点; 缺失小时只列没有任何已采集回合的小时。"""
    await _seed(pg)
    monkeypatch.setattr(stats, "db", pg.db)

    async def redis():
        return facts_refresh_redis

    monkeypatch.setattr(stats, "get_redis", redis)
    monkeypatch.setattr(stats.settings, "trace_backend", "local")
    monkeypatch.setattr(stats.settings, "trace_retention_days", 1)
    conv = (await pg.db.query_raw("SELECT id FROM conversations ORDER BY id LIMIT 1"))[0]["id"]
    since = datetime.now(timezone.utc) - timedelta(days=7)
    before = await stats._trace_coverage(since, 7)
    # 3 天前: 有 id 无行 → 已过期; 精确到一个其他样本不会落入的分钟
    at = (datetime.now(timezone.utc) - timedelta(days=3)).replace(minute=17, second=0, microsecond=0)
    for meta in ({"achievement_turn_id": f"user-turn:{uuid4()}", "trace_id": str(uuid4())},
                 {"achievement_turn_id": f"user-turn:{uuid4()}"}):
        await pg.db.execute_raw(
            "INSERT INTO messages(id, conversation_id, role, content, metadata, created_at) "
            "VALUES ($1, $2, 'assistant', '合成回复', $3::jsonb, $4::timestamp)",
            str(uuid4()), conv, json.dumps(meta), at.replace(tzinfo=None).isoformat())
    after = await stats._trace_coverage(since, 7)
    assert _delta(after, before, "coverage", "expired") == 1
    assert _delta(after, before, "coverage", "rows_missing") == 0
    local_hour = at.astimezone(stats.ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:00")
    assert local_hour in after["missing_hours"]["items"]  # 没 id 的那条仍是缺失
    assert after["turns"]["truncated"] is False and after["turns"]["effective_since"] == since.isoformat()

    monkeypatch.setattr(stats, "_TC_TURN_CAP", 1)
    capped = await stats._trace_coverage(since, 7)
    assert capped["turns"]["truncated"] is True and capped["coverage"]["turns"] == 1
    assert capped["turns"]["effective_since"] > since.isoformat()


async def test_trace_coverage_does_not_report_numbers_when_local_collection_is_off(monkeypatch):
    async def must_not_query(*_a, **_k):
        raise AssertionError("未启用本地采集时不该查库")

    async def broken():
        raise ConnectionError("redis down")

    monkeypatch.setattr(stats.db, "query_raw", must_not_query, raising=False)
    monkeypatch.setattr(stats, "get_redis", broken)
    for backend, status in (("langsmith", "external_langsmith"), ("off", "disabled")):
        monkeypatch.setattr(stats.settings, "trace_backend", backend)
        result = await stats._trace_coverage(datetime.now(timezone.utc), 1)
        assert result["collection"]["status"] == status
        assert result["coverage"] is None and result["missing_hours"] is None and result["turns"] is None


async def test_trace_coverage_reports_unavailable_counters_instead_of_zero(monkeypatch):
    async def broken():
        raise ConnectionError("redis down")

    monkeypatch.setattr(stats, "get_redis", broken)
    result = await stats._trace_write_failures(datetime.now(timezone.utc), 1)
    assert result == {"status": "unavailable", "by_kind": {}, "total": None, "orphaned": None}


async def test_trace_coverage_treats_corrupt_counter_as_unavailable(facts_refresh_redis, monkeypatch):
    async def redis():
        return facts_refresh_redis

    monkeypatch.setattr(stats, "get_redis", redis)
    now = datetime.now(timezone.utc)
    key = TRACE_WRITE_FAILURE_KEY.format(day=now.strftime("%Y-%m-%d"))
    await facts_refresh_redis.hset(key, "corrupt_probe", "not-a-number")
    try:
        result = await stats._trace_write_failures(now, 1)
    finally:
        await facts_refresh_redis.hdel(key, "corrupt_probe")
    assert result["status"] == "unavailable" and result["total"] is None
