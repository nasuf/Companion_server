"""TIME11: 后台时间感知汇总 —— 真实 JWT → 管理员 → 只读聚合 → 隔离 PG。

库是全量回归共用的, 别的用例也会写带诊断的回复: 只断言本用例插入前后的增量。
"""
import json
from uuid import uuid4

import httpx
from fastapi import FastAPI

from app.api.admin import stats
from app.services.auth import create_jwt
from tests.test_beta_daily_facts_postgres import pg, facts_refresh_redis, _seed  # noqa: F401


def _diag(**extra):
    return json.dumps({"response_diagnostics": {"version": 1, **extra}}, ensure_ascii=False)


def _delta(after: dict, before: dict, field: str, key: str) -> int:
    return after[field].get(key, 0) - before[field].get(key, 0)


async def test_time_awareness_summary_counts_and_auth(pg, facts_refresh_redis, monkeypatch):
    ids = await _seed(pg)
    monkeypatch.setattr(stats, "db", pg.db)
    conv = (await pg.db.query_raw("SELECT id FROM conversations ORDER BY id LIMIT 1"))[0]["id"]
    app = FastAPI()
    app.include_router(stats.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated") as client:
        assert (await client.get("/admin-api/stats/time-awareness")).status_code == 401
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.external, "user")
        assert (await client.get("/admin-api/stats/time-awareness")).status_code == 403
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.admin, "admin")
        assert (await client.get("/admin-api/stats/time-awareness", params={"days": 31})).status_code == 422
        before = (await client.get("/admin-api/stats/time-awareness", params={"days": 7})).json()

        rows = [
            _diag(agent_routine={"status": "injected", "phase": "at_work"},
                  temporal_context={"event_check_status": "agent_schedule_evidence",
                                    "query": {"operation": {"type": "as_of", "inferred": True}},
                                    "computation": {"status": "computed", "reply_check": "consistent"}}),
            _diag(agent_routine={"status": "not_derived"},
                  temporal_context={"query": {"operation": {"type": "count"}},
                                    "computation": {"status": "computed", "reply_check": "overclaimed"}},
                  historical_recall_fallthrough="inferred_scope_without_answer"),
            _diag(),  # 旧消息: 没有这些字段
        ]
        for meta in rows:
            await pg.db.execute_raw(
                "INSERT INTO messages(id, conversation_id, role, content, metadata, created_at) "
                "VALUES ($1, $2, 'assistant', '合成回复', $3::jsonb, NOW())", str(uuid4()), conv, meta)
        await pg.db.execute_raw(
            "INSERT INTO messages(id, conversation_id, role, content, metadata, created_at) "
            "VALUES ($1, $2, 'user', '合成', $3::jsonb, NOW())", str(uuid4()), conv,
            _diag(agent_routine={"status": "x"}))
        # 坏数据 (missed_at 不是时间) 不能让整张卡片 503
        owner = (await pg.db.query_raw("SELECT user_id, agent_id FROM conversations WHERE id = $1", conv))[0]
        bad_trigger = str(uuid4())
        await pg.db.execute_raw(
            "INSERT INTO time_triggers(id, ai_agent_id, user_id, trigger_time, action_type, action_data, is_active) "
            "VALUES ($1, $2, $3, NOW(), 'reminder', $4::jsonb, false)",
            bad_trigger, owner["agent_id"], owner["user_id"], json.dumps({"missed_at": "坏数据"}))
        try:
            response = await client.get("/admin-api/stats/time-awareness", params={"days": 7})
        finally:
            # 夹具只清理它自己建的表, 这条要自己删
            await pg.db.execute_raw("DELETE FROM time_triggers WHERE id = $1", bad_trigger)
    assert response.status_code == 200, response.text[:300]
    after = response.json()
    assert "合成" not in response.text
    assert after["replies"]["truncated"] is False
    assert after["replies"]["count"] - before["replies"]["count"] == 3  # 用户消息不算
    d = lambda field, key: _delta(after, before, field, key)  # noqa: E731
    assert d("agent_routine_status", "injected") == 1 and d("agent_routine_status", "not_derived") == 1
    assert d("agent_routine_status", "not_collected") == 1
    assert d("event_check_status", "agent_schedule_evidence") == 1
    # 推断的 as_of 单独一类; 次数问法属于 "不适用", 跟没有诊断块的 "未采集" 分开
    assert d("inferred_as_of", "as_of_inferred") == 1 and d("inferred_as_of", "none") == 1
    assert d("inferred_as_of", "not_collected") == 1
    assert d("computation_reply_check", "consistent") == 1 and d("computation_reply_check", "overclaimed") == 1
    assert d("historical_recall_fallthrough", "inferred_scope_without_answer") == 1
    assert set(after["routine_coverage"]) == {"with_routine", "active_agents"}
    assert set(after["reminders"]) == {"missed_in_window", "paused_now"}
    # 各项分布出自同一批样本, 分母一致
    totals = {k: sum(v.values()) for k, v in after.items()
              if k in {"agent_routine_status", "query_operation", "inferred_as_of", "computation_status"}}
    assert len(set(totals.values())) == 1, totals
