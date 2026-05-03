"""Reminder emit 失败重试 + DLQ + scan_triggers Semaphore 限流测试.

工程扩展, 详见 CLAUDE.md §6 偏离表. 这些都是 spec 没说但 scale 必备的容错:

- LLM Semaphore 限流: scan 里启 N 个 trigger, 同时只 8 个进 LLM 段, 防 dashscope
  rate limit (QPS=60) 被一波 reminder 打爆触发 circuit breaker (影响半径放大)
- emit 失败重试: 之前 silent log + dead 一次性提醒永久丢; 现在 once 路径自动
  retry up to 3 次 (推迟 30s 触发 next scan), 超限进 DLQ
- DLQ: Redis ZSET 存最近 1000 条失败记录, admin 可读 ZRANGE 排查
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.proactive import triggers as trig_mod


_TZ = timezone.utc


def _make_trigger(
    *,
    trigger_id="t-1",
    action_data=None,
    last_fired=None,
    trigger_time=None,
):
    return SimpleNamespace(
        id=trigger_id,
        aiAgentId="agent-A",
        userId="user-1",
        actionData=action_data or {"summary": "喝水", "memory_id": "m1", "recurrence": "once"},
        actionType="reminder",
        triggerTime=trigger_time or datetime(2025, 6, 15, 14, 0, tzinfo=_TZ),
        lastFired=last_fired,
        repeatRule=None,
        isActive=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Semaphore 限流 — 防 N 个 reminder 同时启 N 个 LLM 打爆 rate limit
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_scan_triggers_caps_concurrency_at_semaphore_limit():
    """scan_triggers 里 20 个 trigger 同时到点, Semaphore(8) 限流后任意时刻
    in-flight 不超过 8. 防 dashscope rate limit 被打爆."""
    # Reset module-level semaphore so test gets a fresh one bound to this loop
    trig_mod._trigger_semaphore = None

    in_flight = 0
    max_in_flight = 0
    started = 0

    async def _slow_execute(trig, now):
        nonlocal in_flight, max_in_flight, started
        in_flight += 1
        started += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)  # 模拟 LLM 5s 链路
        in_flight -= 1

    triggers = [_make_trigger(trigger_id=f"t-{i}") for i in range(20)]

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_execute_trigger", side_effect=_slow_execute),
    ):
        mock_db.timetrigger.find_many = AsyncMock(return_value=triggers)
        await trig_mod.scan_triggers()

    assert started == 20, f"全部 trigger 都该被处理; got {started}"
    assert max_in_flight <= trig_mod._TRIGGER_LLM_CONCURRENCY, (
        f"任意时刻 in-flight 不能超过 {trig_mod._TRIGGER_LLM_CONCURRENCY}; "
        f"实际 max={max_in_flight}"
    )


@pytest.mark.asyncio
async def test_scan_triggers_semaphore_isolates_exception_per_trigger():
    """单个 trigger 抛异常 (gather return_exceptions=True) 不能挂起其他 —
    Semaphore release 必须发生 (async with), 否则后续 trigger 永远拿不到 token."""
    trig_mod._trigger_semaphore = None

    completed = 0

    async def _flaky_execute(trig, now):
        nonlocal completed
        if trig.id == "t-bad":
            raise RuntimeError("LLM 炸了")
        await asyncio.sleep(0.01)
        completed += 1

    triggers = [_make_trigger(trigger_id=f"t-{i}") for i in range(5)]
    triggers[2] = _make_trigger(trigger_id="t-bad")

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_execute_trigger", side_effect=_flaky_execute),
    ):
        mock_db.timetrigger.find_many = AsyncMock(return_value=triggers)
        await trig_mod.scan_triggers()

    # 4 个非 t-bad 都该完成 (异常的 t-bad 释放了 sem 让其他继续)
    assert completed == 4, (
        f"异常隔离失败 — Semaphore 没正确 release; completed={completed}"
    )


# ═══════════════════════════════════════════════════════════════════
# Emit 失败重试 (once 路径) — actionData.retry_count 累加, +30s reactivate
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_emit_failure_once_first_attempt_reactivates():
    """once reminder 第一次 emit 失败 → reactivate trigger, retry_count=1,
    triggerTime 推 30s, 不进 DLQ."""
    trigger = _make_trigger(
        action_data={"summary": "喝水", "memory_id": "m1", "recurrence": "once"},
    )
    update_calls = []

    async def _capture_update(*, where, data):
        update_calls.append({"where": where, "data": data})

    dlq_calls = []

    async def _capture_dlq(*args, **kwargs):
        dlq_calls.append(kwargs)

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_push_reminder_dlq", side_effect=_capture_dlq),
    ):
        mock_db.timetrigger.update = AsyncMock(side_effect=_capture_update)
        await trig_mod._handle_emit_failure(
            trigger, "once", RuntimeError("emit 链路 dashscope 503"),
        )

    assert len(update_calls) == 1, "应 reactivate trigger (1 次 update)"
    upd = update_calls[0]["data"]
    assert upd["isActive"] is True
    assert upd["lastFired"] is None  # reset 让 idempotency 守门 (2min 窗) 不误拦
    # triggerTime 推 ~30s
    new_time = upd["triggerTime"]
    delta = new_time - datetime.now(_TZ)
    assert timedelta(seconds=25) <= delta <= timedelta(seconds=35), (
        f"triggerTime 应推 ~30s, 实际 delta={delta}"
    )
    # actionData 含 retry_count=1
    # Json wrapper 在 prisma 里 serialize, 测试时它把 dict 透传
    new_data = upd["actionData"]
    if hasattr(new_data, "data"):  # Json wrapper
        new_data = new_data.data
    assert new_data["retry_count"] == 1
    assert new_data["memory_id"] == "m1"  # 原 data 保留

    assert dlq_calls == [], "第一次失败不该进 DLQ"


@pytest.mark.asyncio
async def test_emit_failure_once_exhausted_goes_to_dlq():
    """once reminder retry_count 已达上限 → 进 DLQ, 不再 reactivate (永久 dead)."""
    trigger = _make_trigger(
        action_data={
            "summary": "喝水", "memory_id": "m1",
            "recurrence": "once", "retry_count": trig_mod.MAX_REMINDER_RETRY,
        },
    )
    update_calls = []
    dlq_calls = []

    async def _capture_dlq(trig, error, *, kind, attempt):
        dlq_calls.append({"trigger_id": trig.id, "kind": kind, "attempt": attempt})

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_push_reminder_dlq", side_effect=_capture_dlq),
    ):
        mock_db.timetrigger.update = AsyncMock(side_effect=lambda **k: update_calls.append(k))
        await trig_mod._handle_emit_failure(
            trigger, "once", RuntimeError("again 503"),
        )

    assert len(update_calls) == 0, "exhausted 不该 reactivate"
    assert len(dlq_calls) == 1
    assert dlq_calls[0]["kind"] == "exhausted"
    assert dlq_calls[0]["attempt"] == trig_mod.MAX_REMINDER_RETRY


@pytest.mark.asyncio
async def test_emit_failure_periodic_does_not_retry():
    """periodic reminder emit 失败 → 不 reactivate (续期 next_occur 已建,
    下周期照常), 仅写 DLQ 留痕方便 admin 排查."""
    trigger = _make_trigger(
        action_data={"summary": "吃药", "memory_id": "m2", "recurrence": "daily"},
    )
    update_calls = []
    dlq_calls = []

    async def _capture_dlq(trig, error, *, kind, attempt):
        dlq_calls.append({"kind": kind})

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_push_reminder_dlq", side_effect=_capture_dlq),
    ):
        mock_db.timetrigger.update = AsyncMock(side_effect=lambda **k: update_calls.append(k))
        await trig_mod._handle_emit_failure(
            trigger, "daily", RuntimeError("emit 失败"),
        )

    assert len(update_calls) == 0, "周期性不 reactivate (下周期照常)"
    assert len(dlq_calls) == 1
    assert dlq_calls[0]["kind"] == "periodic_lost_one"


@pytest.mark.asyncio
async def test_emit_failure_reactivate_failure_falls_to_dlq():
    """reactivate (db.update) 也失败 → 直接进 DLQ, 防陷入"既不响也不在 DLQ"幽灵态."""
    trigger = _make_trigger(
        action_data={"summary": "X", "memory_id": "m3", "recurrence": "once"},
    )
    dlq_calls = []

    async def _capture_dlq(trig, error, *, kind, attempt):
        dlq_calls.append({"kind": kind, "error": error[:50]})

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_push_reminder_dlq", side_effect=_capture_dlq),
    ):
        mock_db.timetrigger.update = AsyncMock(
            side_effect=ConnectionError("DB unreachable"),
        )
        await trig_mod._handle_emit_failure(
            trigger, "once", RuntimeError("original emit failure"),
        )

    assert len(dlq_calls) == 1
    assert dlq_calls[0]["kind"] == "reactivate_failed"
    assert "DB unreachable" in dlq_calls[0]["error"]


# ═══════════════════════════════════════════════════════════════════
# DLQ Redis ZSET 写入 + cap 大小
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_push_dlq_writes_zset_entry_with_metadata():
    """_push_reminder_dlq 写一条 JSON entry 到 Redis ZSET, 含 trigger_id /
    summary / error / kind / attempt / failed_at."""
    trigger = _make_trigger(
        trigger_id="t-failed",
        action_data={
            "summary": "喝水提醒", "memory_id": "mem-x",
            "recurrence": "once",
        },
    )
    zadd_calls = []
    zremrangebyrank_calls = []

    async def _capture_zadd(key, mapping):
        zadd_calls.append({"key": key, "mapping": mapping})

    async def _capture_zremrangebyrank(key, start, stop):
        zremrangebyrank_calls.append({"key": key, "start": start, "stop": stop})

    fake_redis = MagicMock(
        zadd=AsyncMock(side_effect=_capture_zadd),
        zremrangebyrank=AsyncMock(side_effect=_capture_zremrangebyrank),
    )

    with patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                      return_value=fake_redis):
        await trig_mod._push_reminder_dlq(
            trigger, "test error msg", kind="exhausted", attempt=3,
        )

    assert len(zadd_calls) == 1
    call = zadd_calls[0]
    assert call["key"] == trig_mod._DLQ_KEY
    # mapping 是 {json_str: timestamp}
    entries = list(call["mapping"].items())
    assert len(entries) == 1
    payload_str, score = entries[0]
    payload = json.loads(payload_str)
    assert payload["trigger_id"] == "t-failed"
    assert payload["memory_id"] == "mem-x"
    assert payload["summary"] == "喝水提醒"
    assert payload["recurrence"] == "once"
    assert payload["kind"] == "exhausted"
    assert payload["attempt"] == 3
    assert payload["error"] == "test error msg"
    assert "failed_at" in payload
    assert isinstance(score, float)

    # Cap: zremrangebyrank 必须调一次 (-MAX_SIZE-1 = -1001) 防无限增长
    assert len(zremrangebyrank_calls) == 1
    assert zremrangebyrank_calls[0]["stop"] == -(trig_mod._DLQ_MAX_SIZE + 1)


@pytest.mark.asyncio
async def test_push_dlq_redis_failure_does_not_raise():
    """DLQ 写本身失败 (Redis 挂) 不能冒泡 — 否则掩盖原始 emit failure."""
    trigger = _make_trigger()
    fake_redis = MagicMock(zadd=AsyncMock(side_effect=ConnectionError("redis down")))

    with patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                      return_value=fake_redis):
        # 不抛
        await trig_mod._push_reminder_dlq(
            trigger, "err", kind="exhausted", attempt=3,
        )


# ═══════════════════════════════════════════════════════════════════
# 端到端: _handle_reminder_trigger 在 emit 失败时走 _handle_emit_failure
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handle_reminder_trigger_emit_failure_invokes_retry_handler():
    """完整路径: trigger 到点 → pre-check needed → claim → renewal → emit
    抛异常 → 走 _handle_emit_failure (不 silent dead)."""
    now = datetime(2025, 6, 15, 10, 0, tzinfo=_TZ)
    trigger = _make_trigger(
        action_data={"summary": "X", "memory_id": "m1", "recurrence": "once"},
        trigger_time=now - timedelta(seconds=5),
    )

    async def _needed(**kw):
        return {"state": "needed", "new_time": None, "reason": ""}

    failure_calls = []

    async def _capture_failure(trig, recurrence, exc):
        failure_calls.append({
            "trigger_id": trig.id, "recurrence": recurrence,
            "exc_type": type(exc).__name__, "msg": str(exc)[:50],
        })

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value=""),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              side_effect=_needed),
        patch.object(trig_mod, "resolve_workspace_id",
                     new_callable=AsyncMock, return_value="ws-1"),
        # emit 抛 → 应触发 _handle_emit_failure
        patch("app.services.proactive.emit.emit_proactive_message",
              new_callable=AsyncMock,
              side_effect=RuntimeError("simulated emit failure")),
        patch.object(trig_mod, "_handle_emit_failure", side_effect=_capture_failure),
        patch("app.services.proactive.state.get_active_workspace_context",
              new_callable=AsyncMock,
              return_value={"conversation_id": "conv-1"}),
        patch("app.services.chat.intent_replies.reminder_message",
              new_callable=AsyncMock, return_value="该喝水啦~"),
        patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                     return_value=MagicMock(incr=AsyncMock(), expire=AsyncMock())),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        mock_db.aiagent.find_unique = AsyncMock(
            return_value=SimpleNamespace(id="agent-A", name="A"),
        )
        await trig_mod._handle_reminder_trigger(trigger, now)

    assert len(failure_calls) == 1, (
        f"emit 抛后必须调 _handle_emit_failure (不能 silent log + dead); "
        f"got {failure_calls}"
    )
    assert failure_calls[0]["recurrence"] == "once"
    assert "simulated emit failure" in failure_calls[0]["msg"]
