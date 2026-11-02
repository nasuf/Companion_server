"""TIME10: "先别提醒了" 暂停全部有效提醒, "恢复提醒" 只恢复被暂停的。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.chat import intent_handlers as ih
from app.services.chat.intent_handlers import classify_pause_resume
from app.services.reminder import scheduling
from tests.test_memory_evidence_postgres import origins  # noqa: F401

NOW = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("message, action", [
    ("先别提醒我了", "pause"), ("暂停所有提醒", "pause"), ("这几天不用提醒我", "pause"),
    ("提醒先停一下", "pause"), ("把提醒都暂停吧", "pause"), ("暂时先别提醒了", "pause"),
    ("恢复提醒吧", "resume"), ("可以继续提醒了", "resume"), ("提醒恢复一下", "resume"),
    ("把暂停的提醒恢复", "resume"),
    # 带了具体事项或时间的, 走原来的取消 / 新建流程
    ("别提醒我吃药了", None), ("先别提醒我吃药了", None), ("这几天别提醒我喝水了", None),
    ("把吃药的提醒暂停一下", None), ("明天继续提醒我吃药", None), ("10分钟后重新提醒我", None),
    ("明天8点重新提醒我吧", None), ("取消明天的提醒", None), ("提醒我明天开会", None),
    # 否定暂停
    ("别暂停提醒了", None), ("不要暂停提醒", None),
    # 在问或在报告提醒坏了, 不是吩咐
    ("提醒停了？", None), ("我的提醒停了", None), ("提醒都停了", None), ("提醒恢复了吗", None),
    ("提醒开了", None),
])
def test_classify_pause_resume(message, action):
    assert classify_pause_resume(message) == action


async def _trigger(db, uid, aid, wid, *, minutes, recurrence="once", active=True, data=None):
    """一条提醒 + 它对应的提醒记忆 (恢复时会核对记忆还在不在)。"""
    from prisma import Json

    memory = await db.usermemory.create(data={
        "userId": uid, "workspaceId": wid, "content": f"提醒{minutes}", "level": 3,
        "mainCategory": "生活", "subCategory": "提醒",
    })
    return await db.timetrigger.create(data={
        "aiAgentId": aid, "userId": uid, "actionType": "reminder", "isActive": active,
        "triggerTime": NOW + timedelta(minutes=minutes),
        "actionData": Json({"summary": f"提醒{minutes}", "memory_id": memory.id,
                            "memory_side": "user", "recurrence": recurrence, **(data or {})}),
    })


@pytest.mark.asyncio
async def test_pause_resume_cancel_and_undo_on_real_sql(origins, monkeypatch):
    db, uid, wid, *_rest = origins
    aid = _rest[-3][0]
    monkeypatch.setattr(scheduling, "db", db)
    monkeypatch.setattr(ih, "db", db)
    try:
        soon = await _trigger(db, uid, aid, wid, minutes=60)
        daily = await _trigger(db, uid, aid, wid, minutes=30, recurrence="daily")
        later = await _trigger(db, uid, aid, wid, minutes=600)
        # 用户早先取消的: 停用, 没响过, 不带暂停标记 —— 暂停/恢复都不碰它
        cancelled = await _trigger(db, uid, aid, wid, minutes=90, active=False)

        assert await scheduling.pause_reminders(user_id=uid, agent_id=aid, now=NOW) == 3
        assert {t.id for t in await scheduling.find_paused_reminder_triggers(user_id=uid, agent_id=aid)} \
            == {soon.id, daily.id, later.id}
        assert await scheduling.find_active_reminder_triggers(user_id=uid, agent_id=aid) == []

        # 暂停中取消一条: 之后恢复不会把它带回来; 撤回取消则回到暂停状态
        monkeypatch.setattr(ih, "get_current_time", lambda: type("C", (), {"now": NOW})())
        assert await ih._cancel_active_reminders(user_id=uid, agent_id=aid, trigger_ids=[later.id]) == 1
        row = await db.timetrigger.find_unique(where={"id": later.id})
        assert not row.isActive and "paused_at" not in row.actionData
        snapshot = [{"trigger_id": later.id, "action_data": {**dict(later.actionData),
                                                              "paused_at": NOW.isoformat()}}]
        assert await scheduling.reactivate_reminder_triggers(snapshot) == 1
        row = await db.timetrigger.find_unique(where={"id": later.id})
        assert not row.isActive and row.actionData["paused_at"] == NOW.isoformat()
        assert await ih._cancel_active_reminders(user_id=uid, agent_id=aid, trigger_ids=[later.id]) == 1

        # 两小时后恢复: 一次性的已过点记成错过, 每日的挪到下一期, 取消过的不动
        resumed_at = NOW + timedelta(hours=2)
        assert await scheduling.resume_reminders(user_id=uid, agent_id=aid, now=resumed_at) == (1, 1)
        rows = {t.id: t for t in await db.timetrigger.find_many(where={"userId": uid})}
        assert not rows[soon.id].isActive and rows[soon.id].actionData["missed_reason"] == "paused"
        assert rows[daily.id].isActive and resumed_at < rows[daily.id].triggerTime <= resumed_at + timedelta(days=1)
        assert "paused_at" not in rows[daily.id].actionData
        assert not rows[later.id].isActive and not rows[cancelled.id].isActive
        assert await scheduling.find_paused_reminder_triggers(user_id=uid, agent_id=aid) == []

        # 暂停后在 App 里删掉的, 不再算暂停中, 恢复也不会带回来
        gone = await _trigger(db, uid, aid, wid, minutes=900)
        assert await scheduling.pause_reminders(user_id=uid, agent_id=aid, now=NOW) >= 1
        from prisma import Json
        row = await db.timetrigger.find_unique(where={"id": gone.id})
        await db.timetrigger.update(where={"id": gone.id}, data={"actionData": Json(
            {**dict(row.actionData), "deleted_at": NOW.isoformat()})})
        assert gone.id not in {t.id for t in await scheduling.find_paused_reminder_triggers(
            user_id=uid, agent_id=aid)}
    finally:
        await db.timetrigger.delete_many(where={"userId": uid})


@pytest.mark.asyncio
async def test_pause_does_not_overwrite_a_row_changed_after_reading(origins, monkeypatch):
    from prisma import Json

    db, uid, wid, *_rest = origins
    aid = _rest[-3][0]
    monkeypatch.setattr(scheduling, "db", db)
    try:
        row = await _trigger(db, uid, aid, wid, minutes=60)
        stale = await db.timetrigger.find_unique(where={"id": row.id})
        # 读出之后被重试改过 actionData
        await db.timetrigger.update(where={"id": row.id},
                                    data={"actionData": Json({**dict(row.actionData), "retry_count": 1})})
        assert await scheduling._cas_trigger(stale, active=False,
                                             data={**dict(stale.actionData), "paused_at": NOW.isoformat()}) is False
        fresh = await db.timetrigger.find_unique(where={"id": row.id})
        assert fresh.isActive and fresh.actionData["retry_count"] == 1
    finally:
        await db.timetrigger.delete_many(where={"userId": uid})


@pytest.mark.asyncio
async def test_preflight_takes_resume_only_when_something_is_paused(monkeypatch):
    from app.services.chat import preflight

    ctx = preflight.PreflightCtx(conversation_id="c", agent_id="a", user_id="u", agent=None,
                                 tracer=MagicMock(), short_circuit_fn=AsyncMock(return_value=[]))
    monkeypatch.setattr(scheduling, "find_paused_reminder_triggers", AsyncMock(return_value=[]))
    reply = AsyncMock(return_value="好")
    monkeypatch.setattr(ih, "_pause_or_resume_reply", reply)
    assert [e async for e in preflight.resolve_reminder_pause("恢复提醒", ctx)] == []
    assert not ctx.stopped and not reply.await_count  # 交给撤回流程

    monkeypatch.setattr(scheduling, "find_paused_reminder_triggers", AsyncMock(return_value=[object()]))
    [e async for e in preflight.resolve_reminder_pause("恢复提醒", ctx)]
    assert ctx.stopped and reply.await_args.args[0] == "resume"

    # 正在等用户补充提醒时间时, "先别提醒了" 是在回答那个问题
    ctx.stopped = False
    reply.reset_mock()
    monkeypatch.setattr(preflight, "load_pending_action", AsyncMock(return_value={"action": "set_reminder"}))
    [e async for e in preflight.resolve_reminder_pause("先别提醒了", ctx)]
    assert not ctx.stopped and not reply.await_count
