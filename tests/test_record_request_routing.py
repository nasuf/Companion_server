from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ctx():
    from app.services.chat.intent_handlers import ShortCircuitCtx

    return ShortCircuitCtx(
        conversation_id="c1",
        agent_id="a1",
        user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )


async def _fake_finalize_short_circuit(reply, **kwargs):
    yield {"event": "reply", "data": reply}


@pytest.mark.asyncio
async def test_memory_note_record_request_does_not_create_reminder():
    from app.services.chat.intent_handlers import handle_record_request

    ctx = _ctx()
    with (
        patch(
            "app.services.chat.intent_handlers._direct_create_reminder",
            new=AsyncMock(side_effect=AssertionError("should not create reminder")),
        ),
        patch(
            "app.services.chat.intent_handlers.finalize_short_circuit",
            new=_fake_finalize_short_circuit,
        ),
    ):
        handled, gen = await handle_record_request(
            "记一下：我处理关系时会倾向先保留余地",
            ctx,
        )
        assert handled is True
        events = [evt async for evt in gen]

    assert events[0]["event"] == "reply"
    assert ctx.consumed_full_message is True
    assert ctx.last_short_circuit_kind == "record_request_memory_note"


@pytest.mark.asyncio
async def test_self_note_record_request_falls_through_to_normal_chat():
    from app.services.chat.intent_handlers import handle_record_request

    handled, gen = await handle_record_request(
        "你帮我把它压缩成一句我能贴在备忘录里的话",
        _ctx(),
    )
    assert handled is False
    assert gen is None


@pytest.mark.asyncio
async def test_single_reminder_content_update_without_text_saves_pending():
    from datetime import datetime, timezone
    from app.services.chat.intent_handlers import handle_record_request

    fake_trigger = SimpleNamespace(
        id="trig-1",
        triggerTime=datetime(2026, 5, 8, 10, 0, tzinfo=timezone.utc),
        actionData={"summary": "提醒 A", "memory_id": "m-1", "memory_side": "user"},
    )
    saved = []

    async def _find_active_reminders(**_kwargs):
        return [fake_trigger]

    async def _save_pending(conv_id, *, action, candidates=None, summary=None, **_kw):
        saved.append({
            "conv_id": conv_id,
            "action": action,
            "candidates": candidates,
            "summary": summary,
        })

    with (
        patch(
            "app.services.reminder.scheduling.find_active_reminder_triggers",
            side_effect=_find_active_reminders,
        ),
        patch(
            "app.services.memory.interaction.deletion.save_pending_action",
            side_effect=_save_pending,
        ),
        patch(
            "app.services.chat.intent_handlers.finalize_short_circuit",
            new=_fake_finalize_short_circuit,
        ),
    ):
        handled, gen = await handle_record_request("提醒内容改一下", _ctx())
        events = [evt async for evt in gen]

    assert handled is True
    assert events
    assert saved and saved[0]["action"] == "update_reminder_content"
    assert len(saved[0]["candidates"]) == 1
