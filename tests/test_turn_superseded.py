# 首气泡发出前发现更新的用户消息 -> 中止本轮 (TurnSuperseded), 且不留下副作用.
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.chat import reply_post_process as rpp

_SINCE = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)


async def _drain(monkeypatch, *, newer, offset=0):
    mood = AsyncMock()
    monkeypatch.setattr(rpp, 'save_ai_mood', mood)
    monkeypatch.setattr(rpp, 'has_newer_user_message', AsyncMock(return_value=newer))
    monkeypatch.setattr(rpp, '_load_recent_emojis', AsyncMock(return_value=[]))
    monkeypatch.setattr(rpp, 'should_add_emoji', lambda *_a, **_k: False)
    monkeypatch.setattr(rpp, 'should_add_sticker', lambda *_a, **_k: False)
    emitted = []
    events = []
    try:
        async for evt in rpp.emit_replies(
            ['你好'],
            reply_context=None,
            reply_index_offset=offset,
            sub_intent_mode=True,
            agent=None,
            user_message='hi',
            delay_reply_fn=AsyncMock(),
            fallback_fn=AsyncMock(),
            emitted_replies=emitted,
            reply_emotion={'emotion': '中性', 'intensity': 0},
            conversation_id='c1',
            persona_checked=True,
            covered_until_user_ts=_SINCE,
        ):
            events.append(evt)
    except rpp.TurnSuperseded:
        return 'superseded', events, emitted, mood
    return 'done', events, emitted, mood


def test_newer_user_message_aborts_before_first_bubble(monkeypatch):
    status, events, emitted, mood = asyncio.run(_drain(monkeypatch, newer=True))
    assert status == 'superseded'
    assert events == [] and emitted == []
    mood.assert_not_awaited()


def test_no_newer_message_emits_normally(monkeypatch):
    status, events, emitted, mood = asyncio.run(_drain(monkeypatch, newer=False))
    assert status == 'done'
    assert len(events) == 1 and len(emitted) == 1
    mood.assert_awaited_once()


def test_continuation_fragment_never_aborts(monkeypatch):
    # reply_index_offset > 0 说明首气泡已发出, 不能回复一半又撤回
    status, events, _emitted, _mood = asyncio.run(_drain(monkeypatch, newer=True, offset=1))
    assert status == 'done'
    assert len(events) == 1


def test_has_newer_user_message_fails_open(monkeypatch):
    from app.db import db

    monkeypatch.setattr(db, 'query_raw', AsyncMock(side_effect=RuntimeError('boom')), raising=False)
    assert asyncio.run(rpp.has_newer_user_message('c1', _SINCE)) is False


def test_has_newer_user_message_skips_without_inputs():
    assert asyncio.run(rpp.has_newer_user_message(None, _SINCE)) is False
    assert asyncio.run(rpp.has_newer_user_message('c1', None)) is False
