"""T1 回归: offline 礼物/活动消息 + music 事件回复接 trace.

测试手册 §4 缺口修复: 这两组消息此前无 tracer, 15 个 offline.*/music.*
prompt 无法在 trace 面板 debug. 修复后消息 metadata 带 trace_id →
前端 Trace 按钮可点 (与 proactive 路径对齐).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeTracer:
    safe_trace_id = "trace-123"


@asynccontextmanager
async def _fake_traced_session(**kwargs):
    yield _FakeTracer()


@pytest.mark.asyncio
async def test_offline_trace_yields_tracer_with_trace_id():
    from app.services.offline import chat_emit

    with patch(
        "app.services.llm.usage_tracker.traced_usage_session",
        _fake_traced_session,
    ):
        async with chat_emit.offline_trace(
            "gift_sent", conversation_id="c1", agent_id="a1", user_id="u1",
        ) as tracer:
            assert tracer.safe_trace_id == "trace-123"


@pytest.mark.asyncio
async def test_emit_assistant_passes_trace_id_through():
    from app.services.offline import chat_emit

    with patch.object(
        chat_emit, "emit_proactive_message", AsyncMock(return_value="m1"),
    ) as emit:
        result = await chat_emit.emit_assistant(
            conversation_id="c1", user_id="u1", agent_id="a1",
            workspace_id="w1", message="礼物在路上啦",
            real_world_type="gift", source_id="g1",
            trigger_type="gift_sent", trace_id="trace-123",
        )
    assert result == "m1"
    assert emit.call_args.kwargs["trace_id"] == "trace-123"


@pytest.mark.asyncio
async def test_emit_gift_card_passes_trace_id_through():
    from app.services.offline import chat_emit

    with patch.object(
        chat_emit, "emit_proactive_message", AsyncMock(return_value="m1"),
    ) as emit:
        await chat_emit.emit_gift_card(
            conversation_id="c1", user_id="u1", agent_id="a1",
            workspace_id="w1", gift={"id": "g1", "gift_name": "杯子"},
            trigger_type="gift_sent", status_label="在路上",
            message="给你挑了个杯子", trace_id="trace-123",
        )
    assert emit.call_args.kwargs["trace_id"] == "trace-123"


@pytest.mark.asyncio
async def test_emit_assistant_backward_compatible_without_trace():
    """不传 trace_id 时行为不变 (静态文案发射点无 tracer)."""
    from app.services.offline import chat_emit

    with patch.object(
        chat_emit, "emit_proactive_message", AsyncMock(return_value="m1"),
    ) as emit:
        await chat_emit.emit_assistant(
            conversation_id="c1", user_id="u1", agent_id="a1",
            workspace_id="w1", message="没关系，这个先不算。",
            real_world_type="activity", source_id="act1",
            trigger_type="offline_activity_skip",
        )
    assert emit.call_args.kwargs["trace_id"] is None


@pytest.mark.asyncio
async def test_music_exit_reply_metadata_carries_trace_id():
    from app.services import music_status

    track = MagicMock()
    with (
        patch(
            "app.services.llm.usage_tracker.traced_usage_session",
            _fake_traced_session,
        ),
        patch.object(
            music_status, "_render_exit_reply",
            AsyncMock(return_value="我先去忙啦，歌你慢慢听"),
        ),
        patch.object(
            music_status, "_persist_assistant_message",
            AsyncMock(return_value="m1"),
        ) as persist,
        patch.object(music_status.manager, "send_event", AsyncMock()),
    ):
        result = await music_status._emit_rendered_reply(
            conversation_id="c1",
            prompt_key="music.busy_exit",
            user_name="你",
            ai_name="小满",
            activity="开会",
            track=track,
            music_co_listening=False,
        )
    assert result == "m1"
    metadata = persist.call_args.kwargs["metadata"]
    assert metadata["trace_id"] == "trace-123"
    assert metadata["music_prompt_key"] == "music.busy_exit"


@pytest.mark.asyncio
async def test_music_exit_reply_no_trace_id_when_tracing_disabled():
    """tracing 关闭 (safe_trace_id=None) 时 metadata 不带 trace_id 键."""
    from app.services import music_status

    class _NoTracer:
        safe_trace_id = None

    @asynccontextmanager
    async def _disabled_session(**kwargs):
        yield _NoTracer()

    with (
        patch(
            "app.services.llm.usage_tracker.traced_usage_session",
            _disabled_session,
        ),
        patch.object(
            music_status, "_render_exit_reply", AsyncMock(return_value="回复"),
        ),
        patch.object(
            music_status, "_persist_assistant_message",
            AsyncMock(return_value="m1"),
        ) as persist,
        patch.object(music_status.manager, "send_event", AsyncMock()),
    ):
        await music_status._emit_rendered_reply(
            conversation_id="c1", prompt_key="music.busy_exit",
            user_name="你", ai_name="小满", activity="开会",
            track=MagicMock(), music_co_listening=False,
        )
    assert "trace_id" not in persist.call_args.kwargs["metadata"]
