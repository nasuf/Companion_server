"""WS "已读" ack event 测试.

工程扩展 (2026-05): 之前用户发消息到 AI 流式回复中间 1-5s 无任何反馈, 体感
"消息石沉大海". 加 ack event: persist 落库后立刻发, 前端在气泡旁标 ✓✓.

详见 CLAUDE.md §6 偏离表.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.realtime import ws as ws_mod
from app.models.music import MusicCoListeningResponse, MusicTrack
from app.services.interaction import user_turn_aggregation as turn_mod
from app.services.interaction.user_turn_aggregation import UserMessageAggregationPlan


@pytest.fixture
def fake_ws():
    """Mock WebSocket 捕获 send_json 调用."""
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


def _ack_payloads(ws_mock) -> list[dict]:
    """从 send_json 调用历史里提取所有 type='ack' 的 payload."""
    return [
        call.args[0]
        for call in ws_mock.send_json.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "ack"
    ]


def _aggregation_plan(
    route: str,
    *,
    text: str = "测试消息",
    metadata: dict | None = None,
) -> UserMessageAggregationPlan:
    return UserMessageAggregationPlan(
        route=route,
        agent_id="a1",
        user_id="user-1",
        conversation_id="conv-1",
        text=text,
        metadata=metadata or {"queued": True},
        final_message=text,
        final_context={},
        fallback_message=text,
        fallback_context={},
    )


def test_sanitize_component_card_keeps_allowlisted_payload():
    """聊天卡片 metadata 只保留可回放渲染需要的通用字段。"""
    card = ws_mod._sanitize_component_card({
        "type": "time_capsule",
        "title": "未来胶囊",
        "subtitle": "2026年6月1日开启",
        "body": "一段写给未来的话",
        "footer": "时间胶囊 · 已开启",
        "accent": "#7C3CFF",
        "payload": {"capsule_id": "cap-1", "content": "secret", "ignored": "x"},
        "ignored": "x",
    })

    assert card == {
        "version": 1,
        "type": "time_capsule",
        "title": "未来胶囊",
        "subtitle": "2026年6月1日开启",
        "body": "一段写给未来的话",
        "footer": "时间胶囊 · 已开启",
        "accent": "#7C3CFF",
        "payload": {"capsule_id": "cap-1", "content": "secret"},
    }


def test_sanitize_component_card_limits_payload_size():
    card = ws_mod._sanitize_component_card({
        "type": "time_capsule",
        "body": "x",
        "payload": {
            "capsule_id": "cap-1",
            "content": "a" * 1200,
            "nested": {"large": "ignored"},
        },
    })

    assert card is not None
    assert set(card["payload"]) == {"capsule_id", "content"}
    assert len(card["payload"]["content"]) == 1000


def test_sanitize_external_link_card_preserves_app_url():
    card = ws_mod._sanitize_component_card({
        "type": "external_link",
        "title": "头条文章",
        "payload": {
            "link_id": "link-1",
            "app_url": "snssdk141://detail?groupid=7651359327906710016",
            "final_url": "https://www.toutiao.com/article/7651359327906710016/",
            "ignored": "x",
        },
    })

    assert card is not None
    assert card["payload"] == {
        "link_id": "link-1",
        "app_url": "snssdk141://detail?groupid=7651359327906710016",
        "final_url": "https://www.toutiao.com/article/7651359327906710016/",
    }


def test_sanitize_component_card_rejects_unknown_type():
    assert ws_mod._sanitize_component_card({"type": "unknown"}) is None


def test_sanitize_component_card_allows_offline_gift_payload():
    card = ws_mod._sanitize_component_card({
        "type": "offline_gift",
        "title": "暖手宝",
        "subtitle": "在路上",
        "body": "我给你寄了个小东西。",
        "footer": "点击查看礼物详情",
        "accent": "#F6A64B",
        "payload": {
            "gift_id": "gift-1",
            "status": "shipping",
            "status_label": "在路上",
            "gift_name": "暖手宝",
            "image_url": "/offline/gifts/gift-1/image",
            "real_world_type": "gift",
            "ignored": "x",
        },
    })

    assert card is not None
    assert card["type"] == "offline_gift"
    assert card["payload"] == {
        "gift_id": "gift-1",
        "status": "shipping",
        "status_label": "在路上",
        "gift_name": "暖手宝",
        "image_url": "/offline/gifts/gift-1/image",
        "real_world_type": "gift",
    }


def test_weibo_visitor_link_card_needs_refresh():
    link = SimpleNamespace(
        platform="微博",
        title="Sina Visitor System",
        description=(
            "Sina Visitor System https://weibo.com/6305330504/5311661846303790 "
            "https://weibo.com/6305330504/5311661846303790"
        ),
        summary="",
        content_text="",
    )

    assert ws_mod._link_card_needs_refresh(link)


def test_normal_weibo_link_card_does_not_need_refresh():
    link = SimpleNamespace(
        platform="微博",
        title="马斯克又当爹",
        description="马斯克又当爹！这次是和 Neuralink 女高管相关的新闻。",
        summary="马斯克又当爹！这次是和 Neuralink 女高管相关的新闻。",
        content_text="马斯克又当爹！这次是和 Neuralink 女高管相关的新闻。",
    )

    assert not ws_mod._link_card_needs_refresh(link)


def test_sanitize_component_card_allows_checkin_types():
    reminder = ws_mod._sanitize_component_card({
        "type": "checkin_reminder",
        "title": "打卡提醒",
        "subtitle": "今天 23:00",
        "body": "睡前收尾",
        "footer": "一次提醒",
        "accent": "#4F6DF5",
    })
    habit = ws_mod._sanitize_component_card({
        "type": "checkin_habit",
        "title": "习惯打卡",
        "subtitle": "每天",
        "body": "早起",
        "footer": "周期习惯",
        "accent": "#22C66B",
    })

    assert reminder and reminder["type"] == "checkin_reminder"
    assert habit and habit["type"] == "checkin_habit"


def test_sanitize_component_card_allows_music_track_payload():
    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "body": "Focus 频道",
        "footer": "邀请一起听",
        "accent": "#1f6fff",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {
                "id": "track-1",
                "title": "Quiet Realm",
                "artist": "Jamendo Artist",
                "album": "Jamendo Focus",
                "library": "focus",
                "url": "https://cdn.example.test/a.mp3",
                "duration_sec": 240,
                "cover_key": "music-cover-02.jpg",
                "accent_a": "#1f6fff",
                "accent_b": "#18c6c0",
                "source": "jamendo",
                "metadata": {"lyrics": "line"},
            },
        },
    })

    assert card is not None
    assert card["type"] == "music_track"
    assert card["payload"]["intent"] == "invite"
    assert card["payload"]["track"]["id"] == "track-1"
    assert card["payload"]["track"]["metadata"] == {"lyrics": "line"}


def test_sanitize_checkin_card_preserves_weekdays_and_ai_flag():
    card = ws_mod._sanitize_component_card({
        "type": "checkin_habit",
        "title": "习惯打卡",
        "subtitle": "每周五、周六、周日 11:00",
        "body": "Testing - 5",
        "footer": "打卡 · 周期习惯",
        "accent": "#22C66B",
        "payload": {
            "trigger_id": "trigger-1",
            "summary": "Testing - 5",
            "recurrence": "weekly",
            "trigger_time": "2026-06-05T03:00:00Z",
            "habit_weekdays": [5, 6, 7],
            "sent_to_ai": True,
        },
    })

    assert card is not None
    assert card["payload"]["habit_weekdays"] == [5, 6, 7]
    assert card["payload"]["sent_to_ai"] is True


def test_checkin_component_reply_message_prevents_recreating_reminder():
    card = ws_mod._sanitize_component_card({
        "type": "checkin_habit",
        "title": "习惯打卡",
        "subtitle": "每周五、周六、周日 11:00",
        "body": "Testing - 5",
        "footer": "打卡 · 周期习惯",
        "payload": {
            "summary": "Testing - 5",
            "recurrence": "weekly",
            "trigger_time": "2026-06-05T03:00:00Z",
            "habit_weekdays": [5, 6, 7],
        },
    })

    message = ws_mod._component_card_reply_message("原始文字", card)

    assert message is not None
    assert "已创建的周期习惯打卡卡片" in message
    assert "Testing - 5" in message
    assert "每周五、周六、周日 11:00" in message
    assert "2026-06-05T03:00:00Z" not in message
    assert "不要重新创建提醒、不要反问时间" in message


@pytest.mark.asyncio
async def test_handle_message_checkin_card_skips_time_memory_lookup(fake_ws):
    """打卡卡片进入聊天时保留卡片时间语义, 但不触发普通显式时间记忆检索。"""
    agent = SimpleNamespace(id="a1", name="A")
    queued_kwargs = None

    async def _fake_persist(*args, **kwargs):
        return "db-msg-checkin-card-1"

    async def _fake_queue(*args, **kwargs):
        nonlocal queued_kwargs
        queued_kwargs = kwargs

    card = ws_mod._sanitize_component_card({
        "type": "checkin_reminder",
        "title": "打卡提醒",
        "subtitle": "2026年05月30日 21:59",
        "body": "Testing - 9",
        "footer": "打卡 · 一次提醒",
        "payload": {
            "summary": "Testing - 9",
            "recurrence": "once",
            "trigger_time": "2026-05-30T13:59:00Z",
        },
    })
    plan = _aggregation_plan("immediate", text="打卡提醒", metadata={"queued": True})

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(ws_mod, "_queue_reply_or_error", side_effect=_fake_queue),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "自由时间", "type": "leisure"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "自由时间", "type": "leisure", "status": "idle"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "打卡提醒",
            client_id="client-checkin-card-uuid",
            component_card=card,
        )

    assert queued_kwargs is not None
    assert queued_kwargs["reply_context"]["component_card_reply"] is True
    assert queued_kwargs["reply_context"]["skip_time_memory_lookup"] is True
    assert "2026-05-30T13:59:00Z" not in queued_kwargs["user_message"]


@pytest.mark.asyncio
async def test_handle_message_music_card_idle_starts_co_listening(fake_ws):
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-music-card-1"

    async def _fake_assistant(*args, **kwargs):
        return "assistant-music-1"

    event_order: list[str] = []

    async def _fake_send_json(payload):
        if isinstance(payload, dict) and payload.get("type") == "reply":
            event_order.append("agent_reply")

    async def _fake_music_status(*args, **kwargs):
        event_order.append(f"status:{kwargs.get('actor')}")
        return "music-status-1"

    fake_ws.send_json.side_effect = _fake_send_json

    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    })
    plan = _aggregation_plan("immediate", text="一起听")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "_persist_assistant_message", side_effect=_fake_assistant),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "自由时间", "type": "leisure"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "自由时间", "status": "idle"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.services.music.start_co_listening",
            new_callable=AsyncMock,
            return_value=None,
        ) as start_co,
        patch(
            "app.services.music_chat.render_music_reply",
            new_callable=AsyncMock,
            return_value="好呀，一起听。",
        ) as render_reply,
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new=AsyncMock(side_effect=_fake_music_status),
        ) as music_status,
        patch("app.services.chat.post_process._bg_memory_pipeline", new=lambda *_args, **_kwargs: None),
        patch("app.services.runtime.tasks.fire_background", new=lambda *_args, **_kwargs: None),
    ):
        await ws_mod._handle_message(
            fake_ws,
            "conv-1",
            "user-1",
            agent,
            "一起听",
            workspace_id="ws-1",
            component_card=card,
            user_name="Song",
        )

    start_co.assert_awaited_once()
    assert start_co.await_args.kwargs["initiated_by"] == "user_joined"
    render_reply.assert_awaited_once()
    assert render_reply.await_args.args[0] == "music.accept_invite"
    assert music_status.await_count == 2
    first_status = music_status.await_args_list[0].kwargs
    second_status = music_status.await_args_list[1].kwargs
    assert first_status["status"] == "started"
    assert first_status["actor"] == "user"
    assert second_status["status"] == "started"
    assert second_status["actor"] == "agent"
    assert second_status["actor_name"] == "A"
    assert event_order[:3] == ["agent_reply", "status:user", "status:agent"]
    envelopes = [call.args[0] for call in fake_ws.send_json.call_args_list]
    assert any(item.get("type") == "reply" and item["data"]["music_co_listening"] for item in envelopes)


@pytest.mark.asyncio
async def test_handle_message_music_card_while_agent_waiting_only_rejoins_user(fake_ws):
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-music-card-waiting"

    async def _fake_assistant(*args, **kwargs):
        return "assistant-music-waiting"

    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    })
    plan = _aggregation_plan("immediate", text="一起听")
    waiting = MusicCoListeningResponse(
        status="agent_waiting_user",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=False,
        initiated_by="user",
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "_persist_assistant_message", side_effect=_fake_assistant),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "自由时间", "type": "leisure"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "自由时间", "status": "idle"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=waiting,
        ),
        patch("app.services.music.start_co_listening", new_callable=AsyncMock) as start_co,
        patch(
            "app.services.music_chat.render_music_reply",
            new_callable=AsyncMock,
            return_value="好呀，继续听。",
        ),
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new_callable=AsyncMock,
            return_value="music-status-1",
        ) as music_status,
        patch("app.services.chat.post_process._bg_memory_pipeline", new=lambda *_args, **_kwargs: None),
        patch("app.services.runtime.tasks.fire_background", new=lambda *_args, **_kwargs: None),
    ):
        await ws_mod._handle_message(
            fake_ws,
            "conv-1",
            "user-1",
            agent,
            "一起听",
            workspace_id="ws-1",
            component_card=card,
            user_name="Song",
        )

    start_co.assert_awaited_once()
    assert start_co.await_args.kwargs["status"] == "active"
    music_status.assert_awaited_once()
    assert music_status.await_args.kwargs["status"] == "started"
    assert music_status.await_args.kwargs["actor"] == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initiated_by", "expected_actors"),
    [
        ("agent", ["user"]),
        ("agent_auto", ["user", "agent"]),
    ],
)
async def test_user_invite_joins_agent_only_music_session(
    fake_ws,
    initiated_by,
    expected_actors,
):
    agent = SimpleNamespace(id="a1", name="A")
    current = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=initiated_by == "agent_auto",
        initiated_by=initiated_by,
    )
    card = {
        "type": "music_track",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    }
    start_co = AsyncMock(return_value=None)
    music_status = AsyncMock(return_value="music-status-1")

    with (
        patch.object(
            ws_mod,
            "_persist_assistant_message",
            new_callable=AsyncMock,
            return_value="assistant-music-1",
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=current,
        ),
        patch(
            "app.services.music.start_co_listening",
            new=start_co,
        ),
        patch(
            "app.services.music_chat.render_music_reply",
            new_callable=AsyncMock,
            return_value="好呀，一起听。",
        ),
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new=music_status,
        ),
        patch(
            "app.services.chat.post_process._bg_memory_pipeline",
            new=lambda *_args, **_kwargs: None,
        ),
        patch(
            "app.services.notifications.service.notify_agent_message_created",
            new=lambda *_args, **_kwargs: None,
        ),
        patch(
            "app.services.runtime.tasks.fire_background",
            new=lambda *_args, **_kwargs: None,
        ),
    ):
        handled = await ws_mod._handle_music_component_card(
            fake_ws,
            conversation_id="conv-1",
            user_id="user-1",
            agent=agent,
            workspace_id="ws-1",
            user_name="Song",
            user_message_id="user-message-1",
            component_card=card,
            received_status={"status": "idle", "activity": "自由时间"},
        )

    assert handled is True
    start_co.assert_awaited_once()
    assert start_co.await_args.kwargs["initiated_by"] == "user_joined"
    assert [call.kwargs["actor"] for call in music_status.await_args_list] == expected_actors


@pytest.mark.asyncio
async def test_handle_message_music_card_while_active_switches_track_without_status(fake_ws):
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-music-card-switch"

    async def _fake_assistant(*args, **kwargs):
        return "assistant-music-switch"

    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    })
    plan = _aggregation_plan("immediate", text="换这首")
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=True,
        initiated_by="user",
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "_persist_assistant_message", side_effect=_fake_assistant),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "自由时间", "type": "leisure"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "自由时间", "status": "idle"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=active,
        ),
        patch("app.services.music.start_co_listening", new_callable=AsyncMock) as start_co,
        patch(
            "app.services.music_chat.render_music_reply",
            new_callable=AsyncMock,
            return_value="切歌啦，这首也一起听。",
        ) as render_reply,
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new_callable=AsyncMock,
            return_value="music-status-1",
        ) as music_status,
        patch("app.services.chat.post_process._bg_memory_pipeline", new=lambda *_args, **_kwargs: None),
        patch("app.services.runtime.tasks.fire_background", new=lambda *_args, **_kwargs: None),
    ):
        await ws_mod._handle_message(
            fake_ws,
            "conv-1",
            "user-1",
            agent,
            "换这首",
            workspace_id="ws-1",
            component_card=card,
            user_name="Song",
        )

    start_co.assert_awaited_once()
    assert start_co.await_args.kwargs["status"] == "active"
    render_reply.assert_awaited_once()
    assert render_reply.await_args.args[0] == "music.switch_track"
    music_status.assert_not_awaited()
    envelopes = [call.args[0] for call in fake_ws.send_json.call_args_list]
    assert any(item.get("type") == "reply" and item["data"]["music_co_listening"] for item in envelopes)


@pytest.mark.asyncio
async def test_handle_message_music_card_busy_rejects_without_starting(fake_ws):
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-music-card-2"

    async def _fake_assistant(*args, **kwargs):
        return "assistant-music-2"

    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    })
    plan = _aggregation_plan("immediate", text="一起听")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "_persist_assistant_message", side_effect=_fake_assistant),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "写报告", "type": "work"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "写报告", "status": "very_busy"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.services.music.start_co_listening", new_callable=AsyncMock) as start_co,
        patch(
            "app.services.music_chat.render_music_reply",
            new_callable=AsyncMock,
            return_value="我在写报告，忙完听。",
        ) as render_reply,
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new_callable=AsyncMock,
        ) as persist_status,
        patch("app.services.chat.post_process._bg_memory_pipeline", new=lambda *_args, **_kwargs: None),
        patch("app.services.runtime.tasks.fire_background", new=lambda *_args, **_kwargs: None),
    ):
        await ws_mod._handle_message(
            fake_ws,
            "conv-1",
            "user-1",
            agent,
            "一起听",
            workspace_id="ws-1",
            component_card=card,
            user_name="Song",
        )

    start_co.assert_awaited_once()
    assert start_co.await_args.kwargs["status"] == "pending_agent"
    assert start_co.await_args.kwargs["initiated_by"] == "user_pending"
    assert render_reply.await_args.args[0] == "music.busy_reject"
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["actor"] == "user"
    envelopes = [call.args[0] for call in fake_ws.send_json.call_args_list]
    assert any(item.get("type") == "reply" and not item["data"]["music_co_listening"] for item in envelopes)


@pytest.mark.asyncio
async def test_handle_message_music_card_while_agent_waiting_busy_exits_instead_of_pending(fake_ws):
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-music-waiting-busy"

    card = ws_mod._sanitize_component_card({
        "type": "music_track",
        "title": "Quiet Realm",
        "subtitle": "Jamendo Artist",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "track-1", "title": "Quiet Realm"},
        },
    })
    plan = _aggregation_plan("immediate", text="继续听")
    waiting = MusicCoListeningResponse(
        status="agent_waiting_user",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=False,
        initiated_by="user_joined",
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "写报告", "type": "work"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "写报告", "status": "busy"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=waiting,
        ),
        patch("app.services.music.start_co_listening", new_callable=AsyncMock) as start_co,
        patch(
            "app.services.music_status.reconcile_co_listening_for_status",
            new_callable=AsyncMock,
        ) as reconcile,
        patch("app.services.chat.post_process._bg_memory_pipeline", new=lambda *_args, **_kwargs: None),
        patch("app.services.runtime.tasks.fire_background", new=lambda *_args, **_kwargs: None),
    ):
        await ws_mod._handle_message(
            fake_ws,
            "conv-1",
            "user-1",
            agent,
            "继续听",
            workspace_id="ws-1",
            component_card=card,
            user_name="Song",
        )

    start_co.assert_not_awaited()
    reconcile.assert_awaited_once()
    assert reconcile.await_args.kwargs["status_code"] == "busy"
    assert reconcile.await_args.kwargs["activity"] == "写报告"
    envelopes = [call.args[0] for call in fake_ws.send_json.call_args_list]
    assert envelopes[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_turn_aggregation_bypass_for_record_requests():
    """提醒/记忆类控制消息不等待 turn quiet window。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert await turn_mod.should_bypass_user_turn_aggregation("conv-1", "明天提醒我交报告")


@pytest.mark.asyncio
async def test_turn_aggregation_keeps_current_state_queries_in_turn_window():
    """询问 AI 当前状态仍是聊天回合的一部分, 需要等待短 turn window 合并追问。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert not await turn_mod.should_bypass_user_turn_aggregation("conv-1", "你现在在干嘛")


@pytest.mark.asyncio
async def test_turn_aggregation_keeps_schedule_queries_in_turn_window():
    """计划查询是只读聊天意图, 连续追问应先聚合, 避免重复播报日程。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert not await turn_mod.should_bypass_user_turn_aggregation("conv-1", "你明天忙吗")


@pytest.mark.asyncio
async def test_turn_aggregation_bypass_for_pending_confirmations():
    """删除/矛盾等跨消息确认状态下, 下一句需要立即进入 preflight。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value={"id": "pending"}),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert await turn_mod.should_bypass_user_turn_aggregation("conv-1", "对，更新吧")


@pytest.mark.asyncio
async def test_send_ack_happy_path(fake_ws):
    """_send_ack 直接调用 → ws 发出 type=ack envelope, 含 client_id +
    message_id + ISO received_at."""
    await ws_mod._send_ack(fake_ws, message_id="db-123", client_id="client-uuid-456")

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    data = acks[0]["data"]
    assert data["message_id"] == "db-123"
    assert data["client_id"] == "client-uuid-456"
    assert "received_at" in data
    # ISO format check
    assert "T" in data["received_at"]


@pytest.mark.asyncio
async def test_send_ack_without_client_id(fake_ws):
    """前端没传 client_id (旧客户端兼容) → ack 仍发出, client_id=None.
    前端可 fallback 按时间顺序匹配气泡."""
    await ws_mod._send_ack(fake_ws, message_id="db-789", client_id=None)

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    assert acks[0]["data"]["message_id"] == "db-789"
    assert acks[0]["data"]["client_id"] is None


@pytest.mark.asyncio
async def test_send_ack_failure_does_not_raise(fake_ws):
    """ws 已断 / send 抛 → ack 失败不影响主流程 (主路径 reply 仍会发).
    日志 WARN 但不冒泡异常."""
    fake_ws.send_json = AsyncMock(side_effect=ConnectionError("WS closed"))

    # 不抛 — _send_ack 内部 try/except
    await ws_mod._send_ack(fake_ws, message_id="db-1", client_id="c-1")


@pytest.mark.asyncio
async def test_handle_message_fragment_sends_ack_with_client_id(fake_ws):
    """碎片分支 (短消息聚合): persist → 立刻 send_ack (含 client_id) → 然后
    push_pending → 发 pending:aggregating. 顺序很重要 (ack 必须在 pending 前)."""
    agent = SimpleNamespace(id="a1", name="A")
    captured_ack_msg_id = None

    async def _fake_persist(*args, **kwargs):
        return "db-msg-fragment-1"

    plan = _aggregation_plan(
        "fragment_window",
        text="嗯",
        metadata={"fragment": True},
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "嗯",
            client_id="client-frag-uuid",
        )

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1, f"碎片分支必须发 1 个 ack; got {acks}"
    assert acks[0]["data"]["client_id"] == "client-frag-uuid"
    assert acks[0]["data"]["message_id"] == "db-msg-fragment-1"

    # ack 必须在 pending:aggregating 之前发 — 用户体感"已读 → 处理中"
    all_payloads = [
        call.args[0] for call in fake_ws.send_json.call_args_list
        if isinstance(call.args[0], dict)
    ]
    types = [p.get("type") for p in all_payloads]
    ack_idx = types.index("ack")
    pending_idx = types.index("pending")
    assert ack_idx < pending_idx, (
        f"ack 必须在 pending 前发, got types order: {types}"
    )


@pytest.mark.asyncio
async def test_handle_message_fragment_joins_open_turn_window(fake_ws):
    """普通消息后紧接 1-2 字碎片时, 碎片应追加到 turn window, 不另起 5s fragment window。"""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-fragment-join-1"

    plan = _aggregation_plan("turn_window", text="吗")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True) as enqueue_plan,
        patch.object(ws_mod, "_queue_reply", new_callable=AsyncMock) as queue_reply,
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={"delay_seconds": 0}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "吗",
            client_id="client-frag-join-uuid",
        )

    enqueue_plan.assert_awaited_once()
    assert enqueue_plan.await_args.args[0].route == "turn_window"
    queue_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_non_fragment_sends_ack_with_client_id(fake_ws):
    """非碎片分支 (直接入延迟队列): persist → ack → _queue_reply.
    ack 仍带 client_id."""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-full-1"

    plan = _aggregation_plan("immediate", text="完整一句话哈哈")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "_queue_reply", new_callable=AsyncMock),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "完整一句话哈哈",
            client_id="client-full-uuid",
        )

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1, f"非碎片分支必须发 1 个 ack; got {acks}"
    assert acks[0]["data"]["client_id"] == "client-full-uuid"
    assert acks[0]["data"]["message_id"] == "db-msg-full-1"


@pytest.mark.asyncio
async def test_handle_message_persists_component_card_metadata(fake_ws):
    """组件卡片必须跟 client_id 一起进入用户消息 metadata, 供历史记录回放渲染。"""
    agent = SimpleNamespace(id="a1", name="A")
    captured_metadata = None

    async def _fake_persist(*args, **kwargs):
        nonlocal captured_metadata
        captured_metadata = kwargs.get("metadata")
        return "db-msg-card-1"

    card = {
        "version": 1,
        "type": "time_capsule",
        "title": "Hi there",
        "subtitle": "2026年5月28日开启",
        "body": "Hi there🙌",
        "footer": "时间胶囊 · 已开启",
        "accent": "#7C3CFF",
        "payload": {"capsule_id": "cap-1"},
    }
    plan = _aggregation_plan("immediate", text="胶囊消息", metadata={"queued": True})

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(
            ws_mod, "plan_user_message_aggregation",
            new_callable=AsyncMock, return_value=plan,
        ),
        patch.object(ws_mod, "_queue_reply", new_callable=AsyncMock),
        patch.object(
            ws_mod, "get_cached_schedule",
            new_callable=AsyncMock,
            return_value=[{"activity": "自由时间", "type": "leisure"}],
        ),
        patch.object(
            ws_mod, "get_current_status",
            return_value={"activity": "自由时间", "type": "leisure", "status": "idle"},
        ),
        patch.object(
            ws_mod, "build_reply_timing_context",
            new_callable=AsyncMock, return_value={},
        ),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "胶囊消息",
            client_id="client-card-uuid",
            component_card=card,
        )

    assert captured_metadata == {
        "queued": True,
        "client_id": "client-card-uuid",
        "component_card": card,
    }


@pytest.mark.asyncio
async def test_handle_message_normal_turn_aggregates_before_queue(fake_ws):
    """普通非碎片消息先进入 turn quiet window, 不立刻触发 reply。"""
    agent = SimpleNamespace(id="a1", name="A")
    queue_reply = AsyncMock()

    async def _fake_persist(*args, **kwargs):
        return "db-msg-turn-1"

    plan = _aggregation_plan("turn_window", text="我最近在看一部美剧")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True) as enqueue_plan,
        patch.object(ws_mod, "_queue_reply", queue_reply),
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={"delay_seconds": 0}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "我最近在看一部美剧",
            client_id="client-turn-uuid",
        )

    enqueue_plan.assert_awaited_once()
    assert enqueue_plan.await_args.kwargs["message_id"] == "db-msg-turn-1"
    queue_reply.assert_not_awaited()
    payloads = [
        call.args[0] for call in fake_ws.send_json.call_args_list
        if isinstance(call.args[0], dict)
    ]
    assert any(
        p.get("type") == "pending" and p.get("data", {}).get("status") == "aggregating"
        for p in payloads
    )


@pytest.mark.asyncio
async def test_handle_message_default_client_id_none(fake_ws):
    """旧前端不传 client_id (向后兼容): _handle_message 默认 None,
    ack 仍发出 (仅缺 client_id 字段)."""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-1"

    plan = _aggregation_plan(
        "fragment_window",
        text="嗯",
        metadata={"fragment": True},
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        # 不传 client_id (default=None)
        await ws_mod._handle_message(fake_ws, "conv-1", "user-1", agent, "嗯")

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    assert acks[0]["data"]["client_id"] is None
    assert acks[0]["data"]["message_id"] == "db-1"
