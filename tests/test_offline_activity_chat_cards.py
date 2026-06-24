from unittest.mock import AsyncMock

from app.services.offline import activity_service
from app.services.offline.chat_emit import build_activity_component_card


def test_activity_component_card_uses_status_and_image_payload():
    card = build_activity_component_card(
        {
            "id": "activity-1",
            "status": "pending",
            "title": "镇江博物馆常设展",
            "summary": "安静看展",
            "location_name": "镇江博物馆",
            "image_urls": ["/offline/media/user_activity.jpg"],
        },
        status_label="待回复",
    )

    assert card["type"] == "offline_activity"
    assert card["payload"]["activity_id"] == "activity-1"
    assert card["payload"]["image_url"] == "/offline/media/user_activity.jpg"
    assert card["subtitle"] == "待回复 · 镇江博物馆"


async def test_create_recommendation_emits_invite_message_and_activity_card(monkeypatch):
    generated_card = {
        "title": "镇江博物馆常设展",
        "summary": "安静看展",
        "description": "适合独自慢慢逛的常设展。",
        "location_name": "镇江博物馆",
        "image_urls": ["/offline/media/activity.jpg"],
    }
    created_activity = {
        **generated_card,
        "id": "activity-1",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "status": "pending",
    }
    emit = AsyncMock()
    emit_card = AsyncMock()

    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
                "user_location_latitude": 32.19,
                "user_location_longitude": 119.45,
                "user_location_city": "Zhenjiang",
                "user_location_region": "Jiangsu",
            }
        ),
    )
    monkeypatch.setattr(
        activity_service,
        "generate_activity_card",
        AsyncMock(return_value=generated_card),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "create_activity",
        AsyncMock(return_value=created_activity),
    )
    monkeypatch.setattr(
        activity_service,
        "generate_activity_invite_message",
        AsyncMock(return_value="我找到一个镇江博物馆的小计划，像给今天留个安静出口。"),
    )
    monkeypatch.setattr(activity_service, "emit_assistant", emit)
    monkeypatch.setattr(activity_service, "emit_activity_card", emit_card)
    monkeypatch.setattr(activity_service.repo, "update_next_activity_due", AsyncMock())

    result = await activity_service.create_recommendation_for_user(
        user_id="user-1",
        workspace_id="workspace-1",
        source="manual",
    )

    assert result == created_activity
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["message"].startswith("我找到一个镇江博物馆")
    assert emit.await_args.kwargs["trigger_type"] == "offline_activity_recommendation"
    emit_card.assert_awaited_once()
    assert emit_card.await_args.kwargs["activity"] == created_activity
    assert emit_card.await_args.kwargs["trigger_type"] == (
        "offline_activity_recommendation_card"
    )
    assert emit_card.await_args.kwargs["status_label"] == "待回复"


async def test_accept_activity_allows_reaccepting_ignored_activity(monkeypatch):
    activity = {
        "id": "activity-1",
        "status": "ignored",
        "title": "镇江博物馆常设展",
        "summary": "",
        "description": "",
        "workspace_id": "workspace-1",
        "created_at": "2026-06-21T10:00:00Z",
        "updated_at": "2026-06-21T10:00:00Z",
    }
    updated = {**activity, "status": "accepted"}
    feedback = AsyncMock()
    emit = AsyncMock()
    insert_card = AsyncMock()

    monkeypatch.setattr(
        activity_service.repo,
        "get_activity",
        AsyncMock(return_value=activity),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "update_activity_status",
        AsyncMock(return_value=updated),
    )
    monkeypatch.setattr(activity_service.repo, "create_activity_feedback", feedback)
    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
            }
        ),
    )
    monkeypatch.setattr(activity_service.repo, "update_next_activity_due", AsyncMock())
    monkeypatch.setattr(activity_service, "emit_assistant", emit)
    monkeypatch.setattr(activity_service, "insert_user_activity_card", insert_card)
    monkeypatch.setattr(activity_service, "remember_user_event", lambda **_: None)

    result = await activity_service.accept_activity("user-1", "activity-1")

    assert result.status == "accepted"
    assert "重新接受" in feedback.await_args.kwargs["text"]
    insert_card.assert_awaited_once()
    assert insert_card.await_args.kwargs["activity"] == updated
    assert insert_card.await_args.kwargs["trigger_type"] == (
        "offline_activity_reaccepted_card"
    )
    assert insert_card.await_args.kwargs["status_label"] == "已接受"
    assert emit.await_args.kwargs["trigger_type"] == "offline_activity_reaccepted"


async def test_ignore_activity_emits_ignored_activity_card(monkeypatch):
    activity = {
        "id": "activity-1",
        "user_id": "user-1",
        "status": "pending",
        "title": "镇江博物馆常设展",
        "summary": "",
        "description": "",
        "workspace_id": "workspace-1",
        "created_at": "2026-06-21T10:00:00Z",
        "updated_at": "2026-06-21T10:00:00Z",
    }
    updated = {**activity, "status": "ignored"}
    insert_card = AsyncMock()

    monkeypatch.setattr(
        activity_service.repo,
        "get_activity",
        AsyncMock(return_value=activity),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "update_activity_status",
        AsyncMock(return_value=updated),
    )
    monkeypatch.setattr(activity_service.repo, "create_activity_feedback", AsyncMock())
    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
            }
        ),
    )
    monkeypatch.setattr(activity_service.repo, "update_next_activity_due", AsyncMock())
    monkeypatch.setattr(activity_service, "emit_assistant", AsyncMock())
    monkeypatch.setattr(activity_service, "insert_user_activity_card", insert_card)
    monkeypatch.setattr(activity_service, "remember_user_event", lambda **_: None)

    result = await activity_service.ignore_activity("user-1", "activity-1")

    assert result.status == "ignored"
    insert_card.assert_awaited_once()
    assert insert_card.await_args.kwargs["activity"] == updated
    assert insert_card.await_args.kwargs["trigger_type"] == (
        "offline_activity_ignored_card"
    )
    assert insert_card.await_args.kwargs["status_label"] == "暂不考虑"


async def test_complete_activity_emits_completed_card_with_share_metadata(monkeypatch):
    activity = {
        "id": "activity-1",
        "user_id": "user-1",
        "status": "accepted",
        "title": "镇江博物馆常设展",
        "summary": "",
        "description": "",
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "created_at": "2026-06-21T10:00:00Z",
        "updated_at": "2026-06-21T10:00:00Z",
    }
    updated = {**activity, "status": "completed"}
    insert_card = AsyncMock()
    insert_share = AsyncMock()

    monkeypatch.setattr(
        activity_service.repo,
        "get_activity",
        AsyncMock(return_value=activity),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "update_activity_status",
        AsyncMock(return_value=updated),
    )
    monkeypatch.setattr(activity_service.repo, "create_activity_feedback", AsyncMock())
    monkeypatch.setattr(
        activity_service.repo,
        "get_activity_completion_feedback",
        AsyncMock(return_value={"text": "不错", "photo_attachments": []}),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
            }
        ),
    )
    monkeypatch.setattr(activity_service, "insert_user_component_message", insert_share)
    monkeypatch.setattr(activity_service, "emit_assistant", AsyncMock())
    monkeypatch.setattr(activity_service, "insert_user_activity_card", insert_card)
    monkeypatch.setattr(activity_service, "remember_user_event", lambda **_: None)

    result = await activity_service.complete_activity(
        "user-1",
        "activity-1",
        text="不错",
        photo_attachment_ids=[],
    )

    assert result.status == "completed"
    assert result.completion_feedback is not None
    insert_share.assert_awaited_once()
    assert insert_share.await_args.kwargs["metadata"]["trigger_type"] == (
        "offline_activity_completion_share"
    )
    insert_card.assert_awaited_once()
    assert insert_card.await_args.kwargs["activity"] == updated
    assert insert_card.await_args.kwargs["trigger_type"] == (
        "offline_activity_completed_card"
    )
    assert insert_card.await_args.kwargs["status_label"] == "已完成"
