from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.models.game import (
    NativeCreateSessionRequest,
    NativeSessionResponse,
    SudPlayerInfo,
    SudSessionResponse,
)
from app.services.games import native, sud
from app.services.runtime.ws_manager import manager


def test_native_create_session_rejects_difficulty_override():
    with pytest.raises(ValidationError):
        NativeCreateSessionRequest.model_validate(
            {
                "agent_id": "agent-1",
                "game_key": "gomoku",
                "difficulty": "hard",
            }
        )


def test_native_session_contract_contains_no_sud_credentials():
    session = NativeSessionResponse(
        id="session-1",
        game_key="gomoku",
        status="playing",
        user_id="user-1",
        agent_id="agent-1",
        room_id="gomoku-room",
        play_mode="versus",
        ai_level=2,
        user_player=SudPlayerInfo(uid="user-1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:agent-1", nick_name="小芜", is_ai=1),
    ).model_dump()

    assert session["difficulty"] == "normal"
    assert "app_id" not in session
    assert "app_key" not in session
    assert "bundle_id" not in session
    assert "mg_id" not in session
    assert "code" not in session


def test_native_session_conversion_normalizes_legacy_sud_fields():
    legacy = SudSessionResponse(
        id="legacy-native",
        provider="native",
        game_key="go",
        status="playing",
        sdk_enabled=False,
        user_id="user-1",
        agent_id="agent-1",
        app_id="",
        app_key="",
        bundle_id="",
        is_test_env=False,
        mg_id="",
        room_id="go-legacy",
        code="",
        code_expires_at="",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="user-1", nick_name="You"),
        ai_player=SudPlayerInfo(
            uid="agent-1",
            nick_name="Companion",
            is_ai=1,
        ),
    )

    converted = native._as_native_session(legacy)

    assert converted.provider == "native"
    assert converted.game_key == "go"
    assert converted.difficulty == "normal"


@pytest.mark.asyncio
async def test_shared_status_helper_accepts_native_session_without_sud_fields(
    monkeypatch,
):
    previous = NativeSessionResponse(
        id="native-status-session",
        game_key="go",
        status="created",
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
        room_id="go-room",
        play_mode="versus",
        ai_level=1,
        user_player=SudPlayerInfo(uid="user-1", nick_name="You"),
        ai_player=SudPlayerInfo(
            uid="agent-1",
            nick_name="Companion",
            is_ai=1,
        ),
    )
    updated = previous.model_copy(update={"status": "playing"})
    write_message = AsyncMock(return_value=("message-1", True))
    send_event = AsyncMock()
    monkeypatch.setattr(sud, "_write_game_message", write_message)
    monkeypatch.setattr(manager, "send_event", send_event)

    await sud._persist_game_status_to_chat_if_needed(
        previous,
        updated,
        "game_started",
        "playing",
        {"game_title": "围棋"},
    )

    metadata = write_message.await_args.kwargs["metadata"]
    assert metadata["game_title"] == "围棋"
    assert "mg_id" not in metadata
    ws_payload = send_event.await_args.args[2]
    assert ws_payload["game_title"] == "围棋"
    assert "mg_id" not in ws_payload
