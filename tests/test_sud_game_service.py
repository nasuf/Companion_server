from __future__ import annotations

import pytest

from app.models.game import SudPlayerInfo, SudSessionResponse
from app.services.games import sud


class _FakeDb:
    def __init__(self, rows_by_query: list[list[dict]]):
        self.rows_by_query = rows_by_query
        self.calls: list[tuple[str, tuple]] = []

    async def query_raw(self, query: str, *args):
        self.calls.append((query, args))
        return self.rows_by_query.pop(0)


@pytest.mark.asyncio
async def test_resolve_owned_context_rejects_foreign_conversation(monkeypatch):
    fake_db = _FakeDb(
        [
            [
                {
                    "id": "c1",
                    "user_id": "other-user",
                    "agent_id": "agent-1",
                    "workspace_id": "w1",
                }
            ]
        ]
    )
    monkeypatch.setattr(sud, "db", fake_db)

    with pytest.raises(ValueError, match="context_not_found"):
        await sud._resolve_owned_context(
            user_id="user-1",
            agent_id="agent-1",
            workspace_id=None,
            conversation_id="c1",
        )


@pytest.mark.asyncio
async def test_resolve_owned_context_uses_conversation_workspace(monkeypatch):
    fake_db = _FakeDb(
        [
            [
                {
                    "id": "c1",
                    "user_id": "user-1",
                    "agent_id": "agent-1",
                    "workspace_id": "w1",
                }
            ],
            [{"id": "w1", "user_id": "user-1", "agent_id": "agent-1"}],
        ]
    )
    monkeypatch.setattr(sud, "db", fake_db)

    workspace_id, conversation_id = await sud._resolve_owned_context(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id=None,
        conversation_id="c1",
    )

    assert workspace_id == "w1"
    assert conversation_id == "c1"


def test_extract_result_maps_user_outcome():
    session = SudSessionResponse(
        id="s1",
        status="playing",
        sdk_enabled=True,
        user_id="u1",
        agent_id="a1",
        app_id="app",
        app_key="key",
        bundle_id="bundle",
        is_test_env=True,
        mg_id="mg1",
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )

    result = sud._extract_result(
        session,
        {
            "gameRoundId": "round-1",
            "battle_duration": 80,
            "results": [
                {"uid": "u1", "isWin": 2},
                {"uid": "agent:a1", "isWin": 1},
            ],
        },
    )

    assert result["user_outcome"] == "win"
    assert result["duration_seconds"] == 80
    assert result["user"]["uid"] == "u1"
    assert result["ai"]["uid"] == "agent:a1"


def test_sud_noise_events_do_not_generate_chat_reply():
    session = SudSessionResponse(
        id="s1",
        status="playing",
        sdk_enabled=True,
        user_id="u1",
        agent_id="a1",
        app_id="app",
        app_key="key",
        bundle_id="bundle",
        is_test_env=True,
        mg_id="mg1",
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )

    assert (
        sud._reply_for_event(
            session,
            "sud_player_state",
            "mg_common_self_x",
            {"isReady": True},
        )
        is None
    )


def test_only_game_end_replies_are_persisted_to_chat():
    assert not sud._should_persist_reply_to_chat("session_created", None)
    assert not sud._should_persist_reply_to_chat("sdk_ready", None)
    assert not sud._should_persist_reply_to_chat("sud_player_state", "mg_common_self_ready")
    assert sud._should_persist_reply_to_chat("game_settle", "mg_common_game_settle")


@pytest.mark.asyncio
async def test_sud_report_settlement_reply_is_persisted_to_chat(monkeypatch):
    session = SudSessionResponse(
        id="s1",
        status="settled",
        sdk_enabled=True,
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
        app_id="app",
        app_key="key",
        bundle_id="bundle",
        is_test_env=True,
        mg_id="mg1",
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )
    writes = []

    async def fake_write_game_message(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(sud, "_write_game_message", fake_write_game_message)

    await sud._persist_reply_to_chat_if_needed(
        session,
        "sud_game_settle",
        None,
        "这局结束啦。",
    )

    assert writes == [
        {
            "conversation_id": "c1",
            "role": "assistant",
            "content": "这局结束啦。",
            "metadata": {
                "kind": "game",
                "session_id": "s1",
                "event_type": "sud_game_settle",
                "state": None,
            },
        }
    ]


@pytest.mark.asyncio
async def test_refresh_ss_token_accepts_existing_ss_token(monkeypatch):
    async def fake_user_info_from_token(token: str) -> SudPlayerInfo:
        return SudPlayerInfo(uid=sud.decode_token(token)["uid"], nick_name="玩家")

    monkeypatch.setattr(
        sud,
        "user_info_from_token",
        fake_user_info_from_token,
    )

    old_token, _ = sud.make_ss_token(
        uid="user-1",
        session_id="session-1",
        room_id="room-1",
    )

    new_token, expires_at, user_info, old_payload = await sud.refresh_ss_token(old_token)

    new_payload = sud.decode_token(new_token)
    assert expires_at.timestamp() > new_payload["exp"] - 1
    assert old_payload["uid"] == "user-1"
    assert new_payload["uid"] == "user-1"
    assert new_payload["session_id"] == "session-1"
    assert new_payload["room_id"] == "room-1"
    assert user_info.uid == "user-1"
