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
    assert not sud._should_persist_reply_to_chat("game_settle", "mg_common_game_settle")
    assert sud._should_persist_reply_to_chat("game_exited", None)
    assert sud._should_persist_reply_to_chat("sud_game_settle", None)


def test_monster_crush_score_process_tracks_lead_changes():
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
        mg_id=sud.MONSTER_CRUSH_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )

    result = sud._merge_process_result(
        session,
        None,
        "game_player_scores",
        "mg_common_game_player_scores",
        {"scores": [{"uid": "u1", "score": 30}, {"uid": "agent:a1", "score": 40}]},
    )
    result = sud._merge_process_result(
        session,
        result,
        "game_player_scores",
        "mg_common_game_player_scores",
        {"scores": [{"uid": "u1", "score": 70}, {"uid": "agent:a1", "score": 55}]},
    )

    process = result["process"]
    assert process["score_updates"] == 2
    assert process["user_score"] == 70
    assert process["ai_score"] == 55
    assert process["lead_changes"] == 1
    assert process["max_ai_lead"] == 10
    assert process["max_user_lead"] == 15


def test_monster_crush_settlement_reply_uses_process_and_extras():
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
        mg_id=sud.MONSTER_CRUSH_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
        result={
            "process": {
                "user_score": 70,
                "ai_score": 55,
                "lead_changes": 1,
            }
        },
    )

    reply = sud._reply_for_event(
        session,
        "sud_game_settle",
        None,
        {
            "battle_duration": 88,
            "results": [
                {
                    "uid": "u1",
                    "isWin": 2,
                    "extras": '{"numGood":3,"numPerfect":4,"numExcellent":5,"numCrazy":6}',
                },
                {"uid": "agent:a1", "isWin": 1},
            ],
        },
    )

    assert reply is not None
    assert "可以啊，这局你拿下了" in reply
    assert "中间还来回翻过一次节奏" in reply
    assert "你刚才有一波连得挺凶" in reply
    assert "最后分数是" not in reply


def test_settlement_reply_uses_scores_and_extras_without_process_stream():
    result = {
        "user": {"score": 29410},
        "ai": {"score": 41710},
        "user_extras": {
            "numGood": 9,
            "numPerfect": 3,
            "numExcellent": 0,
            "numCrazy": 0,
        },
    }

    fragment = sud._process_reply_fragment(result)

    assert "你那 3 个 Perfect 挺漂亮" in fragment
    assert "分差也没被拉到离谱" in fragment
    assert "29410" not in fragment


def test_abort_reply_is_suppressed_after_terminal_state():
    session = SudSessionResponse(
        id="s1",
        status="settled",
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

    assert sud._reply_for_event(session, "game_exited", None, {"reason": "page_disposed"}) is None


def test_notify_event_type_mapping():
    assert sud._event_type_for_notify("sud.mg.merchant.game.process") == "sud_game_process"
    assert sud._event_type_for_notify("custom.settle") == "sud_game_settle"


def test_empty_default_mg_id_does_not_mark_empty_session_as_gomoku(monkeypatch):
    monkeypatch.setattr(sud.settings, "sud_default_mg_id", "")
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
        mg_id="",
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )

    assert sud._is_gomoku_session(session) is False


def test_gomoku_process_normalizes_moves_and_deduplicates():
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
        mg_id=sud.GOMOKU_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )

    result = sud._merge_process_result(
        session,
        None,
        "move",
        None,
        {"uid": "u1", "move_index": 32, "piece": "X"},
    )
    result = sud._merge_process_result(
        session,
        result,
        "move",
        None,
        {"uid": "u1", "move_index": 32, "piece": "X"},
    )

    gomoku = result["process"]["gomoku"]
    assert gomoku["move_count"] == 1
    assert gomoku["user_moves"] == 1
    assert gomoku["last_move"]["x"] == 2
    assert gomoku["last_move"]["y"] == 2
    assert gomoku["last_move"]["actor"] == "user"


def test_gomoku_settlement_records_winning_line_and_reply_observation():
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
        mg_id=sud.GOMOKU_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
        result={
            "process": {
                "gomoku": {
                    "move_count": 16,
                    "last_move": {"x": 7, "y": 8, "actor": "user"},
                }
            }
        },
    )

    payload = {
        "duration": 120,
        "winningLine": [
            {"x": 3, "y": 6},
            {"x": 4, "y": 6},
            {"x": 5, "y": 6},
            {"x": 6, "y": 6},
            {"x": 7, "y": 6},
        ],
        "results": [
            {"uid": "u1", "isWin": 2},
            {"uid": "agent:a1", "isWin": 1},
        ],
    }
    result = sud._merge_process_result(
        session,
        sud._extract_result(session, payload),
        "sud_game_settle",
        None,
        payload,
        previous_result=session.result,
    )
    reply = sud._reply_for_event(session, "sud_game_settle", None, payload)

    assert result["gomoku"]["win_direction"] == "horizontal"
    assert result["process"]["gomoku"]["winning_line"][0] == {"x": 3, "y": 6}
    assert reply is not None
    assert "这盘不是几手就崩的" in reply


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
async def test_game_status_message_is_deduplicated(monkeypatch):
    previous = SudSessionResponse(
        id="s1",
        status="created",
        sdk_enabled=True,
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
        app_id="app",
        app_key="key",
        bundle_id="bundle",
        is_test_env=True,
        mg_id=sud.MONSTER_CRUSH_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="AI", is_ai=1),
    )
    updated = previous.model_copy(update={"status": "playing"})
    writes = []

    async def fake_status_exists(conversation_id, session_id, status):
        assert conversation_id == "c1"
        assert session_id == "s1"
        assert status == "started"
        return True

    async def fake_write_game_message(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(sud, "_game_status_message_exists", fake_status_exists)
    monkeypatch.setattr(sud, "_write_game_message", fake_write_game_message)

    await sud._persist_game_status_to_chat_if_needed(
        previous,
        updated,
        "game_started",
        None,
        {"game_title": "怪物消消乐"},
    )

    assert writes == []


@pytest.mark.asyncio
async def test_game_status_message_wraps_game_title_in_brackets(monkeypatch):
    previous = SudSessionResponse(
        id="s1",
        status="created",
        sdk_enabled=True,
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
        app_id="app",
        app_key="key",
        bundle_id="bundle",
        is_test_env=True,
        mg_id=sud.MONSTER_CRUSH_MG_ID,
        room_id="room1",
        code="code",
        code_expires_at="2026-05-31T10:00:00+00:00",
        play_mode="versus",
        difficulty="newbie",
        ai_level=1,
        user_player=SudPlayerInfo(uid="u1", nick_name="玩家"),
        ai_player=SudPlayerInfo(uid="agent:a1", nick_name="小芜", is_ai=1),
    )
    updated = previous.model_copy(update={"status": "playing"})
    writes = []

    async def fake_status_exists(*_args):
        return False

    async def fake_write_game_message(**kwargs):
        writes.append(kwargs)
        return "m1"

    monkeypatch.setattr(sud, "_game_status_message_exists", fake_status_exists)
    monkeypatch.setattr(sud, "_write_game_message", fake_write_game_message)

    await sud._persist_game_status_to_chat_if_needed(
        previous,
        updated,
        "game_started",
        None,
        {"game_title": "怪物消消乐"},
    )

    assert writes[0]["content"] == "小芜 和你已进入游戏《怪物消消乐》"
    assert writes[0]["metadata"]["game_title"] == "怪物消消乐"


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
