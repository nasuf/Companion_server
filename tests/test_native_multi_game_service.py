from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from app.models.game import SudPlayerInfo, SudSessionResponse
from app.services.games import native


def _session(game_key: str) -> SudSessionResponse:
    definition = native._definition(game_key)
    return SudSessionResponse(
        id=f"native-{game_key}",
        provider="native",
        game_key=game_key,
        status="playing",
        sdk_enabled=False,
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        conversation_id="conversation-1",
        app_id="",
        app_key="",
        bundle_id="",
        is_test_env=False,
        mg_id="",
        room_id=f"{game_key}-room",
        code="",
        code_expires_at="",
        play_mode=definition.play_mode,
        difficulty="normal",
        ai_level=2,
        user_player=SudPlayerInfo(uid="user-1", nick_name="玩家"),
        ai_player=SudPlayerInfo(
            uid="agent:agent-1",
            nick_name="小芜",
            is_ai=1,
            ai_level=2,
        ),
    )


def _action(
    actor: str,
    number: int,
    *,
    before_hash: str = "",
    after_hash: str,
) -> dict:
    return {
        "action_id": f"action-{number}",
        "actor": actor,
        "from": {"row": number, "col": number},
        "to": {"row": number + 1, "col": number + 1},
        "state_before": {"state_hash": before_hash} if before_hash else {},
        "state_after": {"state_hash": after_hash, "turn": actor},
        "analysis": {"depth": number + 2, "score": number * 10},
    }


def test_native_registry_contains_every_supported_game():
    assert set(native._GAME_DEFINITIONS) == {
        "go",
        "reversi",
        "gomoku",
        "xiangqi",
        "chess",
        "chinese_checkers",
        "match3",
        "minesweeper",
        "number_merge",
        "tetris_duel",
    }
    assert native._definition("match3").play_mode == "cooperate"
    assert native._definition("minesweeper").play_mode == "cooperate"
    assert native._definition("number_merge").play_mode == "cooperate"


@pytest.mark.asyncio
async def test_create_session_writes_session_and_first_event_atomically(monkeypatch):
    class AgentRepo:
        async def find_unique(self, *, where):
            return SimpleNamespace(
                id=where["id"],
                userId="user-1",
                name="小芜",
                avatarUrl=None,
                gender=None,
            )

    class CaptureDb:
        def __init__(self):
            self.aiagent = AgentRepo()
            self.calls = []

        async def query_raw(self, query, *args):
            self.calls.append((query, args))
            return [{}]

    database = CaptureDb()
    expected = _session("gomoku").model_copy(
        update={"id": "created-session", "status": "created"}
    )
    monkeypatch.setattr(native, "db", database)
    monkeypatch.setattr(
        native.sud,
        "build_user_player",
        AsyncMock(return_value=expected.user_player),
    )
    context = AsyncMock(return_value=("workspace-1", "conversation-1"))
    monkeypatch.setattr(native.sud, "_resolve_owned_context", context)
    monkeypatch.setattr(native.sud, "_row_to_session", lambda _row: expected)

    created = await native.create_session(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        conversation_id="conversation-1",
        game_key="gomoku",
    )

    assert created.id == "created-session"
    assert len(database.calls) == 1
    query, args = database.calls[0]
    assert "WITH created_session AS" in query
    assert "INSERT INTO game_events" in query
    assert "CROSS JOIN created_event" in query
    assert args[1] == "gomoku"
    context.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_session_binds_only_the_declared_owned_query_parameters():
    class CaptureDb:
        args = None

        async def query_raw(self, query, *args):
            self.args = args
            return []

    database = CaptureDb()

    with pytest.raises(ValueError, match="session_not_found"):
        await native.get_session(
            "session-1",
            user_id="user-1",
            database=database,
        )

    assert database.args == ("session-1", "user-1")


@pytest.mark.asyncio
async def test_delete_session_is_scoped_to_native_session_owner(monkeypatch):
    class CaptureDb:
        call = None

        async def execute_raw(self, query, *args):
            self.call = (query, args)
            return 1

    database = CaptureDb()
    monkeypatch.setattr(native, "db", database)
    get_session = AsyncMock(return_value=_session("go"))
    monkeypatch.setattr(native, "get_session", get_session)

    await native.delete_session("session-1", user_id="user-1")

    get_session.assert_awaited_once_with("session-1", user_id="user-1")
    assert "provider = 'native'" in database.call[0]
    assert "user_id = $2" in database.call[0]
    assert database.call[1] == ("session-1", "user-1")


@pytest.mark.asyncio
async def test_delete_session_rejects_missing_or_unowned_session(monkeypatch):
    get_session = AsyncMock(side_effect=ValueError("session_not_found"))
    monkeypatch.setattr(native, "get_session", get_session)

    with pytest.raises(ValueError, match="session_not_found"):
        await native.delete_session("session-1", user_id="other-user")

    get_session.assert_awaited_once_with("session-1", user_id="other-user")


@pytest.mark.asyncio
async def test_delete_session_removes_linked_shared_memories_first(monkeypatch):
    class CaptureDb:
        async def execute_raw(self, query, *args):
            return 1

    session = _session("match3").model_copy(
        update={
            "result": {
                "memory_sync": {
                    "status": "stored",
                    "user_memory_id": "memory-user",
                    "ai_memory_id": "memory-ai",
                }
            }
        }
    )
    delete_memory = AsyncMock()
    monkeypatch.setattr(native, "db", CaptureDb())
    monkeypatch.setattr(native, "get_session", AsyncMock(return_value=session))
    monkeypatch.setattr(native.memory_repo, "delete", delete_memory)

    await native.delete_session("session-1", user_id="user-1")

    assert delete_memory.await_args_list == [
        call("memory-user", source="user"),
        call("memory-ai", source="ai"),
    ]


@pytest.mark.asyncio
async def test_list_sessions_filters_out_games_removed_from_the_registry(monkeypatch):
    class CaptureDb:
        call = None

        async def query_raw(self, query, *args):
            self.call = (query, args)
            return []

    database = CaptureDb()
    monkeypatch.setattr(native, "db", database)

    sessions = await native.list_sessions("user-1")

    assert sessions == []
    assert "game_key IN" in database.call[0]
    assert database.call[1] == ("user-1", 50)
    assert all(
        f"'{game_key}'" in database.call[0]
        for game_key in native._GAME_DEFINITIONS
    )


@pytest.mark.parametrize(
    "game_key",
    [
        "go",
        "reversi",
        "xiangqi",
        "chess",
        "chinese_checkers",
        "match3",
        "minesweeper",
        "number_merge",
    ],
)
def test_generic_action_chain_keeps_detailed_state_and_analysis(game_key: str):
    definition = native._definition(game_key)
    result = native._empty_result("normal", definition)
    first = native._validate_generic_action(
        result,
        definition,
        _action("user", 1, after_hash="hash-1"),
    )
    result = native._append_generic_action(result, definition, first)
    second = native._validate_generic_action(
        result,
        definition,
        _action("agent", 2, before_hash="hash-1", after_hash="hash-2"),
    )
    result = native._append_generic_action(result, definition, second)

    game = native._generic_process(result, definition)
    assert game["action_count"] == 2
    assert game["final_state_hash"] == "hash-2"
    assert game["actions"][0]["analysis"] == {"depth": 3, "score": 10}
    assert game["actions"][1]["state_before_hash"] == "hash-1"


def test_generic_action_rejects_broken_state_hash_chain():
    definition = native._definition("chess")
    result = native._empty_result("normal", definition)
    result = native._append_generic_action(
        result,
        definition,
        native._validate_generic_action(
            result,
            definition,
            _action("user", 1, after_hash="hash-1"),
        ),
    )

    with pytest.raises(ValueError, match="invalid_state_hash"):
        native._validate_generic_action(
            result,
            definition,
            _action("agent", 2, before_hash="wrong", after_hash="hash-2"),
        )


def test_generic_action_requires_a_state_after_hash():
    definition = native._definition("chess")
    result = native._empty_result("normal", definition)
    payload = _action("user", 1, after_hash="hash-1")
    payload["state_after"] = {}

    with pytest.raises(ValueError, match="invalid_state_hash"):
        native._validate_generic_action(result, definition, payload)


def test_generic_action_requires_the_previous_hash_after_first_action():
    definition = native._definition("chess")
    result = native._empty_result("normal", definition)
    result = native._append_generic_action(
        result,
        definition,
        native._validate_generic_action(
            result,
            definition,
            _action("user", 1, after_hash="hash-1"),
        ),
    )

    with pytest.raises(ValueError, match="invalid_state_hash"):
        native._validate_generic_action(
            result,
            definition,
            _action("agent", 2, after_hash="hash-2"),
        )


def test_key_moment_auxiliary_event_does_not_duplicate_action_moment():
    definition = native._definition("chinese_checkers")
    result = native._empty_result("normal", definition)
    action = native._validate_generic_action(
        result,
        definition,
        {
            **_action("user", 1, after_hash="hash-1"),
            "moment": {"type": "long_jump"},
        },
    )
    result = native._append_generic_action(result, definition, action)
    result = native._merge_auxiliary_event(
        result,
        "key_moment",
        {"type": "long_jump", "move_number": 1},
    )

    game = native._generic_process(result, definition)
    assert game["key_moments"] == [
        {"type": "long_jump", "action_number": 1, "actor": "user"}
    ]


def test_reversi_action_persists_every_key_moment_once():
    definition = native._definition("reversi")
    result = native._empty_result("normal", definition)
    action = native._validate_generic_action(
        result,
        definition,
        {
            **_action("user", 1, after_hash="hash-1"),
            "moments": [
                {"type": "corner_captured", "at": {"row": 0, "col": 0}},
                {"type": "big_flip", "flipped_count": 9},
            ],
        },
    )

    result = native._append_generic_action(result, definition, action)
    result = native._merge_auxiliary_event(
        result,
        "key_moment",
        {
            "type": "corner_captured",
            "move_number": 1,
            "actor": "user",
            "at": {"row": 0, "col": 0},
        },
    )

    moments = native._generic_process(result, definition)["key_moments"]
    assert [moment["type"] for moment in moments] == [
        "corner_captured",
        "big_flip",
    ]


def test_reversi_terminal_status_matches_reported_outcome():
    definition = native._definition("reversi")

    assert (
        native._validated_generic_outcome(
            {
                "user_outcome": "lose",
                "terminal_state": {"status": "agentWon"},
            },
            definition,
        )
        == "lose"
    )


def test_minesweeper_action_and_inference_keep_full_process_data():
    definition = native._definition("minesweeper")
    result = native._empty_result("normal", definition)
    action = native._validate_generic_action(
        result,
        definition,
        {
            "action_id": "mine-action-1",
            "actor": "user",
            "action": "reveal",
            "at": {"row": 4, "column": 4},
            "revealed_cells": [
                {"row": 4, "column": 4},
                {"row": 4, "column": 5},
            ],
            "state_before": {"state_hash": "initial"},
            "state_after": {"state_hash": "after-1", "revealed_count": 2},
            "analysis": {"constraint_count": 3, "forced_safe_count": 1},
            "moments": [{"type": "zero_expansion", "revealed_count": 9}],
        },
    )
    result = native._append_generic_action(result, definition, action)
    result = native._merge_auxiliary_event(
        result,
        "inference_made",
        {
            "action_number": 1,
            "actor": "agent",
            "mine_probability": 0,
            "algorithm": "constraint_propagation",
        },
    )

    game = native._generic_process(result, definition)
    assert game["action_count"] == 1
    assert game["actions"][0]["revealed_cells"][1] == {
        "row": 4,
        "column": 5,
    }
    assert game["latest_analysis"]["constraint_count"] == 3
    assert game["key_moments"][0]["type"] == "zero_expansion"
    assert game["snapshots"][0]["event_type"] == "inference_made"


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("completed", "win"), ("failed", "lose")],
)
def test_minesweeper_terminal_status_matches_shared_outcome(
    status: str,
    outcome: str,
):
    definition = native._definition("minesweeper")

    assert (
        native._validated_generic_outcome(
            {
                "user_outcome": outcome,
                "terminal_state": {"status": status},
            },
            definition,
        )
        == outcome
    )


def test_minesweeper_terminal_reply_is_cooperative_not_competitive():
    definition = native._definition("minesweeper")
    session = _session("minesweeper")
    result = {
        **native._empty_result("normal", definition),
        "user_outcome": "lose",
    }

    assert "下次" in native._generic_finish_reply(session, definition, result)
    assert "我先赢" not in native._generic_finish_reply(session, definition, result)


def test_number_merge_action_preserves_every_transition_and_merge_value():
    definition = native._definition("number_merge")
    result = native._empty_result("normal", definition)
    payload = {
        "action_id": "merge-action-1",
        "actor": "user",
        "action": "slide",
        "direction": "left",
        "transitions": [
            {
                "from": {"row": 0, "column": 0},
                "to": {"row": 0, "column": 0},
                "value": 2,
                "result_value": 4,
                "merged": True,
            },
            {
                "from": {"row": 0, "column": 1},
                "to": {"row": 0, "column": 0},
                "value": 2,
                "result_value": 4,
                "merged": True,
            },
        ],
        "merged_values": [4],
        "spawn": {"at": {"row": 3, "column": 3}, "value": 2},
        "score_gained": 4,
        "state_before": {"state_hash": "initial"},
        "state_after": {"state_hash": "after-1", "score": 4},
        "analysis": {"empty_cells": 14, "mobility": 3},
        "moments": [{"type": "first_merge", "created_value": 4}],
    }

    action = native._validate_generic_action(result, definition, payload)
    result = native._append_generic_action(result, definition, action)
    result = native._merge_auxiliary_event(
        result,
        "tiles_merged",
        {
            "move_number": 1,
            "actor": "user",
            "values": [4],
            "transitions": payload["transitions"],
        },
    )

    game = native._generic_process(result, definition)
    assert game["actions"][0]["transitions"] == payload["transitions"]
    assert game["actions"][0]["merged_values"] == [4]
    assert game["latest_analysis"]["empty_cells"] == 14
    assert game["key_moments"][0]["type"] == "first_merge"
    assert game["snapshots"][0]["event_type"] == "tiles_merged"


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("completed", "win"), ("failed", "lose")],
)
def test_number_merge_terminal_status_matches_shared_outcome(
    status: str,
    outcome: str,
):
    definition = native._definition("number_merge")

    assert (
        native._validated_generic_outcome(
            {
                "user_outcome": outcome,
                "terminal_state": {"status": status},
            },
            definition,
        )
        == outcome
    )


def test_number_merge_finish_reply_uses_shared_progress():
    definition = native._definition("number_merge")
    session = _session("number_merge")
    result = {
        **native._empty_result("normal", definition),
        "user_outcome": "win",
        "final_payload": {"max_tile": 2048},
    }

    reply = native._generic_finish_reply(session, definition, result)

    assert "2048" in reply
    assert "我们" in reply


def test_generic_terminal_status_must_match_reported_outcome():
    definition = native._definition("match3")

    with pytest.raises(ValueError, match="invalid_outcome"):
        native._validated_generic_outcome(
            {
                "user_outcome": "win",
                "terminal_state": {"status": "failed"},
            },
            definition,
        )


def test_go_terminal_status_matches_reported_outcome():
    definition = native._definition("go")

    assert (
        native._validated_generic_outcome(
            {
                "user_outcome": "win",
                "terminal_state": {"status": "userWon"},
            },
            definition,
        )
        == "win"
    )
    with pytest.raises(ValueError, match="invalid_outcome"):
        native._validated_generic_outcome(
            {
                "user_outcome": "win",
                "terminal_state": {"status": "agentWon"},
            },
            definition,
        )


@pytest.mark.parametrize(
    ("game_key", "status", "outcome"),
    [
        ("go", "userWon", "win"),
        ("reversi", "agentWon", "lose"),
        ("xiangqi", "draw", "draw"),
        ("chess", "userWon", "win"),
        ("chinese_checkers", "agentWon", "lose"),
        ("match3", "completed", "win"),
        ("minesweeper", "failed", "lose"),
        ("number_merge", "completed", "win"),
        ("tetris_duel", "userWon", "win"),
    ],
)
def test_generic_terminal_recovery_uses_persisted_final_state(
    game_key: str,
    status: str,
    outcome: str,
):
    definition = native._definition(game_key)
    result = native._empty_result("normal", definition)
    game = native._generic_process(result, definition)
    game["final_state"] = {"status": status, "state_hash": "terminal-hash"}
    result["process"][game_key] = game

    recovered = native._recover_generic_terminal(result, definition)

    assert recovered == (
        outcome,
        status,
        {"status": status, "state_hash": "terminal-hash"},
    )


def test_generic_terminal_recovery_ignores_a_playing_snapshot():
    definition = native._definition("reversi")
    result = native._empty_result("normal", definition)
    game = native._generic_process(result, definition)
    game["final_state"] = {"status": "playing", "state_hash": "active-hash"}
    result["process"][definition.key] = game

    assert native._recover_generic_terminal(result, definition) is None


@pytest.mark.asyncio
async def test_stale_cleanup_settles_a_generic_terminal_snapshot(monkeypatch):
    definition = native._definition("minesweeper")
    result = native._empty_result("normal", definition)
    game = native._generic_process(result, definition)
    game["final_state"] = {
        "status": "completed",
        "state_hash": "terminal-hash",
        "revealed_count": 71,
    }
    result["process"][definition.key] = game

    class StaleDb:
        async def query_raw(self, query, *args):
            return [{"id": "session-1", "user_id": "user-1", "result": result}]

    calls = []

    async def handle_event(**data):
        calls.append(data)
        return (
            _session("minesweeper").model_copy(update={"status": "settled"}),
            None,
            "event-1",
            False,
        )

    monkeypatch.setattr(native, "db", StaleDb())
    monkeypatch.setattr(native, "handle_event", handle_event)

    closed = await native.abort_stale_sessions()

    assert closed == 1
    assert calls[0]["event_type"] == "game_finished"
    assert calls[0]["state"] == "settled"
    assert calls[0]["payload"] == {
        "user_outcome": "win",
        "reason": "client_disconnected_after_finish",
        "terminal_state": {"status": "completed"},
        "final_state": {
            "status": "completed",
            "state_hash": "terminal-hash",
            "revealed_count": 71,
        },
        "state_after_hash": "terminal-hash",
    }
    assert calls[0]["client_event_id"] == "server-timeout-finish:session-1"


def test_game_start_persists_the_initial_state_for_event_analysis():
    definition = native._definition("number_merge")
    result = native._empty_result("normal", definition)
    initial_state = {
        "state_hash": "initial-hash",
        "turn": "user",
        "board": [0, 0, 2, 0] * 4,
    }

    stored = native._store_initial_state(
        result,
        definition,
        {"initial_state": initial_state},
    )

    game = native._generic_process(stored, definition)
    assert game["final_state"] == initial_state
    assert game["final_state_hash"] == "initial-hash"


def test_terminal_reconciliation_preserves_saved_detailed_actions():
    definition = native._definition("chinese_checkers")
    result = native._empty_result("normal", definition)
    first = native._validate_generic_action(
        result,
        definition,
        _action("user", 1, after_hash="hash-1"),
    )
    result = native._append_generic_action(result, definition, first)
    reported = [
        {
            "actor": "user",
            "from": {"row": 1, "col": 1},
            "to": {"row": 2, "col": 2},
        },
        _action("agent", 2, before_hash="hash-1", after_hash="hash-2"),
    ]

    reconciled = native._reconcile_reported_actions(
        result,
        definition,
        {"actions": reported},
    )
    game = native._generic_process(reconciled, definition)

    assert game["action_count"] == 2
    assert game["recovered_action_count"] == 1
    assert game["actions"][0]["analysis"] == {"depth": 3, "score": 10}


def test_generic_finish_keeps_game_summary_next_to_process_data():
    definition = native._definition("match3")
    result = native._empty_result("normal", definition)
    result = native._finish_generic_result(
        result,
        definition,
        {
            "user_outcome": "win",
            "turn_count": 18,
            "user_score": 6400,
            "agent_score": 5900,
            "total_score": 12300,
            "target_score": 12000,
            "final_state": {"state_hash": "final-hash"},
            "terminal_state": {"status": "completed"},
        },
        outcome="win",
        duration_seconds=183,
    )

    game = native._generic_process(result, definition)
    assert game["summary"]["total_score"] == 12300
    assert game["summary"]["target_score"] == 12000
    assert game["final_state_hash"] == "final-hash"
    assert result["user_outcome"] == "win"


async def test_generic_shared_experience_enters_both_memory_sides(monkeypatch):
    calls = []

    async def remember(**data):
        calls.append(data)
        return {
            "status": "stored",
            "user_memory_id": "user-memory",
            "ai_memory_id": "ai-memory",
            "failed_sides": [],
        }

    monkeypatch.setattr(native, "remember_shared_game_experience", remember)
    definition = native._definition("chinese_checkers")
    result = native._empty_result("normal", definition)
    process = result["process"]["chinese_checkers"]
    process.update(
        {
            "action_count": 21,
            "key_moments": [{"type": "long_jump", "action_number": 12}],
        }
    )
    result.update(
        {
            "user_outcome": "lose",
            "duration_seconds": 245,
            "chinese_checkers": process,
        }
    )

    sync = await native._remember_shared_experience(
        _session("chinese_checkers"),
        result,
    )

    assert sync["status"] == "stored"
    assert len(calls) == 1
    assert "《跳棋》" in calls[0]["user_text"]
    assert "共同经历" in calls[0]["ai_text"]
    assert "连续长跳" in calls[0]["ai_text"]


@pytest.mark.parametrize(
    ("game_key", "moments", "expected"),
    [
        (
            "gomoku",
            [{"type": "double_threat"}, {"type": "blocked_win"}],
            ("双向威胁", "胜负边缘"),
        ),
        (
            "go",
            [{"type": "large_capture"}, {"type": "atari"}],
            ("提掉", "叫吃"),
        ),
        (
            "reversi",
            [{"type": "corner_captured"}, {"type": "forced_pass"}],
            ("角点", "无子可下"),
        ),
        (
            "xiangqi",
            [{"type": "check"}, {"type": "capture"}],
            ("将军", "关键交换"),
        ),
        (
            "chess",
            [{"type": "castling"}, {"type": "promotion"}],
            ("王车易位", "升变"),
        ),
        (
            "chinese_checkers",
            [{"type": "long_jump"}, {"type": "near_finish"}],
            ("连续长跳", "进营"),
        ),
        (
            "match3",
            [{"type": "special_combo"}, {"type": "big_cascade"}],
            ("特殊方块", "连续消除"),
        ),
        (
            "minesweeper",
            [{"type": "forced_deduction"}, {"type": "zero_expansion"}],
            ("严格推出来", "安全区域"),
        ),
        (
            "number_merge",
            [{"type": "target_reached"}, {"type": "board_recovered"}],
            ("目标数字", "救出了空间"),
        ),
        (
            "tetris_duel",
            [{"type": "tetris"}, {"type": "combo"}],
            ("四行同时消除", "连续消行"),
        ),
    ],
)
def test_every_game_extracts_distinctive_shared_memory_moments(
    game_key: str,
    moments: list[dict],
    expected: tuple[str, str],
):
    text = native._memory_moment(moments, game_key)

    assert all(fragment in text for fragment in expected)


@pytest.mark.asyncio
async def test_duplicate_terminal_replay_repairs_chat_side_effects(monkeypatch):
    calls = []
    background = []

    async def persist_status(previous, updated, event_type, state, payload):
        calls.append(("status", previous.status, updated.status, payload["game_title"]))

    async def persist_reply(session, event_type, state, reply):
        calls.append(("reply", event_type, reply))

    monkeypatch.setattr(
        native.sud,
        "_persist_game_status_to_chat_if_needed",
        persist_status,
    )
    monkeypatch.setattr(
        native.sud,
        "_persist_reply_to_chat_if_needed",
        persist_reply,
    )
    monkeypatch.setattr(native, "fire_background", background.append)
    session = _session("chinese_checkers").model_copy(
        update={
            "status": "settled",
            "result": {"memory_sync": {"status": "stored"}},
        }
    )

    await native._ensure_idempotent_side_effects(
        session,
        "game_finished",
        "settled",
        {},
        "这局我先赢一下。",
    )

    assert calls == []
    assert len(background) == 1
    await background[0]

    assert calls == [
        ("status", "playing", "settled", "跳棋"),
        ("reply", "game_finished", "这局我先赢一下。"),
    ]


@pytest.mark.asyncio
async def test_chat_projection_retry_rebuilds_started_and_terminal_messages(
    monkeypatch,
):
    class CaptureDb:
        query = ""

        async def query_raw(self, query, *args):
            self.query = query
            return [{}]

    database = CaptureDb()
    stored = _session("match3").model_copy(
        update={
            "status": "settled",
            "companion_reply": "刚才那一下消得很漂亮。",
        }
    )
    persist = AsyncMock()
    monkeypatch.setattr(native, "db", database)
    monkeypatch.setattr(native.sud, "_row_to_session", lambda _row: stored)
    monkeypatch.setattr(native, "_persist_chat_side_effects", persist)

    attempted = await native.retry_missing_chat_side_effects(limit=5)

    assert attempted == 1
    assert "provider = 'native'" in database.query
    assert "game_status', 'started'" in database.query
    assert persist.await_count == 2
    assert persist.await_args_list[0].args[2] == "game_started"
    assert persist.await_args_list[1].args[2] == "game_finished"


def test_generic_finish_reply_sounds_like_a_companion_not_a_score_report():
    definition = native._definition("match3")
    result = native._empty_result("normal", definition)
    result["user_outcome"] = "win"
    result["process"]["match3"]["key_moments"] = [
        {"type": "big_cascade", "action_number": 9}
    ]

    reply = native._generic_finish_reply(_session("match3"), definition, result)

    assert "我们" in reply
    assert "最后一下" in reply
    assert "分" not in reply
    assert "总结" not in reply
