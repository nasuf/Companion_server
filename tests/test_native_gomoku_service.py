from __future__ import annotations

import pytest

from app.models.game import SudPlayerInfo, SudSessionResponse
from app.services.games import native


def _session(**updates) -> SudSessionResponse:
    base = SudSessionResponse(
        id="native-1",
        provider="native",
        game_key="gomoku",
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
        room_id="gomoku-native",
        code="",
        code_expires_at="",
        play_mode="versus",
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
    return base.model_copy(update=updates)


def test_native_gomoku_normalizes_turn_and_coordinate():
    result = native._empty_result("normal")

    move = native._validate_and_normalize_move(
        result,
        {"actor": "user", "row": 7, "col": 7},
    )

    assert move["move_number"] == 1
    assert move["stone"] == "black"
    assert move["coordinate"] == "H8"


def test_native_gomoku_rejects_wrong_turn_and_occupied_point():
    result = native._empty_result("normal")
    result = native._append_move(
        result,
        native._validate_and_normalize_move(
            result,
            {"actor": "user", "row": 7, "col": 7},
        ),
    )

    try:
        native._validate_and_normalize_move(
            result,
            {"actor": "user", "row": 7, "col": 8},
        )
    except ValueError as exc:
        assert str(exc) == "invalid_turn"
    else:
        raise AssertionError("wrong turn must be rejected")

    try:
        native._validate_and_normalize_move(
            result,
            {"actor": "agent", "row": 7, "col": 7},
        )
    except ValueError as exc:
        assert str(exc) == "occupied_position"
    else:
        raise AssertionError("occupied point must be rejected")


def test_native_gomoku_detects_diagonal_win():
    moves = []
    for index in range(5):
        moves.append({"actor": "user", "row": index + 3, "col": index + 2})
        if index < 4:
            moves.append({"actor": "agent", "row": 0, "col": index})

    winner, line = native._winner(moves)

    assert winner == "user"
    assert len(line) == 5
    assert native._line_direction(line) == "diagonal"


def test_native_gomoku_validates_finished_outcome_from_board():
    result = native._empty_result("hard")
    for index in range(5):
        user_move = native._validate_and_normalize_move(
            result,
            {"actor": "user", "row": 7, "col": index + 3},
        )
        result = native._append_move(result, user_move)
        if index < 4:
            agent_move = native._validate_and_normalize_move(
                result,
                {"actor": "agent", "row": 9, "col": index + 3},
            )
            result = native._append_move(result, agent_move)

    outcome, line = native._validated_outcome(result, {"user_outcome": "win"})

    assert outcome == "win"
    assert native._line_direction(line) == "horizontal"


def test_native_gomoku_reconciles_missing_move_events_from_final_history():
    result = native._empty_result("normal")
    reported = []
    for index in range(5):
        reported.append({"actor": "user", "row": 7, "col": index + 3})
        if index < 4:
            reported.append({"actor": "agent", "row": 9, "col": index + 3})

    for move in reported[:4]:
        result = native._append_move(
            result,
            native._validate_and_normalize_move(result, move),
        )

    reconciled = native._reconcile_reported_moves(result, {"moves": reported})
    gomoku = native._gomoku(reconciled)
    outcome, _ = native._validated_outcome(
        reconciled,
        {"user_outcome": "win"},
    )

    assert outcome == "win"
    assert gomoku["move_count"] == 9
    assert gomoku["recovered_move_count"] == 5


def test_native_gomoku_rejects_final_history_that_rewrites_saved_moves():
    result = native._empty_result("normal")
    result = native._append_move(
        result,
        native._validate_and_normalize_move(
            result,
            {"actor": "user", "row": 7, "col": 7},
        ),
    )

    try:
        native._reconcile_reported_moves(
            result,
            {"moves": [{"actor": "user", "row": 7, "col": 8}]},
        )
    except ValueError as exc:
        assert str(exc) == "invalid_move_history"
    else:
        raise AssertionError("reported history must preserve persisted moves")


def test_native_gomoku_rejects_moves_after_a_detected_win():
    result = native._empty_result("normal")
    for index in range(5):
        result = native._append_move(
            result,
            native._validate_and_normalize_move(
                result,
                {"actor": "user", "row": 7, "col": index + 3},
            ),
        )
        if index < 4:
            result = native._append_move(
                result,
                native._validate_and_normalize_move(
                    result,
                    {"actor": "agent", "row": 9, "col": index + 3},
                ),
            )

    try:
        native._validate_and_normalize_move(
            result,
            {"actor": "agent", "row": 10, "col": 10},
        )
    except ValueError as exc:
        assert str(exc) == "game_already_finished"
    else:
        raise AssertionError("moves after a win must be rejected")


def test_native_finish_reply_sounds_like_shared_play_not_a_report():
    result = native._empty_result("normal")
    result["user_outcome"] = "win"
    result["gomoku"] = {
        "move_count": 19,
        "winning_line": [
            {"x": 4, "y": 4},
            {"x": 5, "y": 5},
            {"x": 6, "y": 6},
            {"x": 7, "y": 7},
            {"x": 8, "y": 8},
        ],
        "key_moments": [{"type": "double_threat", "move_number": 17}],
    }

    reply = native._finish_reply(_session(), result)

    assert "两边一起冒出来" in reply
    assert "19手" not in reply
    assert "总结" not in reply


async def test_shared_experience_enters_both_memory_sides(monkeypatch):
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
    result = native._empty_result("normal")
    result.update({"user_outcome": "lose", "duration_seconds": 132})
    result["gomoku"] = {
        "move_count": 24,
        "key_moments": [{"type": "blocked_win", "move_number": 20}],
    }

    memory_sync = await native._remember_shared_experience(_session(), result)

    assert len(calls) == 1
    assert "共同经历" in calls[0]["ai_text"]
    assert "差一点" not in calls[0]["user_text"]
    assert calls[0]["workspace_id"] == "workspace-1"
    assert memory_sync["status"] == "stored"


def test_memory_sync_merge_keeps_previous_side_and_schedules_failure_retry():
    previous = {
        "status": "partial",
        "user_memory_id": "user-memory",
        "ai_memory_id": None,
        "attempts": 1,
    }

    merged = native._merge_memory_sync(
        previous,
        {
            "status": "failed",
            "user_memory_id": None,
            "ai_memory_id": None,
            "failed_sides": ["ai"],
        },
    )

    assert merged["status"] == "partial"
    assert merged["user_memory_id"] == "user-memory"
    assert merged["failed_sides"] == ["ai"]
    assert merged["attempts"] == 2
    assert merged["next_retry_at"]


def test_memory_sync_preserves_skipped_zero_move_round():
    merged = native._merge_memory_sync(
        {"status": "pending", "attempts": 0},
        {
            "status": "skipped",
            "user_memory_id": None,
            "ai_memory_id": None,
            "failed_sides": [],
        },
    )

    assert merged["status"] == "skipped"
    assert "next_retry_at" not in merged


@pytest.mark.asyncio
async def test_zero_action_round_never_writes_shared_memory(monkeypatch):
    async def remember(*args, **kwargs):
        raise AssertionError("zero-action rounds must not enter long-term memory")

    monkeypatch.setattr(native, "remember_shared_game_experience", remember)
    result = native._empty_result("normal", native._definition("go"))

    memory_sync = await native._remember_shared_experience(
        _session(game_key="go"),
        result,
    )

    assert memory_sync == {
        "status": "skipped",
        "user_memory_id": None,
        "ai_memory_id": None,
        "failed_sides": [],
    }


@pytest.mark.asyncio
async def test_sync_session_memory_persists_delivery_state(monkeypatch):
    session = _session(
        result={
            **native._empty_result("normal"),
            "memory_sync": {
                "status": "pending",
                "user_memory_id": None,
                "ai_memory_id": None,
                "attempts": 0,
            },
        }
    )
    updates = []

    async def claim_memory_sync(*args, **kwargs):
        return session

    async def remember(*args, **kwargs):
        return {
            "status": "stored",
            "user_memory_id": "user-memory",
            "ai_memory_id": "ai-memory",
            "failed_sides": [],
        }

    async def update_session(**data):
        updates.append(data)

    monkeypatch.setattr(native, "_claim_memory_sync", claim_memory_sync)
    monkeypatch.setattr(native, "_remember_shared_experience", remember)
    monkeypatch.setattr(native, "_update_session", update_session)

    synced = await native.sync_session_memory("native-1")

    assert synced["status"] == "stored"
    assert synced["attempts"] == 1
    assert updates[0]["result"]["memory_sync"]["ai_memory_id"] == "ai-memory"


@pytest.mark.asyncio
async def test_partial_memory_retry_only_requests_the_missing_side(monkeypatch):
    session = _session(
        result={
            **native._empty_result("normal"),
            "memory_sync": {
                "status": "syncing",
                "user_memory_id": "user-memory",
                "ai_memory_id": None,
                "failed_sides": ["ai"],
                "attempts": 1,
            },
        }
    )
    requested_sides = []

    async def claim_memory_sync(*args, **kwargs):
        return session

    async def remember(*args, **kwargs):
        requested_sides.append(kwargs["sides"])
        return {
            "status": "stored",
            "user_memory_id": None,
            "ai_memory_id": "ai-memory",
            "failed_sides": [],
        }

    async def update_session(**data):
        return None

    monkeypatch.setattr(native, "_claim_memory_sync", claim_memory_sync)
    monkeypatch.setattr(native, "_remember_shared_experience", remember)
    monkeypatch.setattr(native, "_update_session", update_session)

    synced = await native.sync_session_memory("native-1")

    assert requested_sides == [("ai",)]
    assert synced["status"] == "stored"
    assert synced["user_memory_id"] == "user-memory"
    assert synced["ai_memory_id"] == "ai-memory"


@pytest.mark.asyncio
async def test_memory_sync_claim_is_an_atomic_database_lease(monkeypatch):
    class ClaimDb:
        call = None

        async def query_raw(self, query, *args):
            self.call = (query, args)
            return []

    database = ClaimDb()
    monkeypatch.setattr(native, "db", database)

    claimed = await native._claim_memory_sync("native-1")

    assert claimed is None
    assert "UPDATE game_sessions" in database.call[0]
    assert "lease_until" in database.call[0]
    assert database.call[1][0] == "native-1"


@pytest.mark.asyncio
async def test_abort_stale_sessions_uses_deterministic_server_event(monkeypatch):
    class StaleDb:
        async def query_raw(self, query, *args):
            assert "status = 'playing'" in query
            assert args == (7 * 24 * 60, 20)
            return [{"id": "session-1", "user_id": "user-1", "result": {}}]

    calls = []

    async def handle_event(**data):
        calls.append(data)
        return _session(status="aborted"), None, "event-1", False

    monkeypatch.setattr(native, "db", StaleDb())
    monkeypatch.setattr(native, "handle_event", handle_event)

    closed = await native.abort_stale_sessions()

    assert closed == 1
    assert calls[0]["source"] == "server"
    assert calls[0]["event_type"] == "game_aborted"
    assert calls[0]["payload"]["reason"] == "client_disconnected_timeout"
    assert calls[0]["client_event_id"] == "server-timeout-abort:session-1"


@pytest.mark.asyncio
async def test_abort_stale_sessions_settles_a_persisted_winning_board(monkeypatch):
    result = native._empty_result("normal")
    for index in range(5):
        result = native._append_move(
            result,
            native._validate_and_normalize_move(
                result,
                {"actor": "user", "row": 7, "col": index + 3},
            ),
        )
        if index < 4:
            result = native._append_move(
                result,
                native._validate_and_normalize_move(
                    result,
                    {"actor": "agent", "row": 9, "col": index + 3},
                ),
            )

    class StaleDb:
        async def query_raw(self, query, *args):
            return [{"id": "session-1", "user_id": "user-1", "result": result}]

    calls = []

    async def handle_event(**data):
        calls.append(data)
        return _session(status="settled"), None, "event-1", False

    monkeypatch.setattr(native, "db", StaleDb())
    monkeypatch.setattr(native, "handle_event", handle_event)

    closed = await native.abort_stale_sessions()

    assert closed == 1
    assert calls[0]["event_type"] == "game_finished"
    assert calls[0]["state"] == "settled"
    assert calls[0]["payload"]["user_outcome"] == "win"
    assert calls[0]["client_event_id"] == "server-timeout-finish:session-1"
