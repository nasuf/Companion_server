from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.games import native


def _lock(
    actor: str,
    piece_number: int,
    *,
    lines: int = 0,
    combo: int = 0,
    attack: int = 0,
    top_out: bool = False,
) -> dict:
    return {
        "actor": actor,
        "piece": "t" if actor == "user" else "i",
        "piece_number": piece_number,
        "lines_cleared": lines,
        "score_gained": piece_number * 100 + lines * 500,
        "score_after": piece_number * 100 + lines * 500,
        "lines_after": lines,
        "level_after": 1,
        "combo": combo,
        "back_to_back": lines == 4,
        "attack_sent": attack,
        "drop_distance": 14,
        "board_hash": f"{actor}-{piece_number}",
        "board_after": [0] * 190 + [1] * 10,
        "top_out": top_out,
        "remaining_seconds": 90 - piece_number,
    }


def test_tetris_lock_chain_tracks_both_players_without_repeating_boards():
    definition = native._definition("tetris_duel")
    result = native._empty_result("normal", definition)

    user_lock = native._validate_tetris_lock(
        result,
        definition,
        _lock("user", 1, lines=4, combo=3, attack=5),
    )
    result = native._append_tetris_lock(result, definition, user_lock)
    agent_lock = native._validate_tetris_lock(
        result,
        definition,
        _lock("agent", 1, lines=2, attack=1),
    )
    result = native._append_tetris_lock(result, definition, agent_lock)

    game = native._generic_process(result, definition)
    assert game["action_count"] == 2
    assert game["user_actions"] == 1
    assert game["ai_actions"] == 1
    assert "board_after" not in game["actions"][0]
    assert game["user"]["final_board"] == [0] * 190 + [1] * 10
    assert game["user"]["tetrises"] == 1
    assert game["user"]["max_combo"] == 3
    assert game["user"]["attack_sent"] == 5
    assert {moment["type"] for moment in game["key_moments"]} == {
        "tetris",
        "multi_line_clear",
    }


def test_tetris_rejects_actor_piece_number_gaps_and_invalid_boards():
    definition = native._definition("tetris_duel")
    result = native._empty_result("normal", definition)

    with pytest.raises(ValueError, match="invalid_piece_number"):
        native._validate_tetris_lock(result, definition, _lock("user", 2))

    invalid = _lock("user", 1)
    invalid["board_after"] = [0] * 199
    with pytest.raises(ValueError, match="invalid_board"):
        native._validate_tetris_lock(result, definition, invalid)

    invalid_progress = _lock("user", 1, lines=1)
    invalid_progress["score_gained"] -= 1
    with pytest.raises(ValueError, match="invalid_score_progression"):
        native._validate_tetris_lock(result, definition, invalid_progress)

    invalid_boolean = _lock("user", 1)
    invalid_boolean["top_out"] = "false"
    with pytest.raises(ValueError, match="invalid_top_out"):
        native._validate_tetris_lock(result, definition, invalid_boolean)


def test_tetris_terminal_outcome_and_companion_reply_are_game_specific():
    definition = native._definition("tetris_duel")
    payload = {
        "user_outcome": "win",
        "terminal_state": {"status": "userWon", "reason": "time_limit"},
        "state_after_hash": "terminal-hash",
        "score": {"user": 6400, "agent": 5900},
        "user": {"score": 6400, "lines": 8, "top_out": False},
        "agent": {"score": 5900, "lines": 7, "top_out": False},
    }
    assert native._validated_tetris_outcome(
        native._empty_result("normal", definition), payload
    ) == "win"

    result = native._empty_result("normal", definition)
    result = native._append_tetris_lock(
        result,
        definition,
        native._validate_tetris_lock(
            result,
            definition,
            _lock("user", 1, lines=4, combo=3, attack=5),
        ),
    )
    result = native._finish_generic_result(
        result,
        definition,
        payload,
        outcome="win",
        duration_seconds=90,
    )
    # 打到 6400 分 / 8 行必然锁了很多块。上面只 append 了 1 块 (够测校验逻辑),
    # 但 0-1 步的局现在走"还没真正展开"分支 —— 中途退出也判负, 那些局同样落
    # settled, 给它们回"你越堆越稳"是编造。这里补齐到与分数相称的步数。
    result["process"]["tetris_duel"]["action_count"] = 24
    reply = native._generic_finish_reply(
        SimpleNamespace(ai_player=SimpleNamespace(nick_name="小芜")),
        definition,
        result,
    )

    assert "越堆越稳" in reply
    assert "6400" not in reply
    assert "总结" not in reply


def test_tetris_terminal_rejects_score_or_winner_mismatch():
    definition = native._definition("tetris_duel")
    result = native._empty_result("normal", definition)
    payload = {
        "user_outcome": "win",
        "terminal_state": {"status": "userWon", "reason": "time_limit"},
        "score": {"user": 1000, "agent": 1200},
        "user": {"score": 1000, "lines": 2, "top_out": False},
        "agent": {"score": 1200, "lines": 2, "top_out": False},
    }

    with pytest.raises(ValueError, match="invalid_outcome"):
        native._validated_tetris_outcome(result, payload)

    malformed = {**payload, "score": []}
    with pytest.raises(ValueError, match="invalid_terminal_summary"):
        native._validated_tetris_outcome(result, malformed)


def test_tetris_memory_moments_capture_combo_and_top_out():
    text = native._memory_moment(
        [{"type": "combo"}, {"type": "top_out"}],
        "tetris_duel",
    )

    assert "连续消行" in text
    assert "堆到顶" in text
