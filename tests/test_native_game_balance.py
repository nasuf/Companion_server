from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.games import balance


def test_engine_config_strength_is_monotonic_for_competitive_search():
    easy = balance.build_engine_config("gomoku", 20)
    hard = balance.build_engine_config("gomoku", 80)

    assert hard["search_time_ms"] > easy["search_time_ms"]
    assert hard["max_depth"] >= easy["max_depth"]
    assert hard["near_best_probability"] < easy["near_best_probability"]


def test_cooperative_strength_increases_challenge_instead_of_agent_power():
    gentle = balance.build_engine_config("match3", 20)
    challenging = balance.build_engine_config("match3", 80)

    assert challenging["target_score"] > gentle["target_score"]
    assert challenging["turn_limit"] < gentle["turn_limit"]
    assert (
        challenging["agent_choice_percentile"]
        < gentle["agent_choice_percentile"]
    )


def test_algorithm_override_validation_rejects_unknown_and_unsafe_values():
    with pytest.raises(ValueError, match="unknown_algorithm_parameter"):
        balance.validate_algorithm_overrides("gomoku", {"magic": 1})
    with pytest.raises(ValueError, match="invalid_algorithm_parameter:max_depth"):
        balance.validate_algorithm_overrides("gomoku", {"max_depth": 99})
    with pytest.raises(ValueError, match="invalid_algorithm_parameter:mine_count"):
        balance.validate_algorithm_overrides(
            "minesweeper",
            {"rows": 6, "columns": 6, "mine_count": 30},
        )


@pytest.mark.asyncio
async def test_resolve_for_session_uses_pair_skill_and_freezes_engine_snapshot():
    class FakeDb:
        def __init__(self):
            self.calls = 0

        async def query_raw(self, query, *args):
            self.calls += 1
            assert "LEFT JOIN native_game_skill_states" in query
            return [
                {
                    "mode": "adaptive",
                    "base_strength": 50,
                    "min_strength": 25,
                    "max_strength": 80,
                    "target_user_rate": 0.55,
                    "adjustment_window": 8,
                    "minimum_games": 3,
                    "maximum_step": 4,
                    "algorithm_overrides": {"search_time_ms": 333},
                    "version": 7,
                    "pair_strength": 74,
                    "pair_completed": 12,
                }
            ]

    database = FakeDb()
    snapshot = await balance.resolve_for_session(
        user_id="user-1",
        agent_id="agent-1",
        game_key="reversi",
        database=database,
    )

    assert database.calls == 1
    assert snapshot["config_version"] == 7
    assert snapshot["effective_strength"] == 74
    assert snapshot["completed_games_before"] == 12
    assert snapshot["engine_config"]["search_time_ms"] == 333


@pytest.mark.asyncio
async def test_resolve_for_session_falls_back_to_defaults_on_db_failure():
    class BrokenDb:
        async def query_raw(self, query, *args):
            raise RuntimeError("relation does not exist")

    snapshot = await balance.resolve_for_session(
        user_id="user-1",
        agent_id="agent-1",
        game_key="gomoku",
        database=BrokenDb(),
    )

    assert snapshot["fallback"] is True
    assert snapshot["effective_strength"] == 50
    assert snapshot["engine_config"]["strength"] == 50

    with pytest.raises(ValueError, match="unsupported_game"):
        await balance.resolve_for_session(
            user_id="user-1",
            agent_id="agent-1",
            game_key="not_a_game",
            database=BrokenDb(),
        )


@pytest.mark.asyncio
async def test_completed_session_updates_pair_strength_from_frozen_target():
    class FakeDb:
        def __init__(self):
            self.seeded = None
            self.updated = None

        async def query_raw(self, query, *args):
            assert "FOR UPDATE" in query
            return [
                {
                    "effective_strength": 50,
                    "completed_games": 4,
                    "ewma_user_rate": 0.80,
                    "wins": 3,
                    "losses": 1,
                    "draws": 0,
                }
            ]

        async def execute_raw(self, query, *args):
            if "INSERT INTO native_game_skill_states" in query:
                assert "ON CONFLICT (user_id, agent_id, game_key) DO NOTHING" in query
                self.seeded = args
            else:
                assert "UPDATE native_game_skill_states" in query
                self.updated = args
            return 1

    database = FakeDb()
    session = SimpleNamespace(
        user_id="user-1",
        agent_id="agent-1",
        game_key="gomoku",
        result={
            "user_outcome": "win",
            "balance": {
                "mode": "adaptive",
                "effective_strength": 50,
                "target_user_rate": 0.55,
                "adjustment_window": 10,
                "minimum_games": 3,
                "maximum_step": 5,
                "min_strength": 20,
                "max_strength": 85,
            },
        },
    )

    await balance.record_completed_session(session, database=database)

    assert database.seeded is not None
    assert database.seeded[3] == 50
    assert database.updated is not None
    assert database.updated[3] > 50
    assert database.updated[4] == 5
    assert database.updated[6] == 4


def test_row_config_preserves_zero_values():
    config = balance._row_config(
        {
            "mode": "fixed",
            "base_strength": 0,
            "min_strength": 0,
            "max_strength": 40,
            "target_user_rate": 0.5,
            "adjustment_window": 10,
            "minimum_games": 3,
            "maximum_step": 5,
            "algorithm_overrides": {},
            "version": 2,
        },
        "gomoku",
    )

    assert config.base_strength == 0
    assert config.min_strength == 0


def test_algorithm_override_validation_accepts_integral_floats():
    balance.validate_algorithm_overrides("gomoku", {"search_time_ms": 500.0})
    config = balance.build_engine_config("gomoku", 50, {"search_time_ms": 500.0})
    assert config["search_time_ms"] == 500
    assert isinstance(config["search_time_ms"], int)


def test_minesweeper_validation_holds_at_worst_case_strength():
    # rows/columns below the Flutter engine's floor are rejected outright.
    with pytest.raises(ValueError, match="invalid_algorithm_parameter:rows"):
        balance.validate_algorithm_overrides("minesweeper", {"rows": 5})
    # Small boards must leave room for the strength-100 lerped mine count.
    balance.validate_algorithm_overrides("minesweeper", {"rows": 6, "columns": 6})
    with pytest.raises(ValueError, match="invalid_algorithm_parameter:mine_count"):
        balance.validate_algorithm_overrides(
            "minesweeper",
            {"rows": 6, "columns": 6, "mine_count": 27},
        )
