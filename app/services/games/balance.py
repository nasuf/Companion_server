from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.db import db


logger = logging.getLogger(__name__)


GAME_TITLES: dict[str, str] = {
    "go": "围棋",
    "reversi": "黑白棋",
    "gomoku": "五子棋",
    "xiangqi": "中国象棋",
    "chess": "国际象棋",
    "chinese_checkers": "跳棋",
    "match3": "消消乐",
    "minesweeper": "协作扫雷",
    "number_merge": "数字合并",
    "tetris_duel": "双人方块竞速",
}

COOPERATIVE_GAMES = {"match3", "minesweeper", "number_merge"}

# Minimum wall-clock time (ms) an AI move should take so the opponent reads as
# human rather than instant. Delivered inside engine_config; the client pads
# the remainder if the engine already thought for less. tetris_duel is exempt
# (it paces via agent_move_ms), but the field is still emitted harmlessly.
DEFAULT_MIN_RESPONSE_MS = 900
DEFAULT_MAX_RESPONSE_MS = 1600
MIN_RESPONSE_MS_RANGE = (0, 8000)


@dataclass(frozen=True)
class GameBalanceConfig:
    game_key: str
    mode: str
    base_strength: int
    min_strength: int
    max_strength: int
    target_user_rate: float
    adjustment_window: int
    minimum_games: int
    maximum_step: int
    algorithm_overrides: dict[str, Any]
    min_response_ms: int
    max_response_ms: int
    enabled: bool
    version: int

    def snapshot(self, effective_strength: int) -> dict[str, Any]:
        return {
            "config_version": self.version,
            "mode": self.mode,
            "base_strength": self.base_strength,
            "effective_strength": effective_strength,
            "min_strength": self.min_strength,
            "max_strength": self.max_strength,
            "target_user_rate": self.target_user_rate,
            "adjustment_window": self.adjustment_window,
            "minimum_games": self.minimum_games,
            "min_response_ms": self.min_response_ms,
            "max_response_ms": self.max_response_ms,
            "engine_config": build_engine_config(
                self.game_key,
                effective_strength,
                self.algorithm_overrides,
                self.min_response_ms,
                self.max_response_ms,
            ),
        }


def _default_config(game_key: str) -> GameBalanceConfig:
    cooperative = game_key in COOPERATIVE_GAMES
    return GameBalanceConfig(
        game_key=game_key,
        mode="adaptive",
        base_strength=50,
        min_strength=20,
        max_strength=85,
        target_user_rate=0.70 if cooperative else 0.55,
        adjustment_window=10,
        minimum_games=3,
        maximum_step=5,
        algorithm_overrides={},
        min_response_ms=DEFAULT_MIN_RESPONSE_MS,
        max_response_ms=DEFAULT_MAX_RESPONSE_MS,
        enabled=True,
        version=1,
    )


def _int_field(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _float_field(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _bool_field(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _load_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _row_config(row: Any, game_key: str) -> GameBalanceConfig:
    if not row:
        return _default_config(game_key)
    return GameBalanceConfig(
        game_key=game_key,
        mode=str(row.get("mode") or "adaptive"),
        base_strength=_int_field(row.get("base_strength"), 50),
        min_strength=_int_field(row.get("min_strength"), 20),
        max_strength=_int_field(row.get("max_strength"), 85),
        target_user_rate=_float_field(row.get("target_user_rate"), 0.55),
        adjustment_window=_int_field(row.get("adjustment_window"), 10),
        minimum_games=_int_field(row.get("minimum_games"), 3),
        maximum_step=_int_field(row.get("maximum_step"), 5),
        algorithm_overrides=_load_json(row.get("algorithm_overrides")),
        min_response_ms=_int_field(
            row.get("min_response_ms"), DEFAULT_MIN_RESPONSE_MS
        ),
        max_response_ms=_int_field(
            row.get("max_response_ms"), DEFAULT_MAX_RESPONSE_MS
        ),
        enabled=_bool_field(row.get("enabled"), True),
        version=_int_field(row.get("version"), 1),
    )


async def get_config(game_key: str, *, database: Any | None = None) -> GameBalanceConfig:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    executor = database or db
    rows = await executor.query_raw(
        """
        SELECT game_key, mode, base_strength, min_strength, max_strength,
               target_user_rate, adjustment_window, minimum_games,
               maximum_step, algorithm_overrides, min_response_ms,
               max_response_ms, enabled, version
        FROM native_game_configs
        WHERE game_key = $1
        LIMIT 1
        """,
        game_key,
    )
    return _row_config(rows[0] if rows else None, game_key)


async def resolve_for_session(
    *,
    user_id: str,
    agent_id: str,
    game_key: str,
    database: Any | None = None,
) -> dict[str, Any]:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    try:
        return await _resolve_for_session(
            user_id=user_id,
            agent_id=agent_id,
            game_key=game_key,
            database=database,
        )
    except Exception:
        # Difficulty tuning must never block game creation; fall back to the
        # built-in curve if the config tables are unreachable.
        logger.warning(
            "game balance resolve failed for %s; using default config",
            game_key,
            exc_info=True,
        )
        config = _default_config(game_key)
        snapshot = config.snapshot(config.base_strength)
        snapshot["completed_games_before"] = 0
        snapshot["resolved_at"] = datetime.now(UTC).isoformat()
        snapshot["fallback"] = True
        return snapshot


async def _resolve_for_session(
    *,
    user_id: str,
    agent_id: str,
    game_key: str,
    database: Any | None = None,
) -> dict[str, Any]:
    executor = database or db
    rows = await executor.query_raw(
        """
        SELECT c.mode, c.base_strength, c.min_strength, c.max_strength,
               c.target_user_rate, c.adjustment_window, c.minimum_games,
               c.maximum_step, c.algorithm_overrides, c.min_response_ms,
               c.max_response_ms, c.version,
               s.effective_strength AS pair_strength,
               s.completed_games AS pair_completed
        FROM native_game_configs c
        LEFT JOIN native_game_skill_states s
          ON s.user_id = $1 AND s.agent_id = $2 AND s.game_key = c.game_key
        WHERE c.game_key = $3
        LIMIT 1
        """,
        user_id,
        agent_id,
        game_key,
    )
    row = rows[0] if rows else None
    config = _row_config(row, game_key)
    effective = config.base_strength
    completed_games = 0
    if config.mode == "adaptive":
        pair_strength = row.get("pair_strength") if row else None
        pair_completed = row.get("pair_completed") if row else None
        if row is None:
            # Config row missing (pre-migration bootstrap): the pair skill
            # state may still exist, so fetch it separately.
            skill_rows = await executor.query_raw(
                """
                SELECT effective_strength, completed_games
                FROM native_game_skill_states
                WHERE user_id = $1 AND agent_id = $2 AND game_key = $3
                LIMIT 1
                """,
                user_id,
                agent_id,
                game_key,
            )
            if skill_rows:
                pair_strength = skill_rows[0].get("effective_strength")
                pair_completed = skill_rows[0].get("completed_games")
        effective = _int_field(pair_strength, effective)
        completed_games = _int_field(pair_completed, 0)
    effective = max(config.min_strength, min(config.max_strength, effective))
    snapshot = config.snapshot(effective)
    snapshot["completed_games_before"] = completed_games
    snapshot["resolved_at"] = datetime.now(UTC).isoformat()
    return snapshot


async def record_completed_session(
    session: Any,
    *,
    database: Any | None = None,
) -> None:
    result = _load_json(getattr(session, "result", None))
    balance = _load_json(result.get("balance"))
    if balance.get("mode") != "adaptive":
        return
    outcome = str(result.get("user_outcome") or "").lower()
    if outcome not in {"win", "lose", "draw"}:
        return
    user_score = {"win": 1.0, "draw": 0.5, "lose": 0.0}[outcome]
    target = float(balance.get("target_user_rate") or 0.55)
    minimum_games = int(balance.get("minimum_games") or 3)
    max_step = int(balance.get("maximum_step") or 5)
    min_strength = int(balance.get("min_strength") or 20)
    max_strength = int(balance.get("max_strength") or 85)
    starting_strength = int(balance.get("effective_strength") or 50)
    window = max(2, int(balance.get("adjustment_window") or 10))
    alpha = 2.0 / (window + 1)

    if database is not None:
        await _record_completed_session(
            session=session,
            database=database,
            outcome=outcome,
            user_score=user_score,
            target=target,
            minimum_games=minimum_games,
            max_step=max_step,
            min_strength=min_strength,
            max_strength=max_strength,
            starting_strength=starting_strength,
            alpha=alpha,
        )
        return
    async with db.tx() as tx:
        await _record_completed_session(
            session=session,
            database=tx,
            outcome=outcome,
            user_score=user_score,
            target=target,
            minimum_games=minimum_games,
            max_step=max_step,
            min_strength=min_strength,
            max_strength=max_strength,
            starting_strength=starting_strength,
            alpha=alpha,
        )


async def _record_completed_session(
    *,
    session: Any,
    database: Any,
    outcome: str,
    user_score: float,
    target: float,
    minimum_games: int,
    max_step: int,
    min_strength: int,
    max_strength: int,
    starting_strength: int,
    alpha: float,
) -> None:
    # Seed the row first so the SELECT ... FOR UPDATE below always locks a
    # real row; otherwise two first-time completions could race and lose one
    # game's counts through concurrent upserts.
    await database.execute_raw(
        """
        INSERT INTO native_game_skill_states (
            id, user_id, agent_id, game_key, effective_strength,
            completed_games, ewma_user_rate, wins, losses, draws,
            created_at, updated_at
        ) VALUES (
            gen_random_uuid(), $1, $2, $3, $4, 0, NULL, 0, 0, 0,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
        ON CONFLICT (user_id, agent_id, game_key) DO NOTHING
        """,
        session.user_id,
        session.agent_id,
        session.game_key,
        max(0, min(100, starting_strength)),
    )
    rows = await database.query_raw(
        """
        SELECT effective_strength, completed_games, ewma_user_rate,
               wins, losses, draws
        FROM native_game_skill_states
        WHERE user_id = $1 AND agent_id = $2 AND game_key = $3
        FOR UPDATE
        """,
        session.user_id,
        session.agent_id,
        session.game_key,
    )
    previous = rows[0] if rows else {}
    completed = int(previous.get("completed_games") or 0) + 1
    previous_rate = previous.get("ewma_user_rate")
    ewma = (
        user_score
        if previous_rate is None
        else float(previous_rate) * (1 - alpha) + user_score * alpha
    )
    current_strength = _int_field(
        previous.get("effective_strength"), starting_strength
    )
    step = 0
    if completed >= minimum_games:
        # A 25-point win-rate gap produces roughly a 5-point adjustment.
        step = round((ewma - target) * 20)
        step = max(-max_step, min(max_step, step))
    next_strength = max(
        min_strength,
        min(max_strength, current_strength + step),
    )
    wins = int(previous.get("wins") or 0) + (1 if outcome == "win" else 0)
    losses = int(previous.get("losses") or 0) + (1 if outcome == "lose" else 0)
    draws = int(previous.get("draws") or 0) + (1 if outcome == "draw" else 0)
    await database.execute_raw(
        """
        UPDATE native_game_skill_states SET
            effective_strength = $4,
            completed_games = $5,
            ewma_user_rate = $6,
            wins = $7,
            losses = $8,
            draws = $9,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND agent_id = $2 AND game_key = $3
        """,
        session.user_id,
        session.agent_id,
        session.game_key,
        next_strength,
        completed,
        ewma,
        wins,
        losses,
        draws,
    )


def ai_level_for_strength(strength: int) -> int:
    if strength >= 67:
        return 3
    if strength >= 34:
        return 2
    return 1


def _lerp_int(low: int, high: int, strength: int) -> int:
    return round(low + (high - low) * max(0, min(100, strength)) / 100)


def _lerp(low: float, high: float, strength: int) -> float:
    return low + (high - low) * max(0, min(100, strength)) / 100


def build_engine_config(
    game_key: str,
    strength: int,
    overrides: dict[str, Any] | None = None,
    min_response_ms: int = DEFAULT_MIN_RESPONSE_MS,
    max_response_ms: int = DEFAULT_MAX_RESPONSE_MS,
) -> dict[str, Any]:
    s = max(0, min(100, int(strength)))
    if game_key == "gomoku":
        config = {
            "search_time_ms": _lerp_int(80, 500, s),
            "max_depth": _lerp_int(2, 5, s),
            "root_candidate_limit": _lerp_int(8, 18, s),
            "near_best_probability": round(_lerp(0.30, 0.04, s), 3),
            "near_best_tolerance": _lerp_int(1200, 180, s),
        }
    elif game_key == "reversi":
        config = {
            "search_time_ms": _lerp_int(180, 1200, s),
            "opening_depth": _lerp_int(3, 9, s),
            "midgame_depth": _lerp_int(5, 11, s),
            "exact_solve_empty": _lerp_int(6, 16, s),
            "near_best_probability": round(_lerp(0.28, 0.03, s), 3),
            "near_best_tolerance": _lerp_int(180, 12, s),
        }
    elif game_key == "go":
        config = {
            "search_time_ms": _lerp_int(160, 1200, s),
            "minimum_simulations": _lerp_int(20, 140, s),
            "exploration_constant": round(_lerp(1.55, 1.18, s), 3),
            "branch_limit": _lerp_int(8, 20, s),
            "rollout_depth": _lerp_int(16, 42, s),
            "move_temperature": round(_lerp(1.20, 0.08, s), 3),
        }
    elif game_key in {"chess", "xiangqi"}:
        is_chess = game_key == "chess"
        config = {
            "search_time_ms": _lerp_int(120, 900 if is_chess else 1000, s),
            "max_depth": _lerp_int(2, 6 if is_chess else 5, s),
            "quiescence_depth": _lerp_int(2, 6, s),
            "near_best_probability": round(_lerp(0.28, 0.03, s), 3),
            "near_best_tolerance": _lerp_int(120, 12, s),
        }
    elif game_key == "chinese_checkers":
        config = {
            "search_time_ms": _lerp_int(100, 700, s),
            "max_depth": _lerp_int(2, 5, s),
            "root_candidate_limit": _lerp_int(14, 40, s),
            "branch_limit": _lerp_int(10, 26, s),
            "near_best_probability": round(_lerp(0.30, 0.04, s), 3),
            "near_best_tolerance": _lerp_int(12, 2, s),
        }
    elif game_key == "match3":
        config = {
            "turn_limit": _lerp_int(34, 26, s),
            "target_score": _lerp_int(9000, 15000, s),
            "agent_choice_percentile": round(_lerp(1.0, 0.76, s), 3),
        }
    elif game_key == "minesweeper":
        config = {
            "rows": 9,
            "columns": 9,
            "mine_count": _lerp_int(10, 18, s),
            "require_no_guess": True,
            "generation_attempts": 360,
        }
    elif game_key == "number_merge":
        config = {
            "target": 2048,
            "search_depth_offset": _lerp_int(1, -2, s),
            "near_best_probability": round(_lerp(0.03, 0.28, s), 3),
            "near_best_tolerance_ratio": round(_lerp(0.01, 0.12, s), 3),
        }
    elif game_key == "tetris_duel":
        config = {
            "duration_seconds": 90,
            "agent_move_ms": _lerp_int(1050, 400, s),
            "near_best_probability": round(_lerp(0.30, 0.02, s), 3),
            "near_best_tolerance": round(_lerp(2.8, 0.2, s), 2),
        }
    else:
        raise ValueError("unsupported_game")
    for key, value in (overrides or {}).items():
        if key not in config:
            continue
        expected = config[key]
        if (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and isinstance(value, float)
            and value.is_integer()
        ):
            value = int(value)
        config[key] = value
    # First-class pacing range (not algorithm overrides): set last so nothing
    # in `overrides` can touch it. The client reads both from engine_config and
    # picks a random delay in [min, max] per move so the AI never feels robotic.
    lo = max(0, int(min_response_ms))
    hi = max(lo, int(max_response_ms))
    config["min_response_ms"] = lo
    config["max_response_ms"] = hi
    config["strength"] = s
    return config


def config_payload(config: GameBalanceConfig) -> dict[str, Any]:
    return {
        "game_key": config.game_key,
        "title": GAME_TITLES[config.game_key],
        "play_mode": (
            "cooperate" if config.game_key in COOPERATIVE_GAMES else "versus"
        ),
        "mode": config.mode,
        "base_strength": config.base_strength,
        "min_strength": config.min_strength,
        "max_strength": config.max_strength,
        "target_user_rate": config.target_user_rate,
        "adjustment_window": config.adjustment_window,
        "minimum_games": config.minimum_games,
        "maximum_step": config.maximum_step,
        "algorithm_overrides": config.algorithm_overrides,
        "min_response_ms": config.min_response_ms,
        "max_response_ms": config.max_response_ms,
        "enabled": config.enabled,
        "version": config.version,
        "preview_engine_config": build_engine_config(
            config.game_key,
            config.base_strength,
            config.algorithm_overrides,
            config.min_response_ms,
            config.max_response_ms,
        ),
    }


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    completed = int(metrics.get("completed") or 0)
    wins = int(metrics.get("wins") or 0)
    draws = int(metrics.get("draws") or 0)
    return {
        "completed_30d": completed,
        "wins_30d": wins,
        "losses_30d": int(metrics.get("losses") or 0),
        "draws_30d": draws,
        "user_rate_30d": ((wins + draws * 0.5) / completed) if completed else None,
    }


async def _load_metrics_by_game() -> dict[str, dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT game_key,
               COUNT(*) FILTER (WHERE status = 'settled')::int AS completed,
               COUNT(*) FILTER (
                   WHERE status = 'settled'
                     AND result->>'user_outcome' = 'win'
               )::int AS wins,
               COUNT(*) FILTER (
                   WHERE status = 'settled'
                     AND result->>'user_outcome' = 'lose'
               )::int AS losses,
               COUNT(*) FILTER (
                   WHERE status = 'settled'
                     AND result->>'user_outcome' = 'draw'
               )::int AS draws
        FROM game_sessions
        WHERE provider = 'native'
          AND created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
        GROUP BY game_key
        """
    )
    return {str(row.get("game_key")): row for row in rows}


async def list_admin_configs() -> list[dict[str, Any]]:
    config_rows, metric_rows = await asyncio.gather(
        db.query_raw(
            """
            SELECT game_key, mode, base_strength, min_strength, max_strength,
                   target_user_rate, adjustment_window, minimum_games,
                   maximum_step, algorithm_overrides, min_response_ms,
                   max_response_ms, enabled, version
            FROM native_game_configs
            """
        ),
        _load_metrics_by_game(),
    )
    configs = {str(row.get("game_key")): row for row in config_rows}
    result = []
    for game_key in GAME_TITLES:
        config = _row_config(configs.get(game_key), game_key)
        payload = config_payload(config)
        payload["metrics"] = _metrics_payload(metric_rows.get(game_key, {}))
        result.append(payload)
    return result


async def get_admin_config(game_key: str) -> dict[str, Any]:
    metrics_by_game, config = await asyncio.gather(
        _load_metrics_by_game(),
        get_config(game_key),
    )
    payload = config_payload(config)
    payload["metrics"] = _metrics_payload(metrics_by_game.get(game_key, {}))
    return payload


async def set_enabled(game_key: str, enabled: bool) -> dict[str, Any]:
    """Toggle a game's client visibility without bumping the balance version.

    Visibility is an operational on/off switch, not a difficulty change, so it
    deliberately does NOT write a config version snapshot. Other config columns
    fall back to their table defaults when the row is created for the first time.
    """
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    await db.execute_raw(
        """
        INSERT INTO native_game_configs (game_key, enabled)
        VALUES ($1, $2)
        ON CONFLICT (game_key) DO UPDATE SET
            enabled = EXCLUDED.enabled,
            updated_at = CURRENT_TIMESTAMP
        """,
        game_key,
        enabled,
    )
    return await get_admin_config(game_key)


async def list_public_catalog() -> list[dict[str, Any]]:
    """Game visibility for the client hub: every game with its enabled flag.

    Games without a config row default to enabled so a fresh install shows the
    full catalog before any admin edits.
    """
    rows = await db.query_raw(
        "SELECT game_key, enabled FROM native_game_configs"
    )
    enabled_by_game = {
        str(row.get("game_key")): _bool_field(row.get("enabled"), True) for row in rows
    }
    return [
        {
            "game_key": game_key,
            "title": GAME_TITLES[game_key],
            "enabled": enabled_by_game.get(game_key, True),
        }
        for game_key in GAME_TITLES
    ]


async def publish_config(game_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    validate_algorithm_overrides(
        game_key,
        _load_json(payload.get("algorithm_overrides")),
    )
    async with db.tx() as tx:
        rows = await tx.query_raw(
            "SELECT version FROM native_game_configs WHERE game_key = $1 FOR UPDATE",
            game_key,
        )
        version = int(rows[0].get("version") or 0) + 1 if rows else 1
        config_json = {"game_key": game_key, **payload, "version": version}
        await tx.execute_raw(
            """
            INSERT INTO native_game_config_versions (
                id, game_key, version, config, published_at
            ) VALUES (gen_random_uuid(), $1, $2, $3::jsonb, CURRENT_TIMESTAMP)
            """,
            game_key,
            version,
            json.dumps(config_json, ensure_ascii=False),
        )
        await tx.execute_raw(
            """
            INSERT INTO native_game_configs (
                game_key, mode, base_strength, min_strength, max_strength,
                target_user_rate, adjustment_window, minimum_games,
                maximum_step, algorithm_overrides, min_response_ms,
                max_response_ms, version, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            ON CONFLICT (game_key) DO UPDATE SET
                mode = EXCLUDED.mode,
                base_strength = EXCLUDED.base_strength,
                min_strength = EXCLUDED.min_strength,
                max_strength = EXCLUDED.max_strength,
                target_user_rate = EXCLUDED.target_user_rate,
                adjustment_window = EXCLUDED.adjustment_window,
                minimum_games = EXCLUDED.minimum_games,
                maximum_step = EXCLUDED.maximum_step,
                algorithm_overrides = EXCLUDED.algorithm_overrides,
                min_response_ms = EXCLUDED.min_response_ms,
                max_response_ms = EXCLUDED.max_response_ms,
                version = EXCLUDED.version,
                updated_at = CURRENT_TIMESTAMP
            """,
            game_key,
            payload["mode"],
            payload["base_strength"],
            payload["min_strength"],
            payload["max_strength"],
            payload["target_user_rate"],
            payload["adjustment_window"],
            payload["minimum_games"],
            payload["maximum_step"],
            json.dumps(payload.get("algorithm_overrides") or {}, ensure_ascii=False),
            _int_field(payload.get("min_response_ms"), DEFAULT_MIN_RESPONSE_MS),
            _int_field(payload.get("max_response_ms"), DEFAULT_MAX_RESPONSE_MS),
            version,
        )
    return await get_admin_config(game_key)


async def list_versions(game_key: str, limit: int = 20) -> list[dict[str, Any]]:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    rows = await db.query_raw(
        """
        SELECT version, config, published_at
        FROM native_game_config_versions
        WHERE game_key = $1
        ORDER BY version DESC
        LIMIT $2
        """,
        game_key,
        limit,
    )
    return [
        {
            "version": int(row.get("version") or 0),
            "config": _load_json(row.get("config")),
            "published_at": row.get("published_at"),
        }
        for row in rows
    ]


async def get_version(game_key: str, version: int) -> dict[str, Any]:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    rows = await db.query_raw(
        """
        SELECT config
        FROM native_game_config_versions
        WHERE game_key = $1 AND version = $2
        LIMIT 1
        """,
        game_key,
        version,
    )
    if not rows:
        raise ValueError("config_version_not_found")
    return _load_json(rows[0].get("config"))


_ALGORITHM_RANGES: dict[str, dict[str, tuple[float, float] | None]] = {
    "gomoku": {
        "search_time_ms": (40, 3000),
        "max_depth": (1, 8),
        "root_candidate_limit": (4, 30),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 5000),
    },
    "reversi": {
        "search_time_ms": (40, 5000),
        "opening_depth": (1, 14),
        "midgame_depth": (1, 16),
        "exact_solve_empty": (0, 24),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 1000),
    },
    "go": {
        "search_time_ms": (40, 5000),
        "minimum_simulations": (1, 500),
        "exploration_constant": (0.1, 4),
        "branch_limit": (2, 40),
        "rollout_depth": (4, 100),
        "move_temperature": (0, 3),
    },
    "chess": {
        "search_time_ms": (40, 5000),
        "max_depth": (1, 10),
        "quiescence_depth": (0, 10),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 1000),
    },
    "xiangqi": {
        "search_time_ms": (40, 5000),
        "max_depth": (1, 8),
        "quiescence_depth": (0, 10),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 1000),
    },
    "chinese_checkers": {
        "search_time_ms": (40, 3000),
        "max_depth": (1, 8),
        "root_candidate_limit": (4, 60),
        "branch_limit": (4, 50),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 100),
    },
    "match3": {
        "turn_limit": (5, 100),
        "target_score": (1000, 1000000),
        "agent_choice_percentile": (0, 1),
    },
    "minesweeper": {
        # Lower bound 6 matches the Flutter engine's `rows >= 6` assert;
        # generation runs on the client's main thread, so cap the attempts.
        "rows": (6, 20),
        "columns": (6, 20),
        "mine_count": (1, 100),
        "require_no_guess": None,
        "generation_attempts": (1, 720),
    },
    "number_merge": {
        "target": (128, 65536),
        "search_depth_offset": (-4, 4),
        "near_best_probability": (0, 1),
        "near_best_tolerance_ratio": (0, 1),
    },
    "tetris_duel": {
        "duration_seconds": (15, 600),
        "agent_move_ms": (100, 5000),
        "near_best_probability": (0, 1),
        "near_best_tolerance": (0, 20),
    },
}


def validate_algorithm_overrides(game_key: str, overrides: dict[str, Any]) -> None:
    if game_key not in _ALGORITHM_RANGES:
        raise ValueError("unsupported_game")
    defaults = build_engine_config(game_key, 50)
    ranges = _ALGORITHM_RANGES[game_key]
    unknown = set(overrides) - set(ranges)
    if unknown:
        raise ValueError(f"unknown_algorithm_parameter:{sorted(unknown)[0]}")
    for key, value in overrides.items():
        expected = defaults[key]
        if isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(f"invalid_algorithm_parameter:{key}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid_algorithm_parameter:{key}")
        if not math.isfinite(float(value)):
            raise ValueError(f"invalid_algorithm_parameter:{key}")
        if (
            isinstance(expected, int)
            and not isinstance(value, int)
            # JSON round-trips often turn ints into floats (500 -> 500.0).
            and not float(value).is_integer()
        ):
            raise ValueError(f"invalid_algorithm_parameter:{key}")
        limits = ranges[key]
        if limits is not None and not limits[0] <= float(value) <= limits[1]:
            raise ValueError(f"invalid_algorithm_parameter:{key}")
    if game_key == "minesweeper":
        # Non-overridden values scale with the live strength at session time,
        # so cross-field checks must hold at the strength-100 worst case, not
        # just the strength-50 defaults used for type checking above.
        worst = build_engine_config(game_key, 100)
        rows = int(overrides.get("rows", worst["rows"]))
        columns = int(overrides.get("columns", worst["columns"]))
        mines = int(overrides.get("mine_count", worst["mine_count"]))
        if mines >= rows * columns - 9:
            raise ValueError("invalid_algorithm_parameter:mine_count")
