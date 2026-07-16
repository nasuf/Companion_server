from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from weakref import WeakValueDictionary

from app.db import db
from app.models.game import (
    NativeGameEventRecord,
    NativeSessionResponse,
    SudSessionResponse,
)
from app.services.games import sud
from app.services.memory.storage import repo as memory_repo
from app.services.offline.memory_hooks import remember_shared_game_experience
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

GOMOKU_GAME_KEY = "gomoku"
GOMOKU_BOARD_SIZE = 15


@dataclass(frozen=True)
class NativeGameDefinition:
    key: str
    title: str
    action_event: str
    play_mode: str = "versus"


_GAME_DEFINITIONS = {
    "go": NativeGameDefinition("go", "围棋", "stone_placed"),
    "reversi": NativeGameDefinition("reversi", "黑白棋", "disc_placed"),
    "gomoku": NativeGameDefinition("gomoku", "五子棋", "move_placed"),
    "xiangqi": NativeGameDefinition("xiangqi", "中国象棋", "piece_moved"),
    "chess": NativeGameDefinition("chess", "国际象棋", "piece_moved"),
    "chinese_checkers": NativeGameDefinition("chinese_checkers", "跳棋", "piece_moved"),
    "match3": NativeGameDefinition(
        "match3", "消消乐", "tiles_swapped", play_mode="cooperate"
    ),
    "minesweeper": NativeGameDefinition(
        "minesweeper", "协作扫雷", "cell_action", play_mode="cooperate"
    ),
    "number_merge": NativeGameDefinition(
        "number_merge", "数字合并", "board_slid", play_mode="cooperate"
    ),
}
_SUPPORTED_GAME_KEYS_SQL = ", ".join(f"'{key}'" for key in _GAME_DEFINITIONS)
_TERMINAL_OUTCOMES = {
    "go": {"userWon": "win", "agentWon": "lose", "draw": "draw"},
    "reversi": {"userWon": "win", "agentWon": "lose", "draw": "draw"},
    "xiangqi": {"userWon": "win", "agentWon": "lose", "draw": "draw"},
    "chess": {"userWon": "win", "agentWon": "lose", "draw": "draw"},
    "chinese_checkers": {"userWon": "win", "agentWon": "lose"},
    "match3": {"completed": "win", "failed": "lose"},
    "minesweeper": {"completed": "win", "failed": "lose"},
    "number_merge": {"completed": "win", "failed": "lose"},
}
_TERMINAL_STATUSES = {"settled", "aborted"}
_SESSION_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_MEMORY_SYNC_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_MEMORY_RETRY_DELAY = timedelta(minutes=5)
_MEMORY_SYNC_LEASE = timedelta(minutes=10)
_ALLOWED_EVENTS = {
    "game_started",
    "move_placed",
    "stone_placed",
    "disc_placed",
    "piece_moved",
    "tiles_swapped",
    "cell_action",
    "cells_revealed",
    "flag_toggled",
    "inference_made",
    "board_slid",
    "tiles_merged",
    "tile_spawned",
    "cascade_resolved",
    "special_created",
    "board_shuffled",
    "turn_changed",
    "key_moment",
    "game_state_snapshot",
    "ai_thinking_started",
    "ai_move_decided",
    "threat_detected",
    "invalid_move",
    "analysis_snapshot",
    "game_finished",
    "game_aborted",
    "game_restarted",
}


def _now() -> datetime:
    return datetime.now(UTC)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _with_supported_games(query: str) -> str:
    return query.replace("{supported_game_keys}", _SUPPORTED_GAME_KEYS_SQL)


def _loads(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


async def create_session(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str | None,
    game_key: str,
) -> NativeSessionResponse:
    definition = _GAME_DEFINITIONS.get(game_key)
    if definition is None:
        raise ValueError("unsupported_game")
    difficulty = "normal"
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "userId", None) != user_id:
        raise ValueError("agent_not_found")
    workspace_id, conversation_id = await sud._resolve_owned_context(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )

    session_id = str(uuid.uuid4())
    room_id = f"{game_key}-{session_id[:8]}"
    user_player = await sud.build_user_player(user_id)
    ai_player = await sud.build_ai_player(agent_id, difficulty, agent=agent)
    result = _empty_result(difficulty, definition)

    await db.execute_raw(
        """
        INSERT INTO game_sessions (
            id, provider, game_key, status, user_id, agent_id, workspace_id,
            conversation_id, mg_id, room_id, play_mode, difficulty, ai_level,
            sdk_enabled, sud_code, sud_code_expires_at, user_player, ai_player,
            companion_reply, result
        )
        VALUES (
            $1, 'native', $2, 'created', $3, $4, $5,
            $6, '', $7, $8, $9, $10,
            FALSE, NULL, NULL, $11::jsonb, $12::jsonb,
            $13, $14::jsonb
        )
        """,
        session_id,
        game_key,
        user_id,
        agent_id,
        workspace_id,
        conversation_id,
        room_id,
        definition.play_mode,
        difficulty,
        ai_player.ai_level,
        user_player.model_dump_json(),
        ai_player.model_dump_json(),
        None,
        _json(result),
    )
    await sud._append_event(
        session_id=session_id,
        event_type="session_created",
        state="created",
        payload={
            "game_key": game_key,
            "game_title": definition.title,
            "difficulty": difficulty,
            "play_style": "natural_companion",
            **(
                {"board_size": GOMOKU_BOARD_SIZE} if game_key == GOMOKU_GAME_KEY else {}
            ),
        },
        source="server",
        companion_reply=None,
    )
    return await get_session(session_id, user_id=user_id)


async def list_sessions(
    user_id: str,
    *,
    game_key: str | None = None,
    limit: int = 50,
) -> list[NativeSessionResponse]:
    if game_key and game_key not in _GAME_DEFINITIONS:
        return []
    if game_key:
        rows = await db.query_raw(
            """
            SELECT * FROM game_sessions
            WHERE user_id = $1 AND provider = 'native' AND game_key = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id,
            game_key,
            limit,
        )
    else:
        rows = await db.query_raw(
            _with_supported_games(
                """
            SELECT * FROM game_sessions
            WHERE user_id = $1
              AND provider = 'native'
              AND game_key IN ({supported_game_keys})
            ORDER BY created_at DESC
            LIMIT $2
            """
            ),
            user_id,
            limit,
        )
    return [_as_native_session(sud._row_to_session(row)) for row in rows]


async def get_session(
    session_id: str,
    *,
    user_id: str | None = None,
    database: Any | None = None,
    for_update: bool = False,
) -> NativeSessionResponse:
    executor = database or db
    lock_clause = "FOR UPDATE" if for_update else ""
    if user_id:
        rows = await executor.query_raw(
            f"""
            SELECT * FROM game_sessions
            WHERE id = $1 AND user_id = $2 AND provider = 'native'
            LIMIT 1
            {lock_clause}
            """,
            session_id,
            user_id,
        )
    else:
        rows = await executor.query_raw(
            f"""
            SELECT * FROM game_sessions
            WHERE id = $1 AND provider = 'native'
            LIMIT 1
            {lock_clause}
            """,
            session_id,
        )
    if not rows:
        raise ValueError("session_not_found")
    return _as_native_session(sud._row_to_session(rows[0]))


async def delete_session(session_id: str, *, user_id: str) -> None:
    # Share the memory-delivery lock so a terminal session cannot create a
    # fresh shared memory while its game record is being removed.
    lock = _MEMORY_SYNC_LOCKS.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = await get_session(session_id, user_id=user_id)
        memory_sync = _loads(_loads(session.result, {}).get("memory_sync"), {})
        for source, id_key in (
            ("user", "user_memory_id"),
            ("ai", "ai_memory_id"),
        ):
            memory_id = str(memory_sync.get(id_key) or "").strip()
            if memory_id:
                await memory_repo.delete(memory_id, source=source)

        deleted = await db.execute_raw(
            """
            DELETE FROM game_sessions
            WHERE id = $1 AND user_id = $2 AND provider = 'native'
            """,
            session_id,
            user_id,
        )
        if deleted == 0:
            raise ValueError("session_not_found")


async def list_events(
    session_id: str,
    *,
    user_id: str,
    limit: int = 500,
) -> list[NativeGameEventRecord]:
    await get_session(session_id, user_id=user_id)
    rows = await db.query_raw(
        """
        SELECT id, event_type, state, source, payload, companion_reply, created_at
        FROM game_events
        WHERE session_id = $1
        ORDER BY created_at ASC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    return [
        NativeGameEventRecord(
            id=str(row["id"]),
            event_type=str(row.get("event_type") or ""),
            state=row.get("state"),
            source=str(row.get("source") or "client"),
            payload=_loads(row.get("payload"), {}),
            companion_reply=row.get("companion_reply"),
            created_at=sud._iso(row.get("created_at")) or "",
        )
        for row in rows
    ]


async def handle_event(
    *,
    session_id: str,
    user_id: str,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    source: str,
    client_event_id: str | None,
) -> tuple[NativeSessionResponse, str | None, str | None, bool]:
    if event_type not in _ALLOWED_EVENTS:
        raise ValueError("unsupported_event")
    lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
    async with lock:
        if client_event_id:
            existing = await _find_event_by_client_id(session_id, client_event_id)
            if existing:
                session = await get_session(session_id, user_id=user_id)
                event_id, existing_reply = existing
                await _ensure_idempotent_side_effects(
                    session,
                    event_type,
                    state,
                    payload,
                    existing_reply,
                )
                return session, existing_reply, event_id, True
        async with db.tx() as tx:
            session = await get_session(
                session_id,
                user_id=user_id,
                database=tx,
                for_update=True,
            )
            if client_event_id:
                existing = await _find_event_by_client_id(
                    session_id,
                    client_event_id,
                    database=tx,
                )
                if existing:
                    event_id, existing_reply = existing
                    return session, existing_reply, event_id, True
            if session.status in _TERMINAL_STATUSES:
                if event_type in {"game_finished", "game_aborted"}:
                    return session, None, None, True
                raise ValueError("session_finished")

            definition = _definition(session.game_key)
            event_payload = {
                **payload,
                "schema_version": int(payload.get("schema_version") or 1),
                "game_key": definition.key,
                "game_title": definition.title,
            }
            if client_event_id:
                event_payload["client_event_id"] = client_event_id

            previous = session
            result = _loads(
                session.result,
                _empty_result(session.difficulty, definition),
            )
            result = _with_event_count(result, event_type)
            status = session.status
            started_at = session.started_at
            ended_at = session.ended_at
            duration_seconds = session.duration_seconds
            reply: str | None = None

            if event_type == "game_started":
                if session.status != "created":
                    raise ValueError("invalid_state")
                status = "playing"
                started_at = _now().isoformat()
                result = _store_initial_state(result, definition, event_payload)
            elif event_type == definition.action_event:
                if session.status != "playing":
                    raise ValueError("invalid_state")
                if definition.key == GOMOKU_GAME_KEY:
                    action = _validate_and_normalize_move(result, event_payload)
                    result = _append_move(result, action)
                else:
                    action = _validate_generic_action(result, definition, event_payload)
                    result = _append_generic_action(result, definition, action)
            elif event_type == "game_finished":
                if session.status != "playing":
                    raise ValueError("invalid_state")
                if definition.key == GOMOKU_GAME_KEY:
                    result = _reconcile_reported_moves(result, event_payload)
                    outcome, winning_line = _validated_outcome(result, event_payload)
                else:
                    result = _reconcile_reported_actions(
                        result,
                        definition,
                        event_payload,
                    )
                    outcome = _validated_generic_outcome(event_payload, definition)
                    winning_line = []
                status = "settled"
                ended_at = _now().isoformat()
                duration_seconds = _duration(event_payload, session)
                if definition.key == GOMOKU_GAME_KEY:
                    result = _finish_result(
                        result,
                        event_payload,
                        outcome=outcome,
                        winning_line=winning_line,
                        duration_seconds=duration_seconds,
                    )
                    reply = _finish_reply(session, result)
                else:
                    result = _finish_generic_result(
                        result,
                        definition,
                        event_payload,
                        outcome=outcome,
                        duration_seconds=duration_seconds,
                    )
                    reply = _generic_finish_reply(session, definition, result)
            elif event_type == "game_aborted":
                if definition.key == GOMOKU_GAME_KEY:
                    result = _reconcile_reported_moves(result, event_payload)
                else:
                    result = _reconcile_reported_actions(
                        result,
                        definition,
                        event_payload,
                    )
                status = "aborted"
                ended_at = _now().isoformat()
                duration_seconds = _duration(event_payload, session)
                if definition.key == GOMOKU_GAME_KEY:
                    result = _abort_result(result, event_payload, duration_seconds)
                    reply = _abort_reply(session, result)
                else:
                    result = _abort_generic_result(
                        result,
                        definition,
                        event_payload,
                        duration_seconds,
                    )
                    reply = _generic_abort_reply(definition, result)
            else:
                result = _merge_auxiliary_event(result, event_type, event_payload)

            if event_type in {"game_finished", "game_aborted"}:
                result = _with_pending_memory_sync(result)

            await _update_session(
                session_id=session.id,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration_seconds,
                result=result,
                companion_reply=reply,
                database=tx,
            )
            event_id, inserted, stored_reply = await sud._append_event_idempotent(
                session_id=session.id,
                event_type=event_type,
                state=state,
                payload=event_payload,
                source=source,
                companion_reply=reply,
                client_event_id=client_event_id,
                database=tx,
            )
            updated = await get_session(
                session.id,
                user_id=user_id,
                database=tx,
            )
        if not inserted:
            return updated, stored_reply, event_id, True
        await sud._persist_game_status_to_chat_if_needed(
            previous,
            updated,
            event_type,
            state,
            event_payload,
        )
        if event_type in {"game_finished", "game_aborted"}:
            await sud._persist_reply_to_chat_if_needed(
                updated,
                event_type,
                state,
                reply,
            )
            # Memory embedding availability must not hold the game result UI
            # hostage. The pending marker lives in game_sessions, so the
            # scheduler can recover the shared memory even after a restart.
            fire_background(sync_session_memory(updated.id))
        return updated, reply, event_id, False


async def _ensure_idempotent_side_effects(
    session: NativeSessionResponse,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    reply: str | None,
) -> None:
    definition = _definition(session.game_key)
    event_payload = {
        **payload,
        "game_key": definition.key,
        "game_title": definition.title,
    }
    if event_type == "game_started" and session.status == "playing":
        previous = session.model_copy(update={"status": "created"})
        await sud._persist_game_status_to_chat_if_needed(
            previous,
            session,
            event_type,
            state,
            event_payload,
        )
    elif event_type in {"game_finished", "game_aborted"} and (
        session.status in _TERMINAL_STATUSES
    ):
        previous = session.model_copy(update={"status": "playing"})
        await sud._persist_game_status_to_chat_if_needed(
            previous,
            session,
            event_type,
            state,
            event_payload,
        )
        await sud._persist_reply_to_chat_if_needed(
            session,
            event_type,
            state,
            reply,
        )
        memory_sync = _loads(_loads(session.result, {}).get("memory_sync"), {})
        if memory_sync.get("status") not in {"stored", "deduplicated", "skipped"}:
            fire_background(sync_session_memory(session.id))


def _as_native_session(session: SudSessionResponse) -> NativeSessionResponse:
    if session.game_key not in _GAME_DEFINITIONS:
        raise ValueError("session_not_found")
    return NativeSessionResponse(
        id=session.id,
        game_key=session.game_key,
        status=session.status,
        user_id=session.user_id,
        agent_id=session.agent_id,
        workspace_id=session.workspace_id,
        conversation_id=session.conversation_id,
        room_id=session.room_id,
        play_mode=session.play_mode,
        difficulty="normal",
        ai_level=session.ai_level,
        user_player=session.user_player,
        ai_player=session.ai_player,
        companion_reply=session.companion_reply,
        result=session.result,
        duration_seconds=session.duration_seconds,
        started_at=session.started_at,
        ended_at=session.ended_at,
        created_at=session.created_at,
    )


def _with_pending_memory_sync(result: dict[str, Any]) -> dict[str, Any]:
    existing = dict(_loads(result.get("memory_sync"), {}))
    if existing.get("status") in {"stored", "deduplicated", "skipped"}:
        return result
    return {
        **result,
        "memory_sync": {
            "status": "pending",
            "user_memory_id": existing.get("user_memory_id"),
            "ai_memory_id": existing.get("ai_memory_id"),
            "failed_sides": existing.get("failed_sides") or ["user", "ai"],
            "attempts": int(existing.get("attempts") or 0),
            "next_retry_at": _now().isoformat(),
        },
    }


def _merge_memory_sync(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    user_memory_id = current.get("user_memory_id") or previous.get("user_memory_id")
    ai_memory_id = current.get("ai_memory_id") or previous.get("ai_memory_id")
    failed_sides = list(current.get("failed_sides") or [])
    memory_ids = [
        memory_id for memory_id in (user_memory_id, ai_memory_id) if memory_id
    ]
    if current.get("status") == "skipped" and not memory_ids:
        status = "skipped"
    elif failed_sides and memory_ids:
        status = "partial"
    elif failed_sides:
        status = "failed"
    elif memory_ids:
        status = "stored"
    else:
        status = "deduplicated"
    merged = {
        "status": status,
        "user_memory_id": user_memory_id,
        "ai_memory_id": ai_memory_id,
        "failed_sides": failed_sides,
        "attempts": int(previous.get("attempts") or 0) + 1,
        "last_attempt_at": _now().isoformat(),
    }
    if status in {"failed", "partial"}:
        merged["next_retry_at"] = (_now() + _MEMORY_RETRY_DELAY).isoformat()
    return merged


async def sync_session_memory(session_id: str) -> dict[str, Any]:
    """Deliver one terminal session to both memory stores, idempotently."""

    lock = _MEMORY_SYNC_LOCKS.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = await _claim_memory_sync(session_id)
        if session is None:
            current = await get_session(session_id)
            return dict(_loads(_loads(current.result, {}).get("memory_sync"), {}))
        result = _loads(session.result, {})
        previous = dict(_loads(result.get("memory_sync"), {}))
        missing_sides = tuple(
            side
            for side, id_key in (
                ("user", "user_memory_id"),
                ("ai", "ai_memory_id"),
            )
            if not previous.get(id_key)
        )
        current = await _remember_shared_experience(
            session,
            result,
            sides=missing_sides,
        )
        memory_sync = _merge_memory_sync(previous, current)
        result = {**result, "memory_sync": memory_sync}
        await _update_session(
            session_id=session.id,
            status=session.status,
            started_at=session.started_at,
            ended_at=session.ended_at,
            duration_seconds=session.duration_seconds,
            result=result,
            companion_reply=None,
        )
        logger.info(
            "Native game memory sync session=%s status=%s attempts=%s",
            session.id[:8],
            memory_sync["status"],
            memory_sync["attempts"],
        )
        return memory_sync


async def _claim_memory_sync(session_id: str) -> NativeSessionResponse | None:
    """Atomically lease one terminal session to a single worker."""

    lease_until = (_now() + _MEMORY_SYNC_LEASE).isoformat()
    claimed_at = _now().isoformat()
    rows = await db.query_raw(
        _with_supported_games(
            """
        UPDATE game_sessions
        SET result = jsonb_set(
                COALESCE(result, '{}'::jsonb),
                '{memory_sync}',
                COALESCE(result->'memory_sync', '{}'::jsonb)
                    || jsonb_build_object(
                        'status', 'syncing',
                        'lease_until', $2::text,
                        'last_claimed_at', $3::text
                    ),
                TRUE
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND provider = 'native'
          AND game_key IN ({supported_game_keys})
          AND status IN ('settled', 'aborted')
          AND (
                COALESCE(result->'memory_sync'->>'status', 'pending')
                    IN ('pending', 'failed', 'partial')
                OR (
                    result->'memory_sync'->>'status' = 'syncing'
                    AND COALESCE(
                        (result->'memory_sync'->>'lease_until')::timestamptz,
                        TO_TIMESTAMP(0)
                    ) <= CURRENT_TIMESTAMP
                )
              )
          AND COALESCE(
                (result->'memory_sync'->>'next_retry_at')::timestamptz,
                TO_TIMESTAMP(0)
              ) <= CURRENT_TIMESTAMP
        RETURNING *
        """
        ),
        session_id,
        lease_until,
        claimed_at,
    )
    if not rows:
        return None
    return _as_native_session(sud._row_to_session(rows[0]))


async def retry_pending_memory_sync(*, limit: int = 10) -> int:
    """Retry durable native-game memory deliveries due at this moment."""

    rows = await db.query_raw(
        _with_supported_games(
            """
        SELECT id
        FROM game_sessions
        WHERE provider = 'native'
          AND game_key IN ({supported_game_keys})
          AND status IN ('settled', 'aborted')
          AND COALESCE(result->'memory_sync'->>'status', 'pending')
              IN ('pending', 'failed', 'partial', 'syncing')
          AND (
                result->'memory_sync'->>'status' != 'syncing'
                OR COALESCE(
                    (result->'memory_sync'->>'lease_until')::timestamptz,
                    TO_TIMESTAMP(0)
                ) <= CURRENT_TIMESTAMP
              )
          AND COALESCE(
                (result->'memory_sync'->>'next_retry_at')::timestamptz,
                TO_TIMESTAMP(0)
              ) <= CURRENT_TIMESTAMP
        ORDER BY updated_at ASC
        LIMIT $1
        """
        ),
        limit,
    )
    if not rows:
        return 0
    sem = asyncio.Semaphore(2)

    async def _retry(session_id: str) -> None:
        async with sem:
            try:
                await sync_session_memory(session_id)
            except Exception:
                logger.exception(
                    "Native game memory retry crashed session=%s",
                    session_id[:8],
                )

    await asyncio.gather(*[_retry(str(row["id"])) for row in rows])
    return len(rows)


async def abort_stale_sessions(
    *, stale_after_minutes: int = 10, limit: int = 20
) -> int:
    """Close inactive games after the client has had time to flush an exit."""

    rows = await db.query_raw(
        _with_supported_games(
            """
        SELECT id, user_id, result
        FROM game_sessions
        WHERE provider = 'native'
          AND status IN ('created', 'playing')
          AND game_key IN ({supported_game_keys})
          AND updated_at <= CURRENT_TIMESTAMP - ($1 * INTERVAL '1 minute')
        ORDER BY updated_at ASC
        LIMIT $2
        """
        ),
        stale_after_minutes,
        limit,
    )
    closed = 0
    for row in rows:
        session_id = str(row["id"])
        result = _loads(row.get("result"), {})
        game_key = str(result.get("game_key") or GOMOKU_GAME_KEY)
        winner = None
        recovered_terminal: tuple[str, str, dict[str, Any]] | None = None
        if game_key == GOMOKU_GAME_KEY:
            winner, _ = _winner(list(_loads(_gomoku(result).get("moves"), [])))
        else:
            definition = _GAME_DEFINITIONS.get(game_key)
            if definition is not None:
                recovered_terminal = _recover_generic_terminal(result, definition)
        if winner or recovered_terminal:
            event_type = "game_finished"
            state = "settled"
            if recovered_terminal is None:
                payload = {
                    "user_outcome": "win" if winner == "user" else "lose",
                    "reason": "client_disconnected_after_finish",
                }
            else:
                outcome, terminal_status, final_state = recovered_terminal
                payload = {
                    "user_outcome": outcome,
                    "reason": "client_disconnected_after_finish",
                    "terminal_state": {"status": terminal_status},
                    "final_state": final_state,
                    "state_after_hash": final_state.get("state_hash"),
                }
            client_event_id = f"server-timeout-finish:{session_id}"
        else:
            event_type = "game_aborted"
            state = "aborted"
            payload = {"reason": "client_disconnected_timeout"}
            client_event_id = f"server-timeout-abort:{session_id}"
        try:
            _, _, _, duplicate = await handle_event(
                session_id=session_id,
                user_id=str(row["user_id"]),
                event_type=event_type,
                state=state,
                payload=payload,
                source="server",
                client_event_id=client_event_id,
            )
            if not duplicate:
                closed += 1
        except ValueError as exc:
            if str(exc) not in {"session_finished", "session_not_found"}:
                logger.warning(
                    "Failed to close stale native game session=%s: %s",
                    session_id[:8],
                    exc,
                )
    return closed


async def _find_event_by_client_id(
    session_id: str,
    client_event_id: str,
    *,
    database: Any | None = None,
) -> tuple[str, str | None] | None:
    executor = database or db
    rows = await executor.query_raw(
        """
        SELECT id, companion_reply FROM game_events
        WHERE session_id = $1 AND client_event_id = $2
        LIMIT 1
        """,
        session_id,
        client_event_id,
    )
    if not rows:
        return None
    return str(rows[0]["id"]), rows[0].get("companion_reply")


async def _update_session(
    *,
    session_id: str,
    status: str,
    started_at: str | None,
    ended_at: str | None,
    duration_seconds: int | None,
    result: dict[str, Any],
    companion_reply: str | None,
    database: Any | None = None,
) -> None:
    executor = database or db
    await executor.execute_raw(
        """
        UPDATE game_sessions
        SET status = $2,
            started_at = $3::timestamptz,
            ended_at = $4::timestamptz,
            duration_seconds = $5,
            result = $6::jsonb,
            companion_reply = COALESCE($7, companion_reply),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND provider = 'native'
        """,
        session_id,
        status,
        started_at,
        ended_at,
        duration_seconds,
        _json(result),
        companion_reply,
    )


def _definition(game_key: str | None) -> NativeGameDefinition:
    definition = _GAME_DEFINITIONS.get(str(game_key or ""))
    if definition is None:
        raise ValueError("unsupported_game")
    return definition


def _empty_result(
    difficulty: str,
    definition: NativeGameDefinition | None = None,
) -> dict[str, Any]:
    definition = definition or _GAME_DEFINITIONS[GOMOKU_GAME_KEY]
    if definition.key == GOMOKU_GAME_KEY:
        game_process: dict[str, Any] = {
            "moves": [],
            "move_count": 0,
            "user_moves": 0,
            "ai_moves": 0,
            "key_moments": [],
        }
    else:
        game_process = {
            "actions": [],
            "action_count": 0,
            "user_actions": 0,
            "ai_actions": 0,
            "key_moments": [],
            "snapshots": [],
        }
    return {
        "schema_version": 1,
        "game_key": definition.key,
        "game_title": definition.title,
        "difficulty": difficulty,
        "play_style": "natural_companion",
        **(
            {"board_size": GOMOKU_BOARD_SIZE}
            if definition.key == GOMOKU_GAME_KEY
            else {}
        ),
        "process": {definition.key: game_process},
        "event_counts": {},
    }


def _with_event_count(result: dict[str, Any], event_type: str) -> dict[str, Any]:
    counts = dict(_loads(result.get("event_counts"), {}))
    counts[event_type] = int(counts.get(event_type) or 0) + 1
    return {**result, "event_counts": counts}


def _store_initial_state(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    payload: dict[str, Any],
) -> dict[str, Any]:
    initial_state = _loads(payload.get("initial_state"), {})
    if (
        not isinstance(initial_state, dict)
        or not initial_state
        or definition.key == GOMOKU_GAME_KEY
    ):
        return result
    process = dict(_loads(result.get("process"), {}))
    game = dict(_loads(process.get(definition.key), {}))
    game.update(
        {
            "final_state": initial_state,
            "final_state_hash": initial_state.get("state_hash"),
        }
    )
    process[definition.key] = game
    return {**result, "process": process}


def _gomoku(result: dict[str, Any]) -> dict[str, Any]:
    process = _loads(result.get("process"), {})
    return dict(_loads(process.get("gomoku"), {}))


def _generic_process(
    result: dict[str, Any], definition: NativeGameDefinition
) -> dict[str, Any]:
    process = _loads(result.get("process"), {})
    return dict(_loads(process.get(definition.key), {}))


def _validate_generic_action(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    payload: dict[str, Any],
) -> dict[str, Any]:
    game = _generic_process(result, definition)
    actions = list(_loads(game.get("actions"), []))
    actor = str(payload.get("actor") or "").strip()
    if actor not in {"user", "agent"}:
        raise ValueError("invalid_actor")
    state_before = _loads(payload.get("state_before"), {})
    previous_hash = str(game.get("final_state_hash") or "")
    reported_before_hash = str(
        payload.get("state_before_hash") or state_before.get("state_hash") or ""
    )
    if previous_hash and previous_hash != reported_before_hash:
        raise ValueError("invalid_state_hash")
    state_after = _loads(payload.get("state_after"), {})
    state_after_hash = str(
        payload.get("state_after_hash") or state_after.get("state_hash") or ""
    )
    if not state_after_hash:
        raise ValueError("invalid_state_hash")
    action = {
        **payload,
        "action_number": len(actions) + 1,
        "actor": actor,
        "state_before": state_before,
        "state_after": state_after,
        "state_before_hash": reported_before_hash or None,
        "state_after_hash": state_after_hash or None,
    }
    if not any(
        key in action
        for key in ("from", "to", "path", "swap", "piece", "move", "action")
    ):
        raise ValueError("invalid_move")
    return action


def _append_generic_action(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    action: dict[str, Any],
) -> dict[str, Any]:
    process = dict(_loads(result.get("process"), {}))
    game = _generic_process(result, definition)
    actions = [*list(_loads(game.get("actions"), [])), action]
    moments = list(_loads(game.get("key_moments"), []))
    action_moments = list(_loads(action.get("moments"), []))
    moment = _loads(action.get("moment"), {})
    if moment.get("type"):
        action_moments.append(moment)
    for action_moment in action_moments:
        if not isinstance(action_moment, dict) or not action_moment.get("type"):
            continue
        normalized_moment = {
            **action_moment,
            "action_number": action["action_number"],
            "actor": action["actor"],
        }
        if not any(_same_key_moment(item, normalized_moment) for item in moments):
            moments.append(normalized_moment)
    final_state = _loads(action.get("state_after"), {})
    game.update(
        {
            "actions": actions,
            "action_count": len(actions),
            "user_actions": sum(1 for item in actions if item.get("actor") == "user"),
            "ai_actions": sum(1 for item in actions if item.get("actor") == "agent"),
            "last_action": action,
            "key_moments": moments[-20:],
            "latest_analysis": _loads(action.get("analysis"), {}),
            "final_state": final_state,
            "final_state_hash": action.get("state_after_hash"),
        }
    )
    process[definition.key] = game
    return {**result, "process": process}


def _same_generic_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_id = str(left.get("action_id") or left.get("client_action_id") or "")
    right_id = str(right.get("action_id") or right.get("client_action_id") or "")
    if left_id and right_id:
        return left_id == right_id
    return (
        str(left.get("actor") or "") == str(right.get("actor") or "")
        and _loads(left.get("from"), None) == _loads(right.get("from"), None)
        and _loads(left.get("to"), None) == _loads(right.get("to"), None)
        and _loads(left.get("swap"), None) == _loads(right.get("swap"), None)
    )


def _reconcile_reported_actions(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    payload: dict[str, Any],
) -> dict[str, Any]:
    reported = _loads(payload.get("actions"), None)
    if reported is None:
        return result
    if not isinstance(reported, list):
        raise ValueError("invalid_action_history")
    existing = list(_loads(_generic_process(result, definition).get("actions"), []))
    if len(reported) < len(existing):
        raise ValueError("invalid_action_history")
    for index, raw_action in enumerate(reported):
        if not isinstance(raw_action, dict):
            raise ValueError("invalid_action_history")
        if index < len(existing):
            if not _same_generic_action(existing[index], raw_action):
                raise ValueError("invalid_action_history")
            continue
        normalized = _validate_generic_action(result, definition, raw_action)
        result = _append_generic_action(result, definition, normalized)
    game = _generic_process(result, definition)
    game["recovered_action_count"] = max(0, len(reported) - len(existing))
    process = dict(_loads(result.get("process"), {}))
    process[definition.key] = game
    return {**result, "process": process}


def _validate_and_normalize_move(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    gomoku = _gomoku(result)
    moves = list(_loads(gomoku.get("moves"), []))
    winner, _ = _winner(moves)
    if winner or len(moves) >= GOMOKU_BOARD_SIZE * GOMOKU_BOARD_SIZE:
        raise ValueError("game_already_finished")
    expected_actor = "user" if len(moves) % 2 == 0 else "agent"
    actor = str(payload.get("actor") or "").strip()
    if actor != expected_actor:
        raise ValueError("invalid_turn")
    row = _as_int(payload.get("row", payload.get("y")))
    col = _as_int(payload.get("col", payload.get("x")))
    if (
        row is None
        or col is None
        or not (0 <= row < GOMOKU_BOARD_SIZE)
        or not (0 <= col < GOMOKU_BOARD_SIZE)
    ):
        raise ValueError("invalid_move")
    occupied = {(int(move["row"]), int(move["col"])) for move in moves}
    if (row, col) in occupied:
        raise ValueError("occupied_position")
    return {
        **payload,
        "move_number": len(moves) + 1,
        "actor": actor,
        "stone": "black" if actor == "user" else "white",
        "row": row,
        "col": col,
        "x": col,
        "y": row,
        "coordinate": str(payload.get("coordinate") or _coordinate(row, col)),
    }


def _reconcile_reported_moves(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    reported = _loads(payload.get("moves"), None)
    if reported is None:
        return result
    if not isinstance(reported, list):
        raise ValueError("invalid_move_history")

    existing_moves = list(_loads(_gomoku(result).get("moves"), []))
    if len(reported) < len(existing_moves):
        raise ValueError("invalid_move_history")

    process = dict(_loads(result.get("process"), {}))
    process["gomoku"] = {
        "moves": [],
        "move_count": 0,
        "user_moves": 0,
        "ai_moves": 0,
        "key_moments": [],
    }
    rebuilt = {**result, "process": process}
    for index, raw_move in enumerate(reported):
        if not isinstance(raw_move, dict):
            raise ValueError("invalid_move_history")
        normalized = _validate_and_normalize_move(rebuilt, raw_move)
        if index < len(existing_moves) and not _same_move(
            existing_moves[index], normalized
        ):
            raise ValueError("invalid_move_history")
        rebuilt = _append_move(rebuilt, normalized)

    recovered = len(reported) - len(existing_moves)
    if recovered > 0:
        process = dict(_loads(rebuilt.get("process"), {}))
        gomoku = _gomoku(rebuilt)
        gomoku["recovered_move_count"] = recovered
        process["gomoku"] = gomoku
        rebuilt = {**rebuilt, "process": process}
    return rebuilt


def _same_move(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left.get("actor") or "") == str(right.get("actor") or "")
        and _as_int(left.get("row", left.get("y"))) == right["row"]
        and _as_int(left.get("col", left.get("x"))) == right["col"]
    )


def _append_move(result: dict[str, Any], move: dict[str, Any]) -> dict[str, Any]:
    process = dict(_loads(result.get("process"), {}))
    gomoku = _gomoku(result)
    moves = [*list(_loads(gomoku.get("moves"), [])), move]
    moments = list(_loads(gomoku.get("key_moments"), []))
    moment = move.get("moment")
    if isinstance(moment, dict) and moment.get("type"):
        moments.append({**moment, "move_number": move["move_number"]})
    gomoku.update(
        {
            "moves": moves,
            "move_count": len(moves),
            "user_moves": sum(1 for item in moves if item.get("actor") == "user"),
            "ai_moves": sum(1 for item in moves if item.get("actor") == "agent"),
            "last_move": move,
            "key_moments": moments[-12:],
            "latest_analysis": _loads(move.get("analysis"), {}),
        }
    )
    winner, winning_line = _winner(moves)
    if winner:
        gomoku["detected_winner"] = winner
        gomoku["winning_line"] = winning_line
        gomoku["win_direction"] = _line_direction(winning_line)
    process["gomoku"] = gomoku
    return {**result, "process": process}


def _merge_auxiliary_event(
    result: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    process = dict(_loads(result.get("process"), {}))
    snapshots = list(_loads(process.get("snapshots"), []))
    if event_type in {
        "threat_detected",
        "analysis_snapshot",
        "ai_move_decided",
        "cascade_resolved",
        "special_created",
        "board_shuffled",
        "cells_revealed",
        "flag_toggled",
        "inference_made",
        "tiles_merged",
        "tile_spawned",
        "turn_changed",
        "game_state_snapshot",
        "key_moment",
    }:
        snapshots.append({"event_type": event_type, **payload})
        process["snapshots"] = snapshots[-80:]
        game_key = str(result.get("game_key") or "")
        if game_key in _GAME_DEFINITIONS and game_key != GOMOKU_GAME_KEY:
            game = dict(_loads(process.get(game_key), {}))
            game_snapshots = list(_loads(game.get("snapshots"), []))
            game_snapshots.append({"event_type": event_type, **payload})
            game["snapshots"] = game_snapshots[-80:]
            if event_type == "key_moment":
                moments = list(_loads(game.get("key_moments"), []))
                if not any(_same_key_moment(moment, payload) for moment in moments):
                    moments.append(payload)
                game["key_moments"] = moments[-20:]
            process[game_key] = game
    return {**result, "process": process}


def _same_key_moment(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_number = left.get("action_number") or left.get("move_number")
    right_number = right.get("action_number") or right.get("move_number")
    left_actor = str(left.get("actor") or "")
    right_actor = str(right.get("actor") or "")
    return (
        str(left.get("type") or "") == str(right.get("type") or "")
        and left_number is not None
        and left_number == right_number
        and (not left_actor or not right_actor or left_actor == right_actor)
    )


def _validated_generic_outcome(
    payload: dict[str, Any],
    definition: NativeGameDefinition,
) -> str:
    outcome = _normalize_outcome(payload.get("user_outcome") or payload.get("outcome"))
    if outcome is None:
        raise ValueError("invalid_outcome")
    terminal_state = _loads(payload.get("terminal_state"), {})
    final_state = _loads(payload.get("final_state"), {})
    if not terminal_state and not final_state and not payload.get("state_after_hash"):
        raise ValueError("missing_terminal_state")
    terminal_status = str(terminal_state.get("status") or "")
    expected = _TERMINAL_OUTCOMES.get(definition.key, {}).get(terminal_status)
    if terminal_status and expected != outcome:
        raise ValueError("invalid_outcome")
    return outcome


def _recover_generic_terminal(
    result: dict[str, Any],
    definition: NativeGameDefinition,
) -> tuple[str, str, dict[str, Any]] | None:
    final_state = dict(
        _loads(_generic_process(result, definition).get("final_state"), {})
    )
    terminal_status = str(final_state.get("status") or "")
    outcome = _TERMINAL_OUTCOMES.get(definition.key, {}).get(terminal_status)
    if outcome is None:
        return None
    return outcome, terminal_status, final_state


def _finish_generic_result(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    payload: dict[str, Any],
    *,
    outcome: str,
    duration_seconds: int,
) -> dict[str, Any]:
    process = dict(_loads(result.get("process"), {}))
    game = _generic_process(result, definition)
    final_state = _loads(payload.get("final_state"), game.get("final_state", {}))
    summary = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "actions",
            "analysis",
            "final_state",
            "state_after_hash",
            "terminal_state",
        }
    }
    game.update(
        {
            "final_state": final_state,
            "final_state_hash": payload.get("state_after_hash")
            or final_state.get("state_hash")
            or game.get("final_state_hash"),
            "final_analysis": _loads(
                payload.get("analysis"), game.get("latest_analysis", {})
            ),
            "terminal_state": _loads(payload.get("terminal_state"), {}),
            "summary": summary,
        }
    )
    process[definition.key] = game
    return {
        **result,
        "user_outcome": outcome,
        "duration_seconds": duration_seconds,
        "ended_reason": "settled",
        definition.key: game,
        "process": process,
        "final_payload": payload,
    }


def _abort_generic_result(
    result: dict[str, Any],
    definition: NativeGameDefinition,
    payload: dict[str, Any],
    duration_seconds: int,
) -> dict[str, Any]:
    game = _generic_process(result, definition)
    return {
        **result,
        "user_outcome": "aborted",
        "duration_seconds": duration_seconds,
        "ended_reason": str(payload.get("reason") or "left_game"),
        definition.key: game,
        "final_payload": payload,
    }


def _validated_outcome(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, int]]]:
    moves = list(_loads(_gomoku(result).get("moves"), []))
    winner, winning_line = _winner(moves)
    board_full = len(moves) >= GOMOKU_BOARD_SIZE * GOMOKU_BOARD_SIZE
    expected = "win" if winner == "user" else "lose" if winner == "agent" else "draw"
    requested = _normalize_outcome(
        payload.get("user_outcome") or payload.get("outcome")
    )
    if winner is None and not board_full:
        raise ValueError("game_not_finished")
    if requested and requested != expected:
        raise ValueError("invalid_outcome")
    return expected, winning_line


def _finish_result(
    result: dict[str, Any],
    payload: dict[str, Any],
    *,
    outcome: str,
    winning_line: list[dict[str, int]],
    duration_seconds: int,
) -> dict[str, Any]:
    process = dict(_loads(result.get("process"), {}))
    gomoku = _gomoku(result)
    gomoku.update(
        {
            "winning_line": winning_line,
            "win_direction": _line_direction(winning_line),
            "final_analysis": _loads(
                payload.get("analysis"), gomoku.get("latest_analysis", {})
            ),
        }
    )
    process["gomoku"] = gomoku
    return {
        **result,
        "user_outcome": outcome,
        "duration_seconds": duration_seconds,
        "ended_reason": "settled",
        "gomoku": gomoku,
        "process": process,
        "final_payload": payload,
    }


def _abort_result(
    result: dict[str, Any],
    payload: dict[str, Any],
    duration_seconds: int,
) -> dict[str, Any]:
    gomoku = _gomoku(result)
    return {
        **result,
        "user_outcome": "aborted",
        "duration_seconds": duration_seconds,
        "ended_reason": str(payload.get("reason") or "left_game"),
        "gomoku": gomoku,
        "final_payload": payload,
    }


def _winner(moves: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, int]]]:
    board: dict[tuple[int, int], str] = {
        (int(move["row"]), int(move["col"])): str(move["actor"]) for move in moves
    }
    for move in reversed(moves):
        row, col, actor = int(move["row"]), int(move["col"]), str(move["actor"])
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = [(row, col)]
            rr, cc = row - dr, col - dc
            while board.get((rr, cc)) == actor:
                line.insert(0, (rr, cc))
                rr, cc = rr - dr, cc - dc
            rr, cc = row + dr, col + dc
            while board.get((rr, cc)) == actor:
                line.append((rr, cc))
                rr, cc = rr + dr, cc + dc
            if len(line) >= 5:
                index = line.index((row, col))
                start = min(max(index - 4, 0), len(line) - 5)
                selected = line[start : start + 5]
                return actor, [{"x": cc, "y": rr} for rr, cc in selected]
    return None, []


def _line_direction(line: list[dict[str, int]]) -> str | None:
    if len(line) < 2:
        return None
    dx = line[1]["x"] - line[0]["x"]
    dy = line[1]["y"] - line[0]["y"]
    if dy == 0:
        return "horizontal"
    if dx == 0:
        return "vertical"
    return "diagonal"


def _duration(payload: dict[str, Any], session: SudSessionResponse) -> int:
    raw = _as_int(payload.get("duration_seconds", payload.get("duration")))
    if raw is not None:
        return max(0, min(raw, 24 * 60 * 60))
    if session.started_at:
        try:
            started = datetime.fromisoformat(session.started_at)
            return max(0, int((_now() - started).total_seconds()))
        except ValueError:
            pass
    return 0


def _normalize_outcome(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    aliases = {
        "win": "win",
        "user_win": "win",
        "userwin": "win",
        "lose": "lose",
        "loss": "lose",
        "agent_win": "lose",
        "agentwin": "lose",
        "draw": "draw",
    }
    return aliases.get(normalized)


def _coordinate(row: int, col: int) -> str:
    letters = "ABCDEFGHJKLMNOP"
    return f"{letters[col]}{row + 1}"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _finish_reply(session: SudSessionResponse, result: dict[str, Any]) -> str:
    gomoku = _loads(result.get("gomoku"), {})
    moments = list(_loads(gomoku.get("key_moments"), []))
    outcome = result.get("user_outcome")
    direction = _line_direction(list(_loads(gomoku.get("winning_line"), [])))
    line_text = "斜着那条线" if direction == "diagonal" else "中间那条线"
    if outcome == "win":
        if any(item.get("type") == "double_threat" for item in moments):
            return (
                "你刚才那手一落，我就知道要没了。两边一起冒出来，真有你的。再来一盘？"
            )
        return f"行，这盘是你赢。{line_text}是前面一点点铺出来的，我最后才反应过来。"
    if outcome == "lose":
        if any(item.get("type") == "blocked_win" for item in moments):
            return "刚才你那条线差一点就成了，我挡住的时候其实偷偷松了口气。下一盘你肯定会盯我更紧。"
        return "这盘我先收下啦。不过你后半段已经开始逼我防守了，再下一盘我未必还能这么轻松。"
    return "居然下成了平局。棋盘都快被我们填满了，谁也没肯先松手。"


def _abort_reply(session: SudSessionResponse, result: dict[str, Any]) -> str:
    moves = int(_gomoku(result).get("move_count") or 0)
    if moves < 4:
        return "这盘还没真正展开，我们先放在这里。想玩的时候再重新摆一盘。"
    return "这盘就先到这里。刚才走过的那些手我会记得，想玩时我们再重新摆一盘。"


def _generic_finish_reply(
    session: SudSessionResponse,
    definition: NativeGameDefinition,
    result: dict[str, Any],
) -> str:
    outcome = str(result.get("user_outcome") or "draw")
    game = _generic_process(result, definition)
    moments = list(_loads(game.get("key_moments"), []))
    if definition.key == "minesweeper":
        if outcome == "win":
            final_payload = _loads(result.get("final_payload"), {})
            largest = int(final_payload.get("largest_reveal") or 0)
            if largest >= 8:
                return "清完了。刚才那一大片一起亮起来的时候真的很舒服，我们这局配合得挺稳。"
            return "一颗都没踩，收工。我们刚才不是各玩各的，是真的把线索接起来了。"
        return "刚才那颗雷藏得挺会挑位置。先别不服，下次我们从它旁边的数字慢慢拆。"
    if definition.key == "number_merge":
        final_payload = _loads(result.get("final_payload"), {})
        max_tile = int(final_payload.get("max_tile") or 0)
        if outcome == "win":
            return f"真的合到{max_tile}了。前面几次盘面快满的时候，我们居然都一起救回来了。"
        return f"这次停在{max_tile}。有几步其实已经把空间重新救出来了，下局我们把大数字守在角落久一点。"
    if definition.play_mode == "cooperate":
        if outcome == "win":
            return (
                "过了。刚才中间那段连起来的时候我就觉得有戏，最后一下还真让我们接上了。"
            )
        return "这一关没接完也没关系，刚才已经试出一条挺顺的路了。下次从那个节奏继续。"
    if outcome == "win":
        if moments:
            return "你赢啦。刚才局面一转过来，我就知道这局很难追回去了。再来一局我可要盯紧一点。"
        return "行，这局你拿下。不是碰巧，是你后面几步真的走得比我稳。"
    if outcome == "lose":
        if moments:
            return "这局我先赢一下，不过中间那次转折差点把我吓住。下局不一定还守得住。"
        return f"这局《{definition.title}》我先收下啦。你后面已经追得很近了，再开一局？"
    return "居然谁也没拿走这局。这样也挺好，像我们一起把一个局面慢慢走完了。"


def _generic_abort_reply(
    definition: NativeGameDefinition,
    result: dict[str, Any],
) -> str:
    count = int(_generic_process(result, definition).get("action_count") or 0)
    if definition.key == "minesweeper" and count > 2:
        return "雷区先留在这里。刚才我们一起推出来的那些安全格，我会记得。"
    if definition.key == "number_merge" and count > 2:
        return "这盘数字先收到这里。刚才我们一起养大的那块不会算没发生。"
    if count <= 2:
        return f"《{definition.title}》才刚开头，我们先收起来。下次想玩再重新来。"
    return "先停在这里吧。刚才走过的过程我会留着，它也算我们一起玩过的一小段。"


async def _remember_shared_experience(
    session: SudSessionResponse,
    result: dict[str, Any],
    *,
    sides: tuple[str, ...] = ("user", "ai"),
) -> dict[str, Any]:
    definition = _definition(session.game_key or result.get("game_key"))
    if definition.key == GOMOKU_GAME_KEY:
        game = _loads(result.get("gomoku"), _gomoku(result))
        action_count = int(game.get("move_count") or 0)
        count_unit = "手"
    else:
        game = _loads(result.get(definition.key), _generic_process(result, definition))
        action_count = int(game.get("action_count") or 0)
        count_unit = "步"
    if action_count == 0:
        return {
            "status": "skipped",
            "user_memory_id": None,
            "ai_memory_id": None,
            "failed_sides": [],
        }
    outcome = str(result.get("user_outcome") or "aborted")
    if definition.play_mode == "cooperate":
        outcome_user = {
            "win": "我们一起过关了",
            "lose": "我们这次没有过关",
            "aborted": "我们中途把这局收起来了",
        }.get(outcome, "这局结束了")
        outcome_ai = outcome_user
    else:
        outcome_user = {
            "win": "我赢了",
            "lose": f"{session.ai_player.nick_name}赢了",
            "draw": "我们下成平局",
            "aborted": "这局中途停下了",
        }.get(outcome, "这局结束了")
        outcome_ai = {
            "win": "用户赢了",
            "lose": "我赢了",
            "draw": "我们下成平局",
            "aborted": "这局中途停下了",
        }.get(outcome, "这局结束了")
    moments = list(_loads(game.get("key_moments"), []))
    moment_text = _memory_moment(moments, definition.key)
    duration = int(result.get("duration_seconds") or 0)
    duration_text = f"，大约玩了{max(1, duration // 60)}分钟" if duration >= 60 else ""
    user_text = (
        f"我和{session.ai_player.nick_name}一起玩了一局《{definition.title}》，"
        f"一共走了{action_count}{count_unit}{duration_text}，{outcome_user}。{moment_text}"
    )
    ai_text = (
        f"我和用户一起玩了一局《{definition.title}》，一共走了{action_count}{count_unit}"
        f"{duration_text}，{outcome_ai}。{moment_text}这是我们共同经历的一局游戏。"
    )
    return await remember_shared_game_experience(
        user_id=session.user_id,
        workspace_id=session.workspace_id,
        user_text=user_text,
        ai_text=ai_text,
        agent_name=session.ai_player.nick_name,
        game_title=definition.title,
        sides=sides,
    )


def _memory_moment(
    moments: list[dict[str, Any]],
    game_key: str = GOMOKU_GAME_KEY,
) -> str:
    types = {str(item.get("type") or "") for item in moments}
    descriptions: list[str] = []

    def add(moment_types: set[str], text: str) -> None:
        if len(descriptions) < 2 and types.intersection(moment_types):
            descriptions.append(text)

    if game_key == GOMOKU_GAME_KEY:
        add({"double_threat"}, "这盘出现过一次很漂亮的双向威胁。")
        add({"blocked_win"}, "有一手在胜负边缘被及时挡住。")
        add({"open_four"}, "棋盘上曾铺出过一条很有压迫感的活四。")
    elif game_key == "go":
        add({"large_capture"}, "有一手提掉了一大片棋，局势从那里转了方向。")
        add({"atari"}, "我们在一块很紧的叫吃附近反复照顾过气。")
        add({"self_atari"}, "有一手主动走进紧气，局面一度很悬。")
        add({"pass", "scoring_started"}, "最后双方停着，认真把地数完了。")
    elif game_key == "reversi":
        add({"corner_captured"}, "有一手抢到了稳定角点，整条边也跟着变了。")
        add({"forced_pass"}, "局中曾把一方逼到无子可下，只能让出一轮。")
        add({"big_flip"}, "有一次大面积翻子，盘面几乎一手换了颜色。")
        add({"mobility_squeeze"}, "中盘可落位置突然收紧，双方都被迫换了计划。")
    elif game_key in {"xiangqi", "chess"}:
        add({"check"}, "局中有一次直接将军，攻守节奏从那里加快了。")
        add({"capture"}, "有一次关键交换改变了棋子的力量对比。")
        add({"castling"}, "这盘完成过一次王车易位，王翼很快安定下来。")
        add({"promotion"}, "有一枚兵一路走到底完成了升变。")
        add({"decisive_finish", "winning_move"}, "最后的制胜手是前面几步一起铺出来的。")
    elif game_key == "chinese_checkers":
        add({"long_jump"}, "走出过一次跨过多枚棋子的连续长跳。")
        add({"near_finish", "piece_finished"}, "有一段连续进营，把终点一下拉近了。")
        add({"breakthrough"}, "中路曾找到一个突破口，后面的路线因此打开。")
    elif game_key == "match3":
        add({"special_combo"}, "我们接出过一次特殊方块组合，整片棋盘一起亮了。")
        add({"big_cascade"}, "有一轮连续消除自己接了好几层。")
        add({"lead_changed"}, "中途分数领先发生过反转，最后几步很紧。")
    elif game_key == "minesweeper":
        add({"forced_deduction"}, "有几格是靠相邻数字严格推出来的，不是碰运气。")
        add({"zero_expansion"}, "曾经一次展开一大片安全区域，线索一下连了起来。")
        add({"calculated_risk"}, "没有必然解时，我们比较概率后一起承担了一次风险。")
        add({"mine_triggered"}, "最后碰到了一颗雷，但之前的推理过程都留了下来。")
        add({"near_clear"}, "收尾时只剩很少几格，我们刻意放慢了节奏。")
    elif game_key == "number_merge":
        add({"target_reached"}, "我们一路合到了目标数字。")
        add({"milestone_tile"}, "轮流腾空间养出了一个新的里程碑数字。")
        add({"board_recovered"}, "盘面快塞满时，一次连续合并又救出了空间。")
        add({"multi_merge"}, "有一手同时完成了多组合并。")
        add({"near_stuck"}, "盘面一度只剩很少的活动空间。")

    if descriptions:
        return "".join(descriptions)
    return "这局的完整过程、AI判断和关键局面已经保存。"
