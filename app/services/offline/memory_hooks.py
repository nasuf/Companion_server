from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.storage.persistence import store_memory
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)


def remember_user_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="user",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )


def remember_ai_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="ai",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )


async def remember_shared_game_experience(
    *,
    user_id: str,
    workspace_id: str | None,
    user_text: str,
    ai_text: str,
    agent_name: str,
    game_title: str,
    sides: tuple[str, ...] = ("user", "ai"),
) -> dict[str, Any]:
    """Persist a completed game as a guaranteed two-sided shared memory.

    Game sessions are already structured, verified events. Running them through
    the conversational "remember / do not remember" gate can discard the very
    shared experience the product promises to keep, so this path starts at the
    existing storage layer instead. It still receives taxonomy validation,
    embeddings, semantic reconciliation, changelog, cache invalidation, and
    achievement hooks from ``store_memory``.
    """

    statement_time = datetime.now(UTC)
    topics = [game_title, "共同游戏"]
    entities = [agent_name, game_title]

    async def _store(side: str) -> str | None:
        is_ai = side == "ai"
        text = ai_text if is_ai else user_text
        return await store_memory(
            user_id=user_id,
            content=text,
            summary=text,
            # Individual rounds stay retrievable but do not permanently crowd
            # the AI's L1 core profile. Exact move history remains in the game
            # session and meaningful rounds can rise through normal L2 dynamics.
            level=2,
            importance=0.80 if is_ai else 0.74,
            memory_type="life",
            main_category="生活",
            sub_category="交互" if is_ai else "其他特殊事件",
            source=side,
            statement_time=statement_time,
            workspace_id=workspace_id,
            entities=entities,
            topics=topics,
        )

    requested_sides = tuple(side for side in ("user", "ai") if side in sides)
    raw_results = await asyncio.gather(
        *(_store(side) for side in requested_sides),
        return_exceptions=True,
    )
    ids: dict[str, str | None] = {"user": None, "ai": None}
    errors: list[str] = []
    for side, value in zip(requested_sides, raw_results, strict=True):
        if isinstance(value, BaseException):
            errors.append(side)
            logger.error(
                "Shared game memory failed side=%s user=%s game=%s: %s",
                side,
                user_id[:8],
                game_title,
                value,
            )
        else:
            ids[side] = value

    stored_count = sum(memory_id is not None for memory_id in ids.values())
    if errors and stored_count:
        status = "partial"
    elif errors:
        status = "failed"
    elif stored_count:
        status = "stored"
    else:
        status = "deduplicated"
    return {
        "status": status,
        "user_memory_id": ids["user"],
        "ai_memory_id": ids["ai"],
        "failed_sides": errors,
    }
