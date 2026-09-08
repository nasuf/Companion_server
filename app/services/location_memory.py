"""Structured memory writes for chat location-share cards."""

from __future__ import annotations

import logging
from typing import Any

from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.memory.storage.persistence import store_memory
from app.services.offerings_memory_text import render_component_card_line

logger = logging.getLogger(__name__)

USER_MEMORY_LEVEL = 2
USER_MEMORY_IMPORTANCE = 0.82
AI_MEMORY_LEVEL = 2
AI_MEMORY_IMPORTANCE = 0.55


def build_location_memory_texts(component_card: dict[str, Any]) -> tuple[str, str]:
    """Return (user-side, ai-side) memory text for a location share card."""
    line = render_component_card_line("", component_card).strip()
    place = line.removeprefix("用户分享了当前位置：").strip() or "未知位置"
    return (
        f"用户当前在{place}",
        f"用户向我分享了当前位置：{place}",
    )


async def write_location_share_memories(
    *,
    user_id: str,
    workspace_id: str | None,
    component_card: dict[str, Any],
) -> None:
    """Persist location share as structured memories, bypassing LLM extraction."""
    user_text, ai_text = build_location_memory_texts(component_card)
    try:
        await store_memory(
            user_id,
            user_text,
            level=USER_MEMORY_LEVEL,
            importance=USER_MEMORY_IMPORTANCE,
            main_category="身份",
            sub_category="现居地",
            source="user",
            workspace_id=workspace_id,
            provenance=USER_STATED,
            skip_reconciliation=True,
        )
        await store_memory(
            user_id,
            ai_text,
            level=AI_MEMORY_LEVEL,
            importance=AI_MEMORY_IMPORTANCE,
            main_category="生活",
            sub_category="交互",
            source="ai",
            workspace_id=workspace_id,
            provenance=AI_AUTHORED,
            skip_reconciliation=True,
        )
    except Exception:
        logger.exception(
            "location share memory write failed user=%s",
            str(user_id)[:8],
        )
