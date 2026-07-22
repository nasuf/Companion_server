from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.db import db
from app.services.achievements.definitions import ACHIEVEMENTS
from app.services.achievements.mode import achievement_display_enabled
from app.services.achievements.rule_registry import (
    ACHIEVEMENT_RULES,
    DISABLED_ACHIEVEMENT_IDS,
)
from app.services.achievements.service import list_achievements

router = APIRouter(prefix="/achievements", tags=["achievements"])


def _hidden_achievements_payload() -> dict:
    """Zero-progress payload for the "off" kill switch.

    Silent mode intentionally does NOT use this: the achievement page stays
    fully visible while only chat alerts / push are muted (2026-07-22).
    """
    return {
        "enabled": False,
        "total": len(ACHIEVEMENTS),
        "active_total": sum(1 for rule in ACHIEVEMENT_RULES.values() if rule.enabled),
        "disabled_total": len(DISABLED_ACHIEVEMENT_IDS),
        "unlocked": 0,
        "score": 0,
        "items": [],
    }


@router.get("")
async def get_achievements(
    agent_id: str = Query(...),
    payload: dict = Depends(require_user),
):
    user_id = str(payload.get("sub") or "")
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "userId", None) != user_id:
        raise HTTPException(status_code=404, detail="Agent not found")
    if not await achievement_display_enabled():
        return _hidden_achievements_payload()
    return await list_achievements(user_id=user_id, agent_id=agent_id)
