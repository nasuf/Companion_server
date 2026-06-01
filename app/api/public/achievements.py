from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.db import db
from app.services.achievements.service import list_achievements

router = APIRouter(prefix="/achievements", tags=["achievements"])


@router.get("")
async def get_achievements(
    agent_id: str = Query(...),
    payload: dict = Depends(require_user),
):
    user_id = str(payload.get("sub") or "")
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "userId", None) != user_id:
        raise HTTPException(status_code=404, detail="Agent not found")
    return await list_achievements(user_id=user_id, agent_id=agent_id)
