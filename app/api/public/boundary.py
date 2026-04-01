from fastapi import APIRouter, HTTPException

from app.db import db

from app.services.relationship.boundary import get_patience, get_patience_zone

router = APIRouter(prefix="/boundary", tags=["boundary"])


@router.get("/{agent_id}/{user_id}")
async def get_boundary_status(agent_id: str, user_id: str):
    """获取耐心值和状态区间。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    patience = await get_patience(agent_id, user_id)
    return {
        "patience": patience,
        "zone": get_patience_zone(patience),
    }
