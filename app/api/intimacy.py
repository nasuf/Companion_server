from fastapi import APIRouter, HTTPException

from app.db import db

from app.services.intimacy import get_intimacy_data

router = APIRouter(prefix="/intimacy", tags=["intimacy"])


@router.get("/{agent_id}/{user_id}")
async def get_intimacy(agent_id: str, user_id: str):
    """获取亲密度数据。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    return await get_intimacy_data(agent_id, user_id)
