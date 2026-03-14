from fastapi import APIRouter

from app.services.intimacy import get_intimacy_data

router = APIRouter(prefix="/intimacy", tags=["intimacy"])


@router.get("/{agent_id}/{user_id}")
async def get_intimacy(agent_id: str, user_id: str):
    """获取亲密度数据。"""
    return await get_intimacy_data(agent_id, user_id)
