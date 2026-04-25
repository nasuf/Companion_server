from fastapi import APIRouter, Depends

from app.api.ownership import require_agent_owner, require_user_self
from app.services.relationship.intimacy import get_intimacy_data

router = APIRouter(prefix="/intimacy", tags=["intimacy"])


@router.get("/{agent_id}/{user_id}")
async def get_intimacy(
    agent_id: str,
    user_id: str,
    _agent=Depends(require_agent_owner),
    _user=Depends(require_user_self),
):
    """获取亲密度数据。需 JWT, agent 必须属本人, user_id 必须为本人."""
    return await get_intimacy_data(agent_id, user_id)
