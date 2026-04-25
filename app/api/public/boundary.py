from fastapi import APIRouter, Depends

from app.api.ownership import require_agent_owner, require_user_self
from app.services.interaction.boundary import get_patience, get_patience_zone

router = APIRouter(prefix="/boundary", tags=["boundary"])


@router.get("/{agent_id}/{user_id}")
async def get_boundary_status(
    agent_id: str,
    user_id: str,
    _agent=Depends(require_agent_owner),
    _user=Depends(require_user_self),
):
    """获取耐心值和状态区间。需 JWT, agent 必须属本人, user_id 必须为本人."""
    patience = await get_patience(agent_id, user_id)
    return {
        "patience": patience,
        "zone": get_patience_zone(patience),
    }
