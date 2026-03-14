from fastapi import APIRouter

from app.services.boundary import get_patience, get_patience_zone

router = APIRouter(prefix="/boundary", tags=["boundary"])


@router.get("/{agent_id}/{user_id}")
async def get_boundary_status(agent_id: str, user_id: str):
    """获取耐心值和状态区间。"""
    patience = await get_patience(agent_id, user_id)
    return {
        "patience": patience,
        "zone": get_patience_zone(patience),
    }
