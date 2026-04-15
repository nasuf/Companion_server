from fastapi import APIRouter

from app.db import db
from app.redis_client import redis_health
from app.models.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    pg_ok = db.is_connected()
    redis_ok = await redis_health()

    status = "ok" if (pg_ok and redis_ok) else "degraded"
    return HealthResponse(
        status=status,
        postgres=pg_ok,
        redis=redis_ok,
    )
