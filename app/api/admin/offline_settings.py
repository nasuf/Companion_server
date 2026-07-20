"""Admin API: 系统设置 — 线下真实世界模块主动推送开关.

Endpoints (all admin-only):
  GET /admin-api/offline-settings  — 读取活动推荐 / 礼物推荐当前开关
  PUT /admin-api/offline-settings  — 更新开关 (写 SystemConfig, 实时生效)

开关默认关闭. 关闭后 offline trigger scan 跳过对应分支, 手动/mock 触发接口
拒绝创建; 已有条目的读取/操作接口不受影响.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.jwt_auth import require_admin_jwt
from app.services.offline.module_settings import (
    get_offline_module_flags,
    set_offline_module_flags,
)

router = APIRouter(
    prefix="/admin-api/offline-settings",
    tags=["admin", "offline"],
    dependencies=[Depends(require_admin_jwt)],
)


class OfflineSettingsResponse(BaseModel):
    activity_enabled: bool
    gift_enabled: bool


class OfflineSettingsUpdateRequest(BaseModel):
    # Each field is optional so a single toggle can be updated in isolation.
    activity_enabled: bool | None = None
    gift_enabled: bool | None = None


@router.get("", response_model=OfflineSettingsResponse)
async def get_offline_settings() -> OfflineSettingsResponse:
    return OfflineSettingsResponse(**await get_offline_module_flags())


@router.put("", response_model=OfflineSettingsResponse)
async def update_offline_settings(
    payload: OfflineSettingsUpdateRequest,
) -> OfflineSettingsResponse:
    flags = await set_offline_module_flags(
        activity_enabled=payload.activity_enabled,
        gift_enabled=payload.gift_enabled,
    )
    return OfflineSettingsResponse(**flags)
