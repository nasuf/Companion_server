"""Admin API: 系统设置 — 成就系统全局模式开关.

Endpoints (all admin-only):
  GET /admin-api/achievement-settings  — 读取当前模式 (override / env 默认 / 生效值)
  PUT /admin-api/achievement-settings  — 更新模式 (写 SystemConfig, ~10s 内全部 worker 生效)

模式语义 (详见 app/services/achievements/mode.py 与 docs/achievement_rule_audit.md):
  on     — 完整评估 + 全部用户可见面 (通知/API/时间线/钱包积分).
  silent — H5 静默计算: 解锁照常实时落库 (unlocked_at/conversation_id 即真实
           达成点), 但抑制全部用户可见面; 切回 on 后全量自动呈现, 无需回填.
  off    — 应急停算: 评估与日终任务全停, 日终 checkpoint 冻结待恢复后补算.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.jwt_auth import require_admin_jwt
from app.services.achievements.mode import (
    get_achievement_settings_snapshot,
    set_achievement_mode,
)

router = APIRouter(
    prefix="/admin-api/achievement-settings",
    tags=["admin", "achievements"],
    dependencies=[Depends(require_admin_jwt)],
)


class AchievementSettingsResponse(BaseModel):
    # DB override; null = inherit the .env ACHIEVEMENT_MODE default.
    mode: Literal["on", "silent", "off"] | None
    env_mode: str
    effective_mode: Literal["on", "silent", "off"]


class AchievementSettingsUpdateRequest(BaseModel):
    mode: Literal["on", "silent", "off"]


@router.get("", response_model=AchievementSettingsResponse)
async def get_achievement_settings() -> AchievementSettingsResponse:
    return AchievementSettingsResponse(**await get_achievement_settings_snapshot())


@router.put("", response_model=AchievementSettingsResponse)
async def update_achievement_settings(
    payload: AchievementSettingsUpdateRequest,
) -> AchievementSettingsResponse:
    return AchievementSettingsResponse(**await set_achievement_mode(payload.mode))
