"""Admin API: 礼物管理 — 搜索用户、礼物推荐分析、手动寄送、物流节点推进、撤销。

Endpoints (all admin-only, /admin-api/gifts):
  GET    /users?q=                        搜索用户
  GET    /users/{user_id}                 用户礼物概览（工作区/完整地址/预算/画像/礼物）
  POST   /users/{user_id}/recommendations 生成一批礼物候选（每个带可追溯依据）
  GET    ?status=&user_id=&q=             礼物列表（跨用户，分页，上限 100）
  POST   ""                               手动寄送礼物（request_id 幂等）
  GET    /{gift_id}                       礼物详情（轨迹 + 关联聊天消息）
  PATCH  /{gift_id}                       修改礼物信息 / 图片 / 快递
  POST   /{gift_id}/stage                 推进 / 纠错回退物流节点：下单、运输、派送
                                          （派送会提醒用户签收；签收只能由用户完成）
  POST   /{gift_id}/cancel                撤销礼物

返回的收货地址未脱敏（运营要据此寄快递）；日志里不记录地址与手机号。
这些接口不受「自动送礼」开关影响：开关只管定时触发 + provider 自动下单。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.admin_gifts import (
    AdminGiftActionResponse,
    AdminGiftCancelRequest,
    AdminGiftCreateRequest,
    AdminGiftDetail,
    AdminGiftListResponse,
    AdminGiftRecommendationRequest,
    AdminGiftRecommendationResponse,
    AdminGiftStatusFilter,
    AdminGiftStageRequest,
    AdminGiftUpdateRequest,
    AdminGiftUserOverview,
)
from app.models.admin_wallet import AdminWalletUserSearchItem
from app.services import game_points
from app.services.offline import gift_admin_service

router = APIRouter(
    prefix="/admin-api/gifts",
    tags=["admin", "offline"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("/users", response_model=list[AdminWalletUserSearchItem])
async def search_users(
    q: str = Query(default="", max_length=80),
    limit: int = Query(default=20, ge=1, le=50),
):
    # 与发放钞票/积分共用模糊搜索（用户名 / id / 微信昵称 / 手机号）。
    return await game_points.search_users(q, limit=limit)


@router.get("/users/{user_id}", response_model=AdminGiftUserOverview)
async def user_overview(user_id: str):
    return await gift_admin_service.user_overview(user_id)


@router.post("/users/{user_id}/recommendations", response_model=AdminGiftRecommendationResponse)
async def recommend_gifts(user_id: str, payload: AdminGiftRecommendationRequest):
    return await gift_admin_service.recommend(user_id, payload)


@router.get("", response_model=AdminGiftListResponse)
async def list_gifts(
    status: AdminGiftStatusFilter = Query(default="all"),
    user_id: str | None = Query(default=None, max_length=64),
    q: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
):
    return await gift_admin_service.list_gifts(
        status_filter=status, user_id=user_id, query=q, limit=limit, offset=offset,
    )


@router.post("", response_model=AdminGiftActionResponse)
async def create_gift(payload: AdminGiftCreateRequest, admin: dict = Depends(require_admin_jwt)):
    return await gift_admin_service.create_manual_gift(str(admin["sub"]), payload)


@router.get("/{gift_id}", response_model=AdminGiftDetail)
async def gift_detail(gift_id: str):
    return await gift_admin_service.gift_detail(gift_id)


@router.patch("/{gift_id}", response_model=AdminGiftDetail)
async def update_gift(gift_id: str, payload: AdminGiftUpdateRequest, admin: dict = Depends(require_admin_jwt)):
    return await gift_admin_service.update_gift(str(admin["sub"]), gift_id, payload)


@router.post("/{gift_id}/stage", response_model=AdminGiftActionResponse)
async def set_stage(gift_id: str, payload: AdminGiftStageRequest, admin: dict = Depends(require_admin_jwt)):
    return await gift_admin_service.set_stage(str(admin["sub"]), gift_id, payload.stage)


@router.post("/{gift_id}/cancel", response_model=AdminGiftDetail)
async def cancel_gift(
    gift_id: str,
    payload: AdminGiftCancelRequest,
    admin: dict = Depends(require_admin_jwt),
):
    return await gift_admin_service.cancel_gift(str(admin["sub"]), gift_id, payload.reason)
