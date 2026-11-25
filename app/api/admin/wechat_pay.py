from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.wechat_pay import (
    AdminWechatPayOrderList,
    AdminWechatPayRefundRequest,
    AdminWechatPayRefundResponse,
)
from app.services.payments.wechat import admin as wechat_pay_admin
from app.services.payments.wechat.errors import (
    WechatPayConfigError,
    WechatPayOrderStateError,
    WechatPaySignatureError,
    WechatPayUpstreamError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/wechat-pay",
    tags=["admin", "wechat-pay"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("/orders", response_model=AdminWechatPayOrderList)
async def list_orders(
    status: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    query: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await wechat_pay_admin.list_orders(
        status=status, user_id=user_id, query=query, limit=limit, offset=offset
    )


@router.post("/refunds", response_model=AdminWechatPayRefundResponse)
async def refund_order(
    data: AdminWechatPayRefundRequest,
    payload: dict = Depends(require_admin_jwt),
):
    """人工全额退款（内测规则：未使用的可全额退）。

    这是一次**不可逆的出款**，所以：不允许部分退款金额由前端传入（金额取订单
    原价），同一订单已有进行中的退款单时直接回放那一单，不重复出款。
    """
    actor_id = str(payload.get("sub") or "")
    try:
        result = await wechat_pay_admin.refund_order(
            out_trade_no=data.out_trade_no, actor_id=actor_id, reason=data.reason
        )
    except WechatPayOrderStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WechatPayConfigError as exc:
        logger.exception("wechat pay refund misconfigured")
        raise HTTPException(status_code=503, detail="wechat_pay_unavailable") from exc
    except WechatPaySignatureError as exc:
        # 微信可能已受理：退款单留在 pending，cron 按同一单号查单收尾。
        logger.warning("wechat pay refund response unverifiable")
        raise HTTPException(status_code=502, detail="wechat_pay_refund_uncertain") from exc
    except WechatPayUpstreamError as exc:
        logger.warning("wechat pay refund rejected: %s", exc.code or "unknown")
        raise HTTPException(status_code=502, detail="wechat_pay_refund_failed") from exc
    return AdminWechatPayRefundResponse(**result)
