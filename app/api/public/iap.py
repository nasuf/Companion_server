from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import require_redis
from app.api.jwt_auth import require_user
from app.observability import bind_context
from app.observability.events import EVT_PAYMENT_VERIFY_FAIL, EVT_PAYMENT_VERIFY_OK
from app.models.iap import IapVerifyRequest, IapVerifyResponse
from app.services.payments import grant, notifications
from app.services.payments.errors import (
    AppleVerificationError,
    PaymentError,
    UnknownProductError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/iap/apple", tags=["iap"])


@router.post(
    "/verify",
    response_model=IapVerifyResponse,
    dependencies=[Depends(require_redis)],
)
async def verify_purchase(
    data: IapVerifyRequest,
    payload: dict = Depends(require_user),
):
    """客户端购买/恢复成功后提交 transaction_id，服务端向 Apple 校验并幂等到账。"""
    user_id = str(payload["sub"])
    with bind_context(user_id=user_id):
        try:
            result = await grant.verify_and_grant(user_id, data.transaction_id)
        except UnknownProductError as exc:
            logger.warning(
                "iap verify unknown product",
                extra={"event": EVT_PAYMENT_VERIFY_FAIL, "product_id": exc.product_id},
            )
            raise HTTPException(status_code=404, detail="unknown_product") from exc
        except AppleVerificationError as exc:
            logger.warning(
                "iap verify apple rejected txn",
                extra={
                    "event": EVT_PAYMENT_VERIFY_FAIL,
                    "transaction_id": data.transaction_id,
                },
            )
            raise HTTPException(status_code=402, detail="verification_failed") from exc
        except PaymentError as exc:
            raise HTTPException(status_code=422, detail="payment_error") from exc
        logger.info(
            "iap verify ok",
            extra={
                "event": EVT_PAYMENT_VERIFY_OK,
                "transaction_id": data.transaction_id,
                "kind": result.get("kind"),
            },
        )
        return result


@router.post("/notifications")
async def app_store_notifications(request: Request):
    """App Store Server Notifications V2 webhook。无用户 JWT，鉴权=JWS 验签。

    验签失败 → 401 拒绝伪造；已落库前提下处理异常也回 200（幂等重放补处理），
    避免 Apple 因非 2xx 无限重推。
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_body") from exc
    signed = body.get("signedPayload") if isinstance(body, dict) else None
    if not signed:
        raise HTTPException(status_code=400, detail="missing_signed_payload")
    try:
        await notifications.apply_notification(signed)
    except AppleVerificationError as exc:
        raise HTTPException(status_code=401, detail="invalid_signature") from exc
    except Exception:
        # 已在 apply_notification 内落库；此处兜底不让非验签异常打成非 2xx。
        logger.exception("iap notification handler error")
    return {"ok": True}
