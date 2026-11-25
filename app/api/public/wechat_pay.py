"""服务号 H5 微信支付的公开路由。

与 `/iap/apple/*` 平行且互不影响：Apple 的验签、通知、退款路由一行不动。

两道开关都要过才能下单：
* `WECHATPAY_ENABLED`（部署级，密钥齐备才打开）
* `system_config.real_money_payments_enabled`（运营级，与 Apple 共用，审核期能一键关）
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.deps import require_redis
from app.redis_client import get_redis
from app.api.jwt_auth import require_user
from app.config import WECHATPAY_NOTIFY_ROUTE_PATH, settings
from app.db import db
from app.models.wechat_pay import (
    WechatPayAuthorizeRequest,
    WechatPayAuthorizeResponse,
    WechatPayCatalogResponse,
    WechatPayOrderRequest,
    WechatPayOrderResponse,
    WechatPayOrderStatusResponse,
    WechatPayProductItem,
)
from app.observability import bind_context
from app.observability.events import (
    EVT_PAYMENT_NOTIFICATION_FAIL,
    EVT_PAYMENT_VERIFY_FAIL,
)
from app.services import wallet
from app.services.auth_security import (
    clear_login_failures,
    enforce_login_rate_limit,
    record_login_failure,
)
from app.services.payments.availability import require_real_money_payments
from app.services.payments.wechat import accounts, catalog, notifications, orders
from app.services.payments.wechat.catalog import WechatProduct
from app.services.payments.wechat.errors import (
    WechatPayConfigError,
    WechatPayDecryptError,
    WechatPayOpenidMissingError,
    WechatPayOrderStateError,
    WechatPaySignatureError,
    WechatPayTrialUsedError,
    WechatPayUnknownProductError,
    WechatPayUpstreamError,
)
from app.services.user_profile import resolve_display_identity
from app.services.phone_auth import IdentityConflict
from app.services.wechat_auth import (
    WeChatLoginError,
    bind_wechat_to_user,
    exchange_wechat_h5_code,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pay/wechat", tags=["wechat-pay"])

# 回调体上限：正常回调不到 4KB，给足余量同时挡住超大 body。
_MAX_NOTIFY_BODY_BYTES = 64 * 1024


def _require_enabled() -> None:
    if not settings.wechatpay_enabled:
        raise HTTPException(status_code=503, detail="wechat_pay_disabled")


# 每分钟最多下几单：正常用户点几次就够了，挡住刷单堆积订单行和微信 API 调用。
_ORDER_RATE_PER_MINUTE = 10


async def _enforce_order_rate_limit(user_id: str) -> None:
    redis = await get_redis()
    key = f"wechatpay:order:minute:{user_id}"
    count = int(await redis.incr(key))
    if count == 1:
        await redis.expire(key, 60)
    if count > _ORDER_RATE_PER_MINUTE:
        raise HTTPException(status_code=429, detail="too_many_orders")


def _item(product: WechatProduct, *, purchasable: bool = True, reason: str | None = None):
    return WechatPayProductItem(
        product_id=product.product_id,
        title=product.title,
        price_fen=product.price_fen,
        list_price_fen=product.list_price_fen,
        vip_days=product.vip_days,
        ticket_amount=product.ticket_amount,
        badge=product.badge,
        purchasable=purchasable,
        unavailable_reason=reason,
    )


async def _account_label(user_id: str) -> str | None:
    """付款前展示的"买给哪个账号"：昵称 + 手机号尾号。

    手机号只露后四位（它在 `auth_identities` 里，provider='phone'，见
    `services/phone_auth`）。目的是让用户在付款前确认账号，所以要足以辨认、
    又不把完整号码渲染到页面上。
    """
    user = await db.user.find_unique(where={"id": user_id})
    if user is None:
        return None
    display_name, _ = await resolve_display_identity(user)
    identity = await db.authidentity.find_first(
        where={"userId": user_id, "provider": "phone"}
    )
    phone = getattr(identity, "providerAccountId", None) if identity else None
    tail = f"（手机尾号 {phone[-4:]}）" if isinstance(phone, str) and len(phone) >= 4 else ""
    return f"{display_name or getattr(user, 'username', '') or '当前账号'}{tail}"


@router.get("/catalog", response_model=WechatPayCatalogResponse)
async def wechat_pay_catalog(response: Response, payload: dict = Depends(require_user)):
    """H5 商城页的全部数据。未配置或运营开关关闭时 enabled=false，前端置灰按钮。"""
    response.headers["Cache-Control"] = "no-store"
    user_id = str(payload["sub"])
    app_id = settings.wechat_h5_app_id.strip()

    enabled = settings.wechatpay_enabled
    if enabled:
        try:
            await require_real_money_payments()
        except HTTPException:
            enabled = False

    snapshot = await wallet.full_wallet(user_id)
    trial_available = bool(snapshot.get("vip_trial_available"))

    openid_bound = True
    try:
        await accounts.openid_for_user(user_id, app_id=app_id)
    except WechatPayOpenidMissingError:
        openid_bound = False

    account_label = await _account_label(user_id)
    return WechatPayCatalogResponse(
        enabled=enabled,
        openid_bound=openid_bound,
        vip_plans=[_item(product) for product in catalog.VIP_PRODUCTS],
        trial=_item(
            catalog.TRIAL_PRODUCT,
            purchasable=trial_available,
            reason=None if trial_available else "体验价每个账号仅限一次",
        ),
        ticket_packs=[_item(product) for product in catalog.TICKET_PRODUCTS],
        account_label=account_label,
        vip_is_active=bool(snapshot.get("is_vip")),
        vip_until=snapshot.get("vip_until"),
        ticket_balance=int(snapshot.get("ticket_balance") or 0),
    )


def _log_authorize_conflict(user_id: str) -> None:
    logger.warning(
        "wechat pay authorize rejected: wechat belongs to another account",
        extra={"event": "wechat_pay_authorize_conflict", "user_id": user_id},
    )


@router.post("/authorize", response_model=WechatPayAuthorizeResponse)
async def authorize_wechat_pay(
    data: WechatPayAuthorizeRequest,
    request: Request,
    payload: dict = Depends(require_user),
):
    """登记付款用的服务号 openid（网页授权 snsapi_userinfo 的回跳 code）。

    手机号登录、或登录早于本功能的 H5 会话没有 openid，下单前先走这一步。
    只认已登录账号（JWT）；若这个微信号本身属于另一个账号，返回 409
    `wechat_bound_to_other_account`，不把 openid 记过来——否则钱会付到一个
    用户在 App 里看不到的账号上，这正是要防的"账号分裂"。

    必须拿到 unionid 才放行：`auth_identities.openid` 会被 App 登录覆盖成移动
    应用的 openid，只凭服务号 openid 查不出"这个微信是不是 App 里另一个账号"。
    服务号已绑定开放平台，snsapi_userinfo 授权总会带 unionid；拿不到说明配置
    异常，宁可拒绝也不冒险绑定（fail closed）。
    """
    user_id = str(payload["sub"])
    rate_limit_key = "wechat:payauth"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_h5_code(data.code)
    except WeChatLoginError as exc:
        await record_login_failure(request, rate_limit_key)
        raise HTTPException(status_code=401, detail="wechat_authorization_failed") from exc
    await clear_login_failures(request, rate_limit_key)
    if not token_payload.unionid:
        logger.warning(
            "wechat pay authorize rejected: unionid missing",
            extra={"event": "wechat_pay_authorize_no_unionid", "user_id": user_id},
        )
        raise HTTPException(status_code=409, detail="wechat_unionid_unavailable")

    with bind_context(user_id=user_id):
        owner = await accounts.conflicting_owner(
            user_id=user_id,
            openid=token_payload.openid,
            unionid=token_payload.unionid,
        )
        if owner is not None:
            _log_authorize_conflict(user_id)
            raise HTTPException(status_code=409, detail="wechat_bound_to_other_account")
        # 把这个微信同时绑成当前账号的登录方式：只登记付款 openid 的话，用户之后在
        # App 用同一微信登录会按 unionid 找不到身份而新建一个空账号，VIP 却在这里。
        try:
            await bind_wechat_to_user(user_id, token_payload)
        except IdentityConflict as exc:
            _log_authorize_conflict(user_id)
            raise HTTPException(
                status_code=409, detail="wechat_bound_to_other_account"
            ) from exc
        await accounts.bind_openid(
            user_id=user_id,
            openid=token_payload.openid,
            app_id=settings.wechat_h5_app_id.strip(),
            unionid=token_payload.unionid,
        )
    return WechatPayAuthorizeResponse(openid_bound=True)


@router.post(
    "/orders",
    response_model=WechatPayOrderResponse,
    dependencies=[Depends(require_redis)],
)
async def create_wechat_pay_order(
    data: WechatPayOrderRequest,
    payload: dict = Depends(require_user),
):
    """创建订单并返回 H5 调起支付所需参数。价格一律由服务端目录决定。"""
    _require_enabled()
    await require_real_money_payments()
    user_id = str(payload["sub"])
    app_id = settings.wechat_h5_app_id.strip()
    await _enforce_order_rate_limit(user_id)

    with bind_context(user_id=user_id):
        try:
            openid, unionid = await accounts.payer_for_user(user_id, app_id=app_id)
        except WechatPayOpenidMissingError as exc:
            # 409 让前端区分"需要再授权一次"与"真的失败了"。
            raise HTTPException(status_code=409, detail="wechat_authorization_required") from exc
        # 登记之后这个微信可能已成了别的账号的登录方式：下单前再核一次，宁可拦下
        # 也不让"用 B 的微信给 A 付款"。
        if await accounts.conflicting_owner(user_id=user_id, openid=openid, unionid=unionid):
            _log_authorize_conflict(user_id)
            raise HTTPException(status_code=409, detail="wechat_bound_to_other_account")
        try:
            result = await orders.create_order(
                user_id=user_id, product_id=data.product_id, openid=openid
            )
        except WechatPayUnknownProductError as exc:
            raise HTTPException(status_code=404, detail="unknown_product") from exc
        except WechatPayTrialUsedError as exc:
            raise HTTPException(status_code=409, detail="trial_already_used") from exc
        except WechatPayConfigError as exc:
            logger.exception("wechat pay misconfigured")
            raise HTTPException(status_code=503, detail="wechat_pay_unavailable") from exc
        except (
            WechatPayUpstreamError,
            WechatPayOrderStateError,
            WechatPaySignatureError,
        ) as exc:
            logger.warning(
                "wechat pay prepay failed",
                extra={"event": EVT_PAYMENT_VERIFY_FAIL, "product_id": data.product_id},
            )
            raise HTTPException(status_code=502, detail="wechat_pay_prepay_failed") from exc
    return WechatPayOrderResponse(**result)


@router.get("/orders/{out_trade_no}", response_model=WechatPayOrderStatusResponse)
async def wechat_pay_order_status(
    out_trade_no: str,
    response: Response,
    payload: dict = Depends(require_user),
):
    """H5 支付后轮询。只能查自己的订单。"""
    response.headers["Cache-Control"] = "no-store"
    user_id = str(payload["sub"])
    order = await orders.find_order(out_trade_no)
    # 不是自己的订单与不存在返回同一个 404，避免探测他人订单号。
    if order is None or str(order.get("user_id")) != user_id:
        raise HTTPException(status_code=404, detail="order_not_found")
    snapshot = await wallet.full_wallet(user_id)
    return WechatPayOrderStatusResponse(
        out_trade_no=out_trade_no,
        status=str(order["status"]),
        product_title=str(order["product_title"]),
        amount_fen=int(order["amount_fen"] or 0),
        vip_is_active=bool(snapshot.get("is_vip")),
        vip_until=snapshot.get("vip_until"),
        ticket_balance=int(snapshot.get("ticket_balance") or 0),
    )


@router.post("/notifications")
async def wechat_pay_notification(request: Request):
    """微信支付回调。无鉴权入口，安全完全依赖验签 + 时间戳 + 解密。

    HTTP 语义见 `payments/wechat/notifications` 模块注释。
    """
    raw = await request.body()
    if len(raw) > _MAX_NOTIFY_BODY_BYTES:
        raise HTTPException(status_code=413, detail="payload_too_large")
    body = raw.decode("utf-8", errors="replace")
    try:
        envelope = notifications.verify_and_decrypt(
            body=body,
            timestamp=request.headers.get("Wechatpay-Timestamp"),
            nonce=request.headers.get("Wechatpay-Nonce"),
            signature=request.headers.get("Wechatpay-Signature"),
            serial=request.headers.get("Wechatpay-Serial"),
        )
    except (WechatPaySignatureError, WechatPayDecryptError) as exc:
        logger.warning(
            "wechat pay notification rejected: %s",
            type(exc).__name__,
            extra={"event": EVT_PAYMENT_NOTIFICATION_FAIL},
        )
        # 401：微信会重推，但验签失败本就不会通过；到账由 cron 查单兜底。
        return Response(
            status_code=401,
            media_type="application/json",
            content='{"code":"FAIL","message":"signature verification failed"}',
        )
    except WechatPayConfigError as exc:
        logger.exception("wechat pay notification misconfigured")
        raise HTTPException(status_code=503, detail="wechat_pay_unavailable") from exc

    try:
        await notifications.handle_notification(envelope)
    except Exception:
        # 让微信按退避重推。审计行已落库，process_error 记了原因。
        return Response(
            status_code=500,
            media_type="application/json",
            content='{"code":"FAIL","message":"retry later"}',
        )
    return Response(
        status_code=200, media_type="application/json", content='{"code":"SUCCESS"}'
    )


# 回调路由的实际 path；启动自检用 config 里的同名常量比对 WECHATPAY_NOTIFY_URL。
NOTIFY_ROUTE_PATH = f"{router.prefix}/notifications"
assert NOTIFY_ROUTE_PATH == WECHATPAY_NOTIFY_ROUTE_PATH

__all__ = ["router", "NOTIFY_ROUTE_PATH"]
