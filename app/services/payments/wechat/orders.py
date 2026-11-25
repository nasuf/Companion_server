"""微信支付下单、查单、到账发放。

幂等地基：`wechat_pay_orders.out_trade_no` 唯一 + 事务内 `FOR UPDATE` 回放。
到账只认服务端下单时写定的 `user_id` / `vip_days` / `ticket_amount`，回调里的
金额只用于**校验**（不等则拒绝发放并留痕），不用于决定发多少。
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.db import db
from app.observability.events import EVT_PAYMENT_GRANT
from app.services import wallet
from app.services.payments.wechat import catalog, transport
from app.services.payments.wechat.errors import (
    WechatPayOrderStateError,
    WechatPayTrialUsedError,
    WechatPayUnknownProductError,
    WechatPayUpstreamError,
)
from app.services.runtime.tasks import fire_background
from app.services.vip import grants as vip_grants
from app.services.vip.entitlements import recompute_vip_entitlements

logger = logging.getLogger(__name__)

STATUS_CREATED = "created"
STATUS_PAID = "paid"
STATUS_GRANTED = "granted"
STATUS_CLOSED = "closed"
STATUS_REFUNDED = "refunded"
STATUS_FAILED = "failed"

# 不再发放权益的状态：回调重放只回放状态。
# `closed` 刻意不在其中：关单与用户付款存在竞态（我方刚关、用户在旧收银台上
# 付掉），微信说 SUCCESS 就是真钱到账，必须照常发放，否则钱收了货没发。
_NO_GRANT_STATUSES = frozenset({STATUS_GRANTED, STATUS_REFUNDED, STATUS_FAILED})

# 下单后多久过期：作为 time_expire 传给微信，过了这个点收银台就付不了。
# 关单 cron 的阈值（reconcile.CLOSE_AFTER_MINUTES）必须大于它。
ORDER_TTL_MINUTES = 15

_CST = timezone(timedelta(hours=8))

_JSAPI_PATH = "/v3/pay/transactions/jsapi"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _naive(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _new_out_trade_no() -> str:
    """商户订单号：前缀 + UTC 时间 + 随机。长度 <= 32，仅 ASCII 字母数字。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"BS{stamp}{secrets.token_hex(6).upper()}"


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """落库前的脱敏白名单：只留排查需要的字段，不存 payer 以外的用户信息。"""
    allowed = {
        "appid",
        "mchid",
        "out_trade_no",
        "transaction_id",
        "trade_type",
        "trade_state",
        "trade_state_desc",
        "bank_type",
        "success_time",
        "attach",
    }
    safe = {key: value for key, value in payload.items() if key in allowed}
    amount = payload.get("amount")
    if isinstance(amount, dict):
        safe["amount"] = {
            key: amount[key]
            for key in ("total", "payer_total", "currency", "payer_currency")
            if key in amount
        }
    return safe


async def _trial_unavailable(user_id: str) -> bool:
    """与 catalog 页同一判据：用过体验价或当前已是 VIP 都不能买。"""
    rows = await db.query_raw(
        "SELECT vip_trial_used, vip_until FROM user_wallets WHERE user_id = $1",
        user_id,
    )
    return bool(rows) and not wallet.vip_trial_available_from_row(rows[0])


def _time_expire() -> str:
    """微信要求 RFC3339（yyyy-MM-DDTHH:mm:ss+TIMEZONE）。"""
    expire = datetime.now(_CST) + timedelta(minutes=ORDER_TTL_MINUTES)
    return expire.replace(microsecond=0).isoformat()


async def create_order(
    *, user_id: str, product_id: str, openid: str
) -> dict[str, Any]:
    """下单：服务端定价 → 落 created 订单 → 调微信 JSAPI → 回 H5 调起参数。

    订单先落库再调微信：微信调用失败时订单留在 `created`，由关单 cron 收尾，
    不会出现"用户付了款但我方没有订单"的情况。
    """
    product = catalog.product_for(product_id)
    if product is None:
        raise WechatPayUnknownProductError(product_id)
    is_trial = product.product_id == catalog.TRIAL_PRODUCT_ID
    if is_trial and await _trial_unavailable(user_id):
        raise WechatPayTrialUsedError("体验价每个账号仅限一次")

    await wallet.ensure_wallet(user_id)
    app_id = settings.wechat_h5_app_id.strip()
    mch_id = settings.wechatpay_mch_id.strip()
    out_trade_no = _new_out_trade_no()

    await db.execute_raw(
        """
        INSERT INTO wechat_pay_orders (
            out_trade_no, mch_id, app_id, user_id, openid, product_id,
            product_title, kind, amount_fen, vip_days, ticket_amount, status
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'created')
        """,
        out_trade_no,
        mch_id,
        app_id,
        user_id,
        openid,
        product.product_id,
        product.title,
        product.kind,
        product.price_fen,
        product.vip_days,
        product.ticket_amount,
    )

    body = {
        "appid": app_id,
        "mchid": mch_id,
        "description": f"伴生 · {product.title}",
        "out_trade_no": out_trade_no,
        "notify_url": settings.wechatpay_notify_url.strip(),
        "time_expire": _time_expire(),
        "amount": {"total": product.price_fen, "currency": "CNY"},
        "payer": {"openid": openid},
    }

    try:
        response = await transport.request("POST", _JSAPI_PATH, body=body)
    except WechatPayUpstreamError as exc:
        if exc.status_code is not None and 400 <= exc.status_code < 500:
            # 微信明确拒绝下单：微信侧没有这笔单、也没有 prepay_id，不可能被支付，
            # 直接关掉，不留给对账 cron 查 24 小时的 ORDER_NOT_EXIST。
            await db.execute_raw(
                """
                UPDATE wechat_pay_orders
                SET status = 'closed', closed_at = CURRENT_TIMESTAMP,
                    fail_reason = $2, updated_at = CURRENT_TIMESTAMP
                WHERE out_trade_no = $1 AND status = 'created'
                """,
                out_trade_no,
                f"jsapi_prepay_rejected: {exc.code or exc.status_code}"[:200],
            )
        else:
            await _mark_prepay_failed(out_trade_no)
        raise
    except Exception:
        await _mark_prepay_failed(out_trade_no)
        raise

    prepay_id = response.get("prepay_id")
    if not isinstance(prepay_id, str) or not prepay_id:
        await db.execute_raw(
            """
            UPDATE wechat_pay_orders
            SET fail_reason = 'prepay_id_missing', updated_at = CURRENT_TIMESTAMP
            WHERE out_trade_no = $1 AND status = 'created'
            """,
            out_trade_no,
        )
        raise WechatPayOrderStateError("微信未返回 prepay_id")

    await db.execute_raw(
        """
        UPDATE wechat_pay_orders
        SET prepay_id = $2, updated_at = CURRENT_TIMESTAMP
        WHERE out_trade_no = $1
        """,
        out_trade_no,
        prepay_id,
    )
    logger.info(
        "wechat pay order created user=%s product=%s",
        user_id[:8],
        product.product_id,
        extra={
            "event": "wechat_pay_order_created",
            "user_id": user_id,
            "out_trade_no": out_trade_no,
            "product_id": product.product_id,
        },
    )
    return {
        "out_trade_no": out_trade_no,
        "amount_fen": product.price_fen,
        "pay_params": transport.jsapi_pay_params(prepay_id),
    }


async def _mark_prepay_failed(out_trade_no: str) -> None:
    """结果不确定（超时、5xx、应答验签失败）：订单停在 created，由对账 cron 查单收尾；
    用户重新点购买会得到新订单。"""
    await db.execute_raw(
        """
        UPDATE wechat_pay_orders
        SET fail_reason = 'jsapi_prepay_failed', updated_at = CURRENT_TIMESTAMP
        WHERE out_trade_no = $1 AND status = 'created'
        """,
        out_trade_no,
    )


async def find_order(out_trade_no: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT out_trade_no, user_id, product_id, product_title, status,
               amount_fen, paid_amount_fen, vip_days, ticket_amount,
               transaction_id, trade_state, success_time, granted_at,
               refunded_fen, fail_reason, created_at
        FROM wechat_pay_orders
        WHERE out_trade_no = $1
        """,
        out_trade_no,
    )
    return dict(rows[0]) if rows else None


async def settle_paid_order(
    *,
    out_trade_no: str,
    transaction_id: str | None,
    trade_state: str,
    paid_amount_fen: int | None,
    success_time: datetime | None,
    payload: dict[str, Any],
    notify_id: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """幂等结算一笔已支付订单：校验金额 → 单事务发放权益 → 置 granted。

    回调与主动查单共用。返回 ``{"status": ..., "replay": bool}``。
    金额缺失或币种不是 CNY 与金额不符同等处理：不发放，订单落 failed，后台
    可对其发起退款（见 refunds.create_refund）。
    """
    payload_json = json.dumps(_safe_payload(payload), ensure_ascii=False)
    granted_now = False
    user_id: str | None = None
    grants_vip = False

    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT user_id, product_id, status, amount_fen, vip_days, ticket_amount
            FROM wechat_pay_orders
            WHERE out_trade_no = $1
            FOR UPDATE
            """,
            out_trade_no,
        )
        if not rows:
            # 不是我方订单号：可能是同一商户号下的其它业务，交给调用方决定。
            return {"status": "unknown_order", "replay": False}

        order = rows[0]
        user_id = str(_field(order, "user_id"))
        status = str(_field(order, "status"))
        amount_fen = int(_field(order, "amount_fen") or 0)
        vip_days = int(_field(order, "vip_days") or 0)
        ticket_amount = int(_field(order, "ticket_amount") or 0)
        grants_vip = vip_days > 0

        if status in _NO_GRANT_STATUSES:
            return {"status": status, "replay": True}
        if status == STATUS_CLOSED:
            logger.warning(
                "wechat pay paid after local close order=%s",
                out_trade_no,
                extra={
                    "event": "wechat_pay_paid_after_close",
                    "out_trade_no": out_trade_no,
                    "user_id": user_id,
                },
            )

        # 金额不符一律不发货：既防配置漂移，也防伪造回调压低金额。
        mismatch = None
        if paid_amount_fen is None:
            mismatch = "amount_missing"
        elif paid_amount_fen != amount_fen:
            mismatch = "amount_mismatch"
        elif currency is not None and currency != "CNY":
            mismatch = "currency_mismatch"
        if mismatch is not None:
            await tx.execute_raw(
                """
                UPDATE wechat_pay_orders
                SET status = 'failed', trade_state = $2,
                    paid_amount_fen = $3, fail_reason = $7,
                    transaction_id = COALESCE($4, transaction_id),
                    notify_id = COALESCE($5, notify_id),
                    raw_payload = $6::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE out_trade_no = $1
                """,
                out_trade_no,
                trade_state,
                paid_amount_fen,
                transaction_id,
                notify_id,
                payload_json,
                mismatch,
            )
            logger.error(
                "wechat pay %s order=%s expected=%s paid=%s",
                mismatch,
                out_trade_no,
                amount_fen,
                paid_amount_fen,
                extra={
                    "event": "wechat_pay_amount_mismatch",
                    "out_trade_no": out_trade_no,
                    "user_id": user_id,
                },
            )
            return {"status": STATUS_FAILED, "replay": False}

        is_trial = str(_field(order, "product_id")) == catalog.TRIAL_PRODUCT_ID
        if is_trial:
            # 下单时的检查挡不住"先下两单再分别付款"：这里在钱包行锁内复核，
            # 第二笔不发放、落 failed，由后台原路退回那 1 元。
            await wallet.ensure_wallet(user_id, client=tx)
            locked = await tx.query_raw(
                "SELECT vip_trial_used FROM user_wallets WHERE user_id = $1 FOR UPDATE",
                user_id,
            )
            if locked and bool(_field(locked[0], "vip_trial_used")):
                await tx.execute_raw(
                    """
                    UPDATE wechat_pay_orders
                    SET status = 'failed', trade_state = $2,
                        paid_amount_fen = $3, fail_reason = 'trial_already_used',
                        transaction_id = COALESCE($4, transaction_id),
                        success_time = COALESCE($5::timestamp, success_time),
                        raw_payload = $6::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE out_trade_no = $1
                    """,
                    out_trade_no,
                    trade_state,
                    paid_amount_fen,
                    transaction_id,
                    _naive(success_time),
                    payload_json,
                )
                logger.warning(
                    "wechat pay duplicate trial order=%s",
                    out_trade_no,
                    extra={
                        "event": "wechat_pay_duplicate_trial",
                        "out_trade_no": out_trade_no,
                        "user_id": user_id,
                    },
                )
                return {"status": STATUS_FAILED, "replay": False}

        if ticket_amount > 0:
            await wallet.credit_tickets(
                user_id,
                ticket_amount,
                source=catalog.SOURCE_WECHAT_PAY,
                source_id=out_trade_no,
                metadata={
                    "product_id": str(_field(order, "product_id")),
                    "provider": catalog.PROVIDER_WECHAT,
                    "transaction_id": transaction_id,
                },
                client=tx,
            )
        if is_trial:
            await tx.execute_raw(
                """
                UPDATE user_wallets
                SET vip_trial_used = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $1
                """,
                user_id,
            )

        await tx.execute_raw(
            """
            UPDATE wechat_pay_orders
            SET status = 'granted', trade_state = $2,
                transaction_id = COALESCE($3, transaction_id),
                paid_amount_fen = COALESCE($4, paid_amount_fen),
                success_time = COALESCE($5::timestamp, success_time),
                granted_at = CURRENT_TIMESTAMP,
                closed_at = NULL,
                wallet_ledger_source_id = $1,
                notify_id = COALESCE($6, notify_id),
                fail_reason = NULL,
                raw_payload = $7::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE out_trade_no = $1
            """,
            out_trade_no,
            trade_state,
            transaction_id,
            paid_amount_fen,
            _naive(success_time),
            notify_id,
            payload_json,
        )
        if grants_vip:
            # vip_until 由 entitlements 从所有渠道的已发放订单重算，不就地累加。
            await recompute_vip_entitlements(user_id, client=tx, clear_lapse=False)
        granted_now = True

    if granted_now and grants_vip and user_id is not None:
        # 与 Apple 侧一致：VIP 到账后立即发当月权益，失败不影响已生效 VIP。
        fire_background(_grant_vip_monthly_safe(user_id))

    logger.info(
        "wechat pay grant ok order=%s user=%s",
        out_trade_no,
        (user_id or "")[:8],
        extra={
            "event": EVT_PAYMENT_GRANT,
            "provider": catalog.PROVIDER_WECHAT,
            "out_trade_no": out_trade_no,
            "user_id": user_id,
        },
    )
    return {"status": STATUS_GRANTED, "replay": False}


async def _grant_vip_monthly_safe(user_id: str) -> None:
    try:
        await vip_grants.grant_monthly(user_id)
    except Exception:
        logger.exception("wechat pay monthly vip grant failed user=%s", user_id[:8])


async def mark_order_closed(out_trade_no: str, *, reason: str) -> None:
    """只关未支付订单。已支付/已发放的订单绝不被关单逻辑改状态。"""
    await db.execute_raw(
        """
        UPDATE wechat_pay_orders
        SET status = 'closed', closed_at = CURRENT_TIMESTAMP,
            fail_reason = $2, updated_at = CURRENT_TIMESTAMP
        WHERE out_trade_no = $1 AND status = 'created'
        """,
        out_trade_no,
        reason,
    )


async def mark_refunded_before_grant(
    out_trade_no: str, *, payload: dict[str, Any]
) -> dict[str, Any]:
    """查单发现"已支付又已退款"、而我方从未发放：只记账，不发货。

    只改 created/closed 的订单；已发放的订单走退款回收路径，这里不动。
    """
    rows = await db.query_raw(
        """
        UPDATE wechat_pay_orders
        SET status = 'refunded', trade_state = 'REFUND',
            refunded_fen = amount_fen, refunded_at = CURRENT_TIMESTAMP,
            fail_reason = 'refunded_before_grant', raw_payload = $2::jsonb,
            updated_at = CURRENT_TIMESTAMP
        WHERE out_trade_no = $1 AND status IN ('created', 'closed')
        RETURNING user_id
        """,
        out_trade_no,
        json.dumps(_safe_payload(payload), ensure_ascii=False),
    )
    if not rows:
        current = await find_order(out_trade_no)
        return {"status": str(current["status"]) if current else "unknown_order", "replay": True}
    logger.warning(
        "wechat pay order refunded before grant order=%s",
        out_trade_no,
        extra={
            "event": "wechat_pay_refunded_before_grant",
            "out_trade_no": out_trade_no,
            "user_id": str(_field(rows[0], "user_id")),
        },
    )
    return {"status": STATUS_REFUNDED, "replay": False}
