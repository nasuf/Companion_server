"""App Store Server Notifications V2 处理：验签 → 落库(幂等) → 按类型分派。

webhook 唯一鉴权 = JWS 验签（apple_env.verify_notification）。收到即落 iap_notifications
（notification_uuid 唯一），已收过直接短路。处理失败只记 process_error 不抛，让端点回
200 防 Apple 无限重推——靠 unprocessed 行 + 幂等重放补处理。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from appstoreserverlibrary.models.NotificationTypeV2 import NotificationTypeV2
from appstoreserverlibrary.models.Subtype import Subtype

from app.db import db
from app.observability.events import (
    EVT_PAYMENT_NOTIFICATION,
    EVT_PAYMENT_NOTIFICATION_FAIL,
    EVT_PAYMENT_REFUND,
    EVT_PAYMENT_REVOKE,
    EVT_PAYMENT_SUB_EXPIRE,
    EVT_PAYMENT_SUB_RENEW,
)
from app.services import wallet
from app.services.payments import catalog, grant
from app.services.payments.apple import environment as apple_env

logger = logging.getLogger(__name__)

PROVIDER_APPLE = "apple"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _enum_val(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


async def apply_notification(signed_payload: str) -> None:
    decoded, env = await apple_env.verify_notification(signed_payload)  # 验签失败→抛，端点回 401
    uuid = _field(decoded, "notificationUUID") or ""
    ntype = _enum_val(_field(decoded, "notificationType"))
    subtype = _enum_val(_field(decoded, "subtype"))
    data = _field(decoded, "data")

    txn = None
    renewal = None
    if data is not None and _field(data, "signedTransactionInfo"):
        txn = await asyncio.to_thread(
            apple_env.verify_signed_transaction, _field(data, "signedTransactionInfo"), env
        )
    if data is not None and _field(data, "signedRenewalInfo"):
        renewal = await asyncio.to_thread(
            apple_env.verify_renewal_info, _field(data, "signedRenewalInfo"), env
        )

    original_txn_id = _field(txn, "originalTransactionId") or _field(
        renewal, "originalTransactionId"
    )
    transaction_id = _field(txn, "transactionId")

    is_new = await _insert_notification(
        uuid, ntype, subtype, env, original_txn_id, transaction_id, signed_payload
    )
    logger.info(
        "iap notification %s/%s",
        ntype,
        subtype,
        extra={
            "event": EVT_PAYMENT_NOTIFICATION,
            "notification_type": ntype,
            "notification_subtype": subtype,
            "environment": env,
            "transaction_id": transaction_id,
        },
    )
    if not is_new:
        return  # 幂等：已收到过这条通知

    try:
        await _dispatch(ntype, subtype, txn, renewal, env, uuid, original_txn_id)
        await _mark_processed(uuid, None)
    except Exception as exc:
        logger.exception(
            "iap notification dispatch failed uuid=%s type=%s",
            uuid[:12],
            ntype,
            extra={
                "event": EVT_PAYMENT_NOTIFICATION_FAIL,
                "notification_type": ntype,
                "environment": env,
            },
        )
        await _mark_processed(uuid, str(exc))


async def _dispatch(
    ntype: str | None,
    subtype: str | None,
    txn: Any,
    renewal: Any,
    env: str,
    uuid: str,
    original_txn_id: str | None,
) -> None:
    if ntype in (NotificationTypeV2.SUBSCRIBED.value, NotificationTypeV2.DID_RENEW.value):
        await _grant_from_notification(txn, env, uuid)
        return
    if ntype == NotificationTypeV2.DID_CHANGE_RENEWAL_STATUS.value:
        await _update_renewal_status(original_txn_id, renewal)
        return
    if ntype == NotificationTypeV2.DID_CHANGE_RENEWAL_PREF.value:
        await _update_renewal_pref(original_txn_id, renewal)
        return
    if ntype == NotificationTypeV2.DID_FAIL_TO_RENEW.value and subtype == Subtype.GRACE_PERIOD.value:
        await _set_subscription_state(original_txn_id, "in_grace", renewal=renewal, ntype=ntype, subtype=subtype)
        return
    if ntype in (
        NotificationTypeV2.EXPIRED.value,
        NotificationTypeV2.GRACE_PERIOD_EXPIRED.value,
    ):
        # 不主动清 vip_until：它已是过去时，is_vip 自然为 False；限时钞票由既有
        # vip_expire_clear cron 清，避免与既有清算路径打架。
        await _set_subscription_state(original_txn_id, "expired", ntype=ntype, subtype=subtype)
        logger.info(
            "iap subscription expired otxn=%s",
            (original_txn_id or "")[:12],
            extra={"event": EVT_PAYMENT_SUB_EXPIRE, "environment": env},
        )
        return
    if ntype == NotificationTypeV2.REFUND.value:
        await _handle_refund(txn, env)
        return
    if ntype == NotificationTypeV2.REVOKE.value:
        await _handle_revoke(txn, original_txn_id, env)
        return
    # TEST / PRICE_INCREASE / CONSUMPTION_REQUEST / REFUND_DECLINED 等：仅落库审计。


async def _grant_from_notification(txn: Any, env: str, uuid: str) -> None:
    if txn is None:
        return
    original_txn_id = _field(txn, "originalTransactionId") or _field(txn, "transactionId")
    user_id = await _user_for_original_txn(original_txn_id)
    if not user_id:
        # 通知早于客户端 verify 落库（少见）：跳过，客户端下次 verify 会补到账。
        logger.warning("iap renew notification with no known user otxn=%s", (original_txn_id or "")[:12])
        return
    product = catalog.product_for(_field(txn, "productId") or "")
    if product is None:
        logger.warning("iap renew unknown product %s", _field(txn, "productId"))
        return
    await grant.record_and_grant(user_id, txn, env, product, notification_uuid=uuid)
    logger.info(
        "iap subscription renewed user=%s",
        user_id[:8],
        extra={"event": EVT_PAYMENT_SUB_RENEW, "environment": env},
    )


async def _handle_refund(txn: Any, env: str) -> None:
    if txn is None:
        return
    transaction_id = _field(txn, "transactionId") or ""
    row = await grant._find_transaction(transaction_id)  # noqa: SLF001 (同域复用)
    if row is None:
        return
    if row["status"] in ("refunded", "revoked"):
        return  # 幂等：已清算
    user_id = row["user_id"]
    product = catalog.product_for(row["product_id"])
    async with db.tx() as tx:
        if product is not None and product.grants_tickets:
            await _reverse_tickets(tx, user_id, product.ticket_amount, transaction_id, env)
        await tx.execute_raw(
            """
            UPDATE iap_transactions SET status = 'refunded', updated_at = CURRENT_TIMESTAMP
            WHERE provider = $1 AND transaction_id = $2
            """,
            PROVIDER_APPLE,
            transaction_id,
        )
    if product is not None and product.grants_vip:
        await _expire_vip_now(user_id)
        await _set_subscription_state(
            _field(txn, "originalTransactionId"), "refunded", ntype="REFUND"
        )
    logger.info(
        "iap refund user=%s txn=%s",
        user_id[:8],
        transaction_id[:12],
        extra={"event": EVT_PAYMENT_REFUND, "environment": env},
    )


async def _handle_revoke(txn: Any, original_txn_id: str | None, env: str) -> None:
    user_id = await _user_for_original_txn(original_txn_id)
    if not user_id:
        return
    await _expire_vip_now(user_id)
    await _set_subscription_state(original_txn_id, "revoked", ntype="REVOKE")
    if txn is not None:
        await db.execute_raw(
            """
            UPDATE iap_transactions SET status = 'revoked', updated_at = CURRENT_TIMESTAMP
            WHERE provider = $1 AND transaction_id = $2
            """,
            PROVIDER_APPLE,
            _field(txn, "transactionId") or "",
        )
    logger.info(
        "iap revoke user=%s",
        user_id[:8],
        extra={"event": EVT_PAYMENT_REVOKE, "environment": env},
    )


async def _reverse_tickets(
    tx: Any, user_id: str, amount: int, transaction_id: str, env: str
) -> None:
    """退款反向扣钞票，floor 0（仿 admin_adjust_tickets），source_id 复用 txn 幂等。"""
    if amount <= 0:
        return
    dup = await tx.query_raw(
        """
        SELECT 1 FROM wallet_ledger
        WHERE source = $1 AND source_id = $2 LIMIT 1
        """,
        catalog.SOURCE_APPLE_IAP_REFUND,
        transaction_id,
    )
    if dup:
        return  # 已反向过
    locked = await tx.query_raw(
        "SELECT ticket_balance FROM user_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    if not locked:
        return
    current = int(_field(locked[0], "ticket_balance", 0) or 0)
    new_balance = max(0, current - amount)
    applied = new_balance - current  # 负数或 0
    rows = await tx.query_raw(
        """
        UPDATE user_wallets SET ticket_balance = $2, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING ticket_balance, point_balance, achievement_points_synced
        """,
        user_id,
        new_balance,
    )
    balance = wallet.wallet_balances(rows[0])
    await wallet._record_ledger(  # noqa: SLF001 (同项目跨模块复用 _record_ledger)
        user_id=user_id,
        currency="ticket",
        delta=applied,
        balance_after=balance["ticket_balance"],
        source=catalog.SOURCE_APPLE_IAP_REFUND,
        source_id=transaction_id,
        metadata={"environment": env, "reason": "apple_refund"},
        client=tx,
    )


async def _expire_vip_now(user_id: str) -> None:
    """把 vip_until 设成过去并清限时钞票/vip_grant 批次（复用 clear_on_lapse）。"""
    from app.services.vip import grants as vip_grants

    await db.execute_raw(
        """
        UPDATE user_wallets
        SET vip_until = CURRENT_TIMESTAMP - INTERVAL '1 second',
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        """,
        user_id,
    )
    try:
        await vip_grants.clear_on_lapse(user_id)
    except Exception:
        logger.exception("clear_on_lapse after refund/revoke failed user=%s", user_id[:8])


async def _user_for_original_txn(original_txn_id: str | None) -> str | None:
    if not original_txn_id:
        return None
    rows = await db.query_raw(
        "SELECT user_id FROM iap_subscription_state WHERE original_transaction_id = $1",
        original_txn_id,
    )
    if rows:
        return str(_field(rows[0], "user_id"))
    rows = await db.query_raw(
        """
        SELECT user_id FROM iap_transactions
        WHERE original_transaction_id = $1 ORDER BY created_at ASC LIMIT 1
        """,
        original_txn_id,
    )
    return str(_field(rows[0], "user_id")) if rows else None


async def _set_subscription_state(
    original_txn_id: str | None,
    status: str,
    *,
    renewal: Any = None,
    ntype: str | None = None,
    subtype: str | None = None,
) -> None:
    if not original_txn_id:
        return
    grace = apple_env.ms_to_dt(_field(renewal, "gracePeriodExpiresDate")) if renewal else None
    await db.execute_raw(
        """
        UPDATE iap_subscription_state
        SET status = $2,
            grace_period_expires_date = COALESCE($3, grace_period_expires_date),
            last_notification_type = COALESCE($4, last_notification_type),
            last_notification_subtype = COALESCE($5, last_notification_subtype),
            updated_at = CURRENT_TIMESTAMP
        WHERE original_transaction_id = $1
        """,
        original_txn_id,
        status,
        grace,
        ntype,
        subtype,
    )


async def _update_renewal_status(original_txn_id: str | None, renewal: Any) -> None:
    if not original_txn_id or renewal is None:
        return
    await db.execute_raw(
        """
        UPDATE iap_subscription_state
        SET auto_renew_status = $2,
            last_notification_type = 'DID_CHANGE_RENEWAL_STATUS',
            updated_at = CURRENT_TIMESTAMP
        WHERE original_transaction_id = $1
        """,
        original_txn_id,
        bool(_field(renewal, "autoRenewStatus")),
    )


async def _update_renewal_pref(original_txn_id: str | None, renewal: Any) -> None:
    if not original_txn_id or renewal is None:
        return
    await db.execute_raw(
        """
        UPDATE iap_subscription_state
        SET auto_renew_product_id = $2,
            last_notification_type = 'DID_CHANGE_RENEWAL_PREF',
            updated_at = CURRENT_TIMESTAMP
        WHERE original_transaction_id = $1
        """,
        original_txn_id,
        _field(renewal, "autoRenewProductId"),
    )


async def _insert_notification(
    uuid: str,
    ntype: str | None,
    subtype: str | None,
    env: str,
    original_txn_id: str | None,
    transaction_id: str | None,
    signed_payload: str,
) -> bool:
    """落库；返回 True=新通知（需处理），False=已收到过（幂等短路）。"""
    summary = {
        "notificationType": ntype,
        "subtype": subtype,
        "environment": env,
        "originalTransactionId": original_txn_id,
        "transactionId": transaction_id,
    }
    rows = await db.query_raw(
        """
        INSERT INTO iap_notifications (
            provider, notification_uuid, notification_type, subtype, environment,
            original_transaction_id, transaction_id, signed_payload, decoded_payload
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)
        ON CONFLICT (provider, notification_uuid) DO NOTHING
        RETURNING id
        """,
        PROVIDER_APPLE,
        uuid,
        ntype or "",
        subtype,
        env,
        original_txn_id,
        transaction_id,
        signed_payload,
        json.dumps(summary, ensure_ascii=False),
    )
    return bool(rows)


async def _mark_processed(uuid: str, error: str | None) -> None:
    await db.execute_raw(
        """
        UPDATE iap_notifications
        SET processed_at = CURRENT_TIMESTAMP, process_error = $2
        WHERE provider = $1 AND notification_uuid = $3
        """,
        PROVIDER_APPLE,
        error,
        uuid,
    )
