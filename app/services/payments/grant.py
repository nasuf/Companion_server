"""IAP 到账服务：校验 → 幂等 → 单事务发放 → 落 iap_transactions + wallet_ledger。

只信 transactionId（权益字段全部来自 Apple 校验后的 payload）。幂等地基：
iap_transactions (provider, transaction_id) 唯一约束 + 事务内 INSERT ON CONFLICT
DO NOTHING + FOR UPDATE 回放。verify 端点与 webhook 续期共用 `record_and_grant`。
"""

from __future__ import annotations

import asyncio
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.observability.events import EVT_PAYMENT_GRANT
from app.services import wallet
from app.services.payments import catalog
from app.services.payments.apple import environment as apple_env
from app.services.payments.catalog import IapProduct
from app.services.payments.errors import AppleVerificationError, UnknownProductError
from app.services.runtime.tasks import fire_background
from app.services.vip import grants as vip_grants
from app.services.vip.entitlements import (
    as_utc as _as_utc,
    compute_vip_entitlement_end,
    naive_dt as _naive,
    recompute_vip_entitlements,
    reconcile_vip_entitlements,
    _consumable_vip_floor,
    _subscription_vip_end,
)

logger = logging.getLogger(__name__)

PROVIDER_APPLE = "apple"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _payload_dict(payload: Any) -> dict[str, Any]:
    """把已验签的交易 payload 收敛成可落库/审计的 JSON（只取关心的字段）。"""
    type_val = _field(payload, "type")
    return {
        "productId": _field(payload, "productId"),
        "transactionId": _field(payload, "transactionId"),
        "originalTransactionId": _field(payload, "originalTransactionId"),
        "webOrderLineItemId": _field(payload, "webOrderLineItemId"),
        "expiresDate": _field(payload, "expiresDate"),
        "purchaseDate": _field(payload, "purchaseDate"),
        "quantity": _field(payload, "quantity"),
        "type": getattr(type_val, "value", type_val),
        "environment": _field(payload, "environment"),
        # 实付金额审计：price 是货币最小单位的千分之一（milliunits，¥29→29000），
        # currency=ISO4217（CNY/USD），storefront=ISO 国家码（CHN/USA）。仅新交易有，
        # 供 admin 支付管理展示实付金额。旧行为 None。
        "price": _field(payload, "price"),
        "currency": _field(payload, "currency"),
        "storefront": _field(payload, "storefront"),
    }


async def _snapshot(user_id: str) -> dict[str, Any]:
    """到账后回给客户端的一致快照：新钱包 + 新 VIP 状态。"""
    return {
        "wallet": await wallet.get_balance(user_id),
        "vip": await wallet.full_wallet(user_id),
    }


async def _find_transaction(transaction_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT status, kind, product_id, user_id
        FROM iap_transactions
        WHERE provider = $1 AND transaction_id = $2
        """,
        PROVIDER_APPLE,
        transaction_id,
    )
    return dict(rows[0]) if rows else None


async def resolve_verified_payload(
    transaction_id: str,
    *,
    signed_transaction: str | None = None,
) -> tuple[Any, str]:
    """Verify a transaction via client JWS (fast) or Apple API (fallback)."""
    if signed_transaction:
        last_exc: AppleVerificationError | None = None
        for env_str in apple_env.env_strings_to_try():
            try:
                payload = await asyncio.to_thread(
                    apple_env.verify_signed_transaction, signed_transaction, env_str
                )
                if (_field(payload, "transactionId") or "") != transaction_id:
                    raise AppleVerificationError("transaction_id_mismatch")
                return payload, env_str
            except AppleVerificationError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise AppleVerificationError("jws_verify_failed:no_matching_environment")
    return await apple_env.fetch_and_verify_transaction(transaction_id)


async def user_id_for_app_account_token(token: str | None) -> str | None:
    """Map Apple appAccountToken (client sets to user UUID) to an active user."""
    if not token:
        return None
    try:
        user_id = str(uuid.UUID(str(token).strip()))
    except ValueError:
        return None
    rows = await db.query_raw(
        """
        SELECT id FROM users
        WHERE id = $1::uuid AND status = 'active'
        LIMIT 1
        """,
        user_id,
    )
    return str(_field(rows[0], "id")) if rows else None


async def verify_and_grant(
    user_id: str,
    transaction_id: str,
    *,
    signed_transaction: str | None = None,
) -> dict[str, Any]:
    """客户端购买/恢复后调用：向 Apple 校验该交易并幂等到账。"""
    existing = await _find_transaction(transaction_id)
    if existing and existing["status"] == "granted":
        # 幂等回放：已到账过，直接回当前快照（app 重启会重放未 complete 的交易）。
        # 仍跑 reconcile：历史 bug / 并发可能导致 consumable VIP 未叠上却被标 granted。
        await reconcile_vip_entitlements(user_id)
        return {
            "status": "granted",
            "kind": existing["kind"],
            "replay": True,
            **await _snapshot(user_id),
        }

    payload, env = await resolve_verified_payload(
        transaction_id, signed_transaction=signed_transaction
    )
    product = catalog.product_for(_field(payload, "productId") or "")
    if product is None:
        raise UnknownProductError(_field(payload, "productId") or "")

    return await record_and_grant(user_id, payload, env, product)


async def record_and_grant(
    user_id: str,
    payload: Any,
    environment: str,
    product: IapProduct,
    *,
    notification_uuid: str | None = None,
) -> dict[str, Any]:
    """把一笔已验签交易幂等落库并发放权益。verify 与 webhook 续期共用。"""
    transaction_id = _field(payload, "transactionId") or ""
    original_txn_id = _field(payload, "originalTransactionId") or transaction_id
    quantity = int(_field(payload, "quantity") or 1)
    purchase_dt = apple_env.ms_to_dt(_field(payload, "purchase_date") or _field(payload, "purchaseDate"))
    expires_dt = apple_env.ms_to_dt(_field(payload, "expiresDate"))
    payload_json = json.dumps(_payload_dict(payload), ensure_ascii=False)

    await wallet.ensure_wallet(user_id)

    granted_now = False
    async with db.tx() as tx:
        inserted = await tx.query_raw(
            """
            INSERT INTO iap_transactions (
                provider, transaction_id, original_transaction_id,
                web_order_line_item_id, product_id, kind, environment, user_id,
                quantity, purchase_date, expires_date, status,
                notification_uuid, raw_transaction_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::timestamp,$11::timestamp,'pending',$12,$13::jsonb)
            ON CONFLICT (provider, transaction_id) DO NOTHING
            RETURNING id
            """,
            PROVIDER_APPLE,
            transaction_id,
            original_txn_id,
            _field(payload, "webOrderLineItemId"),
            product.product_id,
            product.kind,
            environment,
            user_id,
            quantity,
            _naive(purchase_dt),
            _naive(expires_dt),
            notification_uuid,
            payload_json,
        )
        if not inserted:
            # 并发 / 重放：行已存在。锁住看状态：
            #   - granted：回放（app 重启会重放未 complete 的交易）
            #   - refunded / revoked / failed：已终态，绝不再发权益（否则退款后
            #     客户端重发同一 txn 会二次到账）
            #   - pending：上次事务中途崩，在本事务补发
            locked = await tx.query_raw(
                """
                SELECT status FROM iap_transactions
                WHERE provider = $1 AND transaction_id = $2
                FOR UPDATE
                """,
                PROVIDER_APPLE,
                transaction_id,
            )
            locked_status = _field(locked[0], "status") if locked else None
            if locked_status != "pending":
                await reconcile_vip_entitlements(user_id)
                return {
                    "status": locked_status or "granted",
                    "kind": product.kind,
                    "replay": True,
                    **await _snapshot(user_id),
                }

        if product.grants_tickets:
            await wallet.credit_tickets(
                user_id,
                product.ticket_amount * quantity,
                source=catalog.SOURCE_APPLE_IAP,
                source_id=transaction_id,
                metadata={
                    "product_id": product.product_id,
                    "environment": environment,
                    "kind": product.kind,
                },
                client=tx,
            )
        elif product.grants_vip:
            await _apply_vip_metadata(
                tx, user_id, product, expires_dt, environment, original_txn_id, transaction_id
            )

        await tx.execute_raw(
            """
            UPDATE iap_transactions
            SET status = 'granted', wallet_ledger_source_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE provider = $1 AND transaction_id = $2
            """,
            PROVIDER_APPLE,
            transaction_id,
        )
        if product.grants_vip:
            await recompute_vip_entitlements(user_id, client=tx, clear_lapse=False)
        granted_now = True

    # VIP 到账后立即发当月权益（限时钞票/音乐券/补签卡），不必等夜间 cron；
    # 后台执行以缩短 verify 热路径；失败不影响已生效 VIP（cron 会补发）。
    if granted_now and product.grants_vip:
        fire_background(_grant_vip_monthly_safe(user_id))

    logger.info(
        "iap grant ok user=%s product=%s kind=%s",
        user_id[:8],
        product.product_id,
        product.kind,
        extra={
            "event": EVT_PAYMENT_GRANT,
            "user_id": user_id,
            "transaction_id": transaction_id,
            "product_id": product.product_id,
            "environment": environment,
            "kind": product.kind,
        },
    )
    return {"status": "granted", "kind": product.kind, "replay": False, **await _snapshot(user_id)}


async def _grant_vip_monthly_safe(user_id: str) -> None:
    try:
        await vip_grants.grant_monthly(user_id)
    except Exception:
        logger.exception(
            "iap vip monthly grant failed, cron will retry user=%s", user_id[:8]
        )


async def heal_subscription_states_for_user(
    user_id: str, *, client: Any | None = None
) -> None:
    """Realign every subscription state row with remaining granted renewal txns."""
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT original_transaction_id
        FROM iap_subscription_state
        WHERE user_id = $1
        """,
        user_id,
    )
    for row in rows:
        original_txn_id = str(_field(row, "original_transaction_id", "") or "")
        if original_txn_id:
            await refresh_subscription_state_from_grants(
                original_txn_id, user_id, client=client
            )


async def refresh_subscription_state_from_grants(
    original_txn_id: str,
    user_id: str,
    *,
    client: Any | None = None,
) -> None:
    """After refund/revoke, realign ``iap_subscription_state`` with remaining grants."""
    executor = client or db
    await executor.query_raw(
        """
        SELECT transaction_id
        FROM iap_transactions
        WHERE user_id = $1 AND original_transaction_id = $2 AND kind = $3
        FOR UPDATE
        """,
        user_id,
        original_txn_id,
        catalog.KIND_SUBSCRIPTION,
    )
    rows = await executor.query_raw(
        """
        SELECT MAX(expires_date) AS max_expires,
               COUNT(*) FILTER (WHERE status = 'granted') AS granted_cnt
        FROM iap_transactions
        WHERE user_id = $1 AND original_transaction_id = $2 AND kind = $3
        """,
        user_id,
        original_txn_id,
        catalog.KIND_SUBSCRIPTION,
    )
    if not rows:
        return
    granted_cnt = int(_field(rows[0], "granted_cnt") or 0)
    if granted_cnt == 0:
        await executor.execute_raw(
            """
            UPDATE iap_subscription_state
            SET status = 'refunded',
                updated_at = CURRENT_TIMESTAMP
            WHERE original_transaction_id = $1
            """,
            original_txn_id,
        )
        return
    max_exp = _as_utc(_field(rows[0], "max_expires"))
    now = datetime.now(timezone.utc)
    status = "active" if max_exp is not None and max_exp > now else "expired"
    await executor.execute_raw(
        """
        UPDATE iap_subscription_state
        SET status = $2,
            expires_date = COALESCE($3::timestamp, expires_date),
            updated_at = CURRENT_TIMESTAMP
        WHERE original_transaction_id = $1
        """,
        original_txn_id,
        status,
        _naive(max_exp),
    )


async def _apply_vip_metadata(
    tx: Any,
    user_id: str,
    product: IapProduct,
    expires_dt: datetime | None,
    environment: str,
    original_txn_id: str,
    transaction_id: str,
) -> None:
    """Subscription state upsert + trial flag; ``vip_until`` is set by recompute."""
    is_trial = product.product_id.endswith(".vip.trial")
    if is_trial:
        await tx.execute_raw(
            """
            UPDATE user_wallets
            SET vip_trial_used = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            """,
            user_id,
        )

    if product.kind == catalog.KIND_SUBSCRIPTION:
        await tx.execute_raw(
            """
            INSERT INTO iap_subscription_state (
                original_transaction_id, provider, user_id, product_id,
                environment, status, auto_renew_status, expires_date,
                latest_transaction_id, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,'active',TRUE,$6::timestamp,$7,CURRENT_TIMESTAMP)
            ON CONFLICT (original_transaction_id) DO UPDATE
            SET status = 'active',
                product_id = EXCLUDED.product_id,
                environment = EXCLUDED.environment,
                expires_date = EXCLUDED.expires_date,
                latest_transaction_id = EXCLUDED.latest_transaction_id,
                auto_renew_status = COALESCE(
                    iap_subscription_state.auto_renew_status, EXCLUDED.auto_renew_status
                ),
                updated_at = CURRENT_TIMESTAMP
            """,
            original_txn_id,
            PROVIDER_APPLE,
            user_id,
            product.product_id,
            environment,
            _naive(expires_dt),
            transaction_id,
        )