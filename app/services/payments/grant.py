"""IAP 到账服务：校验 → 幂等 → 单事务发放 → 落 iap_transactions + wallet_ledger。

只信 transactionId（权益字段全部来自 Apple 校验后的 payload）。幂等地基：
iap_transactions (provider, transaction_id) 唯一约束 + 事务内 INSERT ON CONFLICT
DO NOTHING + FOR UPDATE 回放。verify 端点与 webhook 续期共用 `record_and_grant`。
"""

from __future__ import annotations

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
from app.services.payments.errors import UnknownProductError
from app.services.vip import grants as vip_grants

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


async def verify_and_grant(user_id: str, transaction_id: str) -> dict[str, Any]:
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

    payload, env = await apple_env.fetch_and_verify_transaction(transaction_id)
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
            await _apply_vip(tx, user_id, product, expires_dt, environment, original_txn_id, transaction_id)

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
        granted_now = True

    # VIP 到账后立即发当月权益（限时钞票/音乐券/补签卡），不必等夜间 cron；
    # 独立事务，失败不影响已生效的 VIP（cron 扫到 vip_last_grant_at 到期会补发）。
    if granted_now and product.grants_vip:
        try:
            await vip_grants.grant_monthly(user_id)
        except Exception:
            logger.exception(
                "iap vip monthly grant failed, cron will retry user=%s", user_id[:8]
            )

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
    await reconcile_vip_entitlements(user_id)
    return {"status": "granted", "kind": product.kind, "replay": False, **await _snapshot(user_id)}


async def reconcile_vip_entitlements(user_id: str, *, client: Any | None = None) -> bool:
    """把 user_wallets.vip_until 抬到 consumable VIP 交易隐含的最低值。

    场景：交易已 granted（幂等回放不再 _apply_vip），但 vip_until 被沙盒订阅
    5 分钟续期盖短。读路径与 verify 回放路径都调，自愈而不改 iap_transactions。
    """
    floor = await _consumable_vip_floor(user_id, client=client)
    if floor is None:
        return False
    executor = client or db
    rows = await executor.query_raw(
        "SELECT vip_until FROM user_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    if not rows:
        return False
    current = _as_utc(_field(rows[0], "vip_until"))
    if current is not None and current >= floor:
        return False
    await executor.execute_raw(
        """
        UPDATE user_wallets
        SET vip_until = $2::timestamp, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        """,
        user_id,
        _naive(floor),
    )
    logger.info(
        "iap vip reconcile user=%s until=%s",
        user_id[:8],
        floor.isoformat(),
    )
    return True


async def _consumable_vip_floor(user_id: str, *, client: Any | None = None) -> datetime | None:
    """按 granted consumable VIP 交易顺序叠天数，得到应有的 vip_until 下界。"""
    from datetime import timedelta

    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT product_id, quantity, purchase_date
        FROM iap_transactions
        WHERE user_id = $1 AND status = 'granted' AND kind = $2
        ORDER BY purchase_date ASC NULLS LAST, created_at ASC
        """,
        user_id,
        catalog.KIND_CONSUMABLE,
    )
    running: datetime | None = None
    for row in rows:
        product = catalog.product_for(_field(row, "product_id") or "")
        if product is None or product.vip_days <= 0:
            continue
        qty = int(_field(row, "quantity") or 1)
        purchased = _as_utc(_field(row, "purchase_date"))
        if purchased is None:
            continue
        base = max(purchased, running) if running is not None else purchased
        running = base + timedelta(days=product.vip_days * qty)
    return running


async def _apply_vip(
    tx: Any,
    user_id: str,
    product: IapProduct,
    expires_dt: datetime | None,
    environment: str,
    original_txn_id: str,
    transaction_id: str,
) -> None:
    """设置/延长 vip_until（存 naive UTC，沿用 activate_vip_trial 惯例）。

    - 订阅：vip_until = max(现值, Apple expires_date)，且 upsert 订阅状态表。
    - 消耗型时长包/体验：vip_until = max(now, 现值) + vip_days（叠加）。
    """
    rows = await tx.query_raw(
        "SELECT vip_until FROM user_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    current = _as_utc(_field(rows[0], "vip_until")) if rows else None
    now = datetime.now(timezone.utc)

    if product.kind == catalog.KIND_SUBSCRIPTION:
        target = expires_dt or (now + _days(product.vip_days))
        # Never shorten an existing VIP window (sandbox subs expire in minutes).
        candidates = [target, now]
        if current is not None:
            candidates.append(current)
        consumable_floor = await _consumable_vip_floor(user_id, client=tx)
        if consumable_floor is not None:
            candidates.append(consumable_floor)
        new_until = max(candidates)
    else:
        from datetime import timedelta

        base = max(now, current) if current else now
        new_until = base + timedelta(days=product.vip_days)

    is_trial = product.product_id.endswith(".vip.trial")
    await tx.execute_raw(
        """
        UPDATE user_wallets
        SET vip_until = $2::timestamp,
            vip_trial_used = vip_trial_used OR $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        """,
        user_id,
        new_until.replace(tzinfo=None),
        is_trial,
    )

    if product.kind == catalog.KIND_SUBSCRIPTION:
        await tx.execute_raw(
            """
            INSERT INTO iap_subscription_state (
                original_transaction_id, provider, user_id, product_id,
                environment, status, expires_date, latest_transaction_id, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,'active',$6::timestamp,$7,CURRENT_TIMESTAMP)
            ON CONFLICT (original_transaction_id) DO UPDATE
            SET status = 'active',
                product_id = EXCLUDED.product_id,
                environment = EXCLUDED.environment,
                expires_date = EXCLUDED.expires_date,
                latest_transaction_id = EXCLUDED.latest_transaction_id,
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


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _naive(dt: datetime | None) -> datetime | None:
    """落库前去掉时区：目标列都是 TIMESTAMP WITHOUT TIME ZONE，存 UTC 墙钟。

    prisma query_raw 会把 datetime 序列化成 ISO 字符串；SQL 侧必须写
    ``$N::timestamp`` 让 PG 显式转型（见 conversations.py / last_will.py 惯例）。
    ms_to_dt 返回 aware UTC，落库前转 naive 存 UTC 墙钟；比较逻辑仍用 aware 值。
    """
    return dt.replace(tzinfo=None) if dt is not None else None


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)
