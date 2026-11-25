"""微信支付退款：后台人工发起 + 回调确认后回收权益。

内测规则（2026-11-25 用户确认）：一律全额退款，不判断是否已使用，由后台人工操作。

权益回收的时机很关键：**发起退款时不动权益，等微信回调 REFUND.SUCCESS 到了
才回收**。退款可能被微信拒绝或关闭，提前回收会出现"权益没了但钱也没退"。

钞票回收"能扣多少扣多少"（floor 0），不阻断退款，差额由后台看流水处理，因为
钱已经退给用户了。VIP 每月赠送的钞票/券不回收。

可退的订单：
* `granted`：正常退款，到账后回收权益。
* `failed`：钱收了但没发放（金额不符、重复体验价），原路退回，无权益可回收。

退款请求超时（微信可能已受理）时退款单保持 `pending`，由 cron 用同一个
`out_refund_no` 查单收尾；后台重试也复用这一单，保持微信侧幂等。
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.config import settings
from app.db import db
from app.observability.events import EVT_PAYMENT_REFUND
from app.services import wallet
from app.services.payments.wechat import catalog, transport
from app.services.payments.wechat.errors import (
    WechatPayOrderStateError,
    WechatPayUpstreamError,
)
from app.services.vip.entitlements import recompute_vip_entitlements

logger = logging.getLogger(__name__)

_REFUND_PATH = "/v3/refund/domestic/refunds"
_REFUNDABLE_ORDER_STATUSES = frozenset({"granted", "failed"})

STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_CLOSED = "closed"
STATUS_ABNORMAL = "abnormal"
STATUS_FAILED = "failed"

# 微信 refund_status → 我方 status
_REFUND_STATUS_MAP = {
    "SUCCESS": STATUS_SUCCESS,
    "CLOSED": STATUS_CLOSED,
    "ABNORMAL": STATUS_ABNORMAL,
    "PROCESSING": STATUS_PENDING,
}


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _naive(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _new_out_refund_no() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"RF{stamp}{secrets.token_hex(6).upper()}"


async def create_refund(
    *, out_trade_no: str, actor_id: str, reason: str | None = None
) -> dict[str, Any]:
    """后台发起全额退款。

    幂等与并发：在订单行锁内检查/创建退款单，同一订单已有 pending/success 的
    退款单时直接返回那一单——后台双击、两个管理员同时点都只会有一张退款单。
    已有 pending 单且还没拿到微信受理结果时，用**同一个** out_refund_no 重新提交
    （微信按 out_refund_no 幂等），而不是生成新单。
    """
    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT user_id, status, amount_fen, refunded_fen
            FROM wechat_pay_orders
            WHERE out_trade_no = $1
            FOR UPDATE
            """,
            out_trade_no,
        )
        if not rows:
            raise WechatPayOrderStateError("订单不存在")
        order = rows[0]
        status = str(_field(order, "status"))
        if status not in _REFUNDABLE_ORDER_STATUSES:
            raise WechatPayOrderStateError(f"订单状态 {status} 不允许退款")
        amount_fen = int(_field(order, "amount_fen") or 0)
        if int(_field(order, "refunded_fen") or 0) >= amount_fen:
            raise WechatPayOrderStateError("订单已全额退款")

        existing = await tx.query_raw(
            """
            SELECT out_refund_no, status, refund_id FROM wechat_pay_refunds
            WHERE out_trade_no = $1 AND status IN ('pending', 'success')
            ORDER BY created_at DESC LIMIT 1
            """,
            out_trade_no,
        )
        if existing:
            row = existing[0]
            # 已受理（有 refund_id）或已成功：回放。没受理的 pending 单落到下面重提。
            if _field(row, "refund_id") or str(_field(row, "status")) == STATUS_SUCCESS:
                return {
                    "out_refund_no": str(_field(row, "out_refund_no")),
                    "status": str(_field(row, "status")),
                    "replay": True,
                }
            out_refund_no = str(_field(row, "out_refund_no"))
        else:
            out_refund_no = _new_out_refund_no()
            await tx.execute_raw(
                """
                INSERT INTO wechat_pay_refunds (
                    out_refund_no, out_trade_no, user_id, amount_fen, status, reason, actor_id
                )
                VALUES ($1,$2,$3,$4,'pending',$5,$6)
                """,
                out_refund_no,
                out_trade_no,
                str(_field(order, "user_id")),
                amount_fen,
                reason,
                actor_id,
            )

    body = {
        "out_trade_no": out_trade_no,
        "out_refund_no": out_refund_no,
        "reason": (reason or "内测期用户申请退款")[:80],
        "notify_url": settings.wechatpay_notify_url.strip(),
        "amount": {"refund": amount_fen, "total": amount_fen, "currency": "CNY"},
    }
    try:
        response = await transport.request("POST", _REFUND_PATH, body=body)
    except WechatPayUpstreamError as exc:
        if exc.status_code is not None and 400 <= exc.status_code < 500:
            # 微信明确拒绝（余额不足、订单状态不对等）：确定没受理，落 failed。
            await db.execute_raw(
                """
                UPDATE wechat_pay_refunds
                SET status = 'failed', fail_reason = $2, updated_at = CURRENT_TIMESTAMP
                WHERE out_refund_no = $1 AND status = 'pending'
                """,
                out_refund_no,
                f"{exc.code or exc.status_code}"[:200],
            )
        else:
            # 超时 / 5xx：微信可能已受理。保持 pending，cron 按 out_refund_no 查单收尾。
            await db.execute_raw(
                """
                UPDATE wechat_pay_refunds
                SET fail_reason = $2, updated_at = CURRENT_TIMESTAMP
                WHERE out_refund_no = $1 AND status = 'pending'
                """,
                out_refund_no,
                f"submit_uncertain: {type(exc).__name__}"[:200],
            )
        raise

    refund_status = str(response.get("status") or "PROCESSING")
    mapped = _REFUND_STATUS_MAP.get(refund_status, STATUS_PENDING)
    await db.execute_raw(
        """
        UPDATE wechat_pay_refunds
        SET refund_id = $2, status = $3, raw_payload = $4::jsonb,
            fail_reason = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE out_refund_no = $1 AND status = 'pending'
        """,
        out_refund_no,
        response.get("refund_id"),
        mapped,
        json.dumps(_safe_refund_payload(response), ensure_ascii=False),
    )
    logger.info(
        "wechat pay refund submitted order=%s actor=%s",
        out_trade_no,
        actor_id[:8],
        extra={
            "event": EVT_PAYMENT_REFUND,
            "provider": catalog.PROVIDER_WECHAT,
            "out_trade_no": out_trade_no,
            "out_refund_no": out_refund_no,
        },
    )
    # 微信可能同步就返回 SUCCESS；此时直接回收，不等回调。
    if mapped == STATUS_SUCCESS:
        await _settle_refund_success(
            out_refund_no=out_refund_no,
            success_time=_parse_rfc3339(response.get("success_time")),
        )
    return {"out_refund_no": out_refund_no, "status": mapped, "replay": False}


async def query_and_settle_refund(out_refund_no: str) -> dict[str, Any]:
    """按 out_refund_no 查微信退款单并收尾。回调丢失 / 提交超时的兜底。"""
    response = await transport.request(
        "GET", f"/v3/refund/domestic/refunds/{quote(out_refund_no, safe='')}"
    )
    return await apply_refund_notification("REFUND.QUERY", response)


async def reconcile_pending_refunds(*, limit: int = 50) -> dict[str, int]:
    """扫描卡在 pending 的退款单，逐笔查单。"""
    rows = await db.query_raw(
        """
        SELECT out_refund_no, out_trade_no, actor_id, reason FROM wechat_pay_refunds
        WHERE status = 'pending'
          AND updated_at < CURRENT_TIMESTAMP - make_interval(mins => 2)
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        limit,
    )
    counts = {"scanned": len(rows), "settled": 0, "pending": 0, "failed": 0}
    for row in rows:
        out_refund_no = str(_field(row, "out_refund_no"))
        try:
            result = await query_and_settle_refund(out_refund_no)
            if result.get("status") == STATUS_PENDING:
                counts["pending"] += 1
            else:
                counts["settled"] += 1
        except WechatPayUpstreamError as exc:
            if exc.status_code != 404:
                counts["failed"] += 1
                _log_refund_reconcile_failure(out_refund_no, exc)
            else:
                # 微信侧没有这张退款单：提交时就没送达。用**同一个** out_refund_no
                # 重提（create_refund 对没受理的 pending 单就是这么做的，微信按单号
                # 幂等），不换新单号——万一微信其实受理过，换号会出两笔款。
                try:
                    await create_refund(
                        out_trade_no=str(_field(row, "out_trade_no")),
                        actor_id=str(_field(row, "actor_id") or "reconcile"),
                        reason=_field(row, "reason"),
                    )
                    counts["pending"] += 1
                except Exception as resubmit_exc:
                    counts["failed"] += 1
                    _log_refund_reconcile_failure(out_refund_no, resubmit_exc)
        except Exception as exc:  # 单笔失败不中断整批
            counts["failed"] += 1
            _log_refund_reconcile_failure(out_refund_no, exc)
        # 无论结果都推后 updated_at，避免同一批卡住扫描窗口。
        try:
            await db.execute_raw(
                "UPDATE wechat_pay_refunds SET updated_at = CURRENT_TIMESTAMP "
                "WHERE out_refund_no = $1 AND status = 'pending'",
                out_refund_no,
            )
        except Exception:
            logger.exception("wechat pay refund touch failed refund=%s", out_refund_no)
    return counts


def _log_refund_reconcile_failure(out_refund_no: str, exc: Exception) -> None:
    logger.warning(
        "wechat pay refund reconcile failed refund=%s err=%s",
        out_refund_no,
        type(exc).__name__,
        extra={"event": "wechat_pay_refund_reconcile_failed"},
    )


def _parse_rfc3339(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


async def apply_refund_notification(
    event_type: str, decrypted: dict[str, Any]
) -> dict[str, Any]:
    """处理 REFUND.* 回调。只有 SUCCESS 才回收权益。"""
    out_refund_no = decrypted.get("out_refund_no")
    if not isinstance(out_refund_no, str) or not out_refund_no:
        return {"status": "ignored"}
    # 回调里叫 refund_status，退款查单应答里叫 status。
    refund_status = str(decrypted.get("refund_status") or decrypted.get("status") or "")
    mapped = _REFUND_STATUS_MAP.get(refund_status)
    if mapped is None:
        mapped = STATUS_SUCCESS if event_type == "REFUND.SUCCESS" else STATUS_ABNORMAL

    if mapped != STATUS_SUCCESS:
        await db.execute_raw(
            """
            UPDATE wechat_pay_refunds
            SET status = $2, raw_payload = $3::jsonb, updated_at = CURRENT_TIMESTAMP
            WHERE out_refund_no = $1 AND status = 'pending'
            """,
            out_refund_no,
            mapped,
            json.dumps(_safe_refund_payload(decrypted), ensure_ascii=False),
        )
        return {"status": mapped}

    await _adopt_external_refund(out_refund_no, decrypted)
    return await _settle_refund_success(
        out_refund_no=out_refund_no,
        success_time=_parse_rfc3339(decrypted.get("success_time")),
        payload=decrypted,
    )


async def _adopt_external_refund(out_refund_no: str, decrypted: dict[str, Any]) -> None:
    """商户平台后台直接退的款：我方没有这张退款单，按回调补建后再回收权益。

    只在能按 out_trade_no 找到我方订单时补建；找不到说明不是本业务的订单。
    """
    out_trade_no = decrypted.get("out_trade_no")
    if not isinstance(out_trade_no, str) or not out_trade_no:
        return
    amount = decrypted.get("amount")
    refund_fen = amount.get("refund") if isinstance(amount, dict) else None
    # 同一订单只允许一张进行中的退款单（部分唯一索引）。我方那张若从没被微信
    # 受理（没有 refund_id），说明钱是从商户平台退的：先把它作废，否则补建会撞
    # 索引，回调永远 500、权益永远收不回。
    await db.execute_raw(
        """
        UPDATE wechat_pay_refunds
        SET status = 'closed', fail_reason = 'superseded_by_platform_refund',
            updated_at = CURRENT_TIMESTAMP
        WHERE out_trade_no = $1 AND out_refund_no <> $2
          AND status = 'pending' AND refund_id IS NULL
        """,
        out_trade_no,
        out_refund_no,
    )
    await db.execute_raw(
        """
        INSERT INTO wechat_pay_refunds (
            out_refund_no, refund_id, out_trade_no, user_id, amount_fen,
            status, reason, actor_id
        )
        SELECT $1, $2, o.out_trade_no, o.user_id, COALESCE($3::int, o.amount_fen),
               'pending', 'merchant_platform', NULL
        FROM wechat_pay_orders o
        WHERE o.out_trade_no = $4
        ON CONFLICT DO NOTHING
        """,
        out_refund_no,
        decrypted.get("refund_id") if isinstance(decrypted.get("refund_id"), str) else None,
        refund_fen if isinstance(refund_fen, int) and refund_fen > 0 else None,
        out_trade_no,
    )


async def _settle_refund_success(
    *,
    out_refund_no: str,
    success_time: datetime | None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """幂等回收：扣回钞票 / 重算 VIP，并把订单置 refunded。"""
    payload_json = json.dumps(
        _safe_refund_payload(payload or {}), ensure_ascii=False
    )
    recompute_for: str | None = None
    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT r.out_trade_no, r.user_id, r.amount_fen, r.status,
                   o.ticket_amount, o.vip_days, o.status AS order_status
            FROM wechat_pay_refunds r
            JOIN wechat_pay_orders o ON o.out_trade_no = r.out_trade_no
            WHERE r.out_refund_no = $1
            FOR UPDATE OF r, o
            """,
            out_refund_no,
        )
        if not rows:
            return {"status": "unknown_refund"}
        row = rows[0]
        if str(_field(row, "status")) == STATUS_SUCCESS:
            return {"status": STATUS_SUCCESS, "replay": True}

        out_trade_no = str(_field(row, "out_trade_no"))
        user_id = str(_field(row, "user_id"))
        amount_fen = int(_field(row, "amount_fen") or 0)
        # 只有发放过的订单才有权益可回收；failed 订单（钱收了没发货）只退钱。
        was_granted = str(_field(row, "order_status")) == "granted"
        ticket_amount = int(_field(row, "ticket_amount") or 0) if was_granted else 0
        vip_days = int(_field(row, "vip_days") or 0) if was_granted else 0

        if ticket_amount > 0:
            await _reverse_tickets(
                tx,
                user_id=user_id,
                amount=ticket_amount,
                out_refund_no=out_refund_no,
                out_trade_no=out_trade_no,
            )

        await tx.execute_raw(
            """
            UPDATE wechat_pay_refunds
            SET status = 'success', success_time = COALESCE($2::timestamp, success_time),
                raw_payload = $3::jsonb, updated_at = CURRENT_TIMESTAMP
            WHERE out_refund_no = $1
            """,
            out_refund_no,
            _naive(success_time),
            payload_json,
        )
        await tx.execute_raw(
            """
            UPDATE wechat_pay_orders
            SET status = 'refunded', refunded_fen = LEAST(amount_fen, refunded_fen + $2),
                refunded_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE out_trade_no = $1
            """,
            out_trade_no,
            amount_fen,
        )
        if vip_days > 0:
            # 订单已不是 granted，entitlements 会把这段天数排除后重算 vip_until。
            await recompute_vip_entitlements(user_id, client=tx, clear_lapse=False)
            recompute_for = user_id

    logger.info(
        "wechat pay refund settled refund=%s",
        out_refund_no,
        extra={
            "event": EVT_PAYMENT_REFUND,
            "provider": catalog.PROVIDER_WECHAT,
            "out_refund_no": out_refund_no,
            "user_id": recompute_for,
        },
    )
    return {"status": STATUS_SUCCESS, "replay": False}


async def _reverse_tickets(
    tx: Any,
    *,
    user_id: str,
    amount: int,
    out_refund_no: str,
    out_trade_no: str,
) -> None:
    """退款反向扣钞票，floor 0，source_id=out_refund_no 做幂等。

    与 Apple 侧 `payments/notifications._reverse_tickets` 同一套做法：用户可能已
    把钞票花掉，扣到 0 为止，实际扣减量落 ledger（delta 可能小于 amount），差额
    由后台看流水处理——钱已经退了，不能因为余额不足就阻断退款。
    """
    if amount <= 0:
        return
    dup = await tx.query_raw(
        "SELECT 1 FROM wallet_ledger WHERE source = $1 AND source_id = $2 LIMIT 1",
        catalog.SOURCE_WECHAT_PAY_REFUND,
        out_refund_no,
    )
    if dup:
        return
    locked = await tx.query_raw(
        "SELECT ticket_balance FROM user_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    if not locked:
        return
    current = int(_field(locked[0], "ticket_balance", 0) or 0)
    subunits = wallet.ticket_subunits(amount)
    new_balance = max(0, current - subunits)
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
    await wallet._record_ledger(  # noqa: SLF001 (同项目跨模块复用，见 Apple 侧)
        user_id=user_id,
        currency="ticket",
        delta=applied,
        balance_after=balance["ticket_balance"],
        source=catalog.SOURCE_WECHAT_PAY_REFUND,
        source_id=out_refund_no,
        metadata={"out_trade_no": out_trade_no, "reason": "wechat_pay_refund"},
        client=tx,
    )


def _safe_refund_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "refund_id",
        "out_refund_no",
        "out_trade_no",
        "transaction_id",
        "channel",
        "status",
        "refund_status",
        "success_time",
        "user_received_account",
    }
    safe = {key: value for key, value in payload.items() if key in allowed}
    amount = payload.get("amount")
    if isinstance(amount, dict):
        safe["amount"] = {
            key: amount[key]
            for key in ("total", "refund", "payer_total", "payer_refund", "currency")
            if key in amount
        }
    return safe
