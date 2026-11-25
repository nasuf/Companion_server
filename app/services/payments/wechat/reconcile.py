"""主动查单与关单：回调丢失时的兜底，以及过期未付订单的收尾。

为什么必须有主动查单：回调是**不可靠**的。地址被拦、证书问题、我方短暂 5xx
都会让某笔支付的回调迟到或丢失。只依赖回调就会出现"用户付了钱但权益没到"。
微信侧的查单接口是权威真相，这里按它结算。

为什么关单只处理 `created`：已支付的订单绝不能被关单逻辑改状态，否则会把
到账中的订单误判为失败。关单也会调微信 close 接口，避免用户在旧收银台页面
上又把这笔付掉。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.config import settings
from app.db import db
from app.services.payments.wechat import notifications as wechat_notifications
from app.services.payments.wechat import orders, refunds, transport
from app.services.payments.wechat.errors import WechatPayUpstreamError
from app.services.vip.entitlements import as_utc

logger = logging.getLogger(__name__)

# 微信侧 trade_state → 我方处理
_SETTLE_STATES = frozenset({"SUCCESS"})
_CLOSED_STATES = frozenset({"CLOSED", "REVOKED", "PAYERROR"})

# 多久没动静就去查单。给用户留出扫码付款的时间，避免刚下单就查。
STALE_AFTER_MINUTES = 3
# 多久没付就关单。必须大于下单时传给微信的 time_expire（orders.ORDER_TTL_MINUTES），
# 保证关单时微信侧收银台已经付不了。
CLOSE_AFTER_MINUTES = 30
# 查单持续失败的订单（如 prepay 失败、微信侧根本没有这笔单）超过这个时长就落
# closed，不再占用扫描窗口。
GIVE_UP_AFTER_HOURS = 24


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _query_path(out_trade_no: str) -> str:
    mch_id = settings.wechatpay_mch_id.strip()
    return (
        f"/v3/pay/transactions/out-trade-no/{quote(out_trade_no, safe='')}"
        f"?mchid={quote(mch_id, safe='')}"
    )


def _close_path(out_trade_no: str) -> str:
    return f"/v3/pay/transactions/out-trade-no/{quote(out_trade_no, safe='')}/close"


async def query_and_settle(out_trade_no: str) -> dict[str, Any]:
    """按商户订单号查微信，必要时结算或关单。幂等（复用 settle_paid_order）。"""
    response = await transport.request("GET", _query_path(out_trade_no))
    trade_state = str(response.get("trade_state") or "")

    # REFUND：付了款又退了，而我方还没发放（否则这单不会在 created 里被扫到）。
    # 钱已经回到用户手里，不能再发货——商户平台直接退的款不会回调到我们，补发
    # 之后就再也没有机会收回。
    if trade_state == "REFUND":
        return await orders.mark_refunded_before_grant(out_trade_no, payload=response)

    if trade_state in _SETTLE_STATES:
        amount = response.get("amount")
        paid = None
        currency = None
        if isinstance(amount, dict):
            total = amount.get("total")
            paid = int(total) if isinstance(total, int) else None
            currency = amount.get("currency") if isinstance(amount.get("currency"), str) else None
        transaction_id = response.get("transaction_id")
        return await orders.settle_paid_order(
            out_trade_no=out_trade_no,
            transaction_id=transaction_id if isinstance(transaction_id, str) else None,
            trade_state=trade_state,
            paid_amount_fen=paid,
            success_time=wechat_notifications._parse_rfc3339(  # noqa: SLF001 (同域复用)
                response.get("success_time")
            ),
            payload=response,
            currency=currency,
        )

    if trade_state in _CLOSED_STATES:
        await orders.mark_order_closed(out_trade_no, reason=f"trade_state_{trade_state.lower()}")
        return {"status": "closed", "trade_state": trade_state}

    # NOTPAY / USERPAYING：用户还没付完，什么都不做，下一轮再查。
    return {"status": "pending", "trade_state": trade_state}


# 关单时微信返回这些错误码，说明订单在微信侧确实已经不能付款，可以落 closed。
_CLOSE_OK_CODES = frozenset({"ORDER_CLOSED", "ORDERCLOSED", "ORDER_NOT_EXIST", "ORDERNOTEXIST"})


async def close_order(out_trade_no: str) -> dict[str, Any]:
    """让微信关掉这笔未付订单，确认关掉后才落我方 closed。

    关单与用户付款存在竞态，所以：
    * 微信返回 ORDERPAID（或任何其它错误、网络超时）→ **不落 closed**，立刻再查
      一次单；付了就结算，没付就留到下一轮。
    * 关单成功（204）或订单在微信侧已关闭/不存在 → 落 closed。
    即使这里仍有漏网（关单后极短窗口内付款），`settle_paid_order` 也允许 closed
    订单在微信确认 SUCCESS 后照常发放。
    """
    try:
        await transport.request(
            "POST",
            _close_path(out_trade_no),
            body={"mchid": settings.wechatpay_mch_id.strip()},
            verify_response=False,
        )
    except WechatPayUpstreamError as exc:
        if (exc.code or "").upper() not in _CLOSE_OK_CODES:
            logger.warning(
                "wechat pay close not confirmed order=%s code=%s; re-querying",
                out_trade_no,
                exc.code or type(exc).__name__,
                extra={"event": "wechat_pay_close_unconfirmed", "out_trade_no": out_trade_no},
            )
            return await query_and_settle(out_trade_no)
    await orders.mark_order_closed(out_trade_no, reason="expired_unpaid")
    return {"status": "closed"}


async def _touch(out_trade_no: str) -> None:
    """查单失败也推后 updated_at，让扫描窗口轮到别的订单，不被同一批卡死。"""
    await db.execute_raw(
        "UPDATE wechat_pay_orders SET updated_at = CURRENT_TIMESTAMP "
        "WHERE out_trade_no = $1 AND status = 'created'",
        out_trade_no,
    )


async def reconcile_pending_orders(*, limit: int = 100) -> dict[str, int]:
    """扫描卡住的订单：先对超时未付的关单，再对其余的主动查单。

    返回各类计数供 cron 日志与后台展示——区分"结算成功"、"已关单"、"仍未付"
    和"查单失败"，不把失败混进成功里。
    """
    now = datetime.now(timezone.utc)
    rows = await db.query_raw(
        """
        SELECT out_trade_no, status, created_at
        FROM wechat_pay_orders
        WHERE status = 'created'
          AND updated_at < CURRENT_TIMESTAMP - make_interval(mins => $1::int)
        ORDER BY updated_at ASC
        LIMIT $2
        """,
        STALE_AFTER_MINUTES,
        limit,
    )

    counts = {"scanned": len(rows), "settled": 0, "closed": 0, "pending": 0, "failed": 0}
    for row in rows:
        out_trade_no = str(_field(row, "out_trade_no"))
        # query_raw 把 timestamp 列返回成 ISO 字符串（不是 datetime），as_utc 两种都认。
        created_at = as_utc(_field(row, "created_at"))
        age_minutes = (now - created_at).total_seconds() / 60 if created_at else None

        try:
            # 超时未付：先查一次确认真的没付（防止"刚付上就被关单"），再关。
            result = await query_and_settle(out_trade_no)
            status = str(result.get("status"))
            if status == "pending" and age_minutes is not None and age_minutes > CLOSE_AFTER_MINUTES:
                result = await close_order(out_trade_no)
                status = str(result.get("status"))
            if status == "granted":
                counts["settled"] += 1
            elif status == "closed":
                counts["closed"] += 1
            else:
                counts["pending"] += 1
                await _touch(out_trade_no)
        except Exception as exc:  # 单笔失败不中断整批（含 DB 异常）
            if _is_order_not_exist(exc):
                # 微信侧没有这笔单（下单结果不确定的那类）：不可能被支付，不算故障，
                # 不让它把 cron 健康标红、盖住真问题；24 小时后由下面收尾。
                counts["pending"] += 1
                await _give_up_or_touch(out_trade_no, age_minutes, exc)
                continue
            counts["failed"] += 1
            logger.warning(
                "wechat pay reconcile failed order=%s err=%s",
                out_trade_no,
                type(exc).__name__,
                extra={"event": "wechat_pay_reconcile_failed", "out_trade_no": out_trade_no},
            )
            await _give_up_or_touch(out_trade_no, age_minutes, exc)
    logger.info(
        "wechat pay reconcile done scanned=%(scanned)s settled=%(settled)s "
        "closed=%(closed)s pending=%(pending)s failed=%(failed)s",
        counts,
        extra={"event": "wechat_pay_reconcile", **counts},
    )
    return counts


def _is_order_not_exist(exc: Exception) -> bool:
    return (getattr(exc, "code", None) or "").upper() in {"ORDER_NOT_EXIST", "ORDERNOTEXIST"}


async def _give_up_or_touch(
    out_trade_no: str, age_minutes: float | None, exc: Exception
) -> None:
    """查单持续失败：超过 GIVE_UP_AFTER_HOURS 且微信明确说没有这笔单才放弃。

    只在 ORDER_NOT_EXIST 时关单——这类订单（prepay 失败）在微信侧根本不存在，
    不可能有人付款。其它错误（网络、5xx）只推后重试，绝不关单。
    """
    try:
        if (
            _is_order_not_exist(exc)
            and age_minutes is not None
            and age_minutes > GIVE_UP_AFTER_HOURS * 60
        ):
            await orders.mark_order_closed(out_trade_no, reason="order_not_exist")
        else:
            await _touch(out_trade_no)
    except Exception:
        logger.exception("wechat pay reconcile bookkeeping failed order=%s", out_trade_no)


async def retry_failed_notifications(*, limit: int = 50) -> tuple[int, int]:
    """重放处理失败的回调（我方回了 500，微信可能已放弃重推）。

    按事件类型分流，各自走权威查单，比重放原报文更稳（原报文可能已过时）：
    * TRANSACTION.SUCCESS → 支付查单 `query_and_settle`
    * REFUND.*            → 退款查单 `refunds.query_and_settle_refund`
    其它类型不在这里处理（`_dispatch` 已把它们当场标为已处理）。
    """
    rows = await db.query_raw(
        """
        SELECT notify_id, event_type, out_trade_no,
               decoded_payload->>'out_refund_no' AS out_refund_no
        FROM wechat_pay_notifications
        WHERE processed_at IS NULL
          AND (
            (event_type = 'TRANSACTION.SUCCESS' AND out_trade_no IS NOT NULL)
            OR (event_type LIKE 'REFUND.%' AND decoded_payload ? 'out_refund_no')
          )
        ORDER BY received_at ASC
        LIMIT $1
        """,
        limit,
    )
    ok = 0
    failed = 0
    for row in rows:
        notify_id = str(_field(row, "notify_id"))
        event_type = str(_field(row, "event_type") or "")
        try:
            if event_type.startswith("REFUND."):
                result = await refunds.query_and_settle_refund(
                    str(_field(row, "out_refund_no"))
                )
                # 退款仍在处理中：保持未处理，下一轮再查，绝不提前标记。
                if result.get("status") == refunds.STATUS_PENDING:
                    continue
            else:
                await query_and_settle(str(_field(row, "out_trade_no")))
            await db.execute_raw(
                """
                UPDATE wechat_pay_notifications
                SET processed_at = CURRENT_TIMESTAMP, process_error = NULL
                WHERE notify_id = $1
                """,
                notify_id,
            )
            ok += 1
        except Exception as exc:  # 单条失败不中断整批
            failed += 1
            await db.execute_raw(
                "UPDATE wechat_pay_notifications SET process_error = $2 WHERE notify_id = $1",
                notify_id,
                f"retry: {type(exc).__name__}"[:500],
            )
    return ok, failed
