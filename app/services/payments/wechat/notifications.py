"""微信支付回调处理：验签 → 解密 → 落审计 → 幂等结算。

处理顺序是刻意的：**先验签再落库**。未通过验签的报文可能完全由攻击者构造，
落库只会污染审计表；验签通过后立刻落 `wechat_pay_notifications`，再处理业务，
这样即使业务处理失败，也留下了可重放的原始凭据（process_error 记原因）。

返回给微信的 HTTP 语义：
* 处理成功或确认是重放 → 200 + `{"code": "SUCCESS"}`，微信停止重推。
* 验签/解密失败 → 401。微信对任何非 2xx 都会按退避重推，所以这不能阻止重推；
  真正的兜底是 cron 主动查单（reconcile），回调验签失败不会丢单。
* 业务处理异常 → 500 + `{"code": "FAIL"}`，让微信按退避策略重推。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from app.db import db
from app.observability.events import (
    EVT_PAYMENT_NOTIFICATION,
    EVT_PAYMENT_NOTIFICATION_FAIL,
)
from app.services.payments.wechat import orders, refunds, transport
from app.services.payments.wechat.errors import (
    WechatPayDecryptError,
    WechatPaySignatureError,
)

logger = logging.getLogger(__name__)

EVENT_TRANSACTION_SUCCESS = "TRANSACTION.SUCCESS"
_REFUND_EVENTS = frozenset(
    {"REFUND.SUCCESS", "REFUND.ABNORMAL", "REFUND.CLOSED"}
)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _parse_rfc3339(value: Any) -> datetime | None:
    """微信时间形如 2026-11-25T13:00:00+08:00。解析失败不致命，置空即可。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("wechat pay unparsable timestamp")
        return None


def verify_and_decrypt(
    *,
    body: str,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    serial: str | None,
) -> dict[str, Any]:
    """验签 + 解密。任一环节失败抛错，调用方回 401。"""
    if not timestamp or not nonce or not signature:
        raise WechatPaySignatureError("回调缺少验签头")
    transport.assert_expected_serial(serial)
    transport.assert_fresh_timestamp(timestamp)
    transport.verify_signature(
        timestamp=timestamp, nonce=nonce, body=body, signature=signature
    )
    try:
        envelope = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WechatPayDecryptError("回调外层不是合法 JSON") from exc
    if not isinstance(envelope, dict):
        raise WechatPayDecryptError("回调外层不是 JSON 对象")
    resource = envelope.get("resource")
    if not isinstance(resource, dict):
        raise WechatPayDecryptError("回调缺少 resource")
    envelope["_serial"] = serial
    envelope["_decrypted"] = transport.decrypt_resource(resource)
    return envelope


async def handle_notification(envelope: dict[str, Any]) -> dict[str, Any]:
    """落审计 + 幂等分发。已处理过的 notify_id 直接回放，不重复发货。"""
    notify_id = str(envelope.get("id") or "")
    event_type = str(envelope.get("event_type") or "")
    decrypted: dict[str, Any] = envelope.get("_decrypted") or {}
    out_trade_no = decrypted.get("out_trade_no")
    transaction_id = decrypted.get("transaction_id")

    inserted = await db.query_raw(
        """
        INSERT INTO wechat_pay_notifications (
            notify_id, event_type, summary, out_trade_no, transaction_id,
            decoded_payload
        )
        VALUES ($1,$2,$3,$4,$5,$6::jsonb)
        ON CONFLICT (notify_id) DO NOTHING
        RETURNING id
        """,
        notify_id,
        event_type,
        envelope.get("summary"),
        out_trade_no if isinstance(out_trade_no, str) else None,
        transaction_id if isinstance(transaction_id, str) else None,
        json.dumps(_safe_decrypted(decrypted), ensure_ascii=False),
    )
    if not inserted:
        rows = await db.query_raw(
            "SELECT processed_at FROM wechat_pay_notifications WHERE notify_id = $1",
            notify_id,
        )
        if rows and _field(rows[0], "processed_at") is not None:
            return {"status": "replay", "event_type": event_type}
        # 上次收到了但没处理完（进程中途挂）：继续往下处理，业务层仍是幂等的。

    try:
        result = await _dispatch(event_type, decrypted, notify_id=notify_id)
    except Exception as exc:
        await db.execute_raw(
            """
            UPDATE wechat_pay_notifications
            SET process_error = $2 WHERE notify_id = $1
            """,
            notify_id,
            f"{type(exc).__name__}: {exc}"[:500],
        )
        logger.exception(
            "wechat pay notification failed event=%s",
            event_type,
            extra={
                "event": EVT_PAYMENT_NOTIFICATION_FAIL,
                "notify_id": notify_id,
                "event_type": event_type,
            },
        )
        raise

    await db.execute_raw(
        """
        UPDATE wechat_pay_notifications
        SET processed_at = CURRENT_TIMESTAMP, process_error = NULL
        WHERE notify_id = $1
        """,
        notify_id,
    )
    logger.info(
        "wechat pay notification handled event=%s status=%s",
        event_type,
        result.get("status"),
        extra={
            "event": EVT_PAYMENT_NOTIFICATION,
            "notify_id": notify_id,
            "event_type": event_type,
        },
    )
    return {"event_type": event_type, **result}


async def _dispatch(
    event_type: str, decrypted: dict[str, Any], *, notify_id: str | None = None
) -> dict[str, Any]:
    if event_type == EVENT_TRANSACTION_SUCCESS:
        return await _handle_transaction_success(decrypted, notify_id=notify_id)
    if event_type in _REFUND_EVENTS:
        return await refunds.apply_refund_notification(event_type, decrypted)
    # 未知事件类型：落了审计就算处理完，不让微信无限重推。
    return {"status": "ignored"}


async def _handle_transaction_success(
    decrypted: dict[str, Any], *, notify_id: str | None = None
) -> dict[str, Any]:
    out_trade_no = decrypted.get("out_trade_no")
    if not isinstance(out_trade_no, str) or not out_trade_no:
        return {"status": "ignored"}
    trade_state = str(decrypted.get("trade_state") or "")
    if trade_state != "SUCCESS":
        # SUCCESS 事件里 trade_state 理应为 SUCCESS；不是就不发货，留审计。
        return {"status": "ignored"}
    amount = decrypted.get("amount")
    paid = None
    currency = None
    if isinstance(amount, dict):
        total = amount.get("total")
        paid = int(total) if isinstance(total, int) else None
        currency = amount.get("currency") if isinstance(amount.get("currency"), str) else None
    return await orders.settle_paid_order(
        currency=currency,
        out_trade_no=out_trade_no,
        transaction_id=(
            decrypted.get("transaction_id")
            if isinstance(decrypted.get("transaction_id"), str)
            else None
        ),
        trade_state=trade_state,
        paid_amount_fen=paid,
        success_time=_parse_rfc3339(decrypted.get("success_time")),
        payload=decrypted,
        notify_id=notify_id or None,
    )


def _safe_decrypted(decrypted: dict[str, Any]) -> dict[str, Any]:
    """审计落库前脱敏：不存 payer.openid 之外的任何用户标识。"""
    allowed = {
        "out_trade_no",
        "transaction_id",
        "trade_type",
        "trade_state",
        "trade_state_desc",
        "bank_type",
        "success_time",
        "refund_id",
        "out_refund_no",
        "refund_status",
        "user_received_account",
    }
    safe = {key: value for key, value in decrypted.items() if key in allowed}
    for key in ("amount",):
        value = decrypted.get(key)
        if isinstance(value, dict):
            safe[key] = {
                field: value[field]
                for field in ("total", "payer_total", "refund", "payer_refund", "currency")
                if field in value
            }
    return safe
