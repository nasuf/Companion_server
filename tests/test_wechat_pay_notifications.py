"""微信支付回调：幂等、未知事件、失败留痕，以及对 Apple 链路的零影响。"""

import json

import pytest

from app.services.payments.wechat import notifications


class _FakeDb:
    def __init__(self, query_rows=()):
        self._rows = list(query_rows)
        self.queries: list[tuple[str, tuple]] = []
        self.executes: list[tuple[str, tuple]] = []

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows.pop(0) if self._rows else []

    async def execute_raw(self, sql, *args):
        self.executes.append((sql, args))
        return 1


def _envelope(event_type="TRANSACTION.SUCCESS", **decrypted):
    payload = {
        "out_trade_no": "BS001",
        "transaction_id": "wx-1",
        "trade_state": "SUCCESS",
        "amount": {"total": 3900, "payer_total": 3900, "currency": "CNY"},
        "success_time": "2026-11-25T13:00:00+08:00",
    }
    payload.update(decrypted)
    return {
        "id": "notify-1",
        "event_type": event_type,
        "summary": "支付成功",
        "_decrypted": payload,
    }


async def test_processed_notification_replays_without_settling(monkeypatch):
    """同一 notify_id 重复推送：已处理过的直接回放，不再碰订单。"""
    # 插入冲突（返回空）→ 查到 processed_at 非空。
    fake = _FakeDb(query_rows=[[], [{"processed_at": "2026-11-25T05:00:00"}]])
    monkeypatch.setattr(notifications, "db", fake)
    settled = []
    monkeypatch.setattr(
        notifications.orders,
        "settle_paid_order",
        lambda **kw: settled.append(kw),
    )

    result = await notifications.handle_notification(_envelope())

    assert result["status"] == "replay"
    assert settled == []


async def test_unprocessed_duplicate_continues_processing(monkeypatch):
    """上次收到但没处理完（进程挂）：继续处理，业务层自身幂等。"""
    fake = _FakeDb(query_rows=[[], [{"processed_at": None}]])
    monkeypatch.setattr(notifications, "db", fake)

    async def fake_settle(**kwargs):
        return {"status": "granted", "replay": False}

    monkeypatch.setattr(notifications.orders, "settle_paid_order", fake_settle)

    result = await notifications.handle_notification(_envelope())

    assert result["status"] == "granted"
    assert any("processed_at = CURRENT_TIMESTAMP" in sql for sql, _ in fake.executes)


async def test_settlement_failure_records_error_and_reraises(monkeypatch):
    """处理失败要留 process_error 并抛出，让路由回 500 促使微信重推。"""
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)

    async def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(notifications.orders, "settle_paid_order", boom)

    with pytest.raises(RuntimeError):
        await notifications.handle_notification(_envelope())

    assert any("process_error = $2" in sql for sql, _ in fake.executes)
    assert any("RuntimeError" in str(args) for _, args in fake.executes)


async def test_non_success_trade_state_is_not_granted(monkeypatch):
    """SUCCESS 事件里 trade_state 不是 SUCCESS：不发货，只留审计。"""
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)
    settled = []
    monkeypatch.setattr(
        notifications.orders, "settle_paid_order", lambda **kw: settled.append(kw)
    )

    result = await notifications.handle_notification(
        _envelope(trade_state="NOTPAY")
    )

    assert result["status"] == "ignored"
    assert settled == []


async def test_unknown_event_type_is_acknowledged(monkeypatch):
    """未知事件落审计即视为处理完，避免微信无限重推。"""
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)

    result = await notifications.handle_notification(
        _envelope(event_type="SOMETHING.NEW")
    )

    assert result["status"] == "ignored"
    assert any("processed_at = CURRENT_TIMESTAMP" in sql for sql, _ in fake.executes)


async def test_refund_event_routed_to_refund_handler(monkeypatch):
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)
    seen = {}

    async def fake_refund(event_type, decrypted):
        seen["event_type"] = event_type
        return {"status": "success"}

    monkeypatch.setattr(
        notifications.refunds, "apply_refund_notification", fake_refund
    )

    result = await notifications.handle_notification(
        _envelope(event_type="REFUND.SUCCESS", out_refund_no="RF001")
    )

    assert seen["event_type"] == "REFUND.SUCCESS"
    assert result["status"] == "success"


async def test_audit_payload_is_redacted(monkeypatch):
    """落审计前脱敏：openid 之类的用户标识不进审计表。"""
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)

    async def fake_settle(**kwargs):
        return {"status": "granted"}

    monkeypatch.setattr(notifications.orders, "settle_paid_order", fake_settle)

    envelope = _envelope()
    envelope["_decrypted"]["payer"] = {"openid": "o-secret"}
    envelope["_decrypted"]["attach"] = "internal-note"
    await notifications.handle_notification(envelope)

    insert_sql, insert_args = fake.queries[0]
    assert "INSERT INTO wechat_pay_notifications" in insert_sql
    stored = json.loads(insert_args[-1])
    assert "payer" not in stored
    assert stored["out_trade_no"] == "BS001"


async def test_missing_out_trade_no_is_ignored(monkeypatch):
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)
    envelope = _envelope()
    del envelope["_decrypted"]["out_trade_no"]

    result = await notifications.handle_notification(envelope)
    assert result["status"] == "ignored"


def test_verify_and_decrypt_requires_all_headers():
    from app.services.payments.wechat.errors import WechatPaySignatureError

    with pytest.raises(WechatPaySignatureError):
        notifications.verify_and_decrypt(
            body="{}", timestamp=None, nonce="N", signature="S", serial="X"
        )


def test_wechat_notifications_module_does_not_touch_apple_tables():
    """渠道隔离的静态保证：微信模块不得出现任何 iap_* 表名。"""
    import pathlib

    root = pathlib.Path(notifications.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        # 注释里提到 iap_transactions 是允许的，SQL 语句里出现则不行。
        for keyword in ("FROM iap_", "INTO iap_", "UPDATE iap_"):
            assert keyword not in text, f"{path.name} 碰了 Apple 的表: {keyword}"


async def test_settlement_records_notify_id_on_order(monkeypatch):
    """订单行要回链到触发发货的那条回调（后台排查按 notify_id 对账）。"""
    fake = _FakeDb(query_rows=[[{"id": "row-1"}]])
    monkeypatch.setattr(notifications, "db", fake)
    seen = {}

    async def fake_settle(**kwargs):
        seen.update(kwargs)
        return {"status": "granted", "replay": False}

    monkeypatch.setattr(notifications.orders, "settle_paid_order", fake_settle)
    await notifications.handle_notification(_envelope())
    assert seen["notify_id"] == "notify-1"
