"""微信支付退款：出款幂等、权益回收时机、钞票扣减 floor 0。"""

import pytest

from app.config import settings
from app.services.payments.wechat import catalog, refunds


class _FakeTx:
    def __init__(self, rows):
        self._rows = list(rows)
        self.queries: list[tuple[str, tuple]] = []
        self.executes: list[tuple[str, tuple]] = []

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows.pop(0) if self._rows else []

    async def execute_raw(self, sql, *args):
        self.executes.append((sql, args))
        return 1


class _FakeDb:
    def __init__(self, *, rows=(), tx=None):
        self._rows = list(rows)
        self._tx = tx
        self.queries: list[tuple[str, tuple]] = []
        self.executes: list[tuple[str, tuple]] = []

    def tx(self):
        tx = self._tx

        class _Ctx:
            async def __aenter__(self):
                return tx

            async def __aexit__(self, *exc):
                return False

        return _Ctx()

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows.pop(0) if self._rows else []

    async def execute_raw(self, sql, *args):
        self.executes.append((sql, args))
        return 1


@pytest.fixture(autouse=True)
def pay_settings(monkeypatch):
    monkeypatch.setattr(settings, "wechatpay_notify_url", "https://example.test/notify")
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1007054057")


def _order(status="granted", amount=3900):
    return {"user_id": "u1", "status": status, "amount_fen": amount, "refunded_fen": 0}


def _refund_db(*tx_rows):
    """create_refund 在事务里锁订单行、查/建退款单；之后的更新走 db。"""
    tx = _FakeTx(list(tx_rows))
    return _FakeDb(tx=tx), tx


@pytest.mark.parametrize("status", ["created", "paid", "closed", "refunded"])
async def test_refund_rejects_orders_without_money(monkeypatch, status):
    """未支付、已关闭、已退款的订单没有钱可退。"""
    from app.services.payments.wechat.errors import WechatPayOrderStateError

    fake, _ = _refund_db([_order(status)])
    monkeypatch.setattr(refunds, "db", fake)
    with pytest.raises(WechatPayOrderStateError):
        await refunds.create_refund(out_trade_no="BS001", actor_id="admin-1")


async def test_failed_order_is_refundable(monkeypatch):
    """钱收了但没发货（金额不符/重复体验价）的订单必须能原路退回。"""
    fake, _ = _refund_db([_order("failed", 100)], [])
    monkeypatch.setattr(refunds, "db", fake)

    async def fake_request(method, path, *, body=None, verify_response=True):
        return {"refund_id": "wxrf-1", "status": "PROCESSING"}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    result = await refunds.create_refund(out_trade_no="BS001", actor_id="admin-1")
    assert result["status"] == "pending"


async def test_refund_locks_order_row(monkeypatch):
    """检查与建单在订单行锁内：后台双击 / 两人同时点只会有一张退款单。"""
    fake, tx = _refund_db([_order()], [])
    monkeypatch.setattr(refunds, "db", fake)

    async def fake_request(method, path, *, body=None, verify_response=True):
        return {"refund_id": "wxrf-1", "status": "PROCESSING"}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    await refunds.create_refund(out_trade_no="BS001", actor_id="admin-1")
    assert "FOR UPDATE" in tx.queries[0][0]
    assert any("INSERT INTO wechat_pay_refunds" in sql for sql, _ in tx.executes)


async def test_refund_on_missing_order_rejected(monkeypatch):
    from app.services.payments.wechat.errors import WechatPayOrderStateError

    fake, _ = _refund_db([])
    monkeypatch.setattr(refunds, "db", fake)
    with pytest.raises(WechatPayOrderStateError):
        await refunds.create_refund(out_trade_no="NOPE", actor_id="admin-1")


async def test_accepted_refund_is_replayed_not_resubmitted(monkeypatch):
    """已受理（有 refund_id）的退款单直接回放——重复出款是不可逆的损失。"""
    fake, _ = _refund_db(
        [_order()], [{"out_refund_no": "RF001", "status": "pending", "refund_id": "wxrf-1"}]
    )
    monkeypatch.setattr(refunds, "db", fake)
    called = []
    monkeypatch.setattr(refunds.transport, "request", lambda *a, **k: called.append(a))

    result = await refunds.create_refund(out_trade_no="BS001", actor_id="admin-1")

    assert result == {"out_refund_no": "RF001", "status": "pending", "replay": True}
    assert called == []


async def test_unaccepted_pending_refund_resubmits_same_refund_no(monkeypatch):
    """上次提交超时（没拿到 refund_id）：用同一个 out_refund_no 重提，保持微信侧幂等。"""
    fake, tx = _refund_db(
        [_order()], [{"out_refund_no": "RF001", "status": "pending", "refund_id": None}]
    )
    monkeypatch.setattr(refunds, "db", fake)
    captured = {}

    async def fake_request(method, path, *, body=None, verify_response=True):
        captured["body"] = body
        return {"refund_id": "wxrf-1", "status": "PROCESSING"}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    result = await refunds.create_refund(out_trade_no="BS001", actor_id="admin-1")

    assert captured["body"]["out_refund_no"] == "RF001"
    assert result["out_refund_no"] == "RF001"
    assert not any("INSERT INTO wechat_pay_refunds" in sql for sql, _ in tx.executes)


async def test_refund_amount_comes_from_order_not_caller(monkeypatch):
    """退款金额取订单原价，调用方无法指定金额（防误操作与越权退款）。"""
    fake, _ = _refund_db([_order(amount=8900)], [])
    monkeypatch.setattr(refunds, "db", fake)
    captured = {}

    async def fake_request(method, path, *, body=None, verify_response=True):
        captured["body"] = body
        return {"refund_id": "wxrf-1", "status": "PROCESSING"}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    result = await refunds.create_refund(out_trade_no="BS002", actor_id="admin-1")

    assert captured["body"]["amount"] == {"refund": 8900, "total": 8900, "currency": "CNY"}
    assert result["status"] == "pending"


async def test_pending_refund_does_not_revoke_entitlement_yet(monkeypatch):
    """发起退款时不回收权益：退款可能被拒或关闭，提前回收会两头空。"""
    fake, _ = _refund_db([_order()], [])
    monkeypatch.setattr(refunds, "db", fake)
    recomputed = []
    monkeypatch.setattr(
        refunds, "recompute_vip_entitlements", lambda user_id, **kw: recomputed.append(user_id)
    )

    async def fake_request(method, path, *, body=None, verify_response=True):
        return {"refund_id": "wxrf-1", "status": "PROCESSING"}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    await refunds.create_refund(out_trade_no="BS003", actor_id="admin-1")
    assert recomputed == []


@pytest.mark.parametrize(
    "refund_status,expected",
    [("CLOSED", "closed"), ("ABNORMAL", "abnormal"), ("PROCESSING", "pending")],
)
async def test_non_success_refund_notification_keeps_entitlement(
    monkeypatch, refund_status, expected
):
    fake = _FakeDb()
    monkeypatch.setattr(refunds, "db", fake)
    result = await refunds.apply_refund_notification(
        "REFUND.ABNORMAL",
        {"out_refund_no": "RF001", "refund_status": refund_status},
    )
    assert result["status"] == expected
    # 只更新退款单状态，不动订单和权益。
    assert all("wechat_pay_orders" not in sql for sql, _ in fake.executes)


async def test_refund_success_revokes_vip_and_marks_order(monkeypatch):
    tx = _FakeTx([
        [{
            "out_trade_no": "BS004", "user_id": "u1", "amount_fen": 3900,
            "status": "pending", "ticket_amount": 0, "vip_days": 31,
            "order_status": "granted",
        }],
    ])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))
    recomputed = []

    async def fake_recompute(user_id, **kw):
        recomputed.append(user_id)
        return True

    monkeypatch.setattr(refunds, "recompute_vip_entitlements", fake_recompute)

    result = await refunds.apply_refund_notification(
        "REFUND.SUCCESS",
        {"out_refund_no": "RF001", "refund_status": "SUCCESS",
         "success_time": "2026-11-25T14:00:00+08:00"},
    )

    assert result["status"] == "success"
    assert recomputed == ["u1"]
    assert any("status = 'refunded'" in sql for sql, _ in tx.executes)


async def test_refund_success_is_idempotent(monkeypatch):
    """重复的 REFUND.SUCCESS 不得二次扣减钞票。"""
    tx = _FakeTx([
        [{
            "out_trade_no": "BS005", "user_id": "u1", "amount_fen": 1800,
            "status": "success", "ticket_amount": 180, "vip_days": 0,
            "order_status": "refunded",
        }],
    ])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))

    result = await refunds.apply_refund_notification(
        "REFUND.SUCCESS", {"out_refund_no": "RF001", "refund_status": "SUCCESS"}
    )

    assert result == {"status": "success", "replay": True}
    assert tx.executes == []


async def test_ticket_reversal_floors_at_zero(monkeypatch):
    """用户已把钞票花掉时扣到 0 为止，实际扣减量落 ledger，不阻断退款。"""
    tx = _FakeTx([
        [{
            "out_trade_no": "BS006", "user_id": "u1", "amount_fen": 1800,
            "status": "pending", "ticket_amount": 180, "vip_days": 0,
            "order_status": "granted",
        }],
        [],                                  # 幂等检查：没有反向流水
        [{"ticket_balance": 50}],            # 余额只剩 50（subunits）
        [{"ticket_balance": 0, "point_balance": 0, "achievement_points_synced": 0}],
    ])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))
    ledger = []

    async def fake_record(**kwargs):
        ledger.append(kwargs)

    monkeypatch.setattr(refunds.wallet, "_record_ledger", fake_record)
    monkeypatch.setattr(refunds.wallet, "ticket_subunits", lambda amount: amount)
    monkeypatch.setattr(
        refunds.wallet, "wallet_balances", lambda row: {"ticket_balance": 0}
    )

    result = await refunds.apply_refund_notification(
        "REFUND.SUCCESS", {"out_refund_no": "RF001", "refund_status": "SUCCESS"}
    )

    assert result["status"] == "success"
    assert ledger[0]["delta"] == -50  # 只扣到 0，不是 -180
    assert ledger[0]["source"] == catalog.SOURCE_WECHAT_PAY_REFUND
    assert ledger[0]["source_id"] == "RF001"


async def test_ticket_reversal_skipped_when_already_reversed(monkeypatch):
    """ledger 里已有反向流水时不再扣——幂等靠 (source, source_id)。"""
    tx = _FakeTx([
        [{
            "out_trade_no": "BS007", "user_id": "u1", "amount_fen": 1800,
            "status": "pending", "ticket_amount": 180, "vip_days": 0,
            "order_status": "granted",
        }],
        [{"exists": 1}],  # 已反向过
    ])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))
    ledger = []
    monkeypatch.setattr(
        refunds.wallet, "_record_ledger",
        lambda **kw: ledger.append(kw),
    )

    await refunds.apply_refund_notification(
        "REFUND.SUCCESS", {"out_refund_no": "RF001", "refund_status": "SUCCESS"}
    )
    assert ledger == []


async def test_unknown_refund_no_is_ignored(monkeypatch):
    tx = _FakeTx([[]])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))
    result = await refunds.apply_refund_notification(
        "REFUND.SUCCESS", {"out_refund_no": "RF-NOPE", "refund_status": "SUCCESS"}
    )
    assert result["status"] == "unknown_refund"


async def test_upstream_rejection_marks_refund_failed(monkeypatch):
    """微信明确拒绝（4xx）：确定没受理，退款单落 failed。"""
    from app.services.payments.wechat.errors import WechatPayUpstreamError

    fake, _ = _refund_db([_order()], [])
    monkeypatch.setattr(refunds, "db", fake)

    async def boom(method, path, *, body=None, verify_response=True):
        raise WechatPayUpstreamError("rejected", status_code=403, code="NOTENOUGH")

    monkeypatch.setattr(refunds.transport, "request", boom)

    with pytest.raises(WechatPayUpstreamError):
        await refunds.create_refund(out_trade_no="BS008", actor_id="admin-1")
    assert any("status = 'failed'" in sql for sql, _ in fake.executes)


async def test_timeout_keeps_refund_pending(monkeypatch):
    """超时/5xx：微信可能已受理，不能判 failed（否则钱退了权益留着）。"""
    from app.services.payments.wechat.errors import WechatPayUpstreamError

    fake, _ = _refund_db([_order()], [])
    monkeypatch.setattr(refunds, "db", fake)

    async def boom(method, path, *, body=None, verify_response=True):
        raise WechatPayUpstreamError("微信支付请求失败: ReadTimeout")

    monkeypatch.setattr(refunds.transport, "request", boom)

    with pytest.raises(WechatPayUpstreamError):
        await refunds.create_refund(out_trade_no="BS008", actor_id="admin-1")
    assert not any("status = 'failed'" in sql for sql, _ in fake.executes)
    assert any("submit_uncertain" in str(args) for _, args in fake.executes)


async def test_refund_query_settles_success(monkeypatch):
    """退款查单（cron 兜底）拿到 SUCCESS 时与回调同路径回收权益。"""
    seen = {}

    async def fake_request(method, path, *, body=None, verify_response=True):
        seen["path"] = path
        return {"out_refund_no": "RF001", "out_trade_no": "BS001", "status": "SUCCESS"}

    async def fake_settle(**kwargs):
        seen["settled"] = kwargs["out_refund_no"]
        return {"status": "success", "replay": False}

    monkeypatch.setattr(refunds.transport, "request", fake_request)
    monkeypatch.setattr(refunds, "_settle_refund_success", fake_settle)
    monkeypatch.setattr(refunds, "db", _FakeDb())

    result = await refunds.query_and_settle_refund("RF001")

    assert seen["path"] == "/v3/refund/domestic/refunds/RF001"
    assert seen["settled"] == "RF001"
    assert result["status"] == "success"


async def test_merchant_platform_refund_is_adopted(monkeypatch):
    """商户平台直接退的款：补建退款单再回收，而不是当未知单号丢掉。"""
    fake = _FakeDb()
    monkeypatch.setattr(refunds, "db", fake)

    async def fake_settle(**kwargs):
        return {"status": "success", "replay": False}

    monkeypatch.setattr(refunds, "_settle_refund_success", fake_settle)

    await refunds.apply_refund_notification(
        "REFUND.SUCCESS",
        {
            "out_refund_no": "PLATFORM-1",
            "out_trade_no": "BS001",
            "refund_status": "SUCCESS",
            "amount": {"refund": 3900},
        },
    )
    # 先作废我方从未被受理的 pending 单（否则撞"一单一张进行中退款"的唯一索引）。
    sql, args = fake.executes[0]
    assert "superseded_by_platform_refund" in sql and "refund_id IS NULL" in sql
    assert args == ("BS001", "PLATFORM-1")
    sql, args = fake.executes[1]
    assert "INSERT INTO wechat_pay_refunds" in sql and "ON CONFLICT DO NOTHING" in sql
    assert args[0] == "PLATFORM-1" and args[3] == "BS001"


async def test_refunding_failed_order_does_not_touch_entitlements(monkeypatch):
    """failed 订单从没发过货：退款成功只置 refunded，不扣钞票、不重算 VIP。"""
    tx = _FakeTx([[{
        "out_trade_no": "BS011", "user_id": "u1", "amount_fen": 100,
        "status": "pending", "ticket_amount": 0, "vip_days": 30,
        "order_status": "failed",
    }]])
    monkeypatch.setattr(refunds, "db", _FakeDb(tx=tx))
    recomputed = []

    async def fake_recompute(user_id, **kw):
        recomputed.append(user_id)

    monkeypatch.setattr(refunds, "recompute_vip_entitlements", fake_recompute)
    result = await refunds.apply_refund_notification(
        "REFUND.SUCCESS", {"out_refund_no": "RF011", "refund_status": "SUCCESS"}
    )
    assert result["status"] == "success"
    assert recomputed == []
