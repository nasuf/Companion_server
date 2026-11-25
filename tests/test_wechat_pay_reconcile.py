"""主动查单 / 关单 / 失败回调重放：回调丢失时的兜底不能反过来造成丢单。

重点覆盖审查发现的两条严重路径：
1. 关单与付款竞态：微信说 ORDERPAID 或网络失败时绝不落 closed。
2. 退款回调失败后被支付查单误标为已处理 → 钱退了权益没收。
"""

import pytest

from app.config import settings
from app.services.payments.wechat import reconcile
from app.services.payments.wechat.errors import WechatPayUpstreamError


class _FakeDb:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.queries: list[tuple[str, tuple]] = []
        self.executes: list[tuple[str, tuple]] = []

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows.pop(0) if self._rows else []

    async def execute_raw(self, sql, *args):
        self.executes.append((sql, args))
        return 1


@pytest.fixture(autouse=True)
def pay_settings(monkeypatch):
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1007054057")


@pytest.mark.parametrize(
    "error",
    [
        WechatPayUpstreamError("paid", status_code=400, code="ORDERPAID"),
        WechatPayUpstreamError("微信支付请求失败: ReadTimeout"),
        WechatPayUpstreamError("server", status_code=500, code="SYSTEM_ERROR"),
    ],
)
async def test_close_failure_never_marks_closed(monkeypatch, error):
    """关单没被微信确认时不落 closed，而是立刻再查一次单。"""
    closed, requeried = [], []

    async def fake_request(method, path, **kwargs):
        raise error

    async def fake_mark(out_trade_no, *, reason):
        closed.append(out_trade_no)

    async def fake_query(out_trade_no):
        requeried.append(out_trade_no)
        return {"status": "granted"}

    monkeypatch.setattr(reconcile.transport, "request", fake_request)
    monkeypatch.setattr(reconcile.orders, "mark_order_closed", fake_mark)
    monkeypatch.setattr(reconcile, "query_and_settle", fake_query)

    result = await reconcile.close_order("BS001")

    assert closed == []
    assert requeried == ["BS001"]
    assert result["status"] == "granted"


@pytest.mark.parametrize("code", ["ORDER_CLOSED", "ORDER_NOT_EXIST"])
async def test_close_confirmed_by_wechat_marks_closed(monkeypatch, code):
    closed = []

    async def fake_request(method, path, **kwargs):
        raise WechatPayUpstreamError("x", status_code=400, code=code)

    async def fake_mark(out_trade_no, *, reason):
        closed.append(out_trade_no)

    monkeypatch.setattr(reconcile.transport, "request", fake_request)
    monkeypatch.setattr(reconcile.orders, "mark_order_closed", fake_mark)

    assert (await reconcile.close_order("BS001"))["status"] == "closed"
    assert closed == ["BS001"]


async def test_query_failure_touches_order_so_scan_window_moves(monkeypatch):
    """查单持续失败的订单要推后 updated_at，不能永远占住扫描窗口前 100 名。"""
    from datetime import datetime, timezone

    fake = _FakeDb(rows=[[{
        "out_trade_no": "BS001", "status": "created",
        "created_at": datetime.now(timezone.utc),
    }]])
    monkeypatch.setattr(reconcile, "db", fake)

    async def boom(out_trade_no):
        raise WechatPayUpstreamError("x", status_code=404, code="ORDER_NOT_EXIST")

    monkeypatch.setattr(reconcile, "query_and_settle", boom)

    counts = await reconcile.reconcile_pending_orders(limit=10)

    # 微信侧没有这笔单不是故障（不可能被支付），不计 failed、不把 cron 标红。
    assert counts["failed"] == 0 and counts["pending"] == 1
    assert any("SET updated_at = CURRENT_TIMESTAMP" in sql for sql, _ in fake.executes)


async def test_transient_query_failure_counts_as_failed(monkeypatch):
    from datetime import datetime, timezone

    fake = _FakeDb(rows=[[{
        "out_trade_no": "BS001", "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }]])
    monkeypatch.setattr(reconcile, "db", fake)

    async def boom(out_trade_no):
        raise WechatPayUpstreamError("微信支付请求失败: ReadTimeout")

    monkeypatch.setattr(reconcile, "query_and_settle", boom)

    counts = await reconcile.reconcile_pending_orders(limit=10)
    assert counts["failed"] == 1


async def test_db_error_in_one_order_does_not_abort_batch(monkeypatch):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    fake = _FakeDb(rows=[[
        {"out_trade_no": "BS001", "status": "created", "created_at": now},
        {"out_trade_no": "BS002", "status": "created", "created_at": now},
    ]])
    monkeypatch.setattr(reconcile, "db", fake)
    seen = []

    async def flaky(out_trade_no):
        seen.append(out_trade_no)
        if out_trade_no == "BS001":
            raise RuntimeError("db down")
        return {"status": "granted"}

    monkeypatch.setattr(reconcile, "query_and_settle", flaky)

    counts = await reconcile.reconcile_pending_orders(limit=10)
    assert seen == ["BS001", "BS002"]
    assert counts["failed"] == 1 and counts["settled"] == 1


async def test_refund_notification_retry_uses_refund_query(monkeypatch):
    """REFUND.* 回调重放必须走退款查单，绝不走支付查单。"""
    fake = _FakeDb(rows=[[{
        "notify_id": "n1", "event_type": "REFUND.SUCCESS",
        "out_trade_no": "BS001", "out_refund_no": "RF001",
    }]])
    monkeypatch.setattr(reconcile, "db", fake)
    paid_queries, refund_queries = [], []

    async def fake_pay_query(out_trade_no):
        paid_queries.append(out_trade_no)
        return {"status": "pending"}

    async def fake_refund_query(out_refund_no):
        refund_queries.append(out_refund_no)
        return {"status": "success"}

    monkeypatch.setattr(reconcile, "query_and_settle", fake_pay_query)
    monkeypatch.setattr(reconcile.refunds, "query_and_settle_refund", fake_refund_query)

    ok, failed = await reconcile.retry_failed_notifications(limit=10)

    assert (ok, failed) == (1, 0)
    assert paid_queries == []
    assert refund_queries == ["RF001"]
    assert any("processed_at = CURRENT_TIMESTAMP" in sql for sql, _ in fake.executes)


async def test_pending_refund_notification_stays_unprocessed(monkeypatch):
    """退款还在处理中：不标记已处理，否则微信重推会被当成重放丢掉。"""
    fake = _FakeDb(rows=[[{
        "notify_id": "n1", "event_type": "REFUND.SUCCESS",
        "out_trade_no": "BS001", "out_refund_no": "RF001",
    }]])
    monkeypatch.setattr(reconcile, "db", fake)

    async def fake_refund_query(out_refund_no):
        return {"status": reconcile.refunds.STATUS_PENDING}

    monkeypatch.setattr(reconcile.refunds, "query_and_settle_refund", fake_refund_query)

    await reconcile.retry_failed_notifications(limit=10)
    assert not any("processed_at = CURRENT_TIMESTAMP" in sql for sql, _ in fake.executes)


async def test_retry_query_filters_by_event_type(monkeypatch):
    fake = _FakeDb(rows=[[]])
    monkeypatch.setattr(reconcile, "db", fake)
    await reconcile.retry_failed_notifications(limit=10)
    sql, _ = fake.queries[0]
    assert "event_type = 'TRANSACTION.SUCCESS'" in sql
    assert "event_type LIKE 'REFUND.%'" in sql


async def test_refund_trade_state_never_grants_unsettled_order(monkeypatch):
    """查单得到 REFUND 而我方未发放：钱已退回用户，不能再发货（商户平台退款不回调）。"""
    async def fake_request(method, path, **kwargs):
        return {
            "trade_state": "REFUND", "transaction_id": "wx-1",
            "amount": {"total": 3900, "currency": "CNY"},
        }

    settled = []
    marked = []

    async def fake_settle(**kwargs):
        settled.append(kwargs)
        return {"status": "granted"}

    async def fake_mark(out_trade_no, *, payload):
        marked.append(out_trade_no)
        return {"status": "refunded", "replay": False}

    monkeypatch.setattr(reconcile.transport, "request", fake_request)
    monkeypatch.setattr(reconcile.orders, "settle_paid_order", fake_settle)
    monkeypatch.setattr(reconcile.orders, "mark_refunded_before_grant", fake_mark)

    result = await reconcile.query_and_settle("BS001")
    assert settled == [] and marked == ["BS001"]
    assert result["status"] == "refunded"
