"""微信支付下单与到账：幂等、金额校验、终态保护、权益隔离。

用轻量 fake 事务替代真实 DB（与 `test_iap_grant.py` 同策略），断言落到 SQL 与
参数上：金额不符时必须不发货、终态不得二次发放、回调重放只回放状态。
"""

import pytest

from app.config import settings
from app.services.payments.wechat import catalog, orders


class _FakeTx:
    """记录 SQL 的假事务。`rows` 按调用顺序喂给 query_raw。"""

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
    def __init__(self, tx, *, rows=()):
        self._tx = tx
        self._rows = list(rows)
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


def _order_row(**overrides):
    row = {
        "user_id": "user-1",
        "product_id": "wx.vip.month",
        "status": "created",
        "amount_fen": 3900,
        "vip_days": 31,
        "ticket_amount": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def no_side_effects(monkeypatch):
    """隔掉钱包、权益重算与后台任务——这里只测订单状态机本身。"""
    calls: dict[str, list] = {"credit": [], "recompute": [], "background": []}

    async def fake_credit(user_id, amount, **kwargs):
        calls["credit"].append((user_id, amount, kwargs))
        return {}

    async def fake_ensure(user_id, **kwargs):
        return {}

    async def fake_recompute(user_id, **kwargs):
        calls["recompute"].append(user_id)
        return True

    monkeypatch.setattr(orders.wallet, "credit_tickets", fake_credit)
    monkeypatch.setattr(orders.wallet, "ensure_wallet", fake_ensure)
    monkeypatch.setattr(orders, "recompute_vip_entitlements", fake_recompute)
    monkeypatch.setattr(
        orders, "fire_background", lambda coro: (coro.close(), calls["background"].append(1))
    )
    return calls


async def test_amount_mismatch_never_grants(monkeypatch, no_side_effects):
    """伪造/错配金额的回调必须落 failed，且一分钱权益都不发。"""
    tx = _FakeTx([[_order_row()]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS001",
        transaction_id="wx-1",
        trade_state="SUCCESS",
        paid_amount_fen=1,  # 应付 3900
        success_time=None,
        payload={},
    )

    assert result["status"] == "failed"
    assert no_side_effects["credit"] == []
    assert no_side_effects["recompute"] == []
    assert any(args[-1] == "amount_mismatch" for _, args in tx.executes)
    assert not any("status = 'granted'" in sql for sql, _ in tx.executes)


@pytest.mark.parametrize("terminal", ["granted", "refunded", "failed"])
async def test_terminal_order_replays_without_regranting(
    monkeypatch, no_side_effects, terminal
):
    """已到终态的订单重复回调只回放状态，绝不二次发放（退款后尤其重要）。"""
    tx = _FakeTx([[_order_row(status=terminal, ticket_amount=180, vip_days=0)]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS001",
        transaction_id="wx-1",
        trade_state="SUCCESS",
        paid_amount_fen=3900,
        success_time=None,
        payload={},
    )

    assert result == {"status": terminal, "replay": True}
    assert no_side_effects["credit"] == []
    assert tx.executes == []


async def test_unknown_order_is_not_settled(monkeypatch, no_side_effects):
    """同一商户号下其它业务的订单号：不认、不发货、不报错。"""
    tx = _FakeTx([[]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="OTHER-BIZ",
        transaction_id="wx-1",
        trade_state="SUCCESS",
        paid_amount_fen=3900,
        success_time=None,
        payload={},
    )

    assert result["status"] == "unknown_order"
    assert tx.executes == []


async def test_ticket_order_credits_snapshotted_amount(monkeypatch, no_side_effects):
    """到账数量取订单行快照，不是现查目录——改价不影响历史订单。"""
    tx = _FakeTx([[_order_row(product_id="wx.ticket.180", vip_days=0, ticket_amount=180, amount_fen=1800)]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS002",
        transaction_id="wx-2",
        trade_state="SUCCESS",
        paid_amount_fen=1800,
        success_time=None,
        payload={},
    )

    assert result["status"] == "granted"
    user_id, amount, kwargs = no_side_effects["credit"][0]
    assert (user_id, amount) == ("user-1", 180)
    assert kwargs["source"] == catalog.SOURCE_WECHAT_PAY
    # source_id 用商户订单号 → 与退款的反向流水能对上。
    assert kwargs["source_id"] == "BS002"
    # 钞票订单不碰 VIP。
    assert no_side_effects["recompute"] == []


async def test_vip_order_recomputes_entitlement_in_same_transaction(
    monkeypatch, no_side_effects
):
    tx = _FakeTx([[_order_row()]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS003",
        transaction_id="wx-3",
        trade_state="SUCCESS",
        paid_amount_fen=3900,
        success_time=None,
        payload={},
    )

    assert result["status"] == "granted"
    assert no_side_effects["recompute"] == ["user-1"]
    assert no_side_effects["credit"] == []
    # VIP 到账后触发当月权益发放。
    assert no_side_effects["background"] == [1]


async def test_trial_order_marks_trial_used(monkeypatch, no_side_effects):
    """¥1 体验与 Apple 共用 vip_trial_used，防止两个渠道各买一次。"""
    tx = _FakeTx([[_order_row(product_id=catalog.TRIAL_PRODUCT_ID, amount_fen=100, vip_days=30)]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    await orders.settle_paid_order(
        out_trade_no="BS004",
        transaction_id="wx-4",
        trade_state="SUCCESS",
        paid_amount_fen=100,
        success_time=None,
        payload={},
    )

    assert any("vip_trial_used = TRUE" in sql for sql, _ in tx.executes)


async def test_missing_paid_amount_is_not_granted(monkeypatch, no_side_effects):
    """报文缺 amount 时不能跳过校验：不发放，落 failed（后台可退款）。"""
    tx = _FakeTx([[_order_row()]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS005",
        transaction_id="wx-5",
        trade_state="SUCCESS",
        paid_amount_fen=None,
        success_time=None,
        payload={},
    )
    assert result["status"] == "failed"
    assert no_side_effects["credit"] == [] and no_side_effects["recompute"] == []
    assert any(args[-1] == "amount_missing" for _, args in tx.executes)


async def test_non_cny_currency_is_not_granted(monkeypatch, no_side_effects):
    tx = _FakeTx([[_order_row()]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS005",
        transaction_id="wx-5",
        trade_state="SUCCESS",
        paid_amount_fen=3900,
        success_time=None,
        payload={},
        currency="USD",
    )
    assert result["status"] == "failed"
    assert any(args[-1] == "currency_mismatch" for _, args in tx.executes)


async def test_closed_order_paid_late_is_still_granted(monkeypatch, no_side_effects):
    """关单与付款竞态：我方已 closed，但微信确认 SUCCESS → 钱到了就必须发货。"""
    tx = _FakeTx([[_order_row(status="closed")]])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS009",
        transaction_id="wx-9",
        trade_state="SUCCESS",
        paid_amount_fen=3900,
        success_time=None,
        payload={},
    )
    assert result == {"status": "granted", "replay": False}
    assert no_side_effects["recompute"] == ["user-1"]
    assert any("closed_at = NULL" in sql for sql, _ in tx.executes)


async def test_second_trial_payment_is_not_granted(monkeypatch, no_side_effects):
    """先下两张体验单再分别付款：第二笔在钱包行锁内被拦下，落 failed 供退款。"""
    tx = _FakeTx([
        [_order_row(product_id=catalog.TRIAL_PRODUCT_ID, amount_fen=100, vip_days=30)],
        [{"vip_trial_used": True}],
    ])
    monkeypatch.setattr(orders, "db", _FakeDb(tx))

    result = await orders.settle_paid_order(
        out_trade_no="BS010",
        transaction_id="wx-10",
        trade_state="SUCCESS",
        paid_amount_fen=100,
        success_time=None,
        payload={},
    )
    assert result["status"] == "failed"
    assert no_side_effects["recompute"] == []
    assert any("trial_already_used" in sql for sql, _ in tx.executes)
    assert any("FOR UPDATE" in sql for sql, _ in tx.queries)


async def test_create_order_passes_time_expire(monkeypatch, no_side_effects):
    """time_expire 必须传给微信：否则微信侧 2 小时有效，我方 30 分钟就关单。"""
    fake_db = _FakeDb(_FakeTx([]), rows=[[{"vip_trial_used": False, "vip_until": None}]])
    monkeypatch.setattr(orders, "db", fake_db)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wxtest")
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1900000109")
    captured = {}

    async def fake_request(method, path, *, body=None, verify_response=True):
        captured["body"] = body
        return {"prepay_id": "prepay-1"}

    monkeypatch.setattr(orders.transport, "request", fake_request)
    monkeypatch.setattr(orders.transport, "jsapi_pay_params", lambda prepay_id: {})

    await orders.create_order(user_id="user-1", product_id="wx.vip.month", openid="o-1")

    from datetime import datetime, timedelta

    expire = datetime.fromisoformat(captured["body"]["time_expire"])
    assert expire.utcoffset() == timedelta(hours=8)
    from app.services.payments.wechat import reconcile

    assert orders.ORDER_TTL_MINUTES < reconcile.CLOSE_AFTER_MINUTES


async def test_vip_user_cannot_buy_trial(monkeypatch, no_side_effects):
    """与 catalog 页同一判据：当前已是 VIP 也不能买体验价。"""
    from datetime import datetime, timedelta, timezone

    from app.services.payments.wechat.errors import WechatPayTrialUsedError

    monkeypatch.setattr(
        orders,
        "db",
        _FakeDb(
            _FakeTx([]),
            rows=[[{"vip_trial_used": False, "vip_until": datetime.now(timezone.utc) + timedelta(days=5)}]],
        ),
    )
    with pytest.raises(WechatPayTrialUsedError):
        await orders.create_order(
            user_id="user-1", product_id=catalog.TRIAL_PRODUCT_ID, openid="o-1"
        )


async def test_create_order_rejects_unknown_product(monkeypatch, no_side_effects):
    monkeypatch.setattr(orders, "db", _FakeDb(_FakeTx([])))
    from app.services.payments.wechat.errors import WechatPayUnknownProductError

    with pytest.raises(WechatPayUnknownProductError):
        await orders.create_order(
            user_id="user-1", product_id="wx.vip.forever", openid="o-1"
        )


async def test_create_order_blocks_second_trial(monkeypatch, no_side_effects):
    from app.services.payments.wechat.errors import WechatPayTrialUsedError

    monkeypatch.setattr(
        orders, "db", _FakeDb(_FakeTx([]), rows=[[{"vip_trial_used": True}]])
    )
    with pytest.raises(WechatPayTrialUsedError):
        await orders.create_order(
            user_id="user-1", product_id=catalog.TRIAL_PRODUCT_ID, openid="o-1"
        )


async def test_create_order_uses_server_price(monkeypatch, no_side_effects):
    """下单金额来自服务端目录；客户端无法影响价格。"""
    fake_db = _FakeDb(_FakeTx([]), rows=[[{"vip_trial_used": False}]])
    monkeypatch.setattr(orders, "db", fake_db)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wxtest")
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1900000109")
    monkeypatch.setattr(settings, "wechatpay_notify_url", "https://example.test/notify")

    captured = {}

    async def fake_request(method, path, *, body=None, verify_response=True):
        captured["body"] = body
        return {"prepay_id": "prepay-1"}

    monkeypatch.setattr(orders.transport, "request", fake_request)
    monkeypatch.setattr(
        orders.transport, "jsapi_pay_params", lambda prepay_id: {"package": prepay_id}
    )

    result = await orders.create_order(
        user_id="user-1", product_id="wx.vip.year", openid="o-1"
    )

    assert captured["body"]["amount"] == {"total": 24900, "currency": "CNY"}
    assert result["amount_fen"] == 24900
    assert result["out_trade_no"].startswith("BS")
    # 订单行记录的是服务端价格与权益快照。
    insert_sql, insert_args = fake_db.executes[0]
    assert "INSERT INTO wechat_pay_orders" in insert_sql
    assert 24900 in insert_args and 372 in insert_args


async def test_create_order_marks_failure_when_prepay_rejected(
    monkeypatch, no_side_effects
):
    """微信下单失败时订单停在 created 并留原因，不会凭空成功。"""
    fake_db = _FakeDb(_FakeTx([]), rows=[[{"vip_trial_used": False}]])
    monkeypatch.setattr(orders, "db", fake_db)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wxtest")
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1900000109")

    from app.services.payments.wechat.errors import WechatPayUpstreamError

    async def fake_request(method, path, *, body=None, verify_response=True):
        raise WechatPayUpstreamError("boom")

    monkeypatch.setattr(orders.transport, "request", fake_request)

    with pytest.raises(WechatPayUpstreamError):
        await orders.create_order(
            user_id="user-1", product_id="wx.vip.month", openid="o-1"
        )
    assert any("jsapi_prepay_failed" in sql for sql, _ in fake_db.executes)


async def test_missing_prepay_id_is_not_treated_as_success(monkeypatch, no_side_effects):
    fake_db = _FakeDb(_FakeTx([]), rows=[[{"vip_trial_used": False}]])
    monkeypatch.setattr(orders, "db", fake_db)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wxtest")
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1900000109")

    async def fake_request(method, path, *, body=None, verify_response=True):
        return {}

    monkeypatch.setattr(orders.transport, "request", fake_request)
    from app.services.payments.wechat.errors import WechatPayOrderStateError

    with pytest.raises(WechatPayOrderStateError):
        await orders.create_order(
            user_id="user-1", product_id="wx.vip.month", openid="o-1"
        )
    assert any("prepay_id_missing" in sql for sql, _ in fake_db.executes)


async def test_close_only_touches_unpaid_orders(monkeypatch):
    fake_db = _FakeDb(_FakeTx([]))
    monkeypatch.setattr(orders, "db", fake_db)
    await orders.mark_order_closed("BS006", reason="expired_unpaid")
    sql, _ = fake_db.executes[0]
    # 必须带 status = 'created' 条件，否则会把已支付订单误关。
    assert "status = 'created'" in sql


def test_catalog_prices_match_flutter_tiers():
    """价格与 Flutter 端一致（月 39 / 季 89 / 年 249 / 体验 1；钞票 1/8/18/30/98/198）。"""
    assert [p.price_fen for p in catalog.VIP_PRODUCTS] == [3900, 8900, 24900]
    assert [p.vip_days for p in catalog.VIP_PRODUCTS] == [31, 93, 372]
    assert catalog.TRIAL_PRODUCT.price_fen == 100
    assert catalog.TRIAL_PRODUCT.vip_days == 30
    assert [p.price_fen for p in catalog.TICKET_PRODUCTS] == [
        100, 800, 1800, 3000, 9800, 19800
    ]
    assert [p.ticket_amount for p in catalog.TICKET_PRODUCTS] == [
        10, 80, 180, 300, 980, 1980
    ]


def test_catalog_has_no_subscription_products():
    """内测不开连续包月：出现订阅类商品说明有人误加，退订/对账链路还没做。"""
    assert all(p.kind == catalog.KIND_CONSUMABLE for p in catalog.WECHAT_PRODUCTS.values())


def test_product_ids_do_not_collide_with_apple():
    """两个渠道的 product_id 必须不重叠，否则权益计算会互相误认。"""
    from app.services.payments import catalog as apple_catalog

    assert not set(catalog.WECHAT_PRODUCTS) & set(apple_catalog.APPLE_PRODUCTS)
