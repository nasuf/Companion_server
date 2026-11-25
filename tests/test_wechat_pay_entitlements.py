"""VIP 权益的跨渠道合并：Apple 与微信叠加，且 Apple 既有行为不变。

这是本批唯一改动的共享代码（`vip/entitlements.py`），所以重点证两件事：
1. 没有微信订单时，Apple 的计算结果与改动前完全一致（回归保护）。
2. 两个渠道的时长按购买时间顺序叠加，不互相覆盖、不重复计算。
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.vip import entitlements


class _FakeDb:
    """按 SQL 内容分发返回值，便于只给某一个渠道喂数据。"""

    def __init__(self, *, apple_rows=(), wechat_rows=(), code_rows=()):
        self.apple_rows = list(apple_rows)
        self.wechat_rows = list(wechat_rows)
        self.code_rows = list(code_rows)
        self.sqls: list[str] = []

    async def query_raw(self, sql, *args):
        self.sqls.append(sql)
        if "iap_transactions" in sql and "kind = $2" in sql:
            return self.apple_rows
        if "wechat_pay_orders" in sql:
            return self.wechat_rows
        if "vip_code_redemptions" in sql:
            return self.code_rows
        if "iap_subscription_state" in sql:
            return []
        return []


BASE = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _apple(product_id, purchased, quantity=1):
    return {"product_id": product_id, "quantity": quantity, "purchase_date": purchased}


def _wechat(vip_days, paid_at):
    return {"vip_days": vip_days, "paid_at": paid_at}


async def test_apple_only_result_unchanged(monkeypatch):
    """没有微信订单时，结果就是 Apple 的原有算法：购买时间 + 31 天。"""
    db = _FakeDb(apple_rows=[_apple("com.bansheng.vip.month", BASE)])
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    assert end == BASE + timedelta(days=31)


async def test_wechat_only_uses_order_snapshot(monkeypatch):
    """微信侧天数来自订单行快照，不查目录——改价不改写历史权益。"""
    db = _FakeDb(wechat_rows=[_wechat(93, BASE)])
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    assert end == BASE + timedelta(days=93)


async def test_channels_stack_in_purchase_order(monkeypatch):
    """先 Apple 月卡再微信季卡：季卡从月卡到期日接着算，总计 31+93 天。"""
    db = _FakeDb(
        apple_rows=[_apple("com.bansheng.vip.month", BASE)],
        wechat_rows=[_wechat(93, BASE + timedelta(days=5))],
    )
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    assert end == BASE + timedelta(days=31 + 93)


async def test_wechat_purchased_first_then_apple(monkeypatch):
    """反向顺序同样成立：按时间排序归并，与哪个渠道无关。"""
    db = _FakeDb(
        apple_rows=[_apple("com.bansheng.vip.month", BASE + timedelta(days=10))],
        wechat_rows=[_wechat(30, BASE)],
    )
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    # 微信 30 天到 11/31，Apple 月卡从那里接着 +31。
    assert end == BASE + timedelta(days=30 + 31)


async def test_lapsed_gap_restarts_from_later_purchase(monkeypatch):
    """两次购买之间断过：后一次从自己的购买时间起算，不回填空档。"""
    db = _FakeDb(
        apple_rows=[_apple("com.bansheng.vip.month", BASE)],
        wechat_rows=[_wechat(31, BASE + timedelta(days=200))],
    )
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    assert end == BASE + timedelta(days=200 + 31)


async def test_zero_day_orders_ignored(monkeypatch):
    """钞票订单（vip_days=0）不得影响 VIP 时长。"""
    db = _FakeDb(wechat_rows=[_wechat(0, BASE)])
    monkeypatch.setattr(entitlements, "db", db)

    assert await entitlements._consumable_vip_floor("u1") is None


async def test_order_without_timestamp_is_skipped(monkeypatch):
    """缺时间戳的行跳过而不是当成 now——否则会凭空延长权益。"""
    db = _FakeDb(wechat_rows=[_wechat(31, None)])
    monkeypatch.setattr(entitlements, "db", db)

    assert await entitlements._consumable_vip_floor("u1") is None


async def test_no_orders_anywhere_yields_none(monkeypatch):
    db = _FakeDb()
    monkeypatch.setattr(entitlements, "db", db)

    assert await entitlements._consumable_vip_floor("u1") is None


async def test_wechat_query_only_counts_granted_orders(monkeypatch):
    """SQL 层面必须只看 granted：created/paid/refunded 的订单不能计入权益。"""
    db = _FakeDb(wechat_rows=[])
    monkeypatch.setattr(entitlements, "db", db)
    await entitlements._consumable_vip_floor("u1")

    wechat_sql = next(sql for sql in db.sqls if "wechat_pay_orders" in sql)
    assert "status = 'granted'" in wechat_sql
    assert "vip_days > 0" in wechat_sql


async def test_apple_query_unchanged_by_this_batch(monkeypatch):
    """Apple 的查询条件保持原样：同一张表、同一组条件。"""
    db = _FakeDb()
    monkeypatch.setattr(entitlements, "db", db)
    await entitlements._consumable_vip_floor("u1")

    apple_sql = next(sql for sql in db.sqls if "iap_transactions" in sql)
    assert "status = 'granted'" in apple_sql
    assert "kind = $2" in apple_sql


async def test_paid_vip_end_includes_both_channels(monkeypatch):
    """`compute_paid_vip_end` 是 vip_until 的来源，必须覆盖两个渠道。"""
    db = _FakeDb(wechat_rows=[_wechat(372, BASE)])
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements.compute_paid_vip_end("u1")
    assert end == BASE + timedelta(days=372)


@pytest.mark.parametrize("quantity", [1, 3])
async def test_apple_quantity_still_multiplies(monkeypatch, quantity):
    """Apple 的 quantity 语义不变（一次买多份）。"""
    db = _FakeDb(apple_rows=[_apple("com.bansheng.vip.month", BASE, quantity)])
    monkeypatch.setattr(entitlements, "db", db)

    end = await entitlements._consumable_vip_floor("u1")
    assert end == BASE + timedelta(days=31 * quantity)
