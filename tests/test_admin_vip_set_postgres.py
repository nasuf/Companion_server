"""后台设置 VIP 到期时间走真实 SQL (vip_until 是 timestamp without time zone, 存 UTC)。

之前只有桩掉 admin_set_vip_until 的测试, 参数类型错 (text 写进 timestamp 列) 没人发现。
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.services import store_bundles, wallet
from tests.test_memory_evidence_postgres import origins  # noqa: F401


@pytest.mark.asyncio
async def test_admin_set_vip_until_writes_utc_and_clears(origins, monkeypatch):
    db, uid, *_ = origins
    monkeypatch.setattr(wallet, "db", db)
    try:
        # 北京时间写入, 库里按 UTC 存; 读回来是同一个时刻
        until = datetime(2026, 12, 2, 18, 30, tzinfo=timezone(timedelta(hours=8)))
        result = await wallet.admin_set_vip_until(uid, until)
        assert result["is_vip"] is True
        assert datetime.fromisoformat(result["vip_until"]) == until
        stored = (await db.query_raw("SELECT vip_until::text AS t FROM user_wallets WHERE user_id = $1", uid))[0]
        assert stored["t"].startswith("2026-12-02 10:30:00")

        # 过去的时间 = 结束 VIP
        assert (await wallet.admin_set_vip_until(uid, datetime.now(timezone.utc) - timedelta(days=1)))["is_vip"] is False
        # None = 清空
        cleared = await wallet.admin_set_vip_until(uid, None)
        assert cleared == {"user_id": uid, "is_vip": False, "vip_until": None}
    finally:
        await db.execute_raw("DELETE FROM user_wallets WHERE user_id = $1", uid)


@pytest.mark.asyncio
async def test_vip_trial_writes_vip_until(origins, monkeypatch):
    """体验会员与后台设置写同一列, 一起覆盖。"""
    db, uid, *_ = origins
    monkeypatch.setattr(wallet, "db", db)
    monkeypatch.setattr(store_bundles, "db", db)

    async def no_grant(user_id):
        return None

    monkeypatch.setattr(store_bundles.grants, "grant_monthly", no_grant)
    try:
        await wallet.ensure_wallet(uid)
        await store_bundles.activate_vip_trial(uid)
        row = (await db.query_raw("SELECT vip_until FROM user_wallets WHERE user_id = $1", uid))[0]
        assert wallet.is_vip_from_row(row)
    finally:
        await db.execute_raw("DELETE FROM user_wallets WHERE user_id = $1", uid)
