"""user_id_for_app_account_token 回归：users.id 是 text 列，SQL 不能把参数强转成 uuid。

2026-11-15 生产：`WHERE id = $1::uuid` 触发 `operator does not exist: text = uuid`，
ONE_TIME_CHARGE 通知的兜底路径恒失败。mock 数据库的测试抓不到这类错误，所以分两层：
静态断言（CI 必跑）+ 真实 PostgreSQL（设 BILLING_TEST_DATABASE_URL 指向本地一次性库才运行，只用临时表）。
"""
import os
from datetime import timedelta
from urllib.parse import urlsplit

import pytest
from prisma import Prisma

from app.services.payments import grant

ACTIVE_TOKEN = "8864f12d-0000-4000-8000-000000000001"
SUSPENDED_TOKEN = "8864f12d-0000-4000-8000-000000000002"
UNKNOWN_TOKEN = "8864f12d-0000-4000-8000-000000000003"


class _FakeDb:
    def __init__(self):
        self.calls = []

    async def query_raw(self, sql, *args):
        self.calls.append((sql, args))
        return []


async def test_sql_does_not_cast_param_to_uuid(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(grant, "db", fake)
    assert await grant.user_id_for_app_account_token(ACTIVE_TOKEN) is None
    sql, args = fake.calls[0]
    assert "::uuid" not in sql.lower()
    assert args == (ACTIVE_TOKEN,)


@pytest.mark.parametrize("token", [None, "", "not-a-uuid"])
async def test_invalid_token_skips_query(monkeypatch, token):
    fake = _FakeDb()
    monkeypatch.setattr(grant, "db", fake)
    assert await grant.user_id_for_app_account_token(token) is None
    assert fake.calls == []


@pytest.fixture
async def postgres_users(monkeypatch):
    url = os.getenv("BILLING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("BILLING_TEST_DATABASE_URL must point to a disposable local PostgreSQL")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    try:
        async with database.tx(timeout=timedelta(seconds=30)) as transaction:
            # 与生产一致：users.id 是 TEXT。临时表随事务丢弃，不留永久表。
            await transaction.execute_raw(
                "CREATE TEMP TABLE users (id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active') ON COMMIT DROP"
            )
            await transaction.execute_raw(
                "INSERT INTO users (id, status) VALUES ($1, 'active'), ($2, 'suspended')",
                ACTIVE_TOKEN,
                SUSPENDED_TOKEN,
            )
            monkeypatch.setattr(grant, "db", transaction)
            yield transaction
    finally:
        await database.disconnect()


async def test_postgres_maps_token_to_active_user(postgres_users):
    assert await grant.user_id_for_app_account_token(ACTIVE_TOKEN) == ACTIVE_TOKEN
    assert await grant.user_id_for_app_account_token(ACTIVE_TOKEN.upper()) == ACTIVE_TOKEN
    assert await grant.user_id_for_app_account_token(SUSPENDED_TOKEN) is None
    assert await grant.user_id_for_app_account_token(UNKNOWN_TOKEN) is None
