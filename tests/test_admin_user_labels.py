"""后台用户名标签只是展示增强: 读失败返回空, 接口沿用原 username。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.user_profile import admin_user_labels_for_ids


@pytest.mark.asyncio
async def test_labels_use_the_callers_client():
    client = MagicMock()
    client.user.find_many = AsyncMock(return_value=[SimpleNamespace(id="u1", username="wx_1", displayName=None)])
    client.authidentity.find_many = AsyncMock(return_value=[SimpleNamespace(
        userId="u1", provider="wechat", openid="o", providerAccountId="o", rawProfile={"nickname": "小明"})])
    assert await admin_user_labels_for_ids(["u1", "u1"], database=client) == {"u1": "小明"}
    client.user.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_failure_returns_no_labels():
    client = MagicMock()
    client.user.find_many = AsyncMock(side_effect=RuntimeError("not connected"))
    client.authidentity.find_many = AsyncMock(return_value=[])
    assert await admin_user_labels_for_ids(["u1"], database=client) == {}
    assert await admin_user_labels_for_ids([], database=client) == {}
