"""Admin character profile DELETE 路由测试.

Regression for: DELETE /admin-api/character/profiles/{id} 之前因为
batch_update_profile_status 函数上错误叠了 @router.delete 装饰器, 导致
DELETE 请求被路由到 batch handler 而非删除路由, 前端"永久删除"按钮无效.

profile status 从 (draft / published / archived) 简化为 (draft / published)
后, DELETE 永远是物理删除, 不再接受 ?force=... 参数。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_auth_header


def _admin_hdr() -> dict:
    return make_auth_header("admin-id", role="admin")


@pytest.fixture
def client(api_client):
    return api_client


def test_delete_profile_hard_deletes(client):
    """DELETE 物理删除 profile, 不再有软归档分支。"""
    profile = SimpleNamespace(id="p1", status="published")
    with patch("app.api.admin.character.db") as db_mock:
        db_mock.characterprofile.find_unique = AsyncMock(return_value=profile)
        db_mock.characterprofile.delete = AsyncMock()
        db_mock.characterprofile.update = AsyncMock()

        r = client.delete(
            "/admin-api/character/profiles/p1",
            headers=_admin_hdr(),
        )

    assert r.status_code == 200, r.text
    assert r.json()["action"] == "deleted"
    db_mock.characterprofile.delete.assert_awaited_once_with(where={"id": "p1"})
    # status 简化后不再有 archive update 路径
    db_mock.characterprofile.update.assert_not_called()


def test_delete_profile_not_found_returns_404(client):
    with patch("app.api.admin.character.db") as db_mock:
        db_mock.characterprofile.find_unique = AsyncMock(return_value=None)
        r = client.delete(
            "/admin-api/character/profiles/nonexistent",
            headers=_admin_hdr(),
        )
    assert r.status_code == 404


def test_delete_profile_no_admin_token_401(client):
    r = client.delete("/admin-api/character/profiles/p1")
    assert r.status_code == 401


def test_delete_profile_non_admin_403(client):
    r = client.delete(
        "/admin-api/character/profiles/p1",
        headers=make_auth_header("u1", role="user"),
    )
    assert r.status_code == 403


def test_batch_status_rejects_archived(client):
    """status 简化为 draft / published, archived 应当被 400 拒绝。"""
    with patch("app.api.admin.character.db"):
        r = client.post(
            "/admin-api/character/profiles/batch-status",
            json={"ids": ["p1"], "status": "archived"},
            headers=_admin_hdr(),
        )
    assert r.status_code == 400


def test_batch_status_accepts_draft_and_published(client):
    """draft / published 二态合法。"""
    for status in ("draft", "published"):
        with patch("app.api.admin.character.db") as db_mock:
            db_mock.execute_raw = AsyncMock(return_value=2)
            r = client.post(
                "/admin-api/character/profiles/batch-status",
                json={"ids": ["p1", "p2"], "status": status},
                headers=_admin_hdr(),
            )
        assert r.status_code == 200, f"status={status!r} {r.text}"
