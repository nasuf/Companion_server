"""Admin character profile DELETE 路由测试.

Regression for: DELETE /admin-api/character/profiles/{id} 之前因为
batch_update_profile_status 函数上错误叠了 @router.delete 装饰器, 导致
DELETE 请求被路由到 batch handler 而非 archive_or_delete_profile, force=true
永远到不了硬删除分支, 前端"永久删除"按钮无效.
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


def test_delete_profile_force_true_hard_deletes(client):
    """force=true → 硬删 (走 archive_or_delete_profile, 不是 batch handler)."""
    profile = SimpleNamespace(id="p1", status="published")
    with patch("app.api.admin.character.db") as db_mock:
        db_mock.characterprofile.find_unique = AsyncMock(return_value=profile)
        db_mock.characterprofile.delete = AsyncMock()
        db_mock.characterprofile.update = AsyncMock()

        r = client.delete(
            "/admin-api/character/profiles/p1?force=true",
            headers=_admin_hdr(),
        )

    assert r.status_code == 200, r.text
    assert r.json()["action"] == "deleted"
    db_mock.characterprofile.delete.assert_awaited_once_with(where={"id": "p1"})
    db_mock.characterprofile.update.assert_not_called()


def test_delete_profile_default_archives(client):
    """默认 force=false → 软归档 (status='archived')."""
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
    assert r.json()["action"] == "archived"
    db_mock.characterprofile.update.assert_awaited_once()
    db_mock.characterprofile.delete.assert_not_called()


def test_delete_profile_not_found_returns_404(client):
    with patch("app.api.admin.character.db") as db_mock:
        db_mock.characterprofile.find_unique = AsyncMock(return_value=None)
        r = client.delete(
            "/admin-api/character/profiles/nonexistent?force=true",
            headers=_admin_hdr(),
        )
    assert r.status_code == 404


def test_delete_profile_no_admin_token_401(client):
    r = client.delete("/admin-api/character/profiles/p1?force=true")
    assert r.status_code == 401


def test_delete_profile_non_admin_403(client):
    r = client.delete(
        "/admin-api/character/profiles/p1?force=true",
        headers=make_auth_header("u1", role="user"),
    )
    assert r.status_code == 403
