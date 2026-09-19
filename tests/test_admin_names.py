"""Admin name-library HTTP surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.jwt_auth import require_admin_jwt


def _admin_override():
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {
        "sub": "admin-1",
        "role": "admin",
    }
    return app


def _row(**overrides):
    base = dict(
        id="n1",
        name="陈砚",
        nickname="阿砚",
        gender="male",
        status="active",
        sortOrder=1,
        createdAt="2026-09-19",
        updatedAt="2026-09-19",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_list_names_filters_and_returns_counts(api_client):
    app = _admin_override()
    try:
        with patch("app.api.admin.names.db") as mock_db:
            mock_db.nametemplate.count = AsyncMock(side_effect=[1, 698, 699])
            mock_db.nametemplate.find_many = AsyncMock(return_value=[_row()])
            response = api_client.get("/admin-api/names?gender=male&q=陈")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "陈砚"
    assert body["counts"] == {"male": 698, "female": 699}
    where = mock_db.nametemplate.find_many.await_args.kwargs["where"]
    assert where["gender"] == "male"
    assert where["OR"][0]["name"]["contains"] == "陈"


def test_create_name_rejects_blank(api_client):
    app = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/names",
            json={"name": "  ", "gender": "female"},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
    assert response.status_code == 400


def test_create_name_success(api_client):
    app = _admin_override()
    try:
        with patch("app.api.admin.names.db") as mock_db:
            mock_db.nametemplate.find_first = AsyncMock(return_value=_row(sortOrder=1))
            mock_db.nametemplate.create = AsyncMock(
                return_value=_row(id="n2", name="苏知茉", nickname="小茉", gender="female", sortOrder=0),
            )
            response = api_client.post(
                "/admin-api/names",
                json={"name": "苏知茉", "nickname": "小茉", "gender": "女"},
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 201, response.text
    assert response.json()["gender"] == "female"
    payload = mock_db.nametemplate.create.await_args.kwargs["data"]
    assert payload["gender"] == "female"
    assert payload["sortOrder"] == 0
