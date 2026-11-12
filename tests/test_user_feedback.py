from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from PIL import Image


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 180, 240))
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def test_submit_user_feedback_persists_images(api_client, auth_header, monkeypatch):
    from app.api.public import feedback as feedback_api
    from app.services.feedback import storage

    created: dict = {}
    fake_db = MagicMock()

    async def fake_create(*, data):
        row = MagicMock()
        row.id = "fb-1"
        row.createdAt = MagicMock(isoformat=lambda: "2026-09-07T12:00:00+00:00")
        created["data"] = data
        return row

    fake_db.userfeedback.create = fake_create
    monkeypatch.setattr(feedback_api, "db", fake_db)
    monkeypatch.setattr(storage, "save_feedback_image", lambda **kwargs: "user-1_abc.jpg")

    response = api_client.post(
        "/users/me/feedback",
        data={
            "content": "聊天图片加载失败",
            "contact": "test@example.com",
            "occurred_at": "2026-09-07",
        },
        files=[("images", ("shot.jpg", _jpeg_bytes(), "image/jpeg"))],
        headers=auth_header("user-1"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "fb-1"
    assert created["data"]["imageKeys"] == ["user-1_abc.jpg"]


def test_admin_list_user_feedback(api_client, auth_header, monkeypatch):
    from app.api.admin import user_feedback as admin_feedback

    row = MagicMock()
    row.id = "fb-1"
    row.userId = "user-1"
    row.content = "建议增加夜间模式"
    row.contact = "18800001111"
    row.occurredAt = None
    row.imageKeys = []
    row.status = "open"
    row.appVersion = "0.5.0"
    row.platform = "ios"
    row.createdAt = MagicMock(isoformat=lambda: "2026-09-07T12:00:00+00:00")
    row.updatedAt = MagicMock(isoformat=lambda: "2026-09-07T12:00:00+00:00")
    row.user = MagicMock(username="13800138000", displayName="测试用户")

    fake_db = MagicMock()
    fake_db.userfeedback.count = AsyncMock(return_value=1)
    fake_db.userfeedback.find_many = AsyncMock(return_value=[row])
    monkeypatch.setattr(admin_feedback, "db", fake_db)

    response = api_client.get(
        "/admin-api/user-feedback",
        headers=auth_header("admin-1", role="admin"),
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["content"] == "建议增加夜间模式"


def test_feedback_database_failure_removes_new_files(api_client, auth_header, monkeypatch):
    import pytest
    from unittest.mock import Mock
    from app.api.public import feedback
    database = MagicMock()
    database.userfeedback.create = AsyncMock(side_effect=RuntimeError("synthetic DB failure"))
    monkeypatch.setattr(feedback, "db", database)
    monkeypatch.setattr(feedback.storage, "save_feedback_image", lambda **_: "user-1_new.jpg")
    remove = Mock()
    monkeypatch.setattr(feedback.storage, "delete_feedback_image", remove)
    with pytest.raises(RuntimeError, match="synthetic DB failure"):
        api_client.post("/users/me/feedback", data={"content":"合成反馈内容", "contact":"synthetic"},
            files=[("images", ("image.jpg", _jpeg_bytes(), "image/jpeg"))], headers=auth_header("user-1"))
    remove.assert_called_once_with("user-1_new.jpg")


def test_feedback_category_defaults_and_report(api_client, auth_header, monkeypatch):
    """投诉举报走同一接口, category=report; 旧版 App 不传按意见反馈; 非法值拒绝。"""
    from app.api.public import feedback as feedback_api

    created: list[dict] = []
    fake_db = MagicMock()

    async def fake_create(*, data):
        row = MagicMock()
        row.id = "fb-2"
        row.createdAt = MagicMock(isoformat=lambda: "2026-11-12T12:00:00+00:00")
        created.append(data)
        return row

    fake_db.userfeedback.create = fake_create
    monkeypatch.setattr(feedback_api, "db", fake_db)
    base = {"content": "这个角色说了不当内容", "contact": "13800000000"}
    assert api_client.post("/users/me/feedback", data=base, headers=auth_header("user-1")).status_code == 200
    assert created[-1]["category"] == "feedback"
    assert api_client.post("/users/me/feedback", data={**base, "category": "report"},
                           headers=auth_header("user-1")).status_code == 200
    assert created[-1]["category"] == "report"
    bad = api_client.post("/users/me/feedback", data={**base, "category": "spam"}, headers=auth_header("user-1"))
    assert bad.status_code == 400 and len(created) == 2


def test_admin_list_filters_by_category(api_client, auth_header, monkeypatch):
    from app.api.admin import user_feedback as admin_feedback

    fake_db = MagicMock()
    fake_db.userfeedback.count = AsyncMock(return_value=0)
    fake_db.userfeedback.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(admin_feedback, "db", fake_db)
    headers = auth_header("admin-1", role="admin")
    assert api_client.get("/admin-api/user-feedback?category=report", headers=headers).status_code == 200
    assert fake_db.userfeedback.count.await_args.kwargs["where"] == {"category": "report"}
    assert api_client.get("/admin-api/user-feedback?category=spam", headers=headers).status_code == 400
    row = MagicMock(category="report")
    assert admin_feedback._category(row) == "report"
    assert admin_feedback._category(MagicMock(category=None)) == "feedback"
