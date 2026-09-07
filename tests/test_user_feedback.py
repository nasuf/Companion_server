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
