from unittest.mock import AsyncMock
import base64
import inspect

import pytest

from app.api.public import chat_media
from app.services.chat_media import repo, storage, vision
from app.services.chat_media.prompt import render_message_content_for_prompt


def _attachment(**overrides):
    data = {
        "id": "att-1",
        "user_id": "user-id",
        "conversation_id": "conv-id",
        "message_id": None,
        "kind": "image",
        "name": "photo.jpg",
        "mime": "image/jpeg",
        "size": 4,
        "width": 800,
        "height": 600,
        "storage_key": "user-id_photo.jpg",
        "url": "/chat/media/user-id_photo.jpg",
        "vision_status": "pending",
        "vision_summary": None,
        "vision_error": None,
        "created_at": None,
    }
    data.update(overrides)
    return repo.ChatAttachment(**data)


def test_chat_image_upload_accepts_flutter_base64_data_alias():
    upload = chat_media.ChatImageUpload.model_validate(
        {
            "conversation_id": "conv-id",
            "mime": "image/jpeg",
            "base64Data": base64.b64encode(b"image").decode("ascii"),
        }
    )

    assert upload.base64 == base64.b64encode(b"image").decode("ascii")


def test_chat_image_upload_limit_is_10mb():
    assert storage._MAX_IMAGE_BYTES == 10 * 1024 * 1024


def test_chat_media_route_precedes_chat_conversation_fallback():
    from app.main import app

    paths = [
        getattr(route, "path", "")
        for route in app.routes
        if "POST" in getattr(route, "methods", set())
    ]

    assert paths.index("/chat/media") < paths.index("/chat/{conversation_id}")


@pytest.mark.asyncio
async def test_upload_chat_image_saves_file_and_returns_attachment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(chat_media.repo, "conversation_belongs_to_user", AsyncMock(return_value=True))
    def fake_fire_background(coro):
        if inspect.iscoroutine(coro):
            coro.close()

    monkeypatch.setattr(chat_media, "fire_background", fake_fire_background)

    created = _attachment(size=5, storage_key="user-id_saved.jpg", url="/chat/media/user-id_saved.jpg")
    create_attachment = AsyncMock(return_value=created)
    monkeypatch.setattr(chat_media.repo, "create_attachment", create_attachment)
    monkeypatch.setattr(storage, "storage_key_for", lambda _user_id, _mime: "user-id_saved.jpg")

    response = await chat_media.upload_chat_image(
        chat_media.ChatImageUpload(
            conversation_id="conv-id",
            name="photo.jpg",
            mime="image/jpeg",
            width=800,
            height=600,
            base64=base64.b64encode(b"image").decode("ascii"),
        ),
        user={"sub": "user-id", "role": "user"},
    )

    assert (tmp_path / "user-id_saved.jpg").read_bytes() == b"image"
    create_attachment.assert_awaited_once()
    assert response.id == "att-1"
    assert response.url == "/chat/media/user-id_saved.jpg"


def test_render_message_content_for_prompt_includes_image_summary():
    rendered = render_message_content_for_prompt(
        "看看这个",
        {
            "attachments": [
                {
                    "id": "att-1",
                    "kind": "image",
                    "vision_summary": "画面里是一张日程截图，包含周三下午三点的会议。",
                }
            ]
        },
    )

    assert rendered.startswith("看看这个")
    assert "图片1：画面里是一张日程截图" in rendered


@pytest.mark.asyncio
async def test_doubao_vision_uses_openai_compatible_image_url_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "一张猫的照片。"}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, endpoint, *, headers, json):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(vision.settings, "ark_api_key", "ark-key")
    monkeypatch.setattr(vision.settings, "ark_base_url", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(
        vision.settings,
        "doubao_vision_model",
        "doubao-1-5-vision-pro-32k-250115",
    )
    monkeypatch.setattr(vision.httpx, "AsyncClient", FakeClient)

    result = await vision._call_doubao_vision(
        data_url="data:image/jpeg;base64,abcd",
        user_text="这是什么？",
    )

    assert result == "一张猫的照片。"
    assert captured["endpoint"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer ark-key"
    body = captured["json"]
    assert body["model"] == "doubao-1-5-vision-pro-32k-250115"
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,abcd"},
    }


@pytest.mark.asyncio
async def test_doubao_vision_404_mentions_model_config(monkeypatch):
    class FakeResponse:
        status_code = 404
        text = '{"error":"model not found"}'

        def raise_for_status(self):
            request = vision.httpx.Request(
                "POST",
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            )
            response = vision.httpx.Response(
                404,
                request=request,
                text=self.text,
            )
            raise vision.httpx.HTTPStatusError(
                "not found",
                request=request,
                response=response,
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(vision.settings, "ark_api_key", "ark-key")
    monkeypatch.setattr(vision.settings, "ark_base_url", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(vision.settings, "doubao_vision_model", "bad-model")
    monkeypatch.setattr(vision.httpx, "AsyncClient", FakeClient)

    with pytest.raises(RuntimeError, match="DOUBAO_VISION_MODEL"):
        await vision._call_doubao_vision(
            data_url="data:image/jpeg;base64,abcd",
            user_text="这是什么？",
        )


@pytest.mark.asyncio
async def test_ensure_vision_summaries_skips_without_ark_key(monkeypatch):
    monkeypatch.setattr(vision.settings, "ark_api_key", "")
    update = AsyncMock()
    monkeypatch.setattr(vision.repo, "update_vision_result", update)

    metadata = await vision.ensure_vision_summaries([_attachment()], user_text="")

    assert metadata[0]["vision_status"] == "skipped"
    assert "vision_summary" not in metadata[0]
    update.assert_awaited_once()
