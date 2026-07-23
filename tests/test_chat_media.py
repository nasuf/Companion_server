from unittest.mock import AsyncMock
import base64
import inspect
import io

import pytest
from PIL import Image

from app.api.public import chat_media
from app.services.chat_media import repo, storage, vision
from app.services.chat_media.prompt import render_message_content_for_prompt


def _image_bytes(
    width: int,
    height: int,
    *,
    fmt: str = "JPEG",
    color=(120, 180, 90),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=fmt)
    return buffer.getvalue()


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


def test_process_image_upload_keeps_small_images_byte_identical():
    blob = _image_bytes(640, 480)

    processed = storage.process_image_upload(blob, "image/jpeg")

    assert processed.blob == blob
    assert processed.mime == "image/jpeg"
    assert (processed.width, processed.height) == (640, 480)


def test_process_image_upload_caps_oversized_images():
    blob = _image_bytes(4000, 3000)

    processed = storage.process_image_upload(blob, "image/jpeg")

    assert processed.mime == "image/jpeg"
    assert max(processed.width, processed.height) <= storage._INGEST_MAX_EDGE
    # Aspect ratio preserved (4:3).
    assert abs(processed.width / processed.height - 4 / 3) < 0.01
    assert len(processed.blob) < len(blob)


def test_process_image_upload_keeps_png_alpha():
    buffer = io.BytesIO()
    Image.new("RGBA", (3000, 3000), (10, 20, 30, 128)).save(buffer, format="PNG")

    processed = storage.process_image_upload(buffer.getvalue(), "image/png")

    assert processed.mime == "image/png"
    assert max(processed.width, processed.height) <= storage._INGEST_MAX_EDGE
    assert "A" in Image.open(io.BytesIO(processed.blob)).getbands()


def test_process_image_upload_falls_back_on_undecodable_blob():
    processed = storage.process_image_upload(b"not-an-image", "image/jpeg")

    assert processed.blob == b"not-an-image"
    assert processed.width is None and processed.height is None


def test_generate_thumbnail_blob_caps_edge_and_is_jpeg():
    thumb = storage.generate_thumbnail_blob(_image_bytes(2000, 1000))

    assert thumb is not None
    image = Image.open(io.BytesIO(thumb))
    assert image.format == "JPEG"
    assert max(image.size) <= storage._THUMB_MAX_EDGE
    assert storage.generate_thumbnail_blob(b"noise") is None


def test_thumb_storage_key_keeps_owner_prefix():
    key = "user-id_cabc_0123456789abcdef0123456789abcdef.png"

    thumb_key = storage.thumb_storage_key(key)

    assert thumb_key == "user-id_cabc_0123456789abcdef0123456789abcdef_t.jpg"
    assert thumb_key.startswith("user-id_")


def test_save_image_with_thumbnail_writes_both_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)

    key, processed = storage.save_image_with_thumbnail(
        user_id="user-id",
        conversation_id="conv-id",
        blob=_image_bytes(1200, 900),
        mime="image/jpeg",
    )

    assert (tmp_path / key).exists()
    assert (tmp_path / storage.thumb_storage_key(key)).exists()
    assert (processed.width, processed.height) == (1200, 900)


def test_serve_media_thumb_variant_and_cache_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    original = _image_bytes(1200, 900)
    key, _ = storage.save_image_with_thumbnail(
        user_id="user-id",
        conversation_id="conv-id",
        blob=original,
        mime="image/jpeg",
    )

    thumb_response = storage.serve_media(key, user_id="user-id", variant="thumb")
    original_response = storage.serve_media(key, user_id="user-id")

    assert str(thumb_response.path) == str(tmp_path / storage.thumb_storage_key(key))
    assert str(original_response.path) == str(tmp_path / key)
    assert (
        thumb_response.headers["cache-control"]
        == "private, max-age=31536000, immutable"
    )
    assert (
        original_response.headers["cache-control"]
        == "private, max-age=31536000, immutable"
    )


def test_serve_media_thumb_variant_falls_back_to_original(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    (tmp_path / "user-id_legacy.jpg").write_bytes(b"legacy-bytes")

    response = storage.serve_media(
        "user-id_legacy.jpg", user_id="user-id", variant="thumb"
    )

    assert str(response.path) == str(tmp_path / "user-id_legacy.jpg")


def test_serve_media_rejects_other_users(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    (tmp_path / "owner_file.jpg").write_bytes(b"data")

    with pytest.raises(Exception) as excinfo:
        storage.serve_media("owner_file.jpg", user_id="intruder")

    assert getattr(excinfo.value, "status_code", None) == 403


def test_delete_media_file_removes_thumbnail_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    key, _ = storage.save_image_with_thumbnail(
        user_id="user-id",
        conversation_id="conv-id",
        blob=_image_bytes(800, 600),
        mime="image/jpeg",
    )

    storage.delete_media_file(key)

    assert not (tmp_path / key).exists()
    assert not (tmp_path / storage.thumb_storage_key(key)).exists()


def test_audio_storage_key_has_deterministic_conversation_scope():
    prefix = storage.conversation_storage_prefix("user-id", "conv-id")

    key = storage.storage_key_for(
        "user-id",
        "audio/mp4",
        conversation_id="conv-id",
    )

    assert key.startswith(prefix)
    assert key.endswith(".m4a")
    assert prefix != storage.conversation_storage_prefix("user-id", "other-conv")


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

    def scoped_storage_key(_user_id, _mime, **kwargs):
        assert kwargs["conversation_id"] == "conv-id"
        return "user-id_saved.jpg"

    monkeypatch.setattr(
        storage,
        "storage_key_for",
        scoped_storage_key,
    )

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


@pytest.mark.asyncio
async def test_upload_chat_image_multipart_normalizes_and_stores(
    monkeypatch, tmp_path
):
    from starlette.datastructures import Headers, UploadFile

    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        chat_media.repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )

    def fake_fire_background(coro):
        if inspect.iscoroutine(coro):
            coro.close()

    monkeypatch.setattr(chat_media, "fire_background", fake_fire_background)
    created = _attachment(id="att-mp")
    create_attachment = AsyncMock(return_value=created)
    monkeypatch.setattr(chat_media.repo, "create_attachment", create_attachment)

    upload = UploadFile(
        file=io.BytesIO(_image_bytes(4000, 3000)),
        filename="big-photo.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )
    response = await chat_media.upload_chat_image_multipart(
        file=upload,
        conversation_id="conv-id",
        name=None,
        user={"sub": "user-id", "role": "user"},
    )

    assert response.id == "att-mp"
    kwargs = create_attachment.await_args.kwargs
    assert kwargs["name"] == "big-photo.jpg"
    assert kwargs["mime"] == "image/jpeg"
    assert max(kwargs["width"], kwargs["height"]) <= storage._INGEST_MAX_EDGE
    stored = tmp_path / kwargs["storage_key"]
    assert stored.exists()
    assert kwargs["size"] == stored.stat().st_size
    assert (tmp_path / storage.thumb_storage_key(kwargs["storage_key"])).exists()


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


def test_audio_attachment_metadata_keeps_transcript():
    from app.services.chat_media.prompt import attachment_to_metadata

    metadata = attachment_to_metadata(
        _attachment(
            kind="audio",
            mime="audio/mp4",
            storage_key="user-id_voice.m4a",
            url="/chat/media/user-id_voice.m4a",
            duration_seconds=8,
            transcription_status="ready",
            transcription_text="明天下午三点提醒我开会",
            transcription_model="fun-asr-test",
            transcription_request_id="req-audio",
            vision_status="skipped",
        )
    )

    assert metadata["kind"] == "audio"
    assert metadata["duration_seconds"] == 8
    assert metadata["transcription_text"] == "明天下午三点提醒我开会"
    assert metadata["transcription_request_id"] == "req-audio"


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
