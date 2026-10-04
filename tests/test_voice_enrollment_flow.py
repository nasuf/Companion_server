from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin import tts


@pytest.fixture
def enrollment_flow(monkeypatch):
    profile = {"id": "profile-new", "voice_id": "cloned-new", "source": "cloned"}
    save = AsyncMock(return_value=("tts_enroll_sample.wav", "audio/wav"))
    create = AsyncMock(
        return_value=SimpleNamespace(voice_id="cloned-new", request_id="request-1"),
    )
    delete = AsyncMock()
    database = SimpleNamespace(query_raw=AsyncMock(return_value=[profile]))
    cleanups = []

    def background(coro):
        cleanups.append(coro.cr_code.co_name)
        coro.close()

    monkeypatch.setattr(tts, "save_enrollment_audio", save)
    monkeypatch.setattr(
        tts, "signed_enrollment_url", lambda **kwargs: "https://example.com/sample.wav",
    )
    monkeypatch.setattr(tts, "create_cloned_voice", create)
    monkeypatch.setattr(tts, "delete_cloned_voice", delete)
    monkeypatch.setattr(tts, "db", database)
    monkeypatch.setattr(tts, "fire_background", background)

    app = FastAPI()
    app.include_router(tts.router)
    app.dependency_overrides[tts.require_admin_jwt] = lambda: {"sub": "admin-1"}
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(
            client=client, save=save, create=create, delete=delete,
            database=database, cleanups=cleanups, profile=profile,
        )


def upload(flow, *, consent="true"):
    return flow.client.post(
        "/admin-api/tts/voices/clone",
        data={
            "display_name": "聊天音色", "gender": "female", "prefix": "casual",
            "consent_confirmed": consent,
        },
        files={"file": ("sample.m4a", b"recorded-audio", "audio/mp4")},
    )


def test_multipart_enrollment_persists_voice_and_schedules_sample_cleanup(enrollment_flow):
    flow = enrollment_flow
    response = upload(flow)

    assert response.status_code == 200
    assert response.json() == flow.profile
    flow.save.assert_awaited_once_with(
        blob=b"recorded-audio", mime="audio/mp4", filename="sample.m4a",
    )
    flow.create.assert_awaited_once_with(
        prefix="casual", audio_url="https://example.com/sample.wav",
    )
    args = flow.database.query_raw.await_args.args
    assert args[2:7] == (
        "聊天音色", "qwen-audio-3.0-tts-plus", "cloned-new", "female", "request-1",
    )
    assert args[7] == "admin-1"
    flow.delete.assert_not_awaited()
    assert flow.cleanups == ["delete_enrollment_audio_later"]


def test_enrollment_rejects_missing_consent_before_processing_audio(enrollment_flow):
    flow = enrollment_flow

    assert upload(flow, consent="false").status_code == 422
    flow.save.assert_not_awaited()
    flow.create.assert_not_awaited()


def test_enrollment_rejects_insufficient_speech_before_provider_call(enrollment_flow):
    flow = enrollment_flow
    flow.save.side_effect = ValueError("录音需要至少 5 秒有效人声")

    response = upload(flow)

    assert response.status_code == 422
    assert "5 秒" in response.json()["detail"]
    flow.create.assert_not_awaited()
    flow.database.query_raw.assert_not_awaited()


def test_enrollment_provider_failure_does_not_save_voice(enrollment_flow):
    flow = enrollment_flow
    flow.create.side_effect = tts.SpeechSynthesisError("provider unavailable")

    assert upload(flow).status_code == 502
    flow.database.query_raw.assert_not_awaited()
    assert flow.cleanups == ["delete_enrollment_audio_later"]


def test_enrollment_database_failure_deletes_provider_voice(enrollment_flow):
    flow = enrollment_flow
    flow.database.query_raw.side_effect = RuntimeError("database unavailable")

    response = upload(flow)
    assert response.status_code == 503
    assert response.json()["detail"] == "音色保存失败，本次创建未完成，请稍后重试"
    flow.delete.assert_awaited_once_with("cloned-new")
    assert flow.cleanups == ["delete_enrollment_audio_later"]


def test_enrollment_cleanup_failure_is_logged_without_hiding_save_error(
    enrollment_flow, caplog,
):
    flow = enrollment_flow
    flow.database.query_raw.side_effect = RuntimeError("database unavailable")
    flow.delete.side_effect = RuntimeError("provider unavailable")

    response = upload(flow)

    assert response.status_code == 503
    assert response.json()["detail"] == "音色保存失败，本次创建未完成，请稍后重试"
    assert "provider cleanup failed voice_id=cloned-new" in caplog.text
    assert flow.cleanups == ["delete_enrollment_audio_later"]
