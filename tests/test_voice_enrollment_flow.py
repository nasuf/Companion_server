from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin import tts


@pytest.fixture
def enrollment_flow(monkeypatch):
    profile = {"id": "profile-new", "voice_id": "cloned-new", "source": "cloned"}
    save = AsyncMock(return_value=b"normalized-wav")
    create = AsyncMock(
        return_value=SimpleNamespace(voice_id="cloned-new", request_id="request-1"),
    )
    database = SimpleNamespace(query_raw=AsyncMock(return_value=[profile]))

    monkeypatch.setattr(tts, "prepare_enrollment_audio", save)
    monkeypatch.setattr(tts, "create_cloned_voice", create)
    monkeypatch.setattr(tts, "db", database)

    app = FastAPI()
    app.include_router(tts.router)
    app.dependency_overrides[tts.require_admin_jwt] = lambda: {"sub": "admin-1"}
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(
            client=client, save=save, create=create,
            database=database, profile=profile,
        )


def upload(flow, *, consent="true"):
    return flow.client.post(
        "/admin-api/tts/voices/clone",
        data={
            "display_name": "聊天音色", "gender": "female",
            "consent_confirmed": consent,
        },
        files={"file": ("sample.m4a", b"recorded-audio", "audio/mp4")},
    )


def test_multipart_enrollment_uploads_normalized_audio_and_persists_voice(enrollment_flow):
    flow = enrollment_flow
    response = upload(flow)

    assert response.status_code == 200
    assert response.json() == flow.profile
    flow.save.assert_awaited_once_with(
        blob=b"recorded-audio", mime="audio/mp4", filename="sample.m4a",
    )
    flow.create.assert_awaited_once_with(audio=b"normalized-wav")
    args = flow.database.query_raw.await_args.args
    assert args[2:8] == (
        "聊天音色", "stepfun", "stepaudio-2.5-tts", "cloned-new", "female", "request-1",
    )
    assert args[8] == "admin-1"


def test_enrollment_rejects_missing_consent_before_processing_audio(enrollment_flow):
    flow = enrollment_flow

    assert upload(flow, consent="false").status_code == 422
    flow.save.assert_not_awaited()
    flow.create.assert_not_awaited()


def test_enrollment_rejects_insufficient_speech_before_provider_call(enrollment_flow):
    flow = enrollment_flow
    flow.save.side_effect = ValueError("请录制 5–10 秒清晰人声")

    response = upload(flow)

    assert response.status_code == 422
    assert "5–10 秒" in response.json()["detail"]
    flow.create.assert_not_awaited()
    flow.database.query_raw.assert_not_awaited()


def test_enrollment_provider_failure_does_not_save_voice(enrollment_flow):
    flow = enrollment_flow
    flow.create.side_effect = tts.SpeechSynthesisError("provider unavailable")

    assert upload(flow).status_code == 502
    flow.database.query_raw.assert_not_awaited()


def test_enrollment_database_failure_logs_the_orphaned_provider_voice(
    enrollment_flow, caplog,
):
    """StepFun cannot delete voices: the clone is logged for manual registration."""
    flow = enrollment_flow
    flow.database.query_raw.side_effect = RuntimeError("database unavailable")

    response = upload(flow)

    assert response.status_code == 503
    assert "联系管理员登记" in response.json()["detail"]
    assert "orphaned provider voice_id=cloned-new" in caplog.text
