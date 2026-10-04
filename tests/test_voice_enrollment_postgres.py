"""Clone persistence through Prisma and PostgreSQL, using temporary tables only.

Set TTS_TEST_DATABASE_URL to a disposable local PostgreSQL instance to run.
"""
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

from fastapi import FastAPI
import httpx
from prisma import Prisma
import pytest

from app.api.admin import tts
from app.services.speech_output.voice_enrollment import EnrollmentResult


@pytest.fixture
async def postgres_enrollment(monkeypatch):
    url = os.getenv("TTS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TTS_TEST_DATABASE_URL must point to a disposable local PostgreSQL")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    try:
        async with database.tx(timeout=timedelta(seconds=30)) as transaction:
            # Use the deployed column definitions, without creating permanent tables.
            migration = (
                Path(__file__).resolve().parents[1]
                / "prisma/migrations/20260731153000_qwen_audio_tts_customization/migration.sql"
            ).read_text()
            start = migration.index('CREATE TABLE IF NOT EXISTS "tts_voice_profiles" (')
            end = migration.index("\n);", start) + len("\n);")
            table_sql = migration[start:end].replace(
                'CREATE TABLE IF NOT EXISTS', 'CREATE TEMP TABLE', 1,
            ).removesuffix(";") + " ON COMMIT DROP"
            await transaction.execute_raw(table_sql)
            await transaction.execute_raw(
                "CREATE TEMP TABLE ai_agents (tts_voice_id TEXT) ON COMMIT DROP",
            )
            # Consent times must remain UTC even when the session uses another zone.
            await transaction.execute_raw("SET LOCAL TIME ZONE 'Asia/Shanghai'")
            cleanups = []

            def background(coro):
                cleanups.append(coro.cr_code.co_name)
                coro.close()

            create = AsyncMock(return_value=EnrollmentResult("cloned-test", "request-test"))
            delete = AsyncMock()
            monkeypatch.setattr(tts, "db", transaction)
            monkeypatch.setattr(tts, "save_enrollment_audio", AsyncMock(
                return_value=("tts_enroll_sample.wav", "audio/wav"),
            ))
            monkeypatch.setattr(tts, "signed_enrollment_url", lambda **_: "https://example.com/sample.wav")
            monkeypatch.setattr(tts, "create_cloned_voice", create)
            monkeypatch.setattr(tts, "delete_cloned_voice", delete)
            monkeypatch.setattr(tts, "fire_background", background)
            app = FastAPI()
            app.include_router(tts.router)
            app.dependency_overrides[tts.require_admin_jwt] = lambda: {"sub": "admin-test"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                yield SimpleNamespace(
                    client=client, db=transaction, create=create, delete=delete,
                    cleanups=cleanups,
                )
    finally:
        await database.disconnect()


async def upload(flow, gender="male"):
    return await flow.client.post(
        "/admin-api/tts/voices/clone",
        data={
            "display_name": "日常聊天音色", "gender": gender,
            "prefix": "casual", "consent_confirmed": "true",
        },
        files={"file": ("sample.m4a", b"recorded-audio", "audio/mp4")},
    )


@pytest.mark.parametrize("gender,request_id", [("male", "request-test"), ("female", None)])
async def test_clone_persists_consent_timestamp_and_appears_in_library(
    postgres_enrollment, gender, request_id,
):
    flow = postgres_enrollment
    flow.create.return_value = EnrollmentResult("cloned-test", request_id)
    before = datetime.now(UTC).replace(tzinfo=None)
    response = await upload(flow, gender)
    after = datetime.now(UTC).replace(tzinfo=None)

    assert response.status_code == 200, response.text
    row = response.json()
    assert row["display_name"] == "日常聊天音色"
    assert row["gender"] == gender
    assert row["provider_request_id"] == request_id
    assert row["consent_confirmed_by"] == "admin-test"
    confirmed_at = datetime.fromisoformat(row["consent_confirmed_at"].replace("Z", "+00:00"))
    assert before - timedelta(milliseconds=1) <= confirmed_at.replace(tzinfo=None) <= after
    stored = await flow.db.query_raw(
        "SELECT *, pg_typeof(consent_confirmed_at)::text AS consent_type FROM tts_voice_profiles",
    )
    assert len(stored) == 1
    assert stored[0]["consent_type"] == "timestamp without time zone"
    assert stored[0]["id"] == row["id"]
    library = await flow.client.get("/admin-api/tts/voices")
    assert library.status_code == 200
    assert library.json()["voices"][0]["voice_id"] == "cloned-test"
    assert library.json()["voices"][0]["agent_count"] == 0
    flow.delete.assert_not_awaited()
    assert flow.cleanups == ["delete_enrollment_audio_later"]


async def test_database_rejection_compensates_provider_voice_and_returns_clear_error(
    postgres_enrollment,
):
    flow = postgres_enrollment
    await flow.db.execute_raw(
        "ALTER TABLE tts_voice_profiles ADD CHECK (consent_confirmed_by <> 'admin-test')",
    )
    response = await upload(flow)

    assert response.status_code == 503
    assert response.json()["detail"] == "音色保存失败，本次创建未完成，请稍后重试"
    flow.delete.assert_awaited_once_with("cloned-test")
    assert flow.cleanups == ["delete_enrollment_audio_later"]
