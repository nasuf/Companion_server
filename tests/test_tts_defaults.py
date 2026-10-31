"""Defaults must agree between delivery, the admin editor and stored agents."""
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

from prisma import Prisma
import pytest

from app.api.admin import tts
from app.services.speech_output import style, voices


@pytest.mark.parametrize("rate", [None, 0.9, 1.0, 1.25])
@pytest.mark.parametrize("instruction", [None, "轻松自然"])
async def test_editor_and_delivery_preserve_overrides(monkeypatch, rate, instruction):
    row = {"tts_voice_id": "cloned-voice", "tts_rate": rate,
           "tts_instruction": instruction}
    monkeypatch.setattr(voices, "db", SimpleNamespace(
        query_raw=AsyncMock(side_effect=[[row], [{"exists": 1}]]),
    ))
    monkeypatch.setattr(tts, "db", SimpleNamespace(
        query_raw=AsyncMock(side_effect=[[row], [{"id": "profile"}]]),
    ))
    actual = await voices.get_agent_tts_settings("agent")
    editor = await tts._agent_tts_response("agent")
    assert actual.rate == editor["rate"] == (rate if rate is not None else 0.9)
    assert actual.instruction == editor["instruction"] == instruction
    assert editor["default_instruction"] == style.resolve_style_instruction(None)
    assert style.resolve_style_instruction(actual.instruction) == (
        instruction or editor["default_instruction"]
    )


async def test_default_rate_migration_preserves_custom_parameters():
    url = os.getenv("TTS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TTS_TEST_DATABASE_URL must point to disposable local PostgreSQL")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    try:
        async with database.tx(timeout=timedelta(seconds=30)) as tx:
            await tx.execute_raw("""
                CREATE TEMP TABLE ai_agents (
                    id TEXT PRIMARY KEY, tts_voice_id TEXT DEFAULT 'cloned-voice',
                    tts_rate DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    tts_pitch DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    tts_volume INTEGER NOT NULL DEFAULT 50,
                    tts_seed INTEGER NOT NULL DEFAULT 0, tts_instruction TEXT,
                    tts_auto_emotion BOOLEAN NOT NULL DEFAULT TRUE,
                    tts_emotion_scale DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    updated_at TIMESTAMP DEFAULT '2026-01-01'
                ) ON COMMIT DROP
            """)
            await tx.execute_raw("INSERT INTO ai_agents (id) VALUES ('old-default')")
            await tx.execute_raw("INSERT INTO ai_agents (id,tts_instruction) VALUES ('blank-style','  ')")
            overrides = {
                "tts_rate": 1.25, "tts_pitch": 0.9, "tts_volume": 0,
                "tts_seed": 42, "tts_instruction": "轻松自然",
                "tts_auto_emotion": False, "tts_emotion_scale": 0.0,
            }
            for column, value in overrides.items():
                await tx.execute_raw(
                    f"INSERT INTO ai_agents (id,{column}) VALUES ($1,$2)", column, value,
                )
            before = await tx.query_raw("SELECT * FROM ai_agents WHERE id LIKE 'tts_%' ORDER BY id")
            migration = (Path(__file__).resolve().parents[1] /
                "prisma/migrations/20261031120000_relaxed_tts_defaults/migration.sql").read_text()
            for statement in migration.split(';'):
                if statement.strip():
                    await tx.execute_raw(statement)
            assert await tx.query_raw("SELECT * FROM ai_agents WHERE id LIKE 'tts_%' ORDER BY id") == before
            await tx.execute_raw("INSERT INTO ai_agents (id) VALUES ('new-default')")
            defaults = await tx.query_raw("SELECT tts_rate,tts_voice_id,tts_instruction FROM ai_agents WHERE id NOT LIKE 'tts_%'")
            assert len(defaults) == 3
            assert all(row['tts_rate'] == voices.DEFAULT_SPEECH_RATE == 0.9 for row in defaults)
            assert all(row['tts_voice_id'] == 'cloned-voice' for row in defaults)
            assert all(not (row['tts_instruction'] or '').strip() for row in defaults)
    finally:
        await database.disconnect()
