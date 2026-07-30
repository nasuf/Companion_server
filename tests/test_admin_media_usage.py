"""Tests for GET /admin-api/stats/media-usage (voice/image usage stats).

The endpoint issues nine raw SQL aggregations via asyncio.gather — three over
chat_message_attachments (totals by kind / per-user rollup / distinct-user
count), two over speech_usage filtered to display_mode='text' (voice-to-text
totals / per-user rollup), and one unfiltered speech_usage rollup that is the
ASR billing basis, plus three TTS output rollups — the AsyncMock side_effect
list maps to that call order.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.admin import stats as stats_mod


def _fake_db(side_effect):
    return SimpleNamespace(query_raw=AsyncMock(side_effect=side_effect))


@pytest.mark.asyncio
async def test_media_usage_aggregates_totals_and_users(monkeypatch):
    totals_rows = [
        {"kind": "audio", "count": 4, "total_bytes": 145_408, "total_seconds": 62},
        {"kind": "image", "count": 6, "total_bytes": 25_165_824, "total_seconds": 0},
    ]
    user_rows = [
        {
            "user_id": "u1", "username": "alice",
            "voice_count": 3, "voice_seconds": 50, "voice_bytes": 100_000,
            "image_count": 2, "image_bytes": 20_000_000,
        },
        {
            "user_id": "u2", "username": "bob",
            "voice_count": 1, "voice_seconds": 12, "voice_bytes": 45_408,
            "image_count": 0, "image_bytes": 0,
        },
    ]
    user_total_rows = [{"user_total": 2}]
    # speech_usage voice-to-text aggregations (global + per-user).
    text_totals_rows = [{"count": 5, "total_seconds": 40}]
    text_user_rows = [
        {"user_id": "u1", "voice_text_count": 5, "voice_text_seconds": 40},
    ]
    # Billing basis: every transcription, regardless of display mode.
    asr_rows = [{"count": 9, "total_seconds": 102}]
    tts_totals_rows = [{
        "count": 3,
        "total_milliseconds": 12_500,
        "total_bytes": 180_000,
        "billable_characters": 420,
        "cost_cny": 0.0336,
    }]
    tts_user_rows = [{
        "user_id": "u1",
        "username": "alice",
        "tts_count": 3,
        "tts_milliseconds": 12_500,
        "tts_billable_characters": 420,
        "tts_cost_cny": 0.0336,
    }]
    tts_user_total_rows = [{"user_total": 1}]
    fake_db = _fake_db(
        [
            totals_rows, user_rows, user_total_rows,
            text_totals_rows, text_user_rows, asr_rows,
            tts_totals_rows, tts_user_rows, tts_user_total_rows,
        ]
    )
    monkeypatch.setattr(stats_mod, "db", fake_db)
    monkeypatch.setattr(stats_mod.settings, "asr_price_cny_per_second", 0.001)

    resp = await stats_mod.media_usage(days=7, limit=200, offset=0)

    assert resp["voice"] == {"count": 4, "total_seconds": 62, "total_bytes": 145_408}
    assert resp["voice_text"] == {"count": 5, "total_seconds": 40}
    assert resp["asr"] == {
        "count": 9, "total_seconds": 102,
        "price_cny_per_second": 0.001, "cost_cny": pytest.approx(0.102),
    }
    assert resp["image"] == {"count": 6, "total_bytes": 25_165_824}
    assert resp["tts_output"] == {
        "count": 3,
        "total_milliseconds": 12_500,
        "total_seconds": 12.5,
        "total_bytes": 180_000,
        "billable_characters": 420,
        "cost_cny": pytest.approx(0.0336),
    }
    assert resp["tts_user_total"] == 1
    assert resp["tts_users"][0]["tts_count"] == 3
    assert resp["user_total"] == 2
    assert len(resp["users"]) == 2
    # u1 has voice-to-text usage merged in; u2 has none → zeros.
    assert resp["users"][0] == {
        "user_id": "u1", "username": "alice",
        "voice_count": 3, "voice_seconds": 50, "voice_bytes": 100_000,
        "voice_text_count": 5, "voice_text_seconds": 40,
        "image_count": 2, "image_bytes": 20_000_000,
    }
    assert resp["users"][1]["voice_text_count"] == 0
    assert resp["users"][1]["voice_text_seconds"] == 0
    assert resp["window"]["days"] == 7
    assert resp["window"]["start"] is not None


@pytest.mark.asyncio
async def test_media_usage_days_zero_means_all_history(monkeypatch):
    captured_sql: list[str] = []
    captured_params: list[tuple] = []

    async def fake_query_raw(sql, *params):
        captured_sql.append(sql)
        captured_params.append(params)
        if "user_total" in sql:
            return [{"user_total": 0}]
        return []

    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw),
    )

    resp = await stats_mod.media_usage(days=0, limit=50, offset=0)

    # days=0 → no created_at clause and no bind params anywhere.
    assert all("created_at >=" not in sql for sql in captured_sql)
    assert all(params == () for params in captured_params)
    assert resp["window"]["start"] is None
    # Missing kind rows degrade to zeros, users list empty.
    assert resp["voice"] == {"count": 0, "total_seconds": 0, "total_bytes": 0}
    assert resp["image"] == {"count": 0, "total_bytes": 0}
    assert resp["users"] == []
    assert resp["user_total"] == 0


@pytest.mark.asyncio
async def test_media_usage_window_filter_binds_start_param(monkeypatch):
    captured_params: list[tuple] = []

    async def fake_query_raw(sql, *params):
        captured_params.append(params)
        if "user_total" in sql:
            return [{"user_total": 0}]
        return []

    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw),
    )

    await stats_mod.media_usage(days=3, limit=10, offset=0)

    # All nine queries (attachment + ASR + TTS rollups) share the same single
    # timestamp bind param.
    assert len(captured_params) == 9
    assert all(len(params) == 1 for params in captured_params)


@pytest.mark.asyncio
async def test_media_usage_limit_and_offset_are_inlined_as_integers(monkeypatch):
    captured_sql: list[str] = []

    async def fake_query_raw(sql, *params):
        captured_sql.append(sql)
        if "user_total" in sql:
            return [{"user_total": 0}]
        return []

    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw),
    )

    await stats_mod.media_usage(days=0, limit=25, offset=50)

    user_sql = next(sql for sql in captured_sql if "GROUP BY a.user_id" in sql)
    assert "LIMIT 25 OFFSET 50" in user_sql


@pytest.mark.asyncio
async def test_media_usage_first_page_offset_zero(monkeypatch):
    captured_sql: list[str] = []

    async def fake_query_raw(sql, *params):
        captured_sql.append(sql)
        if "user_total" in sql:
            return [{"user_total": 0}]
        return []

    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw),
    )

    await stats_mod.media_usage(days=0, limit=10, offset=0)

    user_sql = next(sql for sql in captured_sql if "GROUP BY a.user_id" in sql)
    assert "LIMIT 10 OFFSET 0" in user_sql
