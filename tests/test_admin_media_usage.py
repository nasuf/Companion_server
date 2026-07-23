"""Tests for GET /admin-api/stats/media-usage (voice/image usage stats).

The endpoint issues three raw SQL aggregations over chat_message_attachments
(totals by kind / per-user rollup / distinct-user count) via asyncio.gather —
the AsyncMock side_effect list maps to that call order.
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
    fake_db = _fake_db([totals_rows, user_rows, user_total_rows])
    monkeypatch.setattr(stats_mod, "db", fake_db)

    resp = await stats_mod.media_usage(days=7, limit=200)

    assert resp["voice"] == {"count": 4, "total_seconds": 62, "total_bytes": 145_408}
    assert resp["image"] == {"count": 6, "total_bytes": 25_165_824}
    assert resp["user_total"] == 2
    assert len(resp["users"]) == 2
    assert resp["users"][0] == {
        "user_id": "u1", "username": "alice",
        "voice_count": 3, "voice_seconds": 50, "voice_bytes": 100_000,
        "image_count": 2, "image_bytes": 20_000_000,
    }
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

    resp = await stats_mod.media_usage(days=0, limit=50)

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

    await stats_mod.media_usage(days=3, limit=10)

    # All three queries share the same single timestamp bind param.
    assert len(captured_params) == 3
    assert all(len(params) == 1 for params in captured_params)


@pytest.mark.asyncio
async def test_media_usage_limit_is_inlined_as_integer(monkeypatch):
    captured_sql: list[str] = []

    async def fake_query_raw(sql, *params):
        captured_sql.append(sql)
        if "user_total" in sql:
            return [{"user_total": 0}]
        return []

    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw),
    )

    await stats_mod.media_usage(days=0, limit=25)

    user_sql = next(sql for sql in captured_sql if "GROUP BY a.user_id" in sql)
    assert "LIMIT 25" in user_sql
