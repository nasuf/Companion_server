"""Tests for per-game client visibility (admin toggle + public catalog).

Visibility lives on native_game_configs.enabled and is exposed to the client
through GET /games/native/catalog. Toggling it must NOT create a balance config
version (it is an operational on/off, not a difficulty change).
"""

from __future__ import annotations

import pytest

from app.services.games import balance


def test_config_payload_includes_enabled_default_true():
    payload = balance.config_payload(balance._default_config("gomoku"))
    assert payload["enabled"] is True


@pytest.mark.asyncio
async def test_list_public_catalog_defaults_missing_rows_to_enabled(monkeypatch):
    class FakeDb:
        async def query_raw(self, query, *args):
            # Only 'go' has a stored row (disabled); every other game has no row
            # and must default to enabled so a fresh install shows everything.
            return [{"game_key": "go", "enabled": False}]

    monkeypatch.setattr(balance, "db", FakeDb())

    catalog = await balance.list_public_catalog()
    by_key = {row["game_key"]: row for row in catalog}

    assert set(by_key) == set(balance.GAME_TITLES)
    assert by_key["go"]["enabled"] is False
    assert by_key["go"]["title"] == "围棋"
    assert by_key["gomoku"]["enabled"] is True


@pytest.mark.asyncio
async def test_set_enabled_upserts_flag_and_returns_payload(monkeypatch):
    calls: dict[str, tuple] = {}

    class FakeDb:
        async def execute_raw(self, query, *args):
            calls["execute"] = (query, args)
            return 1

        async def query_raw(self, query, *args):
            if "FROM game_sessions" in query:
                return []
            if "FROM native_game_configs" in query:
                return [{"game_key": "gomoku", "enabled": False, "version": 3}]
            return []

    monkeypatch.setattr(balance, "db", FakeDb())

    payload = await balance.set_enabled("gomoku", False)

    # Upsert binds exactly (game_key, enabled); no version column is touched.
    assert calls["execute"][1] == ("gomoku", False)
    assert "version" not in calls["execute"][0]
    assert payload["game_key"] == "gomoku"
    assert payload["enabled"] is False


@pytest.mark.asyncio
async def test_set_enabled_rejects_unknown_game():
    with pytest.raises(ValueError, match="unsupported_game"):
        await balance.set_enabled("not_a_real_game", False)
