"""Achievement runtime mode behavior: on (default) / silent (H5) / off.

Mode resolution order (app/services/achievements/mode.py):
SystemConfig.achievement_mode override (admin console) -> .env default.

Silent mode is the H5 chat-only launch mode: evaluation and unlock rows keep
persisting in real time (accurate unlocked_at + conversation_id), while every
user-facing surface stays suppressed. Switching back to "on" must surface the
accumulated results without any backfill and without retroactive notifications.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.admin import achievement_settings as admin_achievement_settings
from app.api.public import achievements as achievements_api
from app.api.public import conversations
from app.config import settings
from app.services import wallet as wallet_service
from app.services.achievements import engine, mode as mode_module, repository
from jobs import scheduler as scheduler_module


@pytest.fixture(autouse=True)
def _clean_mode_cache():
    """The mode override cache is process-global; never leak across tests."""
    mode_module.reset_achievement_mode_cache()
    yield
    mode_module.reset_achievement_mode_cache()


def _db_with_override(value) -> MagicMock:
    fake_db = MagicMock()
    row = SimpleNamespace(achievementMode=value) if value is not ... else None
    fake_db.systemconfig.find_unique = AsyncMock(return_value=row)
    fake_db.systemconfig.upsert = AsyncMock()
    return fake_db


# ── Mode semantics (pure helpers) ──────────────────────────────────────


def test_mode_capability_matrix():
    assert mode_module.evaluation_enabled_for("on") is True
    assert mode_module.evaluation_enabled_for("silent") is True
    assert mode_module.evaluation_enabled_for("off") is False
    assert mode_module.user_facing_enabled_for("on") is True
    assert mode_module.user_facing_enabled_for("silent") is False
    assert mode_module.user_facing_enabled_for("off") is False


def test_default_env_mode_is_on():
    assert settings.achievement_mode == "on"


# ── Resolution: DB override -> env fallback ────────────────────────────


@pytest.mark.asyncio
async def test_db_override_wins_over_env_default(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "on")
    with patch.object(mode_module, "db", _db_with_override("silent")):
        assert await mode_module.get_achievement_mode() == "silent"
        assert await mode_module.achievement_evaluation_enabled() is True
        assert await mode_module.achievement_user_facing_enabled() is False


@pytest.mark.asyncio
async def test_null_override_inherits_env_default(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "silent")
    with patch.object(mode_module, "db", _db_with_override(None)):
        assert await mode_module.get_achievement_mode() == "silent"


@pytest.mark.asyncio
async def test_invalid_db_value_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "on")
    with patch.object(mode_module, "db", _db_with_override("banana")):
        assert await mode_module.get_achievement_mode() == "on"


@pytest.mark.asyncio
async def test_db_error_falls_back_to_env_without_caching(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "silent")
    fake_db = MagicMock()
    fake_db.systemconfig.find_unique = AsyncMock(side_effect=RuntimeError("db down"))
    with patch.object(mode_module, "db", fake_db):
        assert await mode_module.get_achievement_mode() == "silent"
    # Error path must not populate the cache (recovery picked up immediately).
    assert mode_module._override_cache is None


@pytest.mark.asyncio
async def test_override_reads_are_cached_within_ttl():
    fake_db = _db_with_override("off")
    with patch.object(mode_module, "db", fake_db):
        assert await mode_module.get_achievement_mode() == "off"
        assert await mode_module.get_achievement_mode() == "off"
    fake_db.systemconfig.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_achievement_mode_persists_and_applies_immediately():
    fake_db = _db_with_override(None)
    with patch.object(mode_module, "db", fake_db):
        snapshot = await mode_module.set_achievement_mode("silent")
        # Same-process cache refreshed by the write, no extra read needed.
        assert await mode_module.get_achievement_mode() == "silent"

    upsert_data = fake_db.systemconfig.upsert.await_args.kwargs["data"]
    assert upsert_data["update"] == {"achievementMode": "silent"}
    assert snapshot["mode"] == "silent"
    assert snapshot["effective_mode"] == "silent"


@pytest.mark.asyncio
async def test_set_achievement_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid achievement mode"):
        await mode_module.set_achievement_mode("paused")


# ── Admin API (系统设置) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_get_returns_override_env_and_effective(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "on")
    with patch.object(mode_module, "db", _db_with_override("silent")):
        resp = await admin_achievement_settings.get_achievement_settings()

    assert resp.mode == "silent"
    assert resp.env_mode == "on"
    assert resp.effective_mode == "silent"


@pytest.mark.asyncio
async def test_admin_put_updates_mode(monkeypatch):
    monkeypatch.setattr(settings, "achievement_mode", "on")
    fake_db = _db_with_override(None)
    with patch.object(mode_module, "db", fake_db):
        resp = await admin_achievement_settings.update_achievement_settings(
            admin_achievement_settings.AchievementSettingsUpdateRequest(mode="off")
        )

    fake_db.systemconfig.upsert.assert_awaited_once()
    assert resp.mode == "off"
    assert resp.effective_mode == "off"


# ── Gate wiring: unlock persistence vs user-facing side effects ────────


@pytest.mark.asyncio
async def test_silent_mode_persists_unlock_without_notification_side_effects():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[{"id": "unlock-1", "unlocked_at": datetime(2026, 7, 21, tzinfo=UTC)}]
    )
    fake_db.execute_raw = AsyncMock()
    fake_manager = MagicMock()
    fake_manager.send_event = AsyncMock()
    fake_manager.send_to_workspace = AsyncMock()

    with (
        patch.object(
            repository, "achievement_evaluation_enabled", AsyncMock(return_value=True)
        ),
        patch.object(
            repository, "achievement_user_facing_enabled", AsyncMock(return_value=False)
        ),
        patch.object(repository, "_is_unlock_cached", AsyncMock(return_value=False)),
        patch.object(repository, "_cache_unlocked_achievements", AsyncMock()),
        patch.object(repository, "db", fake_db),
        patch.object(repository, "manager", fake_manager),
        patch(
            "app.services.runtime.tasks.fire_background",
            MagicMock(),
        ) as fire_background,
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=1,
        )

    # Unlock row persisted with real-time unlocked_at/conversation_id ...
    assert unlocked is True
    fake_db.query_raw.assert_awaited_once()
    # ... but zero user-facing side effects fired in silent mode.
    fire_background.assert_not_called()
    fake_manager.send_event.assert_not_awaited()
    fake_manager.send_to_workspace.assert_not_awaited()
    fake_db.execute_raw.assert_not_awaited()  # notified_at untouched


@pytest.mark.asyncio
async def test_on_mode_unlock_still_sends_ws_notification():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[{"id": "unlock-1", "unlocked_at": datetime(2026, 7, 21, tzinfo=UTC)}]
    )
    fake_db.execute_raw = AsyncMock()
    fake_manager = MagicMock()
    fake_manager.send_event = AsyncMock(return_value=True)

    with (
        patch.object(
            repository, "achievement_evaluation_enabled", AsyncMock(return_value=True)
        ),
        patch.object(
            repository, "achievement_user_facing_enabled", AsyncMock(return_value=True)
        ),
        patch.object(repository, "_is_unlock_cached", AsyncMock(return_value=False)),
        patch.object(repository, "_cache_unlocked_achievements", AsyncMock()),
        patch.object(repository, "db", fake_db),
        patch.object(repository, "manager", fake_manager),
        patch(
            "app.services.runtime.tasks.fire_background",
            MagicMock(side_effect=lambda coro: coro.close()),
        ),
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=1,
        )

    assert unlocked is True
    fake_manager.send_event.assert_awaited_once()
    fake_db.execute_raw.assert_awaited_once()  # notified_at stamped


@pytest.mark.asyncio
async def test_off_mode_engine_skips_all_rule_evaluation():
    with (
        patch.object(
            engine, "achievement_evaluation_enabled", AsyncMock(return_value=False)
        ),
        patch.object(engine, "evaluate_user_message", AsyncMock()) as evaluate,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            text="你好",
        )

    evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_silent_mode_engine_still_evaluates_rules():
    with (
        patch.object(
            engine, "achievement_evaluation_enabled", AsyncMock(return_value=True)
        ),
        patch.object(engine, "evaluate_user_message", AsyncMock()) as evaluate,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            text="你好",
        )

    evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_off_mode_unlock_is_rejected_before_any_io():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock()

    with (
        patch.object(
            repository, "achievement_evaluation_enabled", AsyncMock(return_value=False)
        ),
        patch.object(repository, "db", fake_db),
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            achievement_id=1,
        )

    assert unlocked is False
    fake_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_off_mode_engine_gate_resolves_from_db_override(monkeypatch):
    """End-to-end: admin sets off in DB, env stays on, engine really stops."""
    monkeypatch.setattr(settings, "achievement_mode", "on")
    with (
        patch.object(mode_module, "db", _db_with_override("off")),
        patch.object(engine, "evaluate_user_message", AsyncMock()) as evaluate,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            text="你好",
        )

    evaluate.assert_not_awaited()


# ── Wallet point sync ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_silent_mode_wallet_sync_credits_nothing():
    ensure = AsyncMock(
        return_value={
            "ticket_balance": 0,
            "point_balance": 0,
            "achievement_points_synced": 0,
        }
    )

    with (
        patch.object(
            wallet_service,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=False),
        ),
        patch.object(wallet_service, "ensure_wallet", ensure),
        patch.object(wallet_service, "list_achievements", AsyncMock()) as listing,
    ):
        balance = await wallet_service.sync_achievement_points("u1", "a1")

    assert balance["point_balance"] == 0
    listing.assert_not_awaited()
    ensure.assert_awaited_once()


@pytest.mark.asyncio
async def test_switching_back_to_on_credits_full_silent_window_delta():
    """First sync after silent -> on credits the whole accumulated score."""

    class _Tx:
        def __init__(self):
            self.point_balance = 0
            self.synced = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query_raw(self, query: str, *args):
            if "FOR UPDATE" in query:
                return [{
                    "ticket_balance": 0,
                    "point_balance": self.point_balance,
                    "achievement_points_synced": self.synced,
                }]
            if "COALESCE(SUM(delta)" in query:
                return [{"synced": self.synced}]
            if "UPDATE user_wallets" in query:
                delta = int(args[1])
                self.point_balance += delta
                self.synced += delta
                return [{
                    "point_balance": self.point_balance,
                    "achievement_points_synced": self.synced,
                }]
            raise AssertionError(f"Unexpected query: {query}")

        async def execute_raw(self, query: str, *args):
            return 1

    tx = _Tx()
    fake_db = MagicMock()
    fake_db.tx = lambda: tx

    with (
        patch.object(
            wallet_service,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=True),
        ),
        patch.object(wallet_service, "db", fake_db),
        patch.object(
            wallet_service,
            "ensure_wallet",
            AsyncMock(return_value={
                "ticket_balance": 0,
                "point_balance": 0,
                "achievement_points_synced": 0,
            }),
        ),
        patch.object(
            wallet_service,
            "list_achievements",
            AsyncMock(return_value={"score": 730}),
        ),
    ):
        balance = await wallet_service.sync_achievement_points("u1", "a1")

    assert balance["point_balance"] == 730
    assert balance["achievement_points_synced"] == 730


# ── Public API surfaces ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_silent_mode_achievements_api_returns_hidden_payload():
    fake_db = MagicMock()
    fake_db.aiagent.find_unique = AsyncMock(
        return_value=SimpleNamespace(userId="u1")
    )

    with (
        patch.object(
            achievements_api,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=False),
        ),
        patch.object(achievements_api, "db", fake_db),
        patch.object(
            achievements_api, "list_achievements", AsyncMock()
        ) as listing,
    ):
        result = await achievements_api.get_achievements(
            agent_id="a1",
            payload={"sub": "u1"},
        )

    listing.assert_not_awaited()
    assert result["enabled"] is False
    assert result["unlocked"] == 0
    assert result["score"] == 0
    assert result["items"] == []
    assert result["total"] == 97
    assert result["active_total"] == 82
    assert result["disabled_total"] == 15


@pytest.mark.asyncio
async def test_on_mode_achievements_api_returns_real_listing():
    fake_db = MagicMock()
    fake_db.aiagent.find_unique = AsyncMock(
        return_value=SimpleNamespace(userId="u1")
    )
    payload = {"total": 97, "unlocked": 3, "score": 60, "items": []}

    with (
        patch.object(
            achievements_api,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=True),
        ),
        patch.object(achievements_api, "db", fake_db),
        patch.object(
            achievements_api,
            "list_achievements",
            AsyncMock(return_value=payload),
        ) as listing,
    ):
        result = await achievements_api.get_achievements(
            agent_id="a1",
            payload={"sub": "u1"},
        )

    listing.assert_awaited_once_with(user_id="u1", agent_id="a1")
    assert result == payload


@pytest.mark.asyncio
async def test_silent_mode_timeline_skips_achievement_synthesis():
    fake_db = MagicMock()
    fake_db.message.find_many = AsyncMock(return_value=[])

    with (
        patch.object(
            conversations,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=False),
        ),
        patch.object(conversations, "db", fake_db),
        patch.object(
            conversations,
            "_achievement_timeline_items",
            AsyncMock(),
        ) as synthesize,
    ):
        items = await conversations.list_messages(
            conversation_id="c1",
            limit=50,
            offset=0,
            include_metadata=True,
            include_achievements=True,
            include_usage=False,
            conv=SimpleNamespace(userId="u1", agentId="a1"),
            user={"sub": "u1", "role": "user"},
        )

    synthesize.assert_not_awaited()
    assert items == []


@pytest.mark.asyncio
async def test_on_mode_timeline_still_synthesizes_achievements():
    fake_db = MagicMock()
    fake_db.message.find_many = AsyncMock(return_value=[])

    with (
        patch.object(
            conversations,
            "achievement_user_facing_enabled",
            AsyncMock(return_value=True),
        ),
        patch.object(conversations, "db", fake_db),
        patch.object(
            conversations,
            "_achievement_timeline_items",
            AsyncMock(return_value=[]),
        ) as synthesize,
    ):
        await conversations.list_messages(
            conversation_id="c1",
            limit=50,
            offset=0,
            include_metadata=True,
            include_achievements=True,
            include_usage=False,
            conv=SimpleNamespace(userId="u1", agentId="a1"),
            user={"sub": "u1", "role": "user"},
        )

    synthesize.assert_awaited_once()


# ── Daily rollup scheduler ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_mode_rollup_skips_and_freezes_checkpoint():
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.set = AsyncMock()

    async def _run_distributed(_name, _ttl, callback):
        await callback()

    with (
        patch(
            "app.services.achievements.mode.achievement_evaluation_enabled",
            AsyncMock(return_value=False),
        ),
        patch.object(
            scheduler_module,
            "_run_distributed_job",
            AsyncMock(side_effect=_run_distributed),
        ),
        patch.object(
            scheduler_module,
            "get_redis",
            AsyncMock(return_value=fake_redis),
        ),
        patch(
            "app.services.achievements.service.run_daily_rollup",
            AsyncMock(),
        ) as run_rollup,
    ):
        await scheduler_module._run_achievement_daily_rollup()

    run_rollup.assert_not_awaited()
    fake_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_silent_mode_rollup_still_runs_and_advances_checkpoint():
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.set = AsyncMock()

    async def _run_distributed(_name, _ttl, callback):
        await callback()

    with (
        patch(
            "app.services.achievements.mode.achievement_evaluation_enabled",
            AsyncMock(return_value=True),
        ),
        patch.object(
            scheduler_module,
            "_run_distributed_job",
            AsyncMock(side_effect=_run_distributed),
        ),
        patch.object(
            scheduler_module,
            "get_redis",
            AsyncMock(return_value=fake_redis),
        ),
        patch(
            "app.services.achievements.service.run_daily_rollup",
            AsyncMock(),
        ) as run_rollup,
    ):
        await scheduler_module._run_achievement_daily_rollup()

    run_rollup.assert_awaited_once()
    fake_redis.set.assert_awaited_once()
