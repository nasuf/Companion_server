from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.admin import offline_settings as admin_offline
from app.services.offline import module_settings


def _config(activity: bool, gift: bool) -> SimpleNamespace:
    return SimpleNamespace(
        offlineActivityEnabled=activity, offlineGiftEnabled=gift
    )


@pytest.mark.asyncio
async def test_get_flags_reads_singleton_row(monkeypatch):
    monkeypatch.setattr(
        module_settings,
        "db",
        SimpleNamespace(
            systemconfig=SimpleNamespace(
                find_unique=AsyncMock(return_value=_config(True, False))
            )
        ),
    )
    flags = await module_settings.get_offline_module_flags()
    assert flags == {"activity_enabled": True, "gift_enabled": False}


@pytest.mark.asyncio
async def test_get_flags_defaults_false_when_row_missing(monkeypatch):
    monkeypatch.setattr(
        module_settings,
        "db",
        SimpleNamespace(
            systemconfig=SimpleNamespace(find_unique=AsyncMock(return_value=None))
        ),
    )
    flags = await module_settings.get_offline_module_flags()
    assert flags == {"activity_enabled": False, "gift_enabled": False}


@pytest.mark.asyncio
async def test_set_flags_upserts_only_provided_fields(monkeypatch):
    upsert = AsyncMock()
    find_unique = AsyncMock(return_value=_config(True, False))
    monkeypatch.setattr(
        module_settings,
        "db",
        SimpleNamespace(
            systemconfig=SimpleNamespace(upsert=upsert, find_unique=find_unique)
        ),
    )
    result = await module_settings.set_offline_module_flags(activity_enabled=True)

    data = upsert.await_args.kwargs["data"]
    assert data["update"] == {"offlineActivityEnabled": True}
    assert data["create"] == {"id": 1, "offlineActivityEnabled": True}
    assert result == {"activity_enabled": True, "gift_enabled": False}


@pytest.mark.asyncio
async def test_set_flags_noop_when_nothing_provided(monkeypatch):
    upsert = AsyncMock()
    find_unique = AsyncMock(return_value=_config(False, False))
    monkeypatch.setattr(
        module_settings,
        "db",
        SimpleNamespace(
            systemconfig=SimpleNamespace(upsert=upsert, find_unique=find_unique)
        ),
    )
    result = await module_settings.set_offline_module_flags()

    upsert.assert_not_awaited()
    assert result == {"activity_enabled": False, "gift_enabled": False}


@pytest.mark.asyncio
async def test_admin_get_returns_flags(monkeypatch):
    monkeypatch.setattr(
        admin_offline,
        "get_offline_module_flags",
        AsyncMock(return_value={"activity_enabled": True, "gift_enabled": True}),
    )
    resp = await admin_offline.get_offline_settings()
    assert resp.activity_enabled is True
    assert resp.gift_enabled is True


@pytest.mark.asyncio
async def test_admin_put_updates_flags(monkeypatch):
    set_flags = AsyncMock(
        return_value={"activity_enabled": False, "gift_enabled": True}
    )
    monkeypatch.setattr(admin_offline, "set_offline_module_flags", set_flags)
    resp = await admin_offline.update_offline_settings(
        admin_offline.OfflineSettingsUpdateRequest(gift_enabled=True)
    )
    assert set_flags.await_args.kwargs == {
        "activity_enabled": None,
        "gift_enabled": True,
    }
    assert resp.activity_enabled is False
    assert resp.gift_enabled is True
