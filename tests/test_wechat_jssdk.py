from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import wechat_jssdk


def test_url_allowlist_strips_fragment(monkeypatch):
    monkeypatch.setattr(
        wechat_jssdk.settings,
        "wechat_jssdk_allowed_origins",
        "https://banshengcomp.com",
    )
    assert (
        wechat_jssdk.normalize_and_validate_url(
            "https://banshengcomp.com/merchant.html?from=menu#section"
        )
        == "https://banshengcomp.com/merchant.html?from=menu"
    )


def test_url_allowlist_rejects_foreign_origin(monkeypatch):
    monkeypatch.setattr(
        wechat_jssdk.settings,
        "wechat_jssdk_allowed_origins",
        "https://banshengcomp.com",
    )
    with pytest.raises(wechat_jssdk.WeChatJSSDKError):
        wechat_jssdk.normalize_and_validate_url("https://evil.example/merchant.html")


@pytest.mark.asyncio
async def test_build_config_signs_exact_url(monkeypatch):
    monkeypatch.setattr(
        wechat_jssdk.settings,
        "wechat_jssdk_allowed_origins",
        "https://banshengcomp.com",
    )
    monkeypatch.setattr(wechat_jssdk.settings, "wechat_login_enabled", True)
    monkeypatch.setattr(wechat_jssdk.settings, "wechat_h5_app_id", "wx-app")
    monkeypatch.setattr(wechat_jssdk.settings, "wechat_h5_app_secret", "secret")
    monkeypatch.setattr(
        wechat_jssdk, "_jsapi_ticket", AsyncMock(return_value="ticket-1")
    )
    monkeypatch.setattr(wechat_jssdk.time, "time", lambda: 1_700_000_000)
    monkeypatch.setattr(wechat_jssdk.secrets, "token_hex", lambda _: "nonce-1")

    result = await wechat_jssdk.build_config(
        "https://banshengcomp.com/merchant.html#ignored"
    )

    assert result == {
        "app_id": "wx-app",
        "timestamp": 1_700_000_000,
        "nonce_str": "nonce-1",
        "signature": "b3d94db3d96e1d15597dd6e5363199f25f834833",
        "url": "https://banshengcomp.com/merchant.html",
    }
