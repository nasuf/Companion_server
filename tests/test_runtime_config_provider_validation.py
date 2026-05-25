import pytest
from fastapi import HTTPException

from app.api.admin import runtime_config as rc


@pytest.mark.asyncio
async def test_remote_model_must_match_fallback_provider_when_provider_is_null(monkeypatch):
    async def fake_model_provider(identifier: str) -> str | None:
        return {"deepseek-v4-pro": "deepseek"}.get(identifier)

    monkeypatch.setattr(rc, "_model_provider", fake_model_provider)

    payload = rc.ConfigPayload(remote_provider=None, remote_chat_model="deepseek-v4-pro")
    with pytest.raises(HTTPException) as exc:
        await rc._validate_payload_models(payload, fallback_remote_provider="dashscope")

    assert exc.value.status_code == 400
    assert "remote_chat_model" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_remote_model_accepts_explicit_matching_provider(monkeypatch):
    async def fake_model_provider(identifier: str) -> str | None:
        return {"deepseek-v4-pro": "deepseek", "deepseek-v4-flash": "deepseek"}.get(identifier)

    monkeypatch.setattr(rc, "_model_provider", fake_model_provider)

    payload = rc.ConfigPayload(
        remote_provider="deepseek",
        remote_chat_model="deepseek-v4-pro",
        remote_small_model="deepseek-v4-flash",
    )
    await rc._validate_payload_models(payload, fallback_remote_provider="dashscope")
