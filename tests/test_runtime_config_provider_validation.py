import pytest
from fastapi import HTTPException

from app.api.admin import runtime_config as rc


@pytest.mark.asyncio
async def test_remote_model_must_match_fallback_provider_when_provider_is_null(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return (provider, identifier) == ("deepseek", "deepseek-v4-pro")

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    payload = rc.ConfigPayload(remote_provider=None, remote_chat_model="deepseek-v4-pro")
    with pytest.raises(HTTPException) as exc:
        await rc._validate_payload_models(payload, fallback_remote_provider="dashscope")

    assert exc.value.status_code == 400
    assert "remote_chat_model" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_remote_model_accepts_explicit_matching_provider(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return provider == "deepseek" and identifier in {"deepseek-v4-pro", "deepseek-v4-flash"}

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    payload = rc.ConfigPayload(
        remote_provider="deepseek",
        remote_chat_model="deepseek-v4-pro",
        remote_small_model="deepseek-v4-flash",
    )
    await rc._validate_payload_models(payload, fallback_remote_provider="dashscope")


@pytest.mark.asyncio
async def test_same_identifier_can_exist_under_multiple_remote_providers(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return identifier == "deepseek-v4-pro" and provider in {"dashscope", "deepseek"}

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    await rc._validate_payload_models(
        rc.ConfigPayload(remote_provider="deepseek", remote_chat_model="deepseek-v4-pro"),
        fallback_remote_provider="dashscope",
    )
    await rc._validate_payload_models(
        rc.ConfigPayload(remote_provider="dashscope", remote_chat_model="deepseek-v4-pro"),
        fallback_remote_provider="dashscope",
    )
