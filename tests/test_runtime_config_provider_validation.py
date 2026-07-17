import pytest
from fastapi import HTTPException

from app.api.admin import runtime_config as rc


def test_legacy_remote_provider_is_mirrored_to_both_roles():
    data = rc._payload_to_data(rc.ConfigPayload(remote_provider="MiniMax"))

    assert data["remoteProvider"] == "minimax"
    assert data["remoteChatProvider"] == "minimax"
    assert data["remoteSmallProvider"] == "minimax"


def test_explicit_role_provider_is_not_overwritten_by_legacy_field():
    data = rc._payload_to_data(rc.ConfigPayload(
        remote_provider="dashscope",
        remote_chat_provider="ark",
        remote_small_provider="minimax",
    ))

    assert data["remoteChatProvider"] == "ark"
    assert data["remoteSmallProvider"] == "minimax"


def test_explicit_null_role_provider_clears_instead_of_restoring_legacy_field():
    data = rc._payload_to_data(rc.ConfigPayload(
        remote_provider="minimax",
        remote_chat_provider=None,
        remote_small_provider=None,
    ))

    assert data["remoteChatProvider"] is None
    assert data["remoteSmallProvider"] is None


@pytest.mark.asyncio
async def test_remote_model_must_match_fallback_provider_when_provider_is_null(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return (provider, identifier) == ("deepseek", "deepseek-v4-pro")

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    payload = rc.ConfigPayload(remote_provider=None, remote_chat_model="deepseek-v4-pro")
    with pytest.raises(HTTPException) as exc:
        await rc._validate_payload_models(
            payload,
            fallback_remote_chat_provider="dashscope",
            fallback_remote_small_provider="dashscope",
            fallback_remote_chat_model="qwen3.5-plus",
            fallback_remote_small_model="qwen3.5-flash",
        )

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
    await rc._validate_payload_models(
        payload,
        fallback_remote_chat_provider="dashscope",
        fallback_remote_small_provider="dashscope",
        fallback_remote_chat_model="qwen3.5-plus",
        fallback_remote_small_model="qwen3.5-flash",
    )


@pytest.mark.asyncio
async def test_same_identifier_can_exist_under_multiple_remote_providers(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return identifier == "deepseek-v4-pro" and provider in {"dashscope", "deepseek"}

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    await rc._validate_payload_models(
        rc.ConfigPayload(remote_provider="deepseek", remote_chat_model="deepseek-v4-pro"),
        fallback_remote_chat_provider="dashscope",
        fallback_remote_small_provider="dashscope",
        fallback_remote_chat_model="qwen3.5-plus",
        fallback_remote_small_model="deepseek-v4-pro",
    )
    await rc._validate_payload_models(
        rc.ConfigPayload(remote_provider="dashscope", remote_chat_model="deepseek-v4-pro"),
        fallback_remote_chat_provider="dashscope",
        fallback_remote_small_provider="dashscope",
        fallback_remote_chat_model="qwen3.5-plus",
        fallback_remote_small_model="deepseek-v4-pro",
    )


@pytest.mark.asyncio
async def test_chat_and_small_models_can_use_different_remote_providers(monkeypatch):
    seen: list[tuple[str, str]] = []

    async def fake_model_exists(identifier: str, provider: str) -> bool:
        seen.append((provider, identifier))
        return (provider, identifier) in {
            ("minimax", "M2-her"),
            ("dashscope", "qwen3.5-flash"),
        }

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    payload = rc.ConfigPayload(
        remote_chat_provider="minimax",
        remote_small_provider="dashscope",
        remote_chat_model="M2-her",
        remote_small_model="qwen3.5-flash",
    )
    await rc._validate_payload_models(
        payload,
        fallback_remote_chat_provider="deepseek",
        fallback_remote_small_provider="deepseek",
        fallback_remote_chat_model="deepseek-v4-pro",
        fallback_remote_small_model="deepseek-v4-flash",
    )

    assert seen == [
        ("minimax", "M2-her"),
        ("dashscope", "qwen3.5-flash"),
    ]


@pytest.mark.asyncio
async def test_provider_only_override_rejects_inherited_model_from_other_provider(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return (provider, identifier) in {
            ("dashscope", "qwen3.5-plus"),
            ("dashscope", "qwen3.5-flash"),
        }

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    with pytest.raises(HTTPException) as exc:
        await rc._validate_payload_models(
            rc.ConfigPayload(remote_chat_provider="minimax"),
            fallback_remote_chat_provider="dashscope",
            fallback_remote_small_provider="dashscope",
            fallback_remote_chat_model="qwen3.5-plus",
            fallback_remote_small_model="qwen3.5-flash",
        )

    assert "remote_chat_model" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_explicit_null_role_provider_validates_against_parent_not_legacy(monkeypatch):
    async def fake_model_exists(identifier: str, provider: str) -> bool:
        return (provider, identifier) in {
            ("dashscope", "qwen3.5-plus"),
            ("dashscope", "qwen3.5-flash"),
        }

    monkeypatch.setattr(rc, "_model_exists_for_provider", fake_model_exists)

    await rc._validate_payload_models(
        rc.ConfigPayload(
            remote_provider="minimax",
            remote_chat_provider=None,
            remote_small_provider=None,
        ),
        fallback_remote_chat_provider="dashscope",
        fallback_remote_small_provider="dashscope",
        fallback_remote_chat_model="qwen3.5-plus",
        fallback_remote_small_model="qwen3.5-flash",
    )
