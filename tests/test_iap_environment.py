"""Apple 环境路由：production 查不到 → fallback sandbox。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from appstoreserverlibrary.api_client import APIException
from appstoreserverlibrary.models.Environment import Environment

from app.services.payments.apple import environment as env
from app.services.payments.errors import (
    AppleVerificationError,
    TransactionNotFoundError,
)


class _FakeClient:
    def __init__(self, *, not_found: bool = False, raises: APIException | None = None):
        self.not_found = not_found
        self.raises = raises

    def get_transaction_info(self, transaction_id: str):
        if self.raises is not None:
            raise self.raises
        if self.not_found:
            raise APIException(404, 4040010)  # TRANSACTION_ID_NOT_FOUND
        return SimpleNamespace(signedTransactionInfo="jws-sandbox")


@pytest.mark.asyncio
async def test_prod_not_found_falls_back_to_sandbox(monkeypatch):
    monkeypatch.setattr(env.settings, "apple_iap_environment", "auto")
    clients = {
        Environment.PRODUCTION: _FakeClient(not_found=True),
        Environment.SANDBOX: _FakeClient(not_found=False),
    }
    monkeypatch.setattr(env, "build_client", lambda e: clients[e])
    seen = {}

    def _verify(jws, environment):
        seen["jws"] = jws
        seen["env"] = environment
        return SimpleNamespace(productId="p", transactionId="t")

    monkeypatch.setattr(env, "verify_signed_transaction", _verify)

    payload, environment = await env.fetch_and_verify_transaction("txn")

    assert environment == "Sandbox"
    assert seen["jws"] == "jws-sandbox"
    assert seen["env"] == "Sandbox"


@pytest.mark.asyncio
async def test_not_found_in_all_envs_raises(monkeypatch):
    monkeypatch.setattr(env.settings, "apple_iap_environment", "auto")
    monkeypatch.setattr(env, "build_client", lambda e: _FakeClient(not_found=True))
    monkeypatch.setattr(env, "verify_signed_transaction", lambda *a: None)

    with pytest.raises(TransactionNotFoundError):
        await env.fetch_and_verify_transaction("txn")


@pytest.mark.asyncio
async def test_prod_401_falls_back_to_sandbox(monkeypatch):
    """app 未上架时 production 对 JWT 返回 401，不应中止，须回退 sandbox。"""
    monkeypatch.setattr(env.settings, "apple_iap_environment", "auto")
    clients = {
        Environment.PRODUCTION: _FakeClient(raises=APIException(401)),
        Environment.SANDBOX: _FakeClient(not_found=False),
    }
    monkeypatch.setattr(env, "build_client", lambda e: clients[e])
    monkeypatch.setattr(
        env,
        "verify_signed_transaction",
        lambda jws, environment: SimpleNamespace(env=environment),
    )

    payload, environment = await env.fetch_and_verify_transaction("txn")

    assert environment == "Sandbox"
    assert payload.env == "Sandbox"


@pytest.mark.asyncio
async def test_hard_error_in_all_envs_raises_verification_error(monkeypatch):
    """两个环境都硬错（如都 401）→ 报鉴权失败，而非误判交易不存在。"""
    monkeypatch.setattr(env.settings, "apple_iap_environment", "auto")
    monkeypatch.setattr(
        env, "build_client", lambda e: _FakeClient(raises=APIException(401))
    )
    monkeypatch.setattr(env, "verify_signed_transaction", lambda *a: None)

    with pytest.raises(AppleVerificationError):
        await env.fetch_and_verify_transaction("txn")


class _FakeNotifVerifier:
    def __init__(self, *, exc=None, result=None):
        self.exc = exc
        self.result = result
        self.calls = 0

    def verify_and_decode_notification(self, signed):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def _verifiers(monkeypatch, prod, sandbox):
    monkeypatch.setattr(env.settings, 'apple_iap_environment', 'auto')
    table = {Environment.PRODUCTION: prod, Environment.SANDBOX: sandbox}
    monkeypatch.setattr(env, 'build_verifier', lambda e: table[e])


@pytest.mark.asyncio
async def test_sandbox_notification_retries_after_app_identifier_mismatch(monkeypatch):
    # 沙盒通知不带 appAppleId，production 验签器先报 INVALID_APP_IDENTIFIER。
    prod = _FakeNotifVerifier(exc=env.VerificationException(env.VerificationStatus.INVALID_APP_IDENTIFIER))
    sandbox = _FakeNotifVerifier(result=SimpleNamespace(ok=True))
    _verifiers(monkeypatch, prod, sandbox)

    decoded, environment = await env.verify_notification('jws')

    assert environment == 'Sandbox'
    assert decoded.ok is True
    assert prod.calls == 1 and sandbox.calls == 1


@pytest.mark.asyncio
async def test_notification_rejected_when_last_env_also_mismatches(monkeypatch):
    err = env.VerificationException(env.VerificationStatus.INVALID_APP_IDENTIFIER)
    _verifiers(monkeypatch, _FakeNotifVerifier(exc=err), _FakeNotifVerifier(exc=err))

    with pytest.raises(AppleVerificationError):
        await env.verify_notification('jws')


@pytest.mark.asyncio
async def test_tampered_notification_does_not_try_next_env(monkeypatch):
    prod = _FakeNotifVerifier(exc=env.VerificationException(env.VerificationStatus.VERIFICATION_FAILURE))
    sandbox = _FakeNotifVerifier(result=SimpleNamespace(ok=True))
    _verifiers(monkeypatch, prod, sandbox)

    with pytest.raises(AppleVerificationError):
        await env.verify_notification('jws')

    assert sandbox.calls == 0
