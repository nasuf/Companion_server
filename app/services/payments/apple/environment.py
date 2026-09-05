"""Apple 传输层：App Store Server API 客户端 + JWS 验签 + prod/sandbox 路由。

用 Apple 官方 `app-store-server-library`（内建 x5c 链校验→AppleRootCA-G3 + OCSP +
bundleId/env 断言，ES256 JWT 鉴权），**不手写 x5c 验签**（支付安全红线）。

官方库是同步的（get_transaction_info 走网络、verify 会做 OCSP），全部用
`asyncio.to_thread` 包裹，避免阻塞 event loop。**本模块不碰 DB。**

测试通过 monkeypatch 本模块的 `fetch_and_verify_transaction` / `verify_notification`
/ `verify_signed_transaction` / `verify_renewal_info` 绕过真实网络与证书链。
"""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timezone

from appstoreserverlibrary.api_client import (
    AppStoreServerAPIClient,
    APIError,
    APIException,
)
from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.models.JWSRenewalInfoDecodedPayload import (
    JWSRenewalInfoDecodedPayload,
)
from appstoreserverlibrary.models.JWSTransactionDecodedPayload import (
    JWSTransactionDecodedPayload,
)
from appstoreserverlibrary.models.ResponseBodyV2DecodedPayload import (
    ResponseBodyV2DecodedPayload,
)
from appstoreserverlibrary.signed_data_verifier import (
    SignedDataVerifier,
    VerificationException,
    VerificationStatus,
)

from app.config import settings
from app.services.payments.errors import (
    AppleVerificationError,
    TransactionNotFoundError,
)

logger = logging.getLogger(__name__)

# transactionId 在该环境不存在 → 换环境重试（审核期与真实用户会混在同一 build，
# production 查不到就 fallback sandbox）。
_NOT_FOUND_ERRORS = {
    APIError.TRANSACTION_ID_NOT_FOUND,
    APIError.ORIGINAL_TRANSACTION_ID_NOT_FOUND,
}


def _envs_to_try() -> list[Environment]:
    mode = (settings.apple_iap_environment or "auto").strip().lower()
    if mode == "production":
        return [Environment.PRODUCTION]
    if mode == "sandbox":
        return [Environment.SANDBOX]
    # auto：先 production 再 sandbox（Apple 官方推荐顺序）。
    return [Environment.PRODUCTION, Environment.SANDBOX]


@functools.cache
def _root_ca_bytes() -> bytes:
    with open(settings.apple_iap_root_ca_path, "rb") as fh:
        return fh.read()


@functools.cache
def _private_key_bytes() -> bytes:
    if settings.apple_iap_private_key.strip():
        return settings.apple_iap_private_key.encode("utf-8")
    with open(settings.apple_iap_private_key_path, "rb") as fh:
        return fh.read()


@functools.cache
def build_verifier(environment: Environment) -> SignedDataVerifier:
    return SignedDataVerifier(
        [_root_ca_bytes()],
        True,  # enable_online_checks: OCSP 吊销校验
        environment,
        settings.apple_iap_bundle_id,
        settings.apple_iap_app_apple_id or None,
    )


@functools.cache
def build_client(environment: Environment) -> AppStoreServerAPIClient:
    return AppStoreServerAPIClient(
        _private_key_bytes(),
        settings.apple_iap_key_id,
        settings.apple_iap_issuer_id,
        settings.apple_iap_bundle_id,
        environment,
    )


def ms_to_dt(value: int | None) -> datetime | None:
    """Apple 日期是 epoch 毫秒 → 带 tz 的 UTC datetime。"""
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _is_not_found(exc: APIException) -> bool:
    return exc.api_error in _NOT_FOUND_ERRORS or exc.http_status_code == 404


async def fetch_and_verify_transaction(
    transaction_id: str,
) -> tuple[JWSTransactionDecodedPayload, str]:
    """按环境依次向 Apple 查交易并验签。返回 (已验签 payload, environment)。

    只信 transactionId：一切权益字段来自 Apple 返回并验签后的 payload，
    杜绝客户端伪造。查不到 → TransactionNotFoundError；验签失败 → AppleVerificationError。
    """
    last_not_found: APIException | None = None
    for env in _envs_to_try():
        client = build_client(env)
        try:
            resp = await asyncio.to_thread(client.get_transaction_info, transaction_id)
        except APIException as exc:
            if _is_not_found(exc):
                last_not_found = exc
                continue
            raise AppleVerificationError(
                f"apple_api_error:{exc.http_status_code}:{exc.api_error}"
            ) from exc
        signed = resp.signedTransactionInfo
        payload = await asyncio.to_thread(verify_signed_transaction, signed, env.value)
        return payload, env.value
    raise TransactionNotFoundError(
        f"transaction_not_found:{transaction_id}"
    ) from last_not_found


def verify_signed_transaction(jws: str, environment: str) -> JWSTransactionDecodedPayload:
    verifier = build_verifier(Environment(environment))
    try:
        return verifier.verify_and_decode_signed_transaction(jws)
    except VerificationException as exc:
        raise AppleVerificationError(f"jws_verify_failed:{exc.status}") from exc


def verify_renewal_info(jws: str, environment: str) -> JWSRenewalInfoDecodedPayload:
    verifier = build_verifier(Environment(environment))
    try:
        return verifier.verify_and_decode_renewal_info(jws)
    except VerificationException as exc:
        raise AppleVerificationError(f"renewal_verify_failed:{exc.status}") from exc


async def verify_notification(
    signed_payload: str,
) -> tuple[ResponseBodyV2DecodedPayload, str]:
    """验 V2 通知签名。环境未知 → 依次用各环境 verifier 试，任一通过即接受。

    webhook 唯一鉴权就是这一步：验签失败（含所有环境都不符）→ AppleVerificationError，
    上层据此回 401 拒绝伪造。
    """
    envs = _envs_to_try()
    last_exc: AppleVerificationError | None = None
    for env in envs:
        verifier = build_verifier(env)
        try:
            decoded = await asyncio.to_thread(
                verifier.verify_and_decode_notification, signed_payload
            )
            return decoded, env.value
        except VerificationException as exc:
            # 环境不符就换下一个；其它验签失败（篡改/证书链/bundle）直接拒绝。
            if exc.status == VerificationStatus.INVALID_ENVIRONMENT and len(envs) > 1:
                last_exc = AppleVerificationError(f"notif_verify_failed:{exc.status}")
                continue
            raise AppleVerificationError(f"notif_verify_failed:{exc.status}") from exc
    raise last_exc or AppleVerificationError("notif_verify_failed")
