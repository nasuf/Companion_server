"""Minimal Tencent Cloud SMS client (SendSms) with TC3-HMAC-SHA256 signing.

Implements the documented signing algorithm directly over httpx instead of
pulling in tencentcloud-sdk-python — we call exactly one API. Reference:
https://cloud.tencent.com/document/api/382/55981 (SendSms, version 2021-01-11).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import UTC, datetime

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_HOST = "sms.tencentcloudapi.com"
_SERVICE = "sms"
_VERSION = "2021-01-11"
_ACTION = "SendSms"
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


class SmsSendError(Exception):
    """Raised when Tencent Cloud rejects or fails the SendSms call."""


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _tc3_authorization(payload: str, timestamp: int) -> str:
    """Build the TC3-HMAC-SHA256 Authorization header for SendSms."""
    secret_id = settings.tencent_sms_secret_id.strip()
    secret_key = settings.tencent_sms_secret_key.strip()
    date = datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")

    # 1. Canonical request. SendSms uses POST / with JSON body; the signed
    #    headers set must match the headers actually sent.
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_headers = (
        f"content-type:application/json; charset=utf-8\nhost:{_HOST}\n"
    )
    signed_headers = "content-type;host"
    canonical_request = (
        f"POST\n/\n\n{canonical_headers}\n{signed_headers}\n{hashed_payload}"
    )

    # 2. String to sign.
    credential_scope = f"{date}/{_SERVICE}/tc3_request"
    hashed_canonical = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (
        f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n{hashed_canonical}"
    )

    # 3. Signature chain.
    secret_date = _hmac_sha256(f"TC3{secret_key}".encode("utf-8"), date)
    secret_service = _hmac_sha256(secret_date, _SERVICE)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return (
        f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


async def send_sms_code(phone: str, code: str, ttl_minutes: int) -> None:
    """Send a verification code to a mainland-CN number via Tencent Cloud SMS.

    ``phone`` is the bare 11-digit number; the +86 prefix is added here.
    Raises ``SmsSendError`` on transport failures or non-Ok statuses.
    """
    payload = json.dumps(
        {
            "PhoneNumberSet": [f"+86{phone}"],
            "SmsSdkAppId": settings.tencent_sms_sdk_app_id.strip(),
            "SignName": settings.tencent_sms_sign_name.strip(),
            "TemplateId": settings.tencent_sms_template_id.strip(),
            "TemplateParamSet": [code, str(ttl_minutes)],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    timestamp = int(time.time())
    headers = {
        "Authorization": _tc3_authorization(payload, timestamp),
        "Content-Type": "application/json; charset=utf-8",
        "Host": _HOST,
        "X-TC-Action": _ACTION,
        "X-TC-Version": _VERSION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": settings.tencent_sms_region.strip() or "ap-guangzhou",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            response = await client.post(
                f"https://{_HOST}/", content=payload.encode("utf-8"), headers=headers
            )
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Tencent SMS transport failure",
            extra={"event": "sms_send_transport_error", "reason": type(exc).__name__},
        )
        raise SmsSendError("sms_transport_error") from exc

    resp = body.get("Response") or {}
    if resp.get("Error"):
        # e.g. signature errors, quota exhausted, unaudited template.
        logger.warning(
            "Tencent SMS API error",
            extra={
                "event": "sms_send_api_error",
                "code": (resp["Error"] or {}).get("Code"),
            },
        )
        raise SmsSendError(str((resp["Error"] or {}).get("Code") or "sms_api_error"))

    statuses = resp.get("SendStatusSet") or []
    status = statuses[0] if statuses else {}
    if str(status.get("Code")) != "Ok":
        # Per-number failure: carrier rejection, blacklisted number, limits…
        logger.warning(
            "Tencent SMS per-number failure",
            extra={"event": "sms_send_status_error", "code": status.get("Code")},
        )
        raise SmsSendError(str(status.get("Code") or "sms_status_error"))
