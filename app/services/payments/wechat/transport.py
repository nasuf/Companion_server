"""微信支付 APIv3 传输层：请求签名、应答/回调验签、回调报文解密。

安全红线（这些是支付资金安全的地基，改动必须 review）：
* 请求签名用商户私钥 RSA-SHA256（`SHA256-RSA2048`），签名串构造顺序固定为
  method\\nurl\\ntimestamp\\nnonce\\nbody\\n（每行末尾都有换行）。
* 回调验签用**微信支付公钥**（不是商户证书），对 timestamp\\nnonce\\nbody\\n 验签。
  验签失败一律拒绝，绝不"先发货再说"。
* 回调时间戳偏移超过 5 分钟视为重放，拒绝。
* 回调 resource 用 AES-256-GCM 解密，key = APIv3 密钥（32 字节），
  AAD = resource.associated_data，nonce = resource.nonce。

**本模块不碰 DB、不做业务判断**，便于单测直接构造签名与密文。
"""

from __future__ import annotations

import base64
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config import settings
from app.services.payments.wechat.errors import (
    WechatPayConfigError,
    WechatPayDecryptError,
    WechatPaySignatureError,
    WechatPayUpstreamError,
)

logger = logging.getLogger(__name__)

API_BASE = "https://api.mch.weixin.qq.com"
_SIGN_TYPE = "WECHATPAY2-SHA256-RSA2048"
# 微信支付要求回调时间戳与当前时间相差不超过 5 分钟（防重放）。
_TIMESTAMP_TOLERANCE_S = 300
_APIV3_KEY_LENGTH = 32


def _read_key_file(path: str, *, label: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise WechatPayConfigError(f"{label} 不可读: {path}") from exc


def _load_private_key() -> rsa.RSAPrivateKey:
    """商户 API 私钥 apiclient_key.pem。进程内缓存，避免每次下单读盘。"""
    cached = _load_private_key.cache  # type: ignore[attr-defined]
    if cached is not None:
        return cached
    path = settings.wechatpay_private_key_path.strip()
    if not path:
        raise WechatPayConfigError("WECHATPAY_PRIVATE_KEY_PATH 未配置")
    key = serialization.load_pem_private_key(
        _read_key_file(path, label="商户 API 私钥"), password=None
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise WechatPayConfigError("商户 API 私钥不是 RSA 私钥")
    _load_private_key.cache = key  # type: ignore[attr-defined]
    return key


_load_private_key.cache = None  # type: ignore[attr-defined]


def _load_public_key() -> rsa.RSAPublicKey:
    """微信支付公钥 pub_key.pem，用于应答与回调验签。"""
    cached = _load_public_key.cache  # type: ignore[attr-defined]
    if cached is not None:
        return cached
    path = settings.wechatpay_public_key_path.strip()
    if not path:
        raise WechatPayConfigError("WECHATPAY_PUBLIC_KEY_PATH 未配置")
    key = serialization.load_pem_public_key(
        _read_key_file(path, label="微信支付公钥")
    )
    if not isinstance(key, rsa.RSAPublicKey):
        raise WechatPayConfigError("微信支付公钥不是 RSA 公钥")
    _load_public_key.cache = key  # type: ignore[attr-defined]
    return key


_load_public_key.cache = None  # type: ignore[attr-defined]


def reset_key_cache() -> None:
    """测试用：切换 settings 中的密钥路径后清缓存。"""
    _load_private_key.cache = None  # type: ignore[attr-defined]
    _load_public_key.cache = None  # type: ignore[attr-defined]


def _apiv3_key() -> bytes:
    raw = settings.wechatpay_api_v3_key.strip().encode("utf-8")
    if len(raw) != _APIV3_KEY_LENGTH:
        raise WechatPayConfigError("WECHATPAY_API_V3_KEY 必须是 32 字节")
    return raw


def build_signature(
    *,
    method: str,
    url_path: str,
    timestamp: str,
    nonce: str,
    body: str,
) -> str:
    """APIv3 请求签名。签名串每行以 \\n 结尾，GET/DELETE 的 body 为空串。"""
    message = f"{method}\n{url_path}\n{timestamp}\n{nonce}\n{body}\n".encode("utf-8")
    signature = _load_private_key().sign(message, padding.PKCS1v15(), hashes.SHA256())
    return base64.b64encode(signature).decode("ascii")


def build_authorization(*, method: str, url_path: str, body: str) -> str:
    mch_id = settings.wechatpay_mch_id.strip()
    serial_no = settings.wechatpay_cert_serial_no.strip()
    if not mch_id or not serial_no:
        raise WechatPayConfigError("WECHATPAY_MCH_ID / WECHATPAY_CERT_SERIAL_NO 未配置")
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16).upper()
    signature = build_signature(
        method=method, url_path=url_path, timestamp=timestamp, nonce=nonce, body=body
    )
    return (
        f'{_SIGN_TYPE} mchid="{mch_id}",nonce_str="{nonce}",'
        f'signature="{signature}",timestamp="{timestamp}",serial_no="{serial_no}"'
    )


def verify_signature(
    *, timestamp: str, nonce: str, body: str, signature: str
) -> None:
    """用微信支付公钥验签（回调与应答共用）。失败抛 WechatPaySignatureError。"""
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError) as exc:
        raise WechatPaySignatureError("签名不是合法 base64") from exc
    message = f"{timestamp}\n{nonce}\n{body}\n".encode("utf-8")
    # 公钥配错要报配置错误，不能被下面的 except 吞成"验签未通过"误导排查。
    public_key = _load_public_key()
    try:
        public_key.verify(
            signature_bytes, message, padding.PKCS1v15(), hashes.SHA256()
        )
    except Exception as exc:  # InvalidSignature 及其它 crypto 异常
        raise WechatPaySignatureError("验签未通过") from exc


def assert_expected_serial(serial: str | None) -> None:
    """`Wechatpay-Serial` 必须是配置的微信支付公钥 ID。

    不一致说明微信用了别的密钥签名（如平台证书、或公钥已在商户平台轮换），
    本地公钥必然验不过。单独报错便于从日志里一眼看出是"换钥匙了"而不是被伪造。
    未配置公钥 ID 时跳过（只靠验签）。
    """
    expected = settings.wechatpay_public_key_id.strip()
    if expected and (serial or "").strip() != expected:
        raise WechatPaySignatureError(
            f"Wechatpay-Serial 与配置的公钥 ID 不一致: {serial or '(空)'}"
        )


def assert_fresh_timestamp(timestamp: str, *, now: float | None = None) -> None:
    """时间戳防重放。非数字或偏移超过 5 分钟都拒绝。"""
    try:
        sent_at = int(timestamp)
    except (TypeError, ValueError) as exc:
        raise WechatPaySignatureError("回调时间戳不是整数") from exc
    current = time.time() if now is None else now
    if abs(current - sent_at) > _TIMESTAMP_TOLERANCE_S:
        raise WechatPaySignatureError("回调时间戳超出容忍窗口")


def decrypt_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """AES-256-GCM 解密回调 resource。algorithm 非预期值一律拒绝。"""
    algorithm = str(resource.get("algorithm") or "")
    if algorithm != "AEAD_AES_256_GCM":
        raise WechatPayDecryptError(f"不支持的回调加密算法: {algorithm or 'missing'}")
    ciphertext_b64 = resource.get("ciphertext")
    nonce = resource.get("nonce")
    if not isinstance(ciphertext_b64, str) or not isinstance(nonce, str):
        raise WechatPayDecryptError("回调 resource 缺少 ciphertext/nonce")
    associated_data = resource.get("associated_data") or ""
    if not isinstance(associated_data, str):
        raise WechatPayDecryptError("回调 resource 的 associated_data 不是字符串")
    try:
        ciphertext = base64.b64decode(ciphertext_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise WechatPayDecryptError("ciphertext 不是合法 base64") from exc
    try:
        plaintext = AESGCM(_apiv3_key()).decrypt(
            nonce.encode("utf-8"), ciphertext, associated_data.encode("utf-8")
        )
    except (InvalidTag, ValueError) as exc:
        raise WechatPayDecryptError("回调报文解密失败") from exc
    try:
        decoded = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WechatPayDecryptError("回调明文不是合法 JSON") from exc
    if not isinstance(decoded, dict):
        raise WechatPayDecryptError("回调明文不是 JSON 对象")
    return decoded


async def request(
    method: str,
    url_path: str,
    *,
    body: dict[str, Any] | None = None,
    verify_response: bool = True,
) -> dict[str, Any]:
    """调用 APIv3。``url_path`` 必须以 / 开头且含 query（签名串用的就是它）。

    成功返回解析后的 JSON（204 返回空 dict）。4xx/5xx 抛 WechatPayUpstreamError，
    其中带上微信的 code/message 便于后台排查，但不把它直接透给终端用户。
    """
    if not url_path.startswith("/"):
        raise WechatPayConfigError(f"url_path 必须以 / 开头: {url_path}")
    payload = (
        json.dumps(body, ensure_ascii=False, separators=(",", ":")) if body is not None else ""
    )
    authorization = build_authorization(method=method, url_path=url_path, body=payload)
    headers = {
        "Authorization": authorization,
        "Accept": "application/json",
        "User-Agent": "companion-server/wechatpay",
    }
    if payload:
        headers["Content-Type"] = "application/json"
    timeout = httpx.Timeout(settings.wechatpay_timeout_s, connect=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.request(
                method, f"{API_BASE}{url_path}", headers=headers, content=payload or None
            )
    except httpx.HTTPError as exc:
        raise WechatPayUpstreamError(f"微信支付请求失败: {type(exc).__name__}") from exc

    text = response.text or ""
    if response.status_code >= 400:
        code, message = _upstream_error(text)
        raise WechatPayUpstreamError(
            f"微信支付返回 {response.status_code}: {code or 'unknown'}",
            status_code=response.status_code,
            code=code,
            message=message,
        )
    if verify_response:
        _verify_response_signature(response, text)
    if response.status_code == 204 or not text.strip():
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WechatPayUpstreamError("微信支付应答不是合法 JSON") from exc
    if not isinstance(decoded, dict):
        raise WechatPayUpstreamError("微信支付应答不是 JSON 对象")
    return decoded


def _verify_response_signature(response: httpx.Response, text: str) -> None:
    timestamp = response.headers.get("Wechatpay-Timestamp")
    nonce = response.headers.get("Wechatpay-Nonce")
    signature = response.headers.get("Wechatpay-Signature")
    if not timestamp or not nonce or not signature:
        raise WechatPaySignatureError("微信支付应答缺少验签头")
    assert_expected_serial(response.headers.get("Wechatpay-Serial"))
    assert_fresh_timestamp(timestamp)
    verify_signature(timestamp=timestamp, nonce=nonce, body=text, signature=signature)


def _upstream_error(text: str) -> tuple[str | None, str | None]:
    try:
        decoded = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return None, None
    if not isinstance(decoded, dict):
        return None, None
    code = decoded.get("code")
    message = decoded.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


def jsapi_pay_params(prepay_id: str) -> dict[str, str]:
    """H5 调起支付所需参数（用商户私钥对 appid/timestamp/nonce/prepay_id 签名）。"""
    app_id = settings.wechat_h5_app_id.strip()
    if not app_id:
        raise WechatPayConfigError("WECHAT_H5_APP_ID 未配置")
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16).upper()
    package = f"prepay_id={prepay_id}"
    message = f"{app_id}\n{timestamp}\n{nonce}\n{package}\n".encode("utf-8")
    signature = _load_private_key().sign(message, padding.PKCS1v15(), hashes.SHA256())
    return {
        "appId": app_id,
        "timeStamp": timestamp,
        "nonceStr": nonce,
        "package": package,
        "signType": "RSA",
        "paySign": base64.b64encode(signature).decode("ascii"),
    }


def notify_path() -> str:
    """回调地址的 path，用于自检配置是否指向本服务的回调路由。"""
    return urlparse(settings.wechatpay_notify_url.strip()).path or "/"
