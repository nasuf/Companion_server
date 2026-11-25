"""微信支付传输层：签名、验签、防重放、解密。

这些是资金安全的地基，所以用真实的 RSA 密钥和真实的 AES-256-GCM 密文来测，
不 mock 加解密本身——mock 掉就等于没测。
"""

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config import settings
from app.services.payments.wechat import transport
from app.services.payments.wechat.errors import (
    WechatPayConfigError,
    WechatPayDecryptError,
    WechatPaySignatureError,
)

API_V3_KEY = "0123456789abcdef0123456789abcdef"  # 32 字节，测试专用


@pytest.fixture
def wechatpay_keys(tmp_path, monkeypatch):
    """生成一对真实 RSA 密钥：私钥当商户私钥，公钥当"微信支付公钥"。"""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path = tmp_path / "apiclient_key.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path = tmp_path / "pub_key.pem"
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setattr(settings, "wechatpay_private_key_path", str(private_path))
    monkeypatch.setattr(settings, "wechatpay_public_key_path", str(public_path))
    monkeypatch.setattr(settings, "wechatpay_mch_id", "1007054057")
    monkeypatch.setattr(settings, "wechatpay_cert_serial_no", "ABC123")
    monkeypatch.setattr(settings, "wechatpay_api_v3_key", API_V3_KEY)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wxtestappid")
    transport.reset_key_cache()
    yield private_key
    transport.reset_key_cache()


def _sign(private_key, message: str) -> str:
    signature = private_key.sign(
        message.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256()
    )
    return base64.b64encode(signature).decode("ascii")


def test_request_signature_message_layout(wechatpay_keys):
    """签名串必须是 method\\nurl\\ntimestamp\\nnonce\\nbody\\n，顺序与换行都不能错。"""
    signature = transport.build_signature(
        method="POST",
        url_path="/v3/pay/transactions/jsapi",
        timestamp="1764000000",
        nonce="NONCE",
        body='{"a":1}',
    )
    expected_message = 'POST\n/v3/pay/transactions/jsapi\n1764000000\nNONCE\n{"a":1}\n'
    wechatpay_keys.public_key().verify(
        base64.b64decode(signature),
        expected_message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_authorization_header_carries_mchid_and_serial(wechatpay_keys):
    header = transport.build_authorization(
        method="POST", url_path="/v3/pay/transactions/jsapi", body="{}"
    )
    assert header.startswith("WECHATPAY2-SHA256-RSA2048 ")
    assert 'mchid="1007054057"' in header
    assert 'serial_no="ABC123"' in header


def test_verify_signature_accepts_valid_and_rejects_tampered_body(wechatpay_keys):
    timestamp = str(int(time.time()))
    body = '{"id":"notify-1"}'
    signature = _sign(wechatpay_keys, f"{timestamp}\nNONCE\n{body}\n")

    transport.verify_signature(
        timestamp=timestamp, nonce="NONCE", body=body, signature=signature
    )

    # 改一个字节就必须失败——这正是防伪造回调的那道门。
    with pytest.raises(WechatPaySignatureError):
        transport.verify_signature(
            timestamp=timestamp,
            nonce="NONCE",
            body='{"id":"notify-2"}',
            signature=signature,
        )


def test_verify_signature_rejects_wrong_key(wechatpay_keys):
    """用别的私钥签名（= 攻击者自己的密钥）必须被拒。"""
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    timestamp = str(int(time.time()))
    body = '{"id":"forged"}'
    signature = _sign(attacker_key, f"{timestamp}\nNONCE\n{body}\n")
    with pytest.raises(WechatPaySignatureError):
        transport.verify_signature(
            timestamp=timestamp, nonce="NONCE", body=body, signature=signature
        )


def test_verify_signature_rejects_non_base64(wechatpay_keys):
    with pytest.raises(WechatPaySignatureError):
        transport.verify_signature(
            timestamp="1", nonce="N", body="{}", signature="not base64!!"
        )


@pytest.mark.parametrize("offset", [-301, 301, -3600, 3600])
def test_stale_timestamp_rejected(wechatpay_keys, offset):
    now = time.time()
    with pytest.raises(WechatPaySignatureError):
        transport.assert_fresh_timestamp(str(int(now + offset)), now=now)


@pytest.mark.parametrize("offset", [0, -299, 299])
def test_fresh_timestamp_accepted(wechatpay_keys, offset):
    now = time.time()
    transport.assert_fresh_timestamp(str(int(now + offset)), now=now)


def test_non_numeric_timestamp_rejected(wechatpay_keys):
    with pytest.raises(WechatPaySignatureError):
        transport.assert_fresh_timestamp("not-a-number")


def _encrypt(plaintext: dict, *, associated_data: str = "transaction") -> dict:
    nonce = "abcdefghijkl"
    ciphertext = AESGCM(API_V3_KEY.encode()).encrypt(
        nonce.encode(),
        json.dumps(plaintext).encode(),
        associated_data.encode(),
    )
    return {
        "algorithm": "AEAD_AES_256_GCM",
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "nonce": nonce,
        "associated_data": associated_data,
    }


def test_decrypt_resource_roundtrip(wechatpay_keys):
    payload = {"out_trade_no": "BS001", "trade_state": "SUCCESS"}
    assert transport.decrypt_resource(_encrypt(payload)) == payload


def test_decrypt_rejects_wrong_associated_data(wechatpay_keys):
    """AAD 是 GCM 认证的一部分：换了 AAD 就解不开，防止跨类型重放。"""
    resource = _encrypt({"a": 1}, associated_data="transaction")
    resource["associated_data"] = "refund"
    with pytest.raises(WechatPayDecryptError):
        transport.decrypt_resource(resource)


def test_decrypt_rejects_unexpected_algorithm(wechatpay_keys):
    resource = _encrypt({"a": 1})
    resource["algorithm"] = "AES_128_CBC"
    with pytest.raises(WechatPayDecryptError):
        transport.decrypt_resource(resource)


def test_decrypt_rejects_tampered_ciphertext(wechatpay_keys):
    resource = _encrypt({"a": 1})
    raw = bytearray(base64.b64decode(resource["ciphertext"]))
    raw[0] ^= 0xFF
    resource["ciphertext"] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(WechatPayDecryptError):
        transport.decrypt_resource(resource)


@pytest.mark.parametrize("bad_key", ["short", "a" * 31, "a" * 33, ""])
def test_wrong_length_api_v3_key_rejected(wechatpay_keys, monkeypatch, bad_key):
    """长度不对时每笔回调都解不开，必须早失败而不是静默丢单。"""
    monkeypatch.setattr(settings, "wechatpay_api_v3_key", bad_key)
    with pytest.raises(WechatPayConfigError):
        transport.decrypt_resource(_encrypt({"a": 1}))


def test_jsapi_pay_params_signature_layout(wechatpay_keys):
    """H5 调起参数的签名串是 appId\\ntimeStamp\\nnonceStr\\npackage\\n。"""
    params = transport.jsapi_pay_params("prepay-123")
    assert params["appId"] == "wxtestappid"
    assert params["package"] == "prepay_id=prepay-123"
    assert params["signType"] == "RSA"
    message = (
        f"{params['appId']}\n{params['timeStamp']}\n"
        f"{params['nonceStr']}\nprepay_id=prepay-123\n"
    )
    wechatpay_keys.public_key().verify(
        base64.b64decode(params["paySign"]),
        message.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_missing_private_key_path_raises_config_error(monkeypatch):
    monkeypatch.setattr(settings, "wechatpay_private_key_path", "")
    transport.reset_key_cache()
    with pytest.raises(WechatPayConfigError):
        transport.build_signature(
            method="GET", url_path="/v3/x", timestamp="1", nonce="N", body=""
        )
    transport.reset_key_cache()


def test_unreadable_key_path_raises_config_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings, "wechatpay_private_key_path", str(tmp_path / "missing.pem")
    )
    transport.reset_key_cache()
    with pytest.raises(WechatPayConfigError):
        transport.build_signature(
            method="GET", url_path="/v3/x", timestamp="1", nonce="N", body=""
        )
    transport.reset_key_cache()


async def test_request_rejects_relative_url_path(wechatpay_keys):
    with pytest.raises(WechatPayConfigError):
        await transport.request("GET", "v3/pay/transactions")


@pytest.mark.parametrize("serial", [None, "", "PUB_KEY_ID_OTHER", "66B7169FC57DB721"])
def test_unexpected_serial_is_rejected(monkeypatch, serial):
    """微信换了签名密钥（或伪造）时明确报错，而不是笼统的验签失败。"""
    from app.services.payments.wechat.errors import WechatPaySignatureError

    monkeypatch.setattr(settings, "wechatpay_public_key_id", "PUB_KEY_ID_EXPECTED")
    with pytest.raises(WechatPaySignatureError, match="公钥 ID"):
        transport.assert_expected_serial(serial)


def test_expected_serial_passes(monkeypatch):
    monkeypatch.setattr(settings, "wechatpay_public_key_id", "PUB_KEY_ID_EXPECTED")
    transport.assert_expected_serial("PUB_KEY_ID_EXPECTED")
