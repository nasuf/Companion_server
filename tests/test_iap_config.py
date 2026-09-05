"""生产环境启用 Apple IAP 但缺关键配置时必须 fail-fast。"""

from __future__ import annotations

import pytest

from app.config import Settings


def _base_prod_kwargs() -> dict:
    # 满足其它生产必填项，隔离出 apple_iap 校验分支。
    return {
        "app_env": "production",
        "jwt_secret": "x" * 40,
        "cors_allowed_origins": "https://app.example.com",
    }


def test_apple_iap_enabled_missing_keys_raises():
    settings = Settings(**_base_prod_kwargs(), apple_iap_enabled=True)
    with pytest.raises(RuntimeError) as exc:
        settings.validate_security_config()
    assert "APPLE_IAP" in str(exc.value)


def test_apple_iap_enabled_with_keys_ok():
    settings = Settings(
        **_base_prod_kwargs(),
        apple_iap_enabled=True,
        apple_iap_bundle_id="com.bansheng.prod",
        apple_iap_issuer_id="issuer-uuid",
        apple_iap_key_id="KEY123",
        apple_iap_private_key="-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
    )
    # 不应抛错
    settings.validate_security_config()


def test_apple_iap_disabled_skips_check():
    settings = Settings(**_base_prod_kwargs(), apple_iap_enabled=False)
    settings.validate_security_config()
