import pytest

from app.config import Settings


def test_development_allows_local_defaults():
    settings = Settings(_env_file=None)

    settings.validate_security_config()

    assert settings.cors_origins() == ["*"]


def test_legacy_admin_basic_auth_fields_are_removed():
    assert "admin_username" not in Settings.model_fields
    assert "admin_password" not in Settings.model_fields


def test_production_rejects_empty_jwt_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://companion.example.com")
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        settings.validate_security_config()


def test_production_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="CORS_ALLOWED_ORIGINS"):
        settings.validate_security_config()


def test_production_accepts_explicit_security_config(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://app.example.com, https://admin.example.com",
    )
    settings = Settings(_env_file=None)

    settings.validate_security_config()

    assert settings.cors_origins() == [
        "https://app.example.com",
        "https://admin.example.com",
    ]
