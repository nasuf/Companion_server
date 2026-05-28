from app.db import (
    _connection_limit_from_database_url,
    _is_db_pool_exhaustion_error,
    _with_safe_database_params,
    db,
)
from app.config import Settings


def test_with_safe_database_params_caps_oversized_connection_limit():
    url = (
        "postgresql://user:pass@host:5432/db"
        "?sslmode=require&connection_limit=20&pool_timeout=10"
    )

    result = _with_safe_database_params(url)

    assert "connection_limit=3" in result
    assert "pool_timeout=10" in result
    assert "connect_timeout=30" in result


def test_with_safe_database_params_preserves_lower_connection_limit():
    url = "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=2"

    result = _with_safe_database_params(url)

    assert "connection_limit=2" in result


def test_with_safe_database_params_allows_forced_connection_limit():
    url = "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=20"

    result = _with_safe_database_params(url, forced_connection_limit=3)

    assert "connection_limit=3" in result


def test_with_safe_database_params_caps_forced_connection_limit():
    url = "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=20"

    result = _with_safe_database_params(url, forced_connection_limit=15)

    assert "connection_limit=5" in result


def test_with_safe_database_params_honors_configured_safe_cap(monkeypatch):
    monkeypatch.setenv("DB_CONNECTION_LIMIT_MAX", "2")
    url = "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=20"

    result = _with_safe_database_params(url, forced_connection_limit=4)

    assert "connection_limit=2" in result


def test_is_db_pool_exhaustion_error_matches_session_pool_message():
    exc = RuntimeError(
        "FATAL: (EMAXCONNSESSION) max clients reached in session mode - "
        "max clients are limited to pool_size: 15"
    )

    assert _is_db_pool_exhaustion_error(exc)


def test_is_db_pool_exhaustion_error_ignores_unrelated_errors():
    assert not _is_db_pool_exhaustion_error(RuntimeError("syntax error at or near"))


def test_connection_limit_from_database_url_reads_runtime_limit():
    url = "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=3"

    assert _connection_limit_from_database_url(url) == 3


def test_settings_accepts_db_pool_environment(monkeypatch):
    monkeypatch.setenv(
        "MIGRATION_DATABASE_URL",
        "postgresql://user:pass@host:5432/db?sslmode=require&connection_limit=1",
    )
    monkeypatch.setenv("DB_CONNECTION_LIMIT", "3")
    monkeypatch.setenv("DB_CONNECTION_LIMIT_MAX", "5")
    monkeypatch.setenv("DB_MAX_CONCURRENT_QUERIES", "3")
    monkeypatch.setenv("DB_QUERY_MAX_RETRIES", "4")

    settings = Settings(_env_file=None)

    assert settings.migration_database_url.startswith("postgresql://")
    assert settings.db_connection_limit == 3
    assert settings.db_connection_limit_max == 5
    assert settings.db_max_concurrent_queries == 3
    assert settings.db_query_max_retries == 4


def test_prisma_local_engine_http_ignores_proxy_environment():
    assert db._http_config["trust_env"] is False
