"""Validate explicit disposable resources before importing application modules."""

from __future__ import annotations

from contextlib import contextmanager
import ipaddress
import os
import re
import socket
from urllib.parse import unquote, urlsplit
from unittest.mock import patch


def isolated_url(url: str, kind: str) -> str:
    """No implicit settings/.env, hostname resolution, tunnels or default DBs.

    A loopback address alone is insufficient: require the dedicated database
    namespace as well. Operators must create it with Prisma migrations.
    """
    parsed = urlsplit(url)
    allowed = {"postgres": {"postgresql", "postgres"}, "redis": {"redis"}}
    if kind not in allowed or parsed.scheme not in allowed[kind]:
        raise ValueError("Explicit isolated PostgreSQL/Redis URL required")
    if (
        parsed.hostname not in {"127.0.0.1", "::1"}
        or not parsed.port
        or parsed.fragment
    ):
        raise ValueError(
            "Evaluation resources must use literal loopback and an explicit port"
        )
    if parsed.query:
        raise ValueError("Evaluation resource URL options are forbidden")
    if kind == "postgres":
        if not re.fullmatch(
            r"/companion_memory_eval_[a-z0-9_]{1,48}", unquote(parsed.path)
        ):
            raise ValueError(
                "Evaluation requires a disposable companion_memory_eval_* database"
            )
    elif not re.fullmatch(r"/(?:1[0-5])", parsed.path):
        raise ValueError("Evaluation Redis must use an explicit isolated DB 10..15")
    return url


def configure_isolation(database_url: str, redis_url: str) -> None:
    """Call only in a dedicated evaluation process, before any app imports."""
    import sys

    if any(m == "app.config" or m == "app.db" for m in sys.modules):
        raise RuntimeError(
            "Start a fresh process: application configuration already loaded"
        )
    isolated_url(database_url, "postgres")
    isolated_url(redis_url, "redis")
    # Do not inherit a production provider, proxy, callback or telemetry key.
    for key in tuple(os.environ):
        if key.endswith(
            ("_KEY", "TOKEN", "PROXY", "_PASSWORD", "_SECRET")
        ) or key.startswith(
            (
                "DASHSCOPE_",
                "ARK_",
                "ANTHROPIC_",
                "AXIOM_",
                "LANGSMITH_",
                "LANGCHAIN_",
                "WECHAT_",
                "WEAPP_",
                "APNS_",
                "SMS_",
                "IAP_",
                "PAYMENT_",
                "MEM0_",
                "MEMORY_EVAL_",
            )
        ):
            os.environ.pop(key, None)
    os.environ.update(
        APP_ENV="test",
        PYTHON_DOTENV_DISABLED="1",
        DATABASE_URL=database_url,
        DIRECT_DATABASE_URL=database_url,
        MIGRATION_DATABASE_URL=database_url,
        REDIS_URL=redis_url,
        TRACE_BACKEND="off",
        ONLINE_MODEL="false",
        JWT_SECRET="synthetic-memory-evaluation-only",
        APNS_ENABLED="false",
        CHAT_EXECUTOR="legacy",
        TTS_OUTPUT_PROBABILITY="0",
    )
    # PYTHON_DOTENV_DISABLED does not disable pydantic-settings' own reader.
    # Otherwise deleted credentials would be silently reloaded from .env.
    from pydantic_settings.sources import DotEnvSettingsSource

    DotEnvSettingsSource._read_env_files = lambda self: {}


@contextmanager
def loopback_network_fence():
    """Application evaluation may only reach local synthetic sidecars.

    Remote model evaluation is a separate, model-only process using the existing
    graph evaluation's HTTPS allowlist fence. No simultaneous business/model IO.
    """
    violations: list[str] = []

    def guard(original):
        def connect(sock, address):
            if isinstance(address, tuple):
                try:
                    valid = ipaddress.ip_address(address[0]).is_loopback
                except ValueError:
                    valid = False
                if not valid:
                    violations.append("non-loopback socket")
                    raise RuntimeError("Evaluation blocked a non-loopback socket")
            elif sock.family != socket.AF_UNIX:
                violations.append("unknown socket")
                raise RuntimeError("Evaluation blocked an unknown socket")
            return original(sock, address)

        return connect

    with (
        patch.object(socket.socket, "connect", guard(socket.socket.connect)),
        patch.object(socket.socket, "connect_ex", guard(socket.socket.connect_ex)),
    ):
        yield violations
