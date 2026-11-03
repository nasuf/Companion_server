"""Collection of network and compose-timing signals. Every writer is best effort.

Never stored: the raw IP address or any draft/composer text.
- IP: keyed hash (HMAC with the server secret) to match accounts on one address,
  plus the /24 (IPv6 /48) prefix and offline region/ISP.
- Compose timing: the first typing ping after a send starts a clock in Redis;
  the next persisted user message takes it. The app may also report its own
  compose time, largest single insertion (a paste) and edit count.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from app.services.authenticity.ip_region import lookup, network_kind

logger = logging.getLogger(__name__)

COMPOSE_TTL_SECONDS = 600
_CONVERSATION_ID = re.compile(r"[0-9a-fA-F-]{32,36}")
RETENTION_DAYS = 180
_MAX_COMPOSE_MS = COMPOSE_TTL_SECONDS * 1000
_METRIC_LIMITS = {"compose_ms": _MAX_COMPOSE_MS, "max_insert_chars": 100_000, "edit_count": 100_000}


def client_ip(headers: Mapping[str, str], fallback: str | None) -> str | None:
    """nginx overwrites X-Real-IP with the TCP peer on every proxied location.

    X-Forwarded-For is not used: its first entry comes from the client.
    """
    for value in (headers.get("x-real-ip"), fallback):
        value = (value or "").strip()
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            continue
    return None


def _ip_key() -> bytes:
    from app.config import settings

    return hmac.new(settings.jwt_secret.encode(), b"user-network-observation-v1", hashlib.sha256).digest()


def ip_hash(ip: str) -> str:
    return hmac.new(_ip_key(), ip.encode(), hashlib.sha256).hexdigest()[:32]


def ip_prefix(ip: str) -> str:
    address = ipaddress.ip_address(ip)
    bits = 24 if address.version == 4 else 48
    return str(ipaddress.ip_network(f"{address}/{bits}", strict=False))


def client_kind(user_agent: str | None) -> str:
    agent = user_agent or ""
    if agent.startswith("Dart/"):
        return "app"
    if agent.startswith("Mozilla/"):
        return "browser"
    return "other"


def _naive_utc(at: datetime) -> datetime:
    return at.astimezone(timezone.utc).replace(tzinfo=None) if at.tzinfo else at


async def record_network(user_id: str, ip: str, user_agent: str | None, *, source: str,
                         seen_at: datetime | None = None, first_seen_at: datetime | None = None,
                         hits: int = 1, merge: bool = True, database=None) -> None:
    """merge=False keeps an existing row untouched (re-runnable backfill)."""
    from app.db import db

    # A private address is the proxy or Docker gateway, shared by every user:
    # recording it would link all accounts to one "IP".
    if not ipaddress.ip_address(ip).is_global:
        return
    region = lookup(ip)
    last = _naive_utc(seen_at or datetime.now(timezone.utc))
    first = _naive_utc(first_seen_at) if first_seen_at else last
    # INSERT ... SELECT skips an account being erased; the FK cascades the rest.
    await (database or db).execute_raw(
        """INSERT INTO user_network_observations (user_id, ip_hash, ip_prefix, country, province,
             city, isp, network_kind, client_kind, user_agent, source, hits, first_seen_at, last_seen_at)
           SELECT id, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::timestamp, $14::timestamp
           FROM users WHERE id=$1 AND status<>'deleting'
           ON CONFLICT (user_id, ip_hash) DO """ + ("""UPDATE SET
             hits = user_network_observations.hits + EXCLUDED.hits,
             first_seen_at = LEAST(user_network_observations.first_seen_at, EXCLUDED.first_seen_at),
             last_seen_at = GREATEST(user_network_observations.last_seen_at, EXCLUDED.last_seen_at),
             client_kind = EXCLUDED.client_kind, user_agent = EXCLUDED.user_agent,
             source = EXCLUDED.source""" if merge else "NOTHING"),
        user_id, ip_hash(ip), ip_prefix(ip), region.country, region.province, region.city,
        region.isp, network_kind(region), client_kind(user_agent),
        (user_agent or "")[:200] or None, source, hits, first.isoformat(), last.isoformat(),
    )


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def observe_network(user_id: str | None, headers: Mapping[str, str], peer: str | None,
                    *, source: str) -> None:
    """Fire-and-forget network observation for an authenticated user request."""
    ip = client_ip(headers, peer)
    if not user_id or not ip:
        return
    if not _loop_running():
        return
    from app.services.runtime.tasks import fire_background

    fire_background(_observe(user_id, ip, headers.get("user-agent"), source))


async def _observe(user_id: str, ip: str, user_agent: str | None, source: str) -> None:
    try:
        await record_network(user_id, ip, user_agent, source=source)
    except Exception as exc:
        logger.warning("network observation failed: %s", type(exc).__name__)


def _compose_key(user_id: str, conversation_id: str) -> str:
    return f"authn:compose:{user_id}:{conversation_id}"


async def mark_compose_started(user_id: str, conversation_id: str) -> None:
    """First typing ping since the last send starts the clock; heartbeats keep it."""
    if not _CONVERSATION_ID.fullmatch(conversation_id):
        return  # client-supplied; keep the key space to real conversation IDs
    try:
        from app.redis_client import get_redis

        redis = await get_redis()
        await redis.set(_compose_key(user_id, conversation_id), str(int(time.time() * 1000)),
                        nx=True, ex=COMPOSE_TTL_SECONDS)
    except Exception as exc:
        logger.warning("compose clock unavailable: %s", type(exc).__name__)


async def _take_compose_ms(user_id: str, conversation_id: str, received_at: datetime) -> int | None:
    from app.redis_client import get_redis

    redis = await get_redis()
    key = _compose_key(user_id, conversation_id)
    raw = await redis.get(key)
    if raw is None:
        return None
    received_ms = int(received_at.timestamp() * 1000)
    started_ms = int(raw)
    if started_ms > received_ms:
        return None  # the clock belongs to the next message being typed
    await redis.delete(key)
    elapsed = received_ms - started_ms
    return elapsed if elapsed <= _MAX_COMPOSE_MS else None


def client_input_metrics(raw: Any) -> dict[str, int]:
    """Bounded integers from the app; anything else is dropped."""
    if not isinstance(raw, dict):
        return {}
    metrics = {}
    for key, limit in _METRIC_LIMITS.items():
        value = raw.get(key)
        if type(value) is int and 0 <= value <= limit:
            metrics[key] = value
    return metrics


async def record_input(message_id: str, user_id: str, conversation_id: str, text: str,
                       received_at: datetime, raw_metrics: Any = None) -> None:
    from app.db import db

    try:
        try:
            server_ms = await _take_compose_ms(user_id, conversation_id, received_at)
        except Exception:
            server_ms = None
        metrics = client_input_metrics(raw_metrics)
        await db.execute_raw(
            """INSERT INTO user_input_signals (message_id, user_id, created_at, chars,
                 server_compose_ms, client_compose_ms, max_insert_chars, edit_count, source)
               SELECT $1, id, $3::timestamp, $4, $5, $6, $7, $8, 'live'
               FROM users WHERE id=$2 AND status<>'deleting'
               ON CONFLICT (message_id) DO NOTHING""",
            message_id, user_id, _naive_utc(received_at).isoformat(), len(text.strip()), server_ms,
            metrics.get("compose_ms"), metrics.get("max_insert_chars"), metrics.get("edit_count"),
        )
    except Exception as exc:
        logger.warning("input signal failed: %s", type(exc).__name__)


def observe_input(message_id: str | None, user_id: str, conversation_id: str, text: str,
                  received_at: datetime, raw_metrics: Any = None) -> None:
    if not message_id or not text.strip():
        return
    if not _loop_running():
        return
    from app.services.runtime.tasks import fire_background

    fire_background(record_input(message_id, user_id, conversation_id, text,
                                 received_at, raw_metrics))


async def purge_expired(now: datetime | None = None) -> int:
    from datetime import timedelta

    from app.db import db

    cutoff = _naive_utc((now or datetime.now(timezone.utc)) - timedelta(days=RETENTION_DAYS)).isoformat()
    networks = await db.execute_raw(
        "DELETE FROM user_network_observations WHERE last_seen_at < $1::timestamp", cutoff)
    inputs = await db.execute_raw(
        "DELETE FROM user_input_signals WHERE created_at < $1::timestamp", cutoff)
    return int(networks or 0) + int(inputs or 0)
