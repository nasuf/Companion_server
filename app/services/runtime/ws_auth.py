"""Mandatory single-use WebSocket credentials shared across workers."""

import hashlib
import json
import math
import re
import secrets
import time
from dataclasses import dataclass

from app.config import settings
from app.redis_client import get_redis

CHAT_PROTOCOL = "companion.chat.v1"
TICKET_PROTOCOL_PREFIX = "companion.ticket."
TICKET_TTL_SECONDS = 60
_TICKET_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_RATE_LIMIT = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], 60) end
return n
"""


class TicketInvalid(Exception):
    pass


class TicketUnavailable(Exception):
    pass


class TicketRateLimited(Exception):
    pass


@dataclass(frozen=True)
class SocketPrincipal:
    user_id: str
    role: str
    expires_at: float


def _key(ticket: str) -> str:
    return "ws:ticket:" + hashlib.sha256(ticket.encode()).hexdigest()


def _claims(payload: dict) -> SocketPrincipal:
    sub, role, exp = payload.get("sub"), payload.get("role"), payload.get("exp")
    if (not isinstance(sub, str) or not sub or not isinstance(role, str) or role not in {"user", "admin"}
            or not isinstance(exp, (int, float)) or isinstance(exp, bool)
            or not math.isfinite(exp) or exp <= time.time()):
        raise TicketInvalid()
    return SocketPrincipal(sub, role, float(exp))


async def issue_ticket(conversation_id: str, payload: dict) -> dict:
    principal = _claims(payload)
    now = time.time()
    expires_at = min(now + TICKET_TTL_SECONDS, principal.expires_at)
    ticket = secrets.token_urlsafe(32)
    value = json.dumps({"conversation_id": conversation_id, "sub": principal.user_id,
                        "role": principal.role, "exp": principal.expires_at,
                        "ticket_exp": expires_at})
    try:
        redis = await get_redis()
        rate_key = "ws:ticket-rate:" + hashlib.sha256(principal.user_id.encode()).hexdigest()
        if await redis.eval(_RATE_LIMIT, 1, rate_key) > 60:
            raise TicketRateLimited()
        if not await redis.set(_key(ticket), value, ex=max(1, math.ceil(expires_at-now)), nx=True):
            raise TicketUnavailable()
    except (TicketRateLimited, TicketUnavailable):
        raise
    except Exception:
        raise TicketUnavailable() from None
    return {"ticket": ticket, "expires_in": max(1, math.ceil(expires_at-now)),
            "protocol": CHAT_PROTOCOL}


async def consume_ticket(ticket: str, conversation_id: str) -> SocketPrincipal:
    if not isinstance(ticket, str) or not _TICKET_PATTERN.fullmatch(ticket):
        raise TicketInvalid()
    try:
        redis = await get_redis()
        value = await redis.getdel(_key(ticket))  # Single atomic operation across workers.
    except Exception:
        raise TicketUnavailable() from None
    try:
        data = json.loads(value) if value else None
        if (not isinstance(data, dict) or data.get("conversation_id") != conversation_id
                or not isinstance(data.get("ticket_exp"), (int, float))
                or not math.isfinite(data["ticket_exp"]) or data["ticket_exp"] <= time.time()):
            raise TicketInvalid()
        return _claims(data)
    except (ValueError, TypeError, KeyError):
        raise TicketInvalid() from None


def offered_ticket(protocols: list[str]) -> str:
    credentials = [p[len(TICKET_PROTOCOL_PREFIX):] for p in protocols
                   if p.startswith(TICKET_PROTOCOL_PREFIX)]
    if len(credentials) != 1 or CHAT_PROTOCOL not in protocols:
        raise TicketInvalid()
    return credentials[0]


def origin_allowed(origin: str | None) -> bool:
    # Native clients omit Origin. Origin is an extra browser check, never identity.
    if origin is None:
        return True
    allowed = {x.rstrip("/") for x in settings.cors_origins()}
    allowed.update(x.strip().rstrip("/") for x in settings.ws_allowed_origins.split(",") if x.strip())
    return origin.rstrip("/") in allowed or ("*" in allowed and not settings.is_production())
