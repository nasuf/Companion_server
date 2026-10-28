"""Observed provider billing failures; never infer balances from a healthy call.

Only allowlisted API codes are retained. Failures are persisted before failover
(with a one-second bound); healthy traffic does no SQL. Workers refresh the small
active set through the existing service-health job. Recovery requires a complete
success on the SAME route, started after the latest failure. A fallback success,
old in-flight success, empty sampling window or restart cannot clear an incident.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta

from app.db import db

logger = logging.getLogger(__name__)
# Official provider error codes; generic HTTP 429/RPM/TPM limits are not billing.
CODES = {
    "ark": {"AccountOverdueError": "overdue", "OperationDenied.ServiceOverdue": "overdue",
            "SetLimitExceeded": "quota_exhausted"},
    "dashscope": {"Arrearage": "overdue", "insufficient_quota": "quota_exhausted",
                  "AllocationQuota.FreeTierOnly": "quota_exhausted"},
    # DeepSeek documents HTTP 402 as insufficient balance (its JSON code may
    # be the generic invalid_request_error, which must NOT be matched alone).
    "deepseek": {"HTTP402": "overdue"},
}
LIMIT = 64
_active: set[tuple[str, str]] = set()
_pending: dict[tuple[str, str], tuple[str, datetime, datetime]] = {}
_overflow = False


async def observed_call(factory, provider: str, model: str):
    started_at = datetime.now(UTC)
    try:
        response = await factory()
    except Exception as exc:
        await record_failure(provider, model, exc)
        raise
    await record_success(provider, model, started_at)
    return response


async def observed_stream(factory, provider: str, model: str):
    started_at = datetime.now(UTC)
    stream = None
    try:
        stream = factory()
        async for chunk in stream:
            yield chunk
    except Exception as exc:
        await record_failure(provider, model, exc)
        raise
    else:
        await record_success(provider, model, started_at)
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def classify(provider: str, exc: Exception) -> str | None:
    """Inspect structured error codes, never echo exception text or request data."""
    for _ in range(3):
        if provider == "deepseek" and getattr(exc, "status_code", None) == 402:
            return "HTTP402"
        codes = [getattr(exc, "code", None)]
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            codes += [body.get("code"), body.get("type")]
            inner = body.get("error")
            if isinstance(inner, dict):
                codes += [inner.get("code"), inner.get("type")]
        for code in codes:
            if isinstance(code, str) and code in CODES.get(provider, {}):
                return code
        exc = exc.__cause__
        if exc is None:
            break
    return None


def route(provider: str, model: str) -> tuple[str, str] | None:
    if model.startswith(provider + "/"):
        model = model[len(provider) + 1:]
    if provider not in CODES or not re.fullmatch(r"[a-zA-Z0-9._:-]{1,160}", model):
        return None
    return provider, model


async def _persist(key: tuple[str, str], event: tuple[str, datetime, datetime]) -> None:
    code, first_at, last_at = event
    async with asyncio.timeout(1):
        async with db.tx(timeout=timedelta(seconds=1)) as tx:
            await tx.execute_raw("SET LOCAL statement_timeout='700ms'")
            await tx.execute_raw(
                """INSERT INTO llm_provider_incidents(provider,model,code,first_failure_at,last_failure_at)
                VALUES($1,$2,$3,$4::timestamptz,$5::timestamptz)
                ON CONFLICT(provider,model) DO UPDATE SET
                  code=EXCLUDED.code,
                  first_failure_at=CASE WHEN llm_provider_incidents.last_success_started_at > llm_provider_incidents.last_failure_at
                    AND EXCLUDED.last_failure_at > llm_provider_incidents.last_success_started_at
                    THEN EXCLUDED.first_failure_at ELSE LEAST(llm_provider_incidents.first_failure_at,EXCLUDED.first_failure_at) END,
                  last_failure_at=EXCLUDED.last_failure_at
                WHERE EXCLUDED.last_failure_at > llm_provider_incidents.last_failure_at""",
                *key, code, first_at.isoformat(), last_at.isoformat(),
            )
    if _pending.get(key) == event:
        _pending.pop(key, None)


async def record_failure(provider: str, model: str, exc: Exception) -> None:
    global _overflow
    code, key = classify(provider, exc), route(provider, model)
    if not code or not key:
        return
    at = datetime.now(UTC)
    if key not in _pending and len(_pending) >= LIMIT:
        _overflow = True
        logger.error("[LLM-BUDGET] pending observation limit reached; monitoring incomplete")
        return
    _active.add(key)
    previous = _pending.get(key)
    event = (code, previous[1] if previous else at, at)
    _pending[key] = event
    try:
        await _persist(key, event)
    except Exception as error:
        logger.error("[LLM-BUDGET] incident persistence failed type=%s", type(error).__name__)


async def record_success(provider: str, model: str, started_at: datetime) -> None:
    key = route(provider, model)
    if not key or key not in _active or key in _pending:
        return
    try:
        async with asyncio.timeout(1):
            await db.execute_raw(
                """UPDATE llm_provider_incidents SET last_success_started_at=$3::timestamptz
                WHERE provider=$1 AND model=$2 AND last_failure_at < $3::timestamptz
                  AND (last_success_started_at IS NULL OR last_success_started_at < $3::timestamptz)""",
                *key, started_at.isoformat(),
            )
        # Do not clear the local set here: another worker may have just failed.
        # The next SQL refresh reconciles it; redundant recovery updates are safe.
    except Exception as error:
        logger.error("[LLM-BUDGET] recovery persistence failed type=%s", type(error).__name__)


async def refresh_active() -> None:
    """Every worker, outside the globally deduplicated service-health sample."""
    for key, event in list(_pending.items())[:4]:
        try:
            await _persist(key, event)
        except Exception:
            break  # bounded retry next scan; leave evidence pending
    async with asyncio.timeout(2):
        rows = await db.query_raw(
            "SELECT provider,model FROM llm_provider_incidents "
            "WHERE last_success_started_at IS NULL OR last_success_started_at <= last_failure_at "
            "ORDER BY last_failure_at DESC LIMIT 65")
    _active.clear()
    _active.update((r["provider"], r["model"]) for r in rows)
    _active.update(_pending)


async def measure(now: datetime, carry: dict) -> dict:
    rows = await db.query_raw(
        """SELECT provider,model,code,first_failure_at,last_failure_at FROM llm_provider_incidents
        WHERE last_success_started_at IS NULL OR last_success_started_at <= last_failure_at
        ORDER BY last_failure_at DESC LIMIT 21""")
    def stamp(value):
        return value.isoformat() if isinstance(value, datetime) else value
    incidents = [{**r, "reason": CODES.get(r["provider"], {}).get(r["code"], "unknown"),
                  "first_failure_at": stamp(r["first_failure_at"]),
                  "last_failure_at": stamp(r["last_failure_at"])} for r in rows[:20]]
    incomplete = bool(_pending or _overflow or len(rows) > 20)
    return {"status": "unknown" if incomplete else "measured", "count": len(rows), "threshold": 1,
            "alerting": bool(rows or _pending or _overflow), "incidents": incidents,
            "truncated": len(rows) > 20, "pending_observations": len(_pending),
            "balance_status": "not_collected", "recovery": "same_route_later_complete_success"}
