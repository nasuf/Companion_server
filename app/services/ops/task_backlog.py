"""Read-only bounded backlog inspection, retaining exact recovery identities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from app.config import settings
from app.services.runtime.job_queue import read_monitor_jobs

# SQL identifiers come exclusively from this registry, never from request data.
TABLES = {
    "runtime_jobs": (
        "id",
        "available_at",
        "running",
        ("pending", "retry"),
        ("succeeded", "cancelled"),
    ),
    "runtime_outbox": (
        "id",
        "available_at",
        "delivering",
        ("pending",),
        ("delivered", "cancelled"),
    ),
    "message_followup_events": (
        "message_id",
        "next_dispatch_at",
        None,
        ("pending_dispatch", "queued"),
        ("completed", "cancelled"),
    ),
    "turn_followup_events": (
        "turn_id",
        "next_dispatch_at",
        None,
        ("pending_dispatch", "queued"),
        ("completed", "cancelled"),
    ),
    "offline_arrival_followups": (
        "activity_id",
        "next_dispatch_at",
        None,
        ("pending_dispatch", "queued"),
        ("completed", "cancelled"),
    ),
}


async def measure(now: datetime, carry: dict) -> dict:
    from app.services.ops.service_health import _query

    limit = settings.ops_backlog_page_size
    cutoff = now - timedelta(seconds=settings.ops_backlog_seconds)

    async def sql_table(table, spec):
        identity, due, running, pending, terminal = spec
        overdue = f"status=ANY($2::text[]) AND {due}<$1::timestamptz"
        if running:
            overdue += f" OR (status='{running}' AND lease_expires_at<$1::timestamptz)"
        old = (carry.get("ids", {}).get(table) or [])[:limit]
        rows = await _query(
            f"SELECT {identity} AS id,status FROM {table} WHERE ({overdue}) "
            f"ORDER BY {due},{identity} LIMIT $3",
            cutoff.isoformat(),
            list(pending),
            limit + 1,
        )
        previous = (
            await _query(
                f"SELECT {identity} AS id,status FROM {table} "
                f"WHERE {identity}=ANY($1::text[])",
                old,
            )
            if old
            else []
        )
        missing = len(previous) != len(old)
        unresolved = [r["id"] for r in previous if r["status"] not in terminal]
        identities = list(dict.fromkeys(unresolved + [r["id"] for r in rows]))
        return table, {
            "count": len(identities),
            "truncated": len(rows) > limit or len(identities) > limit,
            "missing": missing,
            "ids": identities[:limit],
        }

    results = dict(
        await asyncio.gather(*(sql_table(t, spec) for t, spec in TABLES.items()))
    )
    redis_old = (carry.get("ids", {}).get("redis_runtime") or [])[:limit]
    redis = await read_monitor_jobs(limit=limit, carried=redis_old)
    identities, ages, missing = [], [], False
    for job in redis["items"]:
        if job["missing"]:
            missing = True
            continue
        status = job["status"]
        due = (
            job.get("lease_expires_at")
            if status == "running"
            else job.get("not_before") or job.get("created_at")
        )
        # Queue timestamps are epoch seconds; absent eligibility is unknown.
        if status in {"running", "queued"}:
            try:
                age = now.timestamp() - float(due)
            except (TypeError, ValueError, OverflowError):
                missing = True
                continue
            if age > settings.ops_backlog_seconds:
                identities.append(job["id"])
                ages.append(age)
        if job["id"] in redis_old and status not in {
            "succeeded",
            "resolved",
            "cancelled",
        }:
            identities.append(job["id"])
    identities = list(dict.fromkeys(identities))
    results["redis_runtime"] = {
        "count": len(identities),
        "truncated": redis["truncated"] or len(identities) > limit,
        "missing": missing,
        "ids": identities[:limit],
        "oldest_wait_seconds": max(ages, default=0),
    }
    incomplete = any(r["missing"] or r["truncated"] for r in results.values())
    # Preserve old missing identities, so a later readable sample can resolve them.
    next_ids = {
        t: list(dict.fromkeys((carry.get("ids", {}).get(t) or []) + r["ids"]))[:limit]
        if r["missing"]
        else r["ids"]
        for t, r in results.items()
    }
    count = sum(r["count"] for r in results.values())
    return {
        "status": "unknown" if incomplete else "measured",
        "count": count,
        "threshold": 1,
        "after_seconds": settings.ops_backlog_seconds,
        "sample_limit_per_source": limit,
        "count_is_lower_bound": incomplete,
        "alerting": count > 0,
        "sources": {
            t: {k: v for k, v in r.items() if k != "ids"} for t, r in results.items()
        },
        "_carry": {"ids": next_ids},
    }
