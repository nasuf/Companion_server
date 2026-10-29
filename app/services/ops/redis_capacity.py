"""Bounded Redis sampling and PostgreSQL-backed capacity alert outbox.

The monitored Redis is never used for alert locking or state persistence.
Push provider acceptance is deliberately not described as human receipt.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.db import db
from app.redis_client import get_redis
from app.services.notifications.service import enqueue_notification
from app.services.ops import recovery

WARNING_PERCENT = 70
RECOVERY_PERCENT = 65
SAMPLE_SECONDS = 30
STALE_SECONDS = 120


async def collect_metrics() -> dict:
    sampled_at = datetime.now(UTC).isoformat()
    try:
        async with asyncio.timeout(3):
            redis = await get_redis()
            info = await redis.info()
            keys = await redis.dbsize()
            try:
                errors = await redis.info("errorstats")
                error_counts = {key.removeprefix("errorstat_"): value["count"] for key, value in errors.items()
                                if re.fullmatch(r"errorstat_[A-Za-z0-9_]{1,64}", key)
                                and isinstance(value, dict) and isinstance(value.get("count"), int)
                                and value["count"] >= 0}
                errorstats = {"status": "measured", "counts": dict(list(error_counts.items())[:64]),
                              "truncated": len(error_counts) > 64}
            except Exception as exc:
                errorstats = {"status": "unknown", "error_type": type(exc).__name__}
        used, maximum = info.get("used_memory"), info.get("maxmemory")
        valid = isinstance(used, int) and used >= 0 and isinstance(maximum, int) and maximum >= 0
        hits, misses = info.get("keyspace_hits"), info.get("keyspace_misses")
        total = hits + misses if isinstance(hits, int) and isinstance(misses, int) else 0
        return {
            "available": valid, "sampled_at": sampled_at,
            "capacity_status": "measured" if valid and maximum else "unlimited" if valid else "invalid",
            "used_memory": used, "maxmemory": maximum,
            "mem_percent": used / maximum * 100 if valid and maximum else None,
            "used_memory_rss": info.get("used_memory_rss"),
            "used_memory_peak": info.get("used_memory_peak"),
            "maxmemory_policy": info.get("maxmemory_policy"),
            "run_id": info.get("run_id"), "keys": keys,
            "connected_clients": info.get("connected_clients"),
            "hits": hits, "misses": misses,
            "hit_rate": round(hits / total * 100, 1) if total else None,
            "evicted_keys": info.get("evicted_keys"),
            "uptime_seconds": info.get("uptime_in_seconds"),
            "errorstats": errorstats,
        }
    except Exception as exc:
        return {"available": False, "capacity_status": "unavailable", "sampled_at": sampled_at,
                "mem_percent": None, "error_type": type(exc).__name__}


def level_for(sample: dict, previous: str) -> str:
    percent = sample.get("mem_percent")
    if not sample.get("available") or percent is None:
        return "unknown"
    if percent >= WARNING_PERCENT or (previous == "warning" and percent >= RECOVERY_PERCENT):
        return "warning"
    return "normal"


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def _timestamp(value) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def error_deltas(sample: dict, previous: dict, now: datetime) -> None:
    """Redis restart/reset starts a new baseline; never fabricate lost history."""
    errors = sample.get("errorstats") or {}
    baseline = previous.get("_error_baseline") or {}
    events = list(previous.get("error_events") or [])[-20:]
    if errors.get("status") == "measured" and not errors.get("truncated") and sample.get("run_id"):
        same_run = baseline.get("run_id") == sample["run_id"]
        deltas = {}
        if same_run:
            for kind, value in errors["counts"].items():
                old = baseline.get("counts", {}).get(kind, 0)
                if value > old:
                    deltas[kind] = value - old
        if deltas:
            events.append({"sampled_at": now.isoformat(), "since": baseline["sampled_at"], "counts": deltas})
        sample["_error_baseline"] = {"run_id": sample["run_id"], "counts": errors["counts"], "sampled_at": now.isoformat()}
        errors["delta_state"] = "measured" if same_run else "baseline_only"
    else:
        sample["_error_baseline"] = baseline
    sample["error_events"] = events[-20:]


async def sample_and_notify() -> None:
    """Each API worker calls this; the SQL row lock serializes sample/outbox commits."""
    sample = await collect_metrics()
    now = datetime.now(UTC)
    recipients = list(dict.fromkeys(x.strip() for x in settings.redis_memory_alert_user_ids.split(",") if x.strip()))
    if len(recipients) > 20:
        raise ValueError("Redis alert recipients must be at most 20 admins")
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await tx.execute_raw("SET LOCAL lock_timeout = '2s'")
        await tx.execute_raw("INSERT INTO ops_redis_capacity(id) VALUES ('primary') ON CONFLICT DO NOTHING")
        rows = await tx.query_raw("SELECT * FROM ops_redis_capacity WHERE id='primary' FOR UPDATE")
        row = rows[0]
        stamp = row.get("sampled_at")
        if stamp and (now - _timestamp(stamp)).total_seconds() < SAMPLE_SECONDS - 5:
            return
        admins = await tx.query_raw(
            "SELECT id FROM users WHERE id=ANY($1::text[]) AND role='admin' "
            "AND status='active' AND archived_at IS NULL ORDER BY id LIMIT 20", recipients,
        ) if recipients else []
        active_ids = [r["id"] for r in admins]
        previous = row["level"]
        previous_sample = _object(row.get("snapshot")) or {}
        error_deltas(sample, previous_sample, now)
        level = level_for(sample, previous)
        unknown = level == "unknown"
        # Do not declare recovery after failed/unknown sampling. Preserve the
        # warning latch until a real measurement drops below the recovery band.
        if level == "unknown" and previous == "warning":
            level = "warning"
        old_signature = row.get("notified_signature")
        state = _object(old_signature) or {"level": "normal", "checks": []}
        # Recipient changes are configuration, not incidents or recovery evidence.
        state.pop("recipients", None)
        unknown_streak = int(previous_sample.get("unknown_streak", 0)) + 1 if unknown else 0
        sample["unknown_streak"] = unknown_streak
        recovered, next_state = recovery.advance(level, state, now, interval=SAMPLE_SECONDS)
        notify = recovered
        if level == "warning" and not unknown:
            notify = state.get("level") != "warning"
            next_state = {"level": "warning", "checks": []}
        elif level == "unknown" and unknown_streak >= settings.ops_unknown_samples and state.get("level") == "normal":
            notify, next_state = True, {"level": "unknown", "checks": []}
        notify = notify and bool(active_ids)
        if not active_ids:
            next_state = state
        ids = _object(row["notification_ids"]) or []
        generation = int(row["generation"])
        if notify:
            generation += 1
            ids = []
            title = "Redis 内存预警" if level == "warning" else "Redis 采样异常" if level == "unknown" else "Redis 内存告警恢复"
            percent = sample.get("mem_percent")
            body = (f"内存使用 {percent:.1f}%，预警阈值70%，恢复阈值65%。详情见Web后台资源监控。"
                    if percent is not None else "无法确认Redis内存占用，请检查Web后台资源监控。")
            for uid in active_ids:
                event_id = await enqueue_notification(
                    user_id=uid, type="admin_redis_capacity", title=title, body=body,
                    payload={"type": "admin_redis_capacity", "route": "system", "title": title, "body": body},
                    dedupe_key=f"redis-capacity:{generation}", dispatch_now=False, database=tx,
                )
                if event_id:
                    ids.append(event_id)
        await tx.execute_raw(
            "UPDATE ops_redis_capacity SET sampled_at=$1::timestamptz, snapshot=$2::jsonb, level=$3, "
            "notified_signature=$4, generation=$5, notification_ids=$6::jsonb, recipient_count=$7 WHERE id='primary'",
            now.isoformat(), json.dumps(sample), level, json.dumps(next_state, sort_keys=True),
            generation, json.dumps(ids), len(active_ids),
        )


async def read_status() -> dict:
    base = {"warning_percent": WARNING_PERCENT, "recovery_percent": RECOVERY_PERCENT,
            "recovery_seconds": settings.ops_recovery_seconds, "recovery_samples": settings.ops_recovery_samples,
            "unknown_samples": settings.ops_unknown_samples,
            "sample_interval_seconds": SAMPLE_SECONDS, "stale_after_seconds": STALE_SECONDS,
            "human_receipt": "not_collected"}
    try:
        async with asyncio.timeout(3):
            rows = await db.query_raw("SELECT * FROM ops_redis_capacity WHERE id='primary'")
            if not rows or not rows[0].get("sampled_at"):
                return {**base, "status": "not_sampled", "notifications": []}
            row = rows[0]
            ids = (_object(row["notification_ids"]) or [])[:20]
            events = await db.query_raw(
                "SELECT CASE WHEN status='sent' AND NULLIF(error,'') IS NOT NULL "
                "THEN 'partial' ELSE status END AS status, COUNT(*)::int AS count FROM notification_events "
                "WHERE id=ANY($1::text[]) GROUP BY 1 LIMIT 10", ids,
            ) if ids else []
        stamp = _timestamp(row["sampled_at"])
        snapshot = _object(row["snapshot"])
        snapshot.pop("_error_baseline", None)
        stale = (datetime.now(UTC) - stamp).total_seconds() > STALE_SECONDS
        return {**base, "status": "stale" if stale else "active" if row["recipient_count"] else "unconfigured",
                "level": row["level"], "sampled_at": stamp.isoformat(), "sample": snapshot,
                "recipient_count": row["recipient_count"], "notifications": events,
                "recovery_pending": (_object(row.get("notified_signature")) or {}).get("recovery"),
                "missing_notification_count": max(0, len(ids) - sum(e["count"] for e in events))}
    except Exception as exc:
        return {**base, "status": "unavailable", "error_type": type(exc).__name__, "notifications": []}
