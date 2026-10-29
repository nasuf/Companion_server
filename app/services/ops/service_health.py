"""Service-level checks pushed to administrators: replies, failures, memory, cron.

Every check reads bounded, recent rows. A check that cannot be measured is
"unknown", never zero, and an unknown sample never declares recovery. Recovery
needs evidence, not an emptied window: the messages that raised the alert got
answers, a reply succeeded after the last failure. One incident pushes once;
only a newly failing check or a level change pushes again. Push provider
acceptance is not described as human receipt.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.db import db
from app.services.notifications.service import enqueue_notification
from app.services.ops import provider_budget, provider_failing

SAMPLE_SECONDS = 120
STALE_SECONDS = 600
UNKNOWN_STREAK_TO_NOTIFY = 3  # ~6 min: a single slow query is not an outage

# Calibrated on production 2026-10-27: over 14 days, every user message left
# unanswered beyond 3 minutes was a human-takeover message; 7 of 2,092 replies
# in 30 days carried reply_failed.
UNANSWERED_AFTER = timedelta(minutes=3)
UNANSWERED_WINDOW_SPAN = timedelta(minutes=27)
UNANSWERED_THRESHOLD = 2
CARRY_LIMIT = 20
FAILED_WINDOW = timedelta(minutes=15)
FAILED_THRESHOLD = 2
# A failed batch is retried by the conversation's next memory run, so age alone
# only means the user left. Memory is blocked when the user keeps talking.
MEMORY_BLOCKED_AFTER = timedelta(minutes=30)
MEMORY_BLOCKED_LATER_MESSAGES = 3
MEMORY_BLOCKED_THRESHOLD = 1
CRON_FAILING_THRESHOLD = 1

LABELS = {
    "unanswered": "用户消息未回复",
    "reply_failed": "回复生成失败",
    "memory_blocked": "记忆写入受阻",
    "cron_failing": "定时任务失败",
    "provider_failing": "模型供应商持续失败",
    "provider_budget": "模型供应商欠费或额度耗尽",
}


async def _query(sql: str, *args) -> list[dict]:
    async with db.tx(timeout=timedelta(seconds=5)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout = '3s'")
        return await tx.query_raw(sql, *args)


def _naive_utc(at: datetime) -> str:
    # messages.created_at is UTC without a zone; never rely on the session zone.
    return at.astimezone(UTC).replace(tzinfo=None).isoformat()


def unanswered_after() -> timedelta:
    """Intentional reply delays (admin-tunable up to an hour) are not outages."""
    try:
        from app.services.runtime_config import resolve_config_sync
        config = resolve_config_sync()
        if config.reply_delay_enabled:
            return max(UNANSWERED_AFTER, timedelta(seconds=config.reply_delay_max_seconds) + timedelta(minutes=2))
    except Exception:
        pass
    return UNANSWERED_AFTER


_ACTIVE_UNANSWERED = """FROM messages m JOIN conversations c ON c.id=m.conversation_id
    WHERE m.role='user' AND NOT c.is_deleted AND c.archived_at IS NULL
      AND NOT (COALESCE(m.metadata,'{}'::jsonb) ? 'human_takeover')
      AND NOT EXISTS (SELECT 1 FROM messages a WHERE a.conversation_id=m.conversation_id
                      AND a.role='assistant' AND a.created_at > m.created_at)"""


async def _unanswered(now: datetime, carry: dict) -> dict:
    after = unanswered_after()
    recent = await _query(
        f"""SELECT m.id {_ACTIVE_UNANSWERED} AND m.created_at >= $1::timestamp AND m.created_at < $2::timestamp
        ORDER BY m.created_at LIMIT 50""",
        _naive_utc(now - after - UNANSWERED_WINDOW_SPAN), _naive_utc(now - after))
    # Messages that raised the alert keep counting after the window moves on,
    # until each gets an answer (or its conversation is deleted/archived).
    carried_ids = carry.get("unanswered_ids") or []
    still = await _query(f"SELECT m.id {_ACTIVE_UNANSWERED} AND m.id=ANY($1::text[])",
                         carried_ids) if carried_ids else []
    ids = list(dict.fromkeys([r["id"] for r in recent] + [r["id"] for r in still]))
    alerting = len(ids) >= UNANSWERED_THRESHOLD or bool(still)
    return {"count": len(ids), "threshold": UNANSWERED_THRESHOLD, "carried": len(still),
            "after_minutes": int(after.total_seconds() // 60), "alerting": alerting,
            "_carry": {"unanswered_ids": ids[:CARRY_LIMIT] if alerting else []}}


async def _reply_failed(now: datetime, carry: dict) -> dict:
    rows = await _query(
        """SELECT count(*) FILTER (WHERE metadata->>'reply_failed'='true')::int AS failed,
                  count(*)::int AS replies,
                  max(created_at) FILTER (WHERE metadata->>'reply_failed'='true') AS last_failed
        FROM messages WHERE role='assistant' AND created_at >= $1::timestamp""",
        _naive_utc(now - FAILED_WINDOW))
    row = rows[0]
    last_failed = row["last_failed"]
    last_failed = last_failed.isoformat() if isinstance(last_failed, datetime) else last_failed
    awaiting = False
    if carry.get("last_failed_at") and not row["failed"] >= FAILED_THRESHOLD:
        # Recovery needs a reply that succeeded after the last failure.
        ok = await _query(
            """SELECT EXISTS(SELECT 1 FROM messages WHERE role='assistant' AND created_at > $1::timestamp
                AND COALESCE(metadata->>'reply_failed','') <> 'true') AS ok""",
            max(carry["last_failed_at"], last_failed or ""))
        awaiting = not ok[0]["ok"]
    alerting = row["failed"] >= FAILED_THRESHOLD or awaiting
    return {"count": row["failed"], "replies": row["replies"], "threshold": FAILED_THRESHOLD,
            "window_minutes": int(FAILED_WINDOW.total_seconds() // 60), "awaiting_success": awaiting,
            "alerting": alerting,
            "_carry": {"last_failed_at": (last_failed or carry.get("last_failed_at")) if alerting else None}}


async def _memory_blocked(now: datetime, carry: dict) -> dict:
    rows = await _query(
        """SELECT count(*)::int AS n FROM memory_extraction_batches b
        JOIN conversations c ON c.id=b.conversation_id
        WHERE b.status IN ('preparing','processing') AND b.last_error IS NOT NULL
          AND b.prepared_at < $1::timestamptz AND NOT c.is_deleted AND c.archived_at IS NULL
          AND (SELECT count(*) FROM (SELECT 1 FROM messages m WHERE m.conversation_id=b.conversation_id
                 AND m.role='user' AND m.created_at > (b.prepared_at AT TIME ZONE 'UTC') LIMIT $2) later) >= $2""",
        (now - MEMORY_BLOCKED_AFTER).isoformat(), MEMORY_BLOCKED_LATER_MESSAGES)
    n = rows[0]["n"]
    return {"count": n, "threshold": MEMORY_BLOCKED_THRESHOLD, "alerting": n >= MEMORY_BLOCKED_THRESHOLD,
            "after_minutes": int(MEMORY_BLOCKED_AFTER.total_seconds() // 60),
            "later_messages": MEMORY_BLOCKED_LATER_MESSAGES}


async def _cron_failing(now: datetime, carry: dict) -> dict:
    from app.redis_client import get_redis
    from app.services.ops.cron_health import build_report
    from jobs.scheduler import _JOB_HEALTH_KEY, scheduler, scheduler_job_definitions

    # Read the health hash ourselves: an unreadable or empty record is unknown, not "0 failing".
    raw = await (await get_redis()).hgetall(_JOB_HEALTH_KEY)
    if not raw:
        raise RuntimeError("cron_health_unavailable")
    jobs = scheduler.get_jobs() if scheduler.running else scheduler_job_definitions()
    failing = sorted(j.job_id for j in build_report(raw, jobs, now=now).jobs if j.verdict == "failing")
    return {"count": len(failing), "threshold": CRON_FAILING_THRESHOLD,
            "alerting": len(failing) >= CRON_FAILING_THRESHOLD, "jobs": failing[:10]}


CHECKS = {"unanswered": _unanswered, "reply_failed": _reply_failed,
          "memory_blocked": _memory_blocked, "cron_failing": _cron_failing,
          "provider_budget": provider_budget.measure, "provider_failing": provider_failing.measure}


async def collect_metrics(now: datetime | None = None, carry: dict | None = None) -> dict:
    now = now or datetime.now(UTC)
    carry = carry or {}

    async def measure(name, check) -> tuple[str, dict]:
        try:
            async with asyncio.timeout(6):
                return name, {"status": "measured", **await check(now, carry.get(name) or {})}
        except Exception as exc:
            return name, {"status": "unknown", "error_type": type(exc).__name__}

    results = dict(await asyncio.gather(*(measure(n, c) for n, c in CHECKS.items())))
    # An unmeasured check keeps its previous evidence for the next sample.
    next_carry = {name: result.pop("_carry", None) if result["status"] == "measured" else carry.get(name)
                  for name, result in results.items()}
    return {"sampled_at": now.isoformat(), "checks": results,
            "carry": {k: v for k, v in next_carry.items() if v}}


def level_for(sample: dict, previous: str) -> str:
    checks = sample["checks"].values()
    if any(c.get("alerting") for c in checks):
        return "warning"
    if any(c["status"] != "measured" for c in checks):
        # Without a full measurement a warning stays latched; never claim recovery.
        return "warning" if previous == "warning" else "unknown"
    return "normal"


def alerting_checks(sample: dict) -> list[str]:
    return sorted(name for name, c in sample["checks"].items() if c.get("alerting"))


def decide(level: str, sample: dict, notified: dict, unknown_streak: int) -> tuple[bool, dict]:
    """Whether to push, and the incident state after this sample.

    `notified` is {"level", "checks"} as last pushed. A shrinking failing set,
    a latched warning or a recipient change never pushes.
    """
    previous = notified.get("level", "normal")
    if level == "warning":
        failing = set(alerting_checks(sample))
        # A second provider/model incident must notify even while another route
        # is already blocked. Recovery/reoccurrence is a distinct incident too.
        for incident in sample["checks"].get("provider_budget", {}).get("incidents", []):
            failing.add(f"provider_budget:{incident['provider']}/{incident['model']}:{incident['first_failure_at']}")
        for incident in sample["checks"].get("provider_failing", {}).get("incidents", []):
            failing.add(f"provider_failing:{incident['provider']}/{incident['model']}/{incident['role']}:{incident['first_failure_at']}")
            if incident.get("fallback") == "failed":
                failing.add(f"provider_failing:{incident['provider']}/{incident['model']}/{incident['role']}:{incident['first_failure_at']}:backup_failed")
        known = set(notified.get("checks", [])) if previous == "warning" else set()
        return previous != "warning" or bool(failing - known), {"level": "warning", "checks": sorted(known | failing)}
    if level == "unknown":
        if previous == "normal" and unknown_streak >= UNKNOWN_STREAK_TO_NOTIFY:
            return True, {"level": "unknown", "checks": []}
        return False, notified
    return previous != "normal", {"level": "normal", "checks": []}


def message_for(level: str, sample: dict) -> tuple[str, str]:
    checks = sample["checks"]
    if level == "warning":
        parts = []
        for name in alerting_checks(sample):
            c = checks[name]
            if name == "unanswered":
                parts.append(f"{c['count']}条用户消息超过{c['after_minutes']}分钟未回复")
            elif name == "reply_failed":
                parts.append(f"{c['window_minutes']}分钟内{c['count']}条回复生成失败（共{c['replies']}条回复）"
                             if c["count"] >= c["threshold"] else "回复失败后尚未出现成功回复")
            elif name == "memory_blocked":
                parts.append(f"{c['count']}个会话的记忆写入失败超过{c['after_minutes']}分钟，用户仍在聊天")
            elif name == "cron_failing":
                parts.append(f"{c['count']}个定时任务失败：{'、'.join(c['jobs'][:3])}")
            elif name == "provider_budget":
                incidents = c.get("incidents", [])
                parts.append("模型调用出现欠费或额度耗尽：" + "、".join(
                    f"{i['provider']}/{i['model']}（{'欠费' if i['reason'] == 'overdue' else '额度耗尽'}）"
                    for i in incidents[:3]) if incidents else "模型费用故障记录不完整，请检查监控")
            elif name == "provider_failing":
                for incident in c.get("incidents", [])[:3]:
                    outcome = {"succeeded": "最近故障请求已由备用模型接住", "failed": "备用也失败",
                               "not_attempted": "未尝试备用", "cancelled": "备用调用被取消",
                               "retry_succeeded": "最近故障请求重试后成功"}.get(incident.get("fallback"), "备用结果尚未确认")
                    cause = "熔断器已打开" if incident.get("reason") == "circuit_open" else (
                        f"{c['window_seconds'] / 60:g}分钟内失败{incident['failures']}次/{incident['calls']}次调用")
                    parts.append(f"{_route_label(incident)}{cause}；{outcome}")
        return "服务告警", "；".join(parts) + "。详情见Web后台资源监控。"
    if level == "unknown":
        missing = [LABELS[n] for n, c in checks.items() if c["status"] != "measured"]
        return "服务巡检异常", f"连续多次无法完成：{'、'.join(missing)}。不代表服务正常，请检查Web后台资源监控。"
    dormant = [r for r in checks.get("provider_failing", {}).get("routes", []) if r.get("dormant")]
    if dormant:
        # No call ever reached these routes again: stopping the alert is not recovery evidence.
        idle = checks["provider_failing"].get("dormant_seconds", 3600) / 60
        names = "、".join(_route_label(r) for r in dormant[:3])
        return "服务告警解除", (f"{names}已超过{idle:g}分钟没有调用，无法确认是否恢复，已停止告警；"
                                "其余检查正常。详情见Web后台资源监控。")
    return "服务告警恢复", "此前告警的问题已有恢复证据，各项服务检查正常。"


def _route_label(route: dict) -> str:
    role = route.get("role_label") or provider_failing.ROLE_LABELS.get(route.get("role"), route.get("role"))
    return f"{role}（{route['provider']}/{route['model']}）"


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def _timestamp(value) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


async def sample_and_notify() -> None:
    """Every worker calls this every 30 s; the row lock and interval sample once per 2 min.

    It rides the local Redis-health job, so it keeps running while Redis is down.
    """
    # Each worker needs recovery candidates even when a peer owns the sample.
    # A failed refresh must not prevent the other service checks from running.
    try:
        await provider_budget.refresh_active()
    except Exception:
        pass
    try:
        await provider_failing.flush()
    except Exception:
        pass  # pending batch retained; measurement exposes missing/stale collection
    now = datetime.now(UTC)
    rows = await db.query_raw("SELECT sampled_at, snapshot FROM ops_service_health WHERE id='primary'")
    if rows and rows[0].get("sampled_at") and (now - _timestamp(rows[0]["sampled_at"])).total_seconds() < SAMPLE_SECONDS - 10:
        return  # cheap pre-check; re-verified under the lock below
    previous = _object(rows[0]["snapshot"]) if rows else {}
    sample = await collect_metrics(now, (previous or {}).get("carry"))
    recipients = list(dict.fromkeys(x.strip() for x in settings.redis_memory_alert_user_ids.split(",") if x.strip()))
    if len(recipients) > 20:
        raise ValueError("Service alert recipients must be at most 20 admins")
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await tx.execute_raw("SET LOCAL lock_timeout = '2s'")
        await tx.execute_raw("INSERT INTO ops_service_health(id) VALUES ('primary') ON CONFLICT DO NOTHING")
        row = (await tx.query_raw("SELECT * FROM ops_service_health WHERE id='primary' FOR UPDATE"))[0]
        stamp = row.get("sampled_at")
        if stamp and (now - _timestamp(stamp)).total_seconds() < SAMPLE_SECONDS - 10:
            return
        admins = await tx.query_raw(
            "SELECT id FROM users WHERE id=ANY($1::text[]) AND role='admin' "
            "AND status='active' AND archived_at IS NULL ORDER BY id LIMIT 20", recipients,
        ) if recipients else []
        active_ids = [r["id"] for r in admins]
        level = level_for(sample, row["level"])
        streak = int((_object(row["snapshot"]) or {}).get("unknown_streak", 0)) + 1 if level == "unknown" else 0
        sample["unknown_streak"] = streak
        notified = _object(row.get("notified_signature")) or {"level": "normal", "checks": []}
        notify, next_notified = decide(level, sample, notified, streak)
        notify = notify and bool(active_ids)
        if not active_ids:
            next_notified = notified  # nobody was told; keep the last delivered state
        ids = _object(row["notification_ids"]) or []
        generation = int(row["generation"])
        if notify:
            generation += 1
            ids = []
            title, body = message_for(level, sample)
            for uid in active_ids:
                event_id = await enqueue_notification(
                    user_id=uid, type="admin_service_health", title=title, body=body,
                    payload={"type": "admin_service_health", "route": "system", "title": title, "body": body},
                    dedupe_key=f"service-health:{generation}", dispatch_now=False, database=tx,
                )
                if event_id:
                    ids.append(event_id)
        await tx.execute_raw(
            "UPDATE ops_service_health SET sampled_at=$1::timestamptz, snapshot=$2::jsonb, level=$3, "
            "notified_signature=$4, generation=$5, notification_ids=$6::jsonb, recipient_count=$7 WHERE id='primary'",
            now.isoformat(), json.dumps(sample), level, json.dumps(next_notified, sort_keys=True),
            generation, json.dumps(ids), len(active_ids),
        )


async def read_status() -> dict:
    base = {"sample_interval_seconds": SAMPLE_SECONDS, "stale_after_seconds": STALE_SECONDS,
            "labels": LABELS, "human_receipt": "not_collected"}
    try:
        async with asyncio.timeout(3):
            rows = await db.query_raw("SELECT * FROM ops_service_health WHERE id='primary'")
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
        stale = (datetime.now(UTC) - stamp).total_seconds() > STALE_SECONDS
        snapshot = _object(row["snapshot"]) or {}
        # Message IDs used as recovery evidence stay server-side.
        snapshot.pop("carry", None)
        return {**base, "status": "stale" if stale else "active" if row["recipient_count"] else "unconfigured",
                "level": row["level"], "sampled_at": stamp.isoformat(), "sample": snapshot,
                "recipient_count": row["recipient_count"], "notifications": events,
                "missing_notification_count": max(0, len(ids) - sum(e["count"] for e in events))}
    except Exception as exc:
        return {**base, "status": "unavailable", "error_type": type(exc).__name__, "notifications": []}
