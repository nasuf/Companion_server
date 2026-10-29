"""Durable admin-evaluation queue and a single expiring cluster lease."""

from datetime import UTC, datetime, timedelta
import json
from uuid import uuid4

from app.config import settings
from app.db import db
from app.services.quality.policy import local_day

LEASE_SECONDS = 180


class LostLease(RuntimeError):
    pass


async def enqueue(
    kind, source_key, source_version, snapshot, *, now=None, delay=0, for_day=None
):
    if kind not in {"daily", "prompt", "routing", "nightly", "calibration"}:
        raise ValueError("invalid_quality_job_kind")
    now = now or datetime.now(UTC)
    ident = uuid4().hex
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        # Serialise prompt debounce with claim(), including concurrent saves.
        await tx.query_raw(
            "SELECT id FROM quality_executor WHERE id='primary' FOR UPDATE"
        )
        if kind == "prompt":
            await tx.execute_raw(
                """UPDATE quality_jobs SET status='skipped',report='{"reason":"superseded_by_newer_prompt"}'::jsonb,
              updated_at=NOW() WHERE kind='prompt' AND source_key=$1 AND status='pending' AND source_version<>$2""",
                source_key,
                source_version,
            )
        rows = await tx.query_raw(
            """INSERT INTO quality_jobs(id,kind,source_key,source_version,local_date,snapshot,not_before)
          VALUES($1,$2,$3,$4,$5::date,$6::jsonb,$7::timestamptz)
          ON CONFLICT(kind,source_key,source_version) DO UPDATE SET source_key=EXCLUDED.source_key RETURNING id""",
            ident,
            kind,
            source_key,
            source_version,
            str(for_day if for_day is not None else local_day(now)),
            json.dumps(snapshot),
            (now + timedelta(seconds=delay)).isoformat(),
        )
    return rows[0]["id"]


async def claim(*, job_id=None):
    token = uuid4().hex
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        lock = (
            await tx.query_raw(
                "SELECT *,lease_until>NOW() AS active FROM quality_executor WHERE id='primary' FOR UPDATE"
            )
        )[0]
        if lock["active"]:
            return None
        if lock["job_id"]:
            # Never repeat an ambiguous provider request. Its reservation remains
            # charged; unfinished work is visible instead of silently retried.
            interrupted = await tx.query_raw(
                "UPDATE quality_jobs SET status='interrupted',lease_until=NULL,updated_at=NOW() WHERE id=$1 AND status='running' RETURNING id",
                lock["job_id"],
            )
            if interrupted:
                await notify(tx, lock["job_id"], "interrupted")
            await tx.execute_raw(
                "UPDATE quality_samples SET status='unknown' WHERE job_id=$1 AND status='pending'",
                lock["job_id"],
            )
        rows = await tx.query_raw(
            """SELECT * FROM quality_jobs WHERE status='pending' AND source_key<>'quality-watch-v1' AND (kind<>'nightly' OR $1::text IS NOT NULL) AND not_before<=NOW()
          AND ($1::text IS NULL OR id=$1) ORDER BY created_at,id LIMIT 1 FOR UPDATE""",
            job_id,
        )
        if not rows:
            return None
        row = rows[0]
        await tx.execute_raw(
            "UPDATE quality_executor SET job_id=$1,lease_token=$2,lease_until=NOW()+make_interval(secs=>$3) WHERE id='primary'",
            row["id"],
            token,
            LEASE_SECONDS,
        )
        await tx.execute_raw(
            "UPDATE quality_jobs SET status='running',lease_token=$2,lease_until=NOW()+make_interval(secs=>$3),updated_at=NOW() WHERE id=$1",
            row["id"],
            token,
            LEASE_SECONDS,
        )
        return {**row, "lease_token": token, "status": "running"}


async def require_lease(tx, job_id, token):
    rows = await tx.query_raw(
        "SELECT id FROM quality_executor WHERE id='primary' AND job_id=$1 AND lease_token=$2 AND lease_until>NOW() FOR UPDATE",
        job_id,
        token,
    )
    if not rows:
        raise LostLease("quality_executor_lease_lost")


async def renew(job_id, token):
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await require_lease(tx, job_id, token)
        await tx.execute_raw(
            "UPDATE quality_executor SET lease_until=NOW()+make_interval(secs=>$1) WHERE id='primary'",
            LEASE_SECONDS,
        )
        await tx.execute_raw(
            "UPDATE quality_jobs SET lease_until=NOW()+make_interval(secs=>$2) WHERE id=$1 AND status='running'",
            job_id,
            LEASE_SECONDS,
        )


async def finish(job, status, report):
    if status not in {
        "completed",
        "failed",
        "budget_exhausted",
        "not_configured",
        "skipped",
        "interrupted",
    }:
        raise ValueError("invalid_terminal_status")
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await require_lease(tx, job["id"], job["lease_token"])
        await tx.execute_raw(
            "UPDATE quality_jobs SET status=$2,report=report||$3::jsonb,lease_until=NULL,updated_at=NOW() WHERE id=$1",
            job["id"],
            status,
            json.dumps(report),
        )
        await tx.execute_raw(
            "UPDATE quality_executor SET job_id=NULL,lease_token=NULL,lease_until=NULL WHERE id='primary'"
        )
        if status in {"failed", "budget_exhausted", "interrupted"}:
            await notify(tx, job["id"], status)


async def notify(tx, job_id, reason):
    from app.services.notifications.service import enqueue_notification

    ids = list(
        dict.fromkeys(
            x.strip()
            for x in settings.redis_memory_alert_user_ids.split(",")
            if x.strip()
        )
    )[:20]
    admins = await tx.query_raw(
        "SELECT id FROM users WHERE id=ANY($1::text[]) AND role='admin' AND status='active' AND archived_at IS NULL",
        ids,
    )
    title = "质量评测预算已用尽" if reason == "budget_exhausted" else "质量评测需要检查"
    body = (
        "后台质量评测已停止，请查看质量管理中的结果和未完成项；不会自动阻止发布或回滚。"
    )
    for admin in admins:
        await enqueue_notification(
            user_id=admin["id"],
            type="admin_service_health",
            title=title,
            body=body,
            payload={
                "type": "admin_service_health",
                "route": "system",
                "title": title,
                "body": body,
            },
            dedupe_key=f"quality:{job_id}:{reason}",
            dispatch_now=False,
            database=tx,
        )
