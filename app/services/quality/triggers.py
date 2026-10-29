"""Admin-side polling; no hooks or work added to a user chat request."""

from datetime import UTC, datetime, timedelta
import hashlib
import json
from app.config import settings
from app.db import db
from app.services import runtime_config
from app.services.quality import jobs, snapshots
from app.services.quality.policy import local_day

CURSOR = "quality-watch-v1"


async def poll(now=None):
    now = now or datetime.now(UTC)
    await runtime_config.ensure_loaded()
    cfg = runtime_config.resolve_config_sync(None)
    route = [
        cfg.remote_chat_provider,
        cfg.remote_chat_model,
        cfg.remote_small_provider,
        cfg.remote_small_model,
        cfg.online_model,
    ]
    versions = await db.query_raw("""SELECT DISTINCT ON (prompt_key) id,prompt_key,change_type,created_at
      FROM prompt_template_versions ORDER BY prompt_key,created_at DESC,id DESC LIMIT 501""")
    if len(versions) > 500:
        raise ValueError("prompt_watch_exceeds_bound")
    current = dict(versions={r["prompt_key"]: r["id"] for r in versions}, route=route)
    # Cursor is a skipped administrative bookkeeping job, never an eval pass.
    cursor = await jobs.enqueue("routing", CURSOR, "1", current, now=now)
    rows = await db.query_raw(
        "SELECT snapshot,status,updated_at FROM quality_jobs WHERE id=$1", cursor
    )
    previous = rows[0]["snapshot"]
    if rows[0]["status"] == "pending":
        await db.execute_raw(
            "UPDATE quality_jobs SET status='skipped',report='{\"reason\":\"initial_watch_cursor\"}'::jsonb WHERE id=$1",
            cursor,
        )
        return
    changed = [
        r
        for r in versions
        if previous.get("versions", {}).get(r["prompt_key"]) != r["id"]
        and (
            r["change_type"] == "manual_save"
            or r["change_type"] == "reset_default"
            or r["change_type"].startswith("restore:")
        )
    ]
    if changed or previous.get("route") != route:
        snapshot = await snapshots.capture()
        for row in changed:
            await jobs.enqueue(
                "prompt",
                row["prompt_key"],
                row["id"],
                snapshot,
                now=now,
                delay=settings.quality_prompt_debounce_seconds,
            )
        if previous.get("route") != route:
            signature = hashlib.sha256(
                json.dumps(
                    [route, previous.get("route"), str(rows[0]["updated_at"])]
                ).encode()
            ).hexdigest()
            await jobs.enqueue(
                "routing",
                "global_model_route",
                signature,
                {
                    **snapshot,
                    "before_route": previous.get("route"),
                    "after_route": route,
                },
                now=now,
            )
    # Only advance after durable jobs exist; repeats are idempotent.
    await db.execute_raw(
        "UPDATE quality_jobs SET snapshot=$2::jsonb,updated_at=NOW() WHERE id=$1",
        cursor,
        json.dumps(current),
    )
    # After 04:45 local time the previous day is settled. One catch-up date only.
    local = now.astimezone(__import__("zoneinfo").ZoneInfo(settings.schedule_timezone))
    if (local.hour, local.minute) >= (4, 45):
        day = local_day(now) - timedelta(days=1)
        # A second worker can claim immediately after enqueue commits. Publish
        # the settled date in the INSERT, never patch it after the job is visible.
        await jobs.enqueue("daily", "daily", str(day), {}, now=now, for_day=day)
