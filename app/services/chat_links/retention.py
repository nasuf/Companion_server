"""Bounded, operator-triggered maintenance of unreferenced link resources.

Historical messages are the retention boundary, not the mutable URL cache's
single message pointer. SQL admission and legacy JSON writes share the migration
guard's transaction lock. File retirement commits before unlink and is retried
after crashes; only explicitly registered cover objects are ever unlinked.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import shutil
import stat
from typing import Any
from uuid import uuid4

from app.db import db
from app.services.chat_media import storage
from app.services.runtime import user_erasure

POLICY_VERSION = "link_retention_v1"
RETENTION_DAYS = 7
SCAN_LIMIT = 100
COVER_BUDGET_BYTES = 1024 ** 3
DISK_RESERVE_BYTES = 1024 ** 3
LOCK_ID = 87231615


class MaintenanceBusy(Exception):
    pass


@asynccontextmanager
async def locked(database):
    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout='3000ms'")
        await tx.execute_raw("SET LOCAL lock_timeout='1500ms'")
        rows = await tx.query_raw("SELECT pg_try_advisory_xact_lock($1::bigint) AS acquired", LOCK_ID)
        if not rows[0]["acquired"]:
            raise MaintenanceBusy()
        yield tx


async def reserve_cover(key: str, size: int, *, database=None) -> bool:
    """Reserve bytes before creating a file, including ambiguous commit cases."""
    if not 0 < size <= storage._MAX_IMAGE_BYTES:
        return False
    async with locked(db if database is None else database) as tx:
        rows = await tx.query_raw(
            "UPDATE chat_link_storage_state SET reserved_bytes=reserved_bytes+$1::bigint "
            "WHERE id=1 AND reserved_bytes+$1::bigint <= $2::bigint RETURNING id", size, COVER_BUDGET_BYTES)
        if not rows:
            return False
        await tx.execute_raw(
            "INSERT INTO chat_link_cover_objects(storage_key,image_url,byte_size) VALUES ($1,$2,$3::bigint)",
            key, storage.media_url(key), size)
    return True


async def write_cover(*, user_id: str, conversation_id: str | None, blob: bytes,
                      mime: str, database=None) -> str | None:
    async with user_erasure.owner_write_scope(user_id, database=db if database is None else database):
        return await _write_cover(user_id=user_id, conversation_id=conversation_id,
                                 blob=blob, mime=mime, database=database)


async def _write_cover(*, user_id: str, conversation_id: str | None, blob: bytes,
                       mime: str, database=None) -> str | None:
    storage.validate_image_size(blob)
    storage._MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(storage._MEDIA_DIR).free < DISK_RESERVE_BYTES + len(blob):
        return None
    key = storage.storage_key_for(user_id, mime, conversation_id=conversation_id)
    if not await reserve_cover(key, len(blob), database=database):
        return None
    # Exclusive creation prevents overwriting an existing file or symlink.
    # A write failure leaves a ledger entry for bounded, idempotent recovery.
    def write():
        with storage.storage_path(key).open("xb") as stream:
            stream.write(blob)
    # Keep the user fence until the disk write has actually finished, including
    # cancellation. An abandoned thread must not recreate a deleted cover later.
    task = asyncio.create_task(asyncio.to_thread(write))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    return key


_CACHE_REFERENCE = """(
    l.message_id IS NOT NULL OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{link_card,id}'=l.id) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{component_card,payload,link_id}'=l.id) OR
    EXISTS (SELECT 1 FROM chat_link_message_bindings b JOIN chat_link_snapshots s ON s.id=b.snapshot_id
            WHERE s.link_card_id=l.id)
)"""
_SNAPSHOT_REFERENCE = """(
    EXISTS (SELECT 1 FROM chat_link_message_bindings WHERE snapshot_id=s.id) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{link_card,snapshot_id}'=s.id) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{component_card,payload,snapshot_id}'=s.id) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{link_card,id}'=s.link_card_id
            AND metadata #>> '{link_card,snapshot_id}' IS NULL) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{component_card,payload,link_id}'=s.link_card_id
            AND metadata #>> '{component_card,payload,snapshot_id}' IS NULL) OR
    EXISTS (SELECT 1 FROM chat_link_cards WHERE id=s.link_card_id AND message_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM messages WHERE id=message_id
                AND metadata #>> '{link_card,snapshot_id}' IS NOT NULL))
)"""
_COVER_REFERENCE = """(
    EXISTS (SELECT 1 FROM chat_link_cards WHERE image_url=o.image_url) OR
    EXISTS (SELECT 1 FROM chat_link_snapshots WHERE payload->>'image_url'=o.image_url) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{link_card,image_url}'=o.image_url) OR
    EXISTS (SELECT 1 FROM messages WHERE metadata #>> '{component_card,payload,image_url}'=o.image_url)
)"""


async def _sample(tx, cursors: dict) -> dict[str, list[dict]]:
    # Scan by indexed identity, then evaluate age/references on at most 100 rows.
    # Persisted cursors prevent protected old rows from starving later orphans.
    queries = {
        "snapshots": ("chat_link_snapshots", "s", "id", "recorded_at", _SNAPSHOT_REFERENCE),
        "previews": ("chat_link_cards", "l", "id", "updated_at", _CACHE_REFERENCE),
        "covers": ("chat_link_cover_objects", "o", "storage_key", "created_at", _COVER_REFERENCE),
    }
    result = {}
    for kind, (table, alias, key, age, reference) in queries.items():
        # Identifiers are constants above, never client input.
        extra = ", state, byte_size" if kind == "covers" else ""
        clock = "CURRENT_TIMESTAMP" if kind == "covers" else "(CURRENT_TIMESTAMP AT TIME ZONE 'UTC')"
        rows = await tx.query_raw(
            f"WITH page AS MATERIALIZED (SELECT * FROM {table} WHERE {key}>$1 ORDER BY {key} LIMIT $2) "
            f"SELECT {key} AS id, {age} < {clock} - INTERVAL '7 days' AS old, "
            f"{reference} AS protected {extra} FROM page {alias} ORDER BY {key}",
            str(cursors.get(kind, "")), SCAN_LIMIT)
        result[kind] = rows
    return result


def _summary(rows: dict) -> dict:
    return {kind: {
        "sampled": len(items), "protected": sum(bool(x["protected"]) for x in items),
        "eligible": sum(bool(x["old"] and not x["protected"]) for x in items),
        "page_full": len(items) == SCAN_LIMIT,
    } for kind, items in rows.items()}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False)


async def status(*, database=None) -> dict:
    client = db if database is None else database
    async with client.tx(timeout=timedelta(seconds=10)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout='3000ms'")
        state = (await tx.query_raw("SELECT * FROM chat_link_storage_state WHERE id=1"))[0]
        sample = _summary(await _sample(tx, state["cursors"]))
        sizes = await tx.query_raw(
            "SELECT pg_total_relation_size('chat_link_cards') AS previews, "
            "pg_total_relation_size('chat_link_snapshots') AS snapshots, "
            "pg_total_relation_size('chat_link_message_bindings') AS bindings")
    return {"sampled_at": _now(), "policy": {
        "version": POLICY_VERSION, "retention_days": RETENTION_DAYS,
        "scan_limit_per_kind": SCAN_LIMIT, "cover_budget_bytes": COVER_BUDGET_BYTES,
        "retiring_retry_limit": SCAN_LIMIT,
        "disk_reserve_bytes": DISK_RESERVE_BYTES, "execution": "manual",
        "legacy_unregistered_files": "retained_not_measured",
    }, "sample": sample, "database_bytes": sizes[0],
        "managed_cover_reserved_bytes": int(state["reserved_bytes"]),
        "last_preview": state["last_preview"], "last_run": state["last_run"],
        "running_since": str(state["running_since"]) if state["running_since"] else None}


async def preview(*, database=None) -> dict:
    client = db if database is None else database
    report = await status(database=client)
    result = {"status": "completed", "at": report["sampled_at"], "sample": report["sample"]}
    await client.execute_raw("UPDATE chat_link_storage_state SET last_preview=$1::jsonb WHERE id=1", _json(result))
    return result


def _unlink(key: str):
    # Never follow a symlink, including a replacement between stat and unlink.
    # unlink itself removes the entry, not a symlink's target; reject non-files
    # and symlinks so unexpected filesystem objects remain visible as failures.
    path = storage.storage_path(key)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError("unexpected_cover_file_type")
    path.unlink(missing_ok=True)


async def execute(*, database=None) -> dict:
    client = db if database is None else database
    run_id = str(uuid4())
    result: dict[str, Any] = {"status": "running", "at": _now(), "sql_outcome": "not_started", "deleted": {
        "snapshots": 0, "previews": 0, "covers": 0}, "file_failures": 0}
    async with locked(client) as tx:
        rows = await tx.query_raw(
            "UPDATE chat_link_storage_state SET run_id=$1,running_since=NOW(),last_run=$2::jsonb "
            "WHERE id=1 AND (run_id IS NULL OR running_since<NOW()-INTERVAL '10 minutes') RETURNING id",
            run_id, _json(result))
        if not rows:
            raise MaintenanceBusy()
    try:
        async with locked(client) as tx:
            state = (await tx.query_raw("SELECT cursors FROM chat_link_storage_state WHERE id=1 AND run_id=$1", run_id))[0]
            samples = await _sample(tx, state["cursors"])
            result["sample"] = _summary(samples)
            result["sql_outcome"] = "unknown"
            for kind in ("snapshots", "previews"):
                table = "chat_link_snapshots" if kind == "snapshots" else "chat_link_cards"
                eligible = [x["id"] for x in samples[kind] if x["old"] and not x["protected"]]
                # Cache deletion never cascades an unbounded snapshot set. Skip
                # independent writers holding row locks while waiting for guard.
                extra = (" AND NOT EXISTS (SELECT 1 FROM chat_link_snapshots WHERE link_card_id=t.id)"
                         if kind == "previews" else "")
                deleted = await tx.query_raw(
                    f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} t WHERE id=ANY($1::text[]) "
                    f"{extra} FOR UPDATE SKIP LOCKED) RETURNING id", eligible)
                result["deleted"][kind] = len(deleted)
            eligible = [x["id"] for x in samples["covers"] if x["old"] and not x["protected"]]
            await tx.execute_raw("UPDATE chat_link_cover_objects SET state='deleting' WHERE storage_key=ANY($1::text[])", eligible)
            # Retry previously committed retirements even after process death.
            files = await tx.query_raw("SELECT storage_key,byte_size FROM chat_link_cover_objects "
                                      "WHERE state='deleting' AND storage_key>$1 ORDER BY storage_key LIMIT $2",
                                      str(state["cursors"].get("retiring", "")), SCAN_LIMIT)
            cursors = {kind: items[-1]["id"] if len(items) == SCAN_LIMIT else ""
                       for kind, items in samples.items()}
            cursors["retiring"] = files[-1]["storage_key"] if len(files) == SCAN_LIMIT else ""
            await tx.execute_raw("UPDATE chat_link_storage_state SET cursors=$1::jsonb WHERE id=1 AND run_id=$2",
                                 _json(cursors), run_id)
        result["sql_outcome"] = "committed"
        for row in files:
            try:
                await asyncio.to_thread(_unlink, row["storage_key"])
            except (OSError, ValueError):
                result["file_failures"] += 1
                continue
            async with locked(client) as tx:
                removed = await tx.query_raw("DELETE FROM chat_link_cover_objects WHERE storage_key=$1 "
                                             "AND state='deleting' RETURNING byte_size", row["storage_key"])
                if removed:
                    await tx.execute_raw("UPDATE chat_link_storage_state SET reserved_bytes=reserved_bytes-$1::bigint WHERE id=1",
                                         int(removed[0]["byte_size"]))
                    result["deleted"]["covers"] += 1
        result["status"] = "failed" if result["file_failures"] else "completed"
    except asyncio.CancelledError:
        result["status"] = "interrupted"
        raise
    except Exception as exc:
        # No SQL/URL/file paths in the public report. Committed retirements are
        # still safe to retry; rolled-back SQL is never reported as successful.
        result["status"] = "failed"
        result["reason"] = type(exc).__name__
        raise
    finally:
        result["finished_at"] = _now()
        await client.execute_raw("UPDATE chat_link_storage_state SET last_run=$1::jsonb,run_id=NULL,running_since=NULL "
                                 "WHERE id=1 AND run_id=$2", _json(result), run_id)
    return result
