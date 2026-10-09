"""Cumulative lifecycle; usage receipts and values commit in one transaction.

Scope locks followed by ordered row locks serialize rewards and singleton
promotions. Thirty-day receipts outlive the seven-day usage replay horizon.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.memory.lifecycle.value import (
    ACCESS_CEILING, ACCESS_REWARD, CONTRIBUTION_REWARD, VALUE_MAX,
    decayed_value, next_level,
)
from app.services.memory.taxonomy import L1_SINGLETON_SUBS

logger = logging.getLogger(__name__)
_TABLES = {"user": "memories_user", "ai": "memories_ai"}
_BATCH_SIZE = 250
USAGE_MAX_AGE_DAYS = 7
RECEIPT_RETENTION_DAYS = 30


def _signals(contributed_ids: list[str], accessed_ids: list[str]) -> dict[str, bool]:
    signals = {mid: False for mid in accessed_ids if mid}
    signals.update({mid: True for mid in contributed_ids if mid})
    return signals


def _stamp(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def scope_lock_key(source: str, user_id: str, workspace_id: str | None) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(
        [_TABLES[source], user_id, workspace_id], ensure_ascii=False,
    ).encode()).digest()[:8], "big", signed=True)


async def _apply_batch(
    *, source: str, user_id: str, workspace_id: str | None, ids: list[str],
    signals: dict[str, bool] | None = None, event_id: str | None = None,
    older_than_days: int = 30,
) -> dict:
    table = _TABLES[source]
    stats = {"side": source, "total": 0, "promoted": 0, "demoted": 0, "adjusted": 0}
    lock_key = scope_lock_key(source, user_id, workspace_id)
    async with db.tx(timeout=timedelta(seconds=15)) as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
        params = [ids, user_id, workspace_id]
        if signals is not None:
            params.append(event_id)
            guard = f"""AND EXISTS (
              SELECT 1 FROM messages msg JOIN conversations c ON c.id=msg.conversation_id
              WHERE msg.id=$4 AND msg.role='user' AND c.user_id=$2
                AND c.workspace_id IS NOT DISTINCT FROM $3::text AND NOT c.is_deleted
                AND msg.created_at >= CURRENT_TIMESTAMP - INTERVAL '{USAGE_MAX_AGE_DAYS} days'
                AND (c.workspace_id IS NULL OR EXISTS (
                  SELECT 1 FROM chat_workspaces w WHERE w.id=c.workspace_id
                    AND w.user_id=c.user_id AND w.agent_id=c.agent_id AND w.status='active'
                ))
            )"""
        else:
            guard = f"AND COALESCE(value_updated_at, created_at) < CURRENT_TIMESTAMP - INTERVAL '{older_than_days} days'"
        rows = await tx.query_raw(
            f"""SELECT id, level, importance, current_score, value_updated_at, created_at,
                       main_category, sub_category, provenance
                FROM {table} WHERE id=ANY($1::text[]) AND user_id=$2
                  AND workspace_id IS NOT DISTINCT FROM $3::text AND NOT is_archived
                  AND sub_category IS DISTINCT FROM '提醒' {guard}
                ORDER BY id FOR UPDATE""", *params,
        )
        if not rows:
            return stats
        previous = {}
        if signals is not None:
            previous = {r["memory_id"]: r for r in await tx.query_raw(
                "SELECT memory_id, contributed, reward, created_at FROM memory_usage_receipts "
                "WHERE event_id=$1 AND memory_side=$2 AND memory_id=ANY($3::text[])",
                event_id, source, [r["id"] for r in rows],
            )}
        occupied = {(r["main_category"], r["sub_category"]) for r in await tx.query_raw(
            f"SELECT DISTINCT main_category, sub_category FROM {table} WHERE user_id=$1 "
            "AND workspace_id IS NOT DISTINCT FROM $2::text AND level=1 AND NOT is_archived",
            user_id, workspace_id,
        )}
        now = datetime.now(UTC)
        updates, receipts, changes = [], [], []
        rows.sort(key=lambda r: (-float(r["current_score"] if r["current_score"] is not None else r["importance"]), r["id"]))
        for row in rows:
            mid, old_level = row["id"], row["level"]
            old_receipt = previous.get(mid)
            contributed = signals is not None and signals[mid]
            if old_receipt and (old_receipt["contributed"] or not contributed):
                continue
            anchor = _stamp(row["value_updated_at"] or row["created_at"])
            base = float(row["current_score"] if row["current_score"] is not None else row["importance"])
            value = decayed_value(base, max(0, (now - anchor).total_seconds()) / 86400)
            if signals is not None:
                reward = CONTRIBUTION_REWARD if contributed else ACCESS_REWARD * max(0, ACCESS_CEILING - value)
                # Upgrade an earlier candidate credit, never award the event twice.
                previous_credit = decayed_value(float(old_receipt["reward"]), max(
                    0, (now - _stamp(old_receipt["created_at"])).total_seconds(),
                ) / 86400) if old_receipt else 0.0
                value += reward - previous_credit
                receipts.append((event_id, source, mid, contributed, reward))
            value = max(0.0, min(VALUE_MAX, value))
            category = (row["main_category"], row["sub_category"])
            singleton = category in L1_SINGLETON_SUBS
            level = next_level(value, old_level, protected=singleton and old_level == 1)
            if row["provenance"] == "daily_summary" and old_level != 1:
                level = max(2, level)
            if level == 1 and old_level != 1 and singleton:
                if category in occupied:
                    level = old_level
                else:
                    occupied.add(category)
            updates.append((mid, value, level, max(now, anchor)))
            stats["total"] += 1
            if level != old_level:
                operation = "promote" if level < old_level else "demote"
                stats["promoted" if operation == "promote" else "demoted"] += 1
                changes.append((str(uuid.uuid4()), user_id, workspace_id, mid,
                                operation, f"level={old_level}", f"level={level}"))
            else:
                stats["adjusted"] += 1
        if not updates:
            return stats
        await tx.execute_raw(
            f"""UPDATE {table} m SET current_score=u.value, level=u.level,
                    value_updated_at=u.stamp FROM (
                SELECT unnest($1::text[]) AS id, unnest($2::float8[]) AS value,
                       unnest($3::int[]) AS level, unnest($4::timestamp[]) AS stamp
              ) u WHERE m.id=u.id""",
            [r[0] for r in updates], [r[1] for r in updates],
            [r[2] for r in updates], [r[3].replace(tzinfo=None).isoformat() for r in updates],
        )
        if receipts:
            await tx.execute_raw(
                """INSERT INTO memory_usage_receipts(event_id,memory_side,memory_id,contributed,reward)
                   SELECT unnest($1::text[]),unnest($2::text[]),unnest($3::text[]),
                          unnest($4::bool[]),unnest($5::float8[])
                   ON CONFLICT(event_id,memory_side,memory_id) DO UPDATE
                     SET contributed=EXCLUDED.contributed,reward=EXCLUDED.reward""",
                *[[r[i] for r in receipts] for i in range(5)],
            )
        if changes:
            await tx.execute_raw(
                """INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,old_value,new_value)
                   SELECT unnest($1::text[]),unnest($2::text[]),unnest($3::text[]),unnest($4::text[]),
                          unnest($5::text[]),unnest($6::text[]),unnest($7::text[])""",
                *[[r[i] for r in changes] for i in range(7)],
            )
    return stats


async def record_memory_usage(
    *, contributed_ids: list[str] | None = None, accessed_ids: list[str] | None = None,
    event_id: str | None = None, user_id: str | None = None, workspace_id: str | None = None,
) -> int:
    signals = _signals(contributed_ids or [], accessed_ids or [])
    if not signals:
        return 0
    if not event_id or not user_id:
        logger.warning("memory usage skipped: missing stable event or owner")
        return 0
    if len(signals) > 500:
        logger.warning("memory usage skipped: candidate bound exceeded")
        return 0
    total = 0
    for source in _TABLES:
        for offset in range(0, len(signals), _BATCH_SIZE):
            try:
                stats = await _apply_batch(source=source, user_id=user_id, workspace_id=workspace_id,
                    ids=list(signals)[offset:offset + _BATCH_SIZE], signals=signals, event_id=event_id)
                total += stats["total"]
            except Exception:
                logger.warning("memory usage transaction failed (%s)", source, exc_info=True)
    return total


async def sweep_stale_values(
    *, older_than_days: int = 30, limit: int = 1000, user_id: str | None = None,
    sources: tuple[str, ...] = ("user", "ai"),
) -> dict:
    """Oldest first; bounded IDs then small scoped batches, never ORM full loads."""
    if not 1 <= older_than_days <= 3650 or not 1 <= limit <= 5000:
        raise ValueError("Invalid bounded maintenance parameters")
    result = {"scanned": 0, "demoted": 0, "promoted": 0, "adjusted": 0}
    for source in sources:
        table = _TABLES[source]
        owner_guard = "AND user_id=$1" if user_id else ""
        candidates = await db.query_raw(
            f"""SELECT id,user_id,workspace_id FROM {table} WHERE NOT is_archived
                AND sub_category IS DISTINCT FROM '提醒'
                AND COALESCE(value_updated_at,created_at) < CURRENT_TIMESTAMP - INTERVAL '{older_than_days} days'
                {owner_guard} ORDER BY COALESCE(value_updated_at,created_at),id LIMIT {limit}""",
            *([user_id] if user_id else []),
        )
        buckets = defaultdict(list)
        for row in candidates:
            buckets[(row["user_id"], row["workspace_id"])].append(row["id"])
        side_stats = {"side": source, "total": 0, "promoted": 0, "demoted": 0, "adjusted": 0}
        for (owner, workspace), ids in buckets.items():
            for offset in range(0, len(ids), _BATCH_SIZE):
                stats = await _apply_batch(source=source, user_id=owner, workspace_id=workspace,
                    ids=ids[offset:offset + _BATCH_SIZE], older_than_days=older_than_days)
                for key in ("total", "promoted", "demoted", "adjusted"):
                    side_stats[key] += stats[key]
        result[source] = side_stats
        result["scanned"] += side_stats["total"]
        for key in ("demoted", "promoted", "adjusted"):
            result[key] += side_stats[key]
        logger.info("L2 adjustment [%s] complete: %s", source, side_stats)
    result["swept_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return result


async def purge_usage_receipts(*, limit: int = 5000) -> int:
    if not 1 <= limit <= 5000:
        raise ValueError("Invalid receipt retention batch")
    return await db.execute_raw(
        f"""DELETE FROM memory_usage_receipts WHERE (event_id,memory_side,memory_id) IN (
              SELECT event_id,memory_side,memory_id FROM memory_usage_receipts
              WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '{RECEIPT_RETENTION_DAYS} days'
              ORDER BY created_at LIMIT {limit})""",
    )
