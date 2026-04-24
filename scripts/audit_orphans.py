"""审计并清理 DB + Redis 中与已删除 agent 相关的孤儿数据.

默认 dry-run 只报告; `--clean` 真正删除.

Usage:
    uv run python scripts/audit_orphans.py            # 只扫描, 不删
    uv run python scripts/audit_orphans.py --clean    # 真删
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db import db
from app.redis_client import get_redis, close_redis


# agent_id 键位于 Redis key 的第 N 段 (冒号分隔, 0-indexed)
# 例: "emotion:{agent_id}" → pos=1, "memgen:lock:{agent_id}" → pos=2
_REDIS_PATTERNS: list[tuple[str, int]] = [
    ("emotion:*", 1),
    ("life_overview:*", 1),
    ("patience:*", 1),
    ("attack_history:*", 1),
    ("intimacy:*", 1),
    ("topic_intimacy:*", 1),
    ("last_reply:*", 1),
    ("trigger_last:*", 1),
    ("trigger_count:*", 1),
    ("proactive_count:*", 1),
    ("proactive_2day:*", 1),
    ("schedule:*", 1),
    ("schedule_adj:*", 1),
    ("trait_adj:*", 1),
    ("trait_adj_week:*", 1),
    ("pending:msgs:*", 2),
    ("pending:conv:*", 2),
    ("pending:ctx:*", 2),
    ("memgen:lock:*", 2),
    ("memgen:report:*", 2),
    ("provision_progress:*", 1),
]

# conversation_id 键
_REDIS_CONV_PATTERNS: list[tuple[str, int]] = [
    ("topics:*", 1),
    ("context_window:*", 1),
    ("delayed:msgs:*", 1),
]

# DB 表: (table, agent_id_column)
_DB_AGENT_TABLES: list[tuple[str, str]] = [
    ("chat_workspaces", "agent_id"),
    ("conversations", "agent_id"),
    ("intimacies", "agent_id"),
    ("patience_states", "agent_id"),
    ("ai_emotion_states", "agent_id"),
    ("ai_daily_schedules", "agent_id"),
    ("trait_feedback_logs", "agent_id"),
    ("proactive_chat_logs", "agent_id"),
    ("proactive_event_logs", "agent_id"),
    ("proactive_states", "agent_id"),
    ("proactive_counters", "agent_id"),
    ("time_triggers", "ai_agent_id"),
    ("user_portraits", "agent_id"),
    ("schedule_adjust_logs", "agent_id"),
]


async def _load_valid_ids() -> tuple[set[str], set[str]]:
    """从 DB 取 (valid_agent_ids, valid_conversation_ids)."""
    rows = await db.query_raw('SELECT id FROM ai_agents')
    valid_agents = {r["id"] for r in (rows or [])}

    rows = await db.query_raw('SELECT id FROM conversations')
    valid_convs = {r["id"] for r in (rows or [])}

    return valid_agents, valid_convs


async def _audit_db(valid_agents: set[str], clean: bool) -> dict[str, int]:
    """扫 DB 表里 agent_id 不在 valid_agents 的行."""
    findings: dict[str, int] = {}
    for table, col in _DB_AGENT_TABLES:
        try:
            rows = await db.query_raw(f'SELECT DISTINCT "{col}" AS aid FROM "{table}"')
            all_aids = {r["aid"] for r in (rows or []) if r.get("aid")}
            orphans = all_aids - valid_agents
            if not orphans:
                continue
            # 统计孤儿行数
            cnt_row = await db.query_raw(
                f'SELECT COUNT(*) AS c FROM "{table}" WHERE "{col}" = ANY($1::text[])',
                list(orphans),
            )
            count = cnt_row[0]["c"] if cnt_row else 0
            findings[table] = count
            if clean and count > 0:
                await db.execute_raw(
                    f'DELETE FROM "{table}" WHERE "{col}" = ANY($1::text[])',
                    list(orphans),
                )
        except Exception as e:
            print(f"  ! {table} audit failed: {e}")
    return findings


async def _audit_db_memories(valid_workspaces: set[str], clean: bool) -> dict[str, int]:
    """memories_user / memories_ai / memory_embeddings 按 workspace 扫孤儿."""
    findings: dict[str, int] = {}
    for table in ("memories_user", "memories_ai", "user_profiles", "memory_changelogs", "memory_entities"):
        try:
            rows = await db.query_raw(f'SELECT DISTINCT workspace_id AS wid FROM "{table}" WHERE workspace_id IS NOT NULL')
            all_wids = {r["wid"] for r in (rows or []) if r.get("wid")}
            orphans = all_wids - valid_workspaces
            if not orphans:
                continue
            cnt_row = await db.query_raw(
                f'SELECT COUNT(*) AS c FROM "{table}" WHERE workspace_id = ANY($1::text[])',
                list(orphans),
            )
            count = cnt_row[0]["c"] if cnt_row else 0
            findings[table] = count
            if clean and count > 0:
                # memory_embeddings 跟着 memory_id 走, 先收 id
                if table in ("memories_user", "memories_ai"):
                    mem_rows = await db.query_raw(
                        f'SELECT id FROM "{table}" WHERE workspace_id = ANY($1::text[])',
                        list(orphans),
                    )
                    mem_ids = [r["id"] for r in (mem_rows or [])]
                    if mem_ids:
                        await db.execute_raw(
                            'DELETE FROM memory_embeddings WHERE memory_id = ANY($1::text[])',
                            mem_ids,
                        )
                await db.execute_raw(
                    f'DELETE FROM "{table}" WHERE workspace_id = ANY($1::text[])',
                    list(orphans),
                )
        except Exception as e:
            print(f"  ! {table} audit failed: {e}")
    return findings


async def _audit_redis(valid_agents: set[str], valid_convs: set[str], clean: bool) -> dict[str, int]:
    """扫 Redis 所有已知 pattern, 找第 N 段不在 valid 集合的 key."""
    redis = await get_redis()
    findings: dict[str, int] = {}

    async def _scan(pattern: str, id_pos: int, valid: set[str], label: str) -> None:
        orphan_keys: list[str] = []
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=500)
            for raw in keys:
                k = raw.decode() if isinstance(raw, bytes) else raw
                parts = k.split(":")
                if len(parts) <= id_pos:
                    continue
                the_id = parts[id_pos]
                if the_id not in valid:
                    orphan_keys.append(k)
            if cursor == 0:
                break
        if not orphan_keys:
            return
        findings[label] = findings.get(label, 0) + len(orphan_keys)
        if clean:
            # 分批删, Redis 单次 DEL 参数上限
            for i in range(0, len(orphan_keys), 500):
                await redis.delete(*orphan_keys[i:i+500])

    for pattern, pos in _REDIS_PATTERNS:
        await _scan(pattern, pos, valid_agents, pattern)
    for pattern, pos in _REDIS_CONV_PATTERNS:
        await _scan(pattern, pos, valid_convs, pattern)

    # pending:delayed ZSET: 成员格式 "{agent_id}:{user_id}"
    try:
        members = await redis.zrange("pending:delayed", 0, -1)
        orphan_members = []
        for raw in members:
            m = raw.decode() if isinstance(raw, bytes) else raw
            aid = m.split(":")[0]
            if aid not in valid_agents:
                orphan_members.append(m)
        if orphan_members:
            findings["pending:delayed (ZSET member)"] = len(orphan_members)
            if clean:
                await redis.zrem("pending:delayed", *orphan_members)
    except Exception as e:
        print(f"  ! pending:delayed audit failed: {e}")

    # delayed:due ZSET: 成员 = conversation_id
    try:
        members = await redis.zrange("delayed:due", 0, -1)
        orphan_members = [
            (m.decode() if isinstance(m, bytes) else m)
            for m in members
        ]
        orphan_members = [m for m in orphan_members if m not in valid_convs]
        if orphan_members:
            findings["delayed:due (ZSET member)"] = len(orphan_members)
            if clean:
                await redis.zrem("delayed:due", *orphan_members)
    except Exception as e:
        print(f"  ! delayed:due audit failed: {e}")

    return findings


async def main(clean: bool) -> None:
    await db.connect()
    try:
        print(f"{'CLEANING' if clean else 'AUDIT (dry-run)'} — use --clean to actually delete\n")

        valid_agents, valid_convs = await _load_valid_ids()
        print(f"Active: {len(valid_agents)} agents, {len(valid_convs)} conversations\n")

        ws_rows = await db.query_raw(
            'SELECT id FROM chat_workspaces WHERE agent_id = ANY($1::text[])',
            list(valid_agents),
        )
        valid_workspaces = {r["id"] for r in (ws_rows or [])}

        print("=== DB: tables keyed by agent_id ===")
        db_findings = await _audit_db(valid_agents, clean)
        for table, count in sorted(db_findings.items()):
            print(f"  {table}: {count} orphan rows")
        if not db_findings:
            print("  (none)")

        print("\n=== DB: tables keyed by workspace_id ===")
        mem_findings = await _audit_db_memories(valid_workspaces, clean)
        for table, count in sorted(mem_findings.items()):
            print(f"  {table}: {count} orphan rows")
        if not mem_findings:
            print("  (none)")

        print("\n=== Redis ===")
        redis_findings = await _audit_redis(valid_agents, valid_convs, clean)
        for pattern, count in sorted(redis_findings.items()):
            print(f"  {pattern}: {count} orphan keys")
        if not redis_findings:
            print("  (none)")

        total_db = sum(db_findings.values()) + sum(mem_findings.values())
        total_redis = sum(redis_findings.values())
        print(f"\nTotal: {total_db} DB rows + {total_redis} Redis keys")
        if not clean and (total_db or total_redis):
            print("Re-run with --clean to delete.")
    finally:
        await close_redis()
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Actually delete (default: dry-run)")
    args = parser.parse_args()
    asyncio.run(main(args.clean))
