"""清空所有数据库（PostgreSQL + Redis），保留表结构。

Usage:
    cd Companion_server
    python -m scripts.clear_database
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 按外键依赖顺序排列：子表在前，父表在后
_TABLES = [
    "stickers",
    "memory_mentions",
    "memory_entities",
    "memory_embeddings",
    "memory_changelogs",
    "memories_user",
    "memories_ai",
    "memories_backup",
    "time_triggers",
    "messages",
    "proactive_chat_logs",
    "trait_feedback_logs",
    "schedule_adjust_logs",
    "ai_daily_schedules",
    "ai_emotion_states",
    "user_portraits",
    "user_profiles",
    "intimacies",
    "conversations",
    "ai_agents",
    "users",
]


async def _clear_postgres():
    from app.db import db

    print("[PostgreSQL]")
    await db.connect()
    try:
        for table in _TABLES:
            try:
                cnt = await db.execute_raw(f"DELETE FROM {table}")
                print(f"  {table}: deleted {cnt} rows" if cnt else f"  {table}: empty")
            except Exception as e:
                print(f"  {table}: ERROR - {e}")
    finally:
        await db.disconnect()


async def _clear_redis():
    from app.redis_client import get_redis, close_redis

    print("[Redis]")
    try:
        r = await get_redis()
        count = await r.dbsize()
        await r.flushdb()
        print(f"  flushed {count} keys")
    except Exception as e:
        print(f"  ERROR - {e}")
    finally:
        await close_redis()


async def main():
    await _clear_postgres()
    await _clear_redis()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
