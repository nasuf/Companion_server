"""清空数据库所有表数据，保留表结构。

Usage:
    cd Server
    python -m scripts.clear_database
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 按外键依赖顺序排列：子表在前，父表在后
_TABLES = [
    "memory_embeddings",
    "memory_changelogs",
    "memories_user",
    "memories_ai",
    "memories_backup",
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


async def main():
    from app.db import db

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

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
