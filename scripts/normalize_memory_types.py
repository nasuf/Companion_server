"""One-time migration: normalize Memory.type values to standard English enum.

Standard types: identity, emotion, preference, life, thought, consolidated.

Run from Server directory:
    python -m scripts.normalize_memory_types
"""

import asyncio
import os
import sys

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.memory.storage.persistence import _TYPE_NORMALIZE_MAP

_TABLES = ["memories_user", "memories_ai"]


async def main():
    from app.db import db

    await db.connect()

    try:
        for table in _TABLES:
            print(f"\n=== {table} ===")

            results = await db.query_raw(
                f"SELECT DISTINCT type, COUNT(*) as cnt FROM {table} WHERE type IS NOT NULL GROUP BY type ORDER BY cnt DESC"
            )
            print("Current type distribution:")
            for r in results:
                t = r["type"]
                mapped = _TYPE_NORMALIZE_MAP.get(t, t)
                marker = "" if t == mapped else f" → {mapped}"
                print(f"  {t}: {r['cnt']}{marker}")

            updated_total = 0
            for old_type, new_type in _TYPE_NORMALIZE_MAP.items():
                if old_type == new_type:
                    continue
                result = await db.execute_raw(
                    f"UPDATE {table} SET type = $1 WHERE type = $2",
                    new_type, old_type,
                )
                if result > 0:
                    print(f"  Updated {result} rows: {old_type} → {new_type}")
                    updated_total += result

            print(f"Total updated: {updated_total} rows")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
