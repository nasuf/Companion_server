"""Migrate data from unified 'memories' table to split 'memories_user' + 'memories_ai' tables.

Usage:
    cd Server
    python -m scripts.split_memory_tables

Prerequisites:
    - Run `prisma db push` first to create the new tables
    - The old 'memories' table must still exist with the 'source' column
"""

import asyncio
import sys
from pathlib import Path

# Add Server/ to path so app modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db


async def main():
    await db.connect()

    try:
        # Count source distribution
        rows = await db.query_raw(
            "SELECT source, COUNT(*) as cnt FROM memories GROUP BY source"
        )
        print("Source distribution in memories table:")
        for r in rows:
            print(f"  {r['source']}: {r['cnt']}")

        total_old = sum(int(r["cnt"]) for r in rows)
        print(f"  Total: {total_old}")

        if total_old == 0:
            print("No data to migrate.")
            return

        # Migrate user memories
        user_count = await db.execute_raw("""
            INSERT INTO memories_user (id, user_id, type, level, content, summary, importance, mention_count, is_archived, created_at, updated_at)
            SELECT id, user_id, type, level, content, summary, importance, mention_count, is_archived, created_at, updated_at
            FROM memories
            WHERE source = 'user'
            ON CONFLICT (id) DO NOTHING
        """)
        print(f"Migrated {user_count} rows to memories_user")

        # Migrate AI memories
        ai_count = await db.execute_raw("""
            INSERT INTO memories_ai (id, user_id, type, level, content, summary, importance, mention_count, is_archived, created_at, updated_at)
            SELECT id, user_id, type, level, content, summary, importance, mention_count, is_archived, created_at, updated_at
            FROM memories
            WHERE source = 'ai'
            ON CONFLICT (id) DO NOTHING
        """)
        print(f"Migrated {ai_count} rows to memories_ai")

        # Verify counts
        user_verify = await db.query_raw("SELECT COUNT(*) as cnt FROM memories_user")
        ai_verify = await db.query_raw("SELECT COUNT(*) as cnt FROM memories_ai")
        new_total = int(user_verify[0]["cnt"]) + int(ai_verify[0]["cnt"])

        print(f"\nVerification:")
        print(f"  memories_user: {user_verify[0]['cnt']}")
        print(f"  memories_ai:   {ai_verify[0]['cnt']}")
        print(f"  New total:     {new_total}")
        print(f"  Old total:     {total_old}")

        if new_total == total_old:
            print("\n✓ Migration successful! Row counts match.")
            print("\nNext steps:")
            print("  1. Verify the app works correctly with new tables")
            print("  2. Optionally rename old table: ALTER TABLE memories RENAME TO memories_backup;")
            print("  3. Remove the Memory model from schema.prisma and re-generate")
        else:
            print(f"\n⚠ Row count mismatch! Expected {total_old}, got {new_total}")
            print("  Some rows may have been skipped due to ON CONFLICT DO NOTHING (duplicate IDs)")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
