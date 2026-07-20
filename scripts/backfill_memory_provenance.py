"""One-shot backfill for the memory `provenance` column (Phase 2).

Conservative heuristics — wrong guesses are worse than NULL, so anything
ambiguous stays NULL:

- memories_user: every row was extracted from user statements (chat pipeline,
  contradiction resolutions) → user_stated.
- memories_ai, L1 rows created within 30 minutes of the workspace's FIRST AI
  memory: those are the provisioning seed batch → profile_seed.
- memories_ai other rows: could be daily-summary trivia or chat-time
  self-memory — indistinguishable after the fact → left NULL.

Usage (from Companion_server/):
    .venv/bin/python scripts/backfill_memory_provenance.py           # dry-run
    .venv/bin/python scripts/backfill_memory_provenance.py --apply  # write
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db, ensure_connected  # noqa: E402

_USER_SQL = """
UPDATE memories_user SET provenance = 'user_stated' WHERE provenance IS NULL
"""

_USER_COUNT_SQL = """
SELECT COUNT(*)::int AS n FROM memories_user WHERE provenance IS NULL
"""

# Seed batch detection: rows within 30 minutes of the workspace's first AI row.
_AI_SEED_SQL = """
UPDATE memories_ai m
SET provenance = 'profile_seed'
FROM (
    SELECT workspace_id, MIN(created_at) AS first_at
    FROM memories_ai
    GROUP BY workspace_id
) f
WHERE m.workspace_id = f.workspace_id
  AND m.provenance IS NULL
  AND m.level = 1
  AND m.created_at < f.first_at + INTERVAL '30 minutes'
"""

_AI_SEED_COUNT_SQL = """
SELECT COUNT(*)::int AS n
FROM memories_ai m
JOIN (
    SELECT workspace_id, MIN(created_at) AS first_at
    FROM memories_ai
    GROUP BY workspace_id
) f ON m.workspace_id = f.workspace_id
WHERE m.provenance IS NULL
  AND m.level = 1
  AND m.created_at < f.first_at + INTERVAL '30 minutes'
"""


async def main(apply: bool) -> None:
    await ensure_connected()

    user_rows = await db.query_raw(_USER_COUNT_SQL)
    ai_seed_rows = await db.query_raw(_AI_SEED_COUNT_SQL)
    user_n = int(user_rows[0]["n"]) if user_rows else 0
    ai_seed_n = int(ai_seed_rows[0]["n"]) if ai_seed_rows else 0

    print(f"memories_user NULL → user_stated : {user_n} rows")
    print(f"memories_ai   NULL → profile_seed: {ai_seed_n} rows (L1 within 30min of first)")
    print("memories_ai remaining NULL rows stay NULL (daily_summary vs chat ambiguous)")

    if not apply:
        print("\nDry-run only. Re-run with --apply to write.")
        return

    updated_user = await db.execute_raw(_USER_SQL)
    updated_seed = await db.execute_raw(_AI_SEED_SQL)
    print(f"\nApplied: user_stated={updated_user}, profile_seed={updated_seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default dry-run)")
    args = parser.parse_args()
    asyncio.run(main(args.apply))
