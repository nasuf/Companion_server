#!/usr/bin/env python3
"""Backfill offering message content and 馈赠 memories.

Usage:
  PYTHONPATH=/app python scripts/backfill_offering_memories.py [--limit 500]
  PYTHONPATH=/app python scripts/backfill_offering_memories.py --content-only
  PYTHONPATH=/app python scripts/backfill_offering_memories.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.db import db
from app.services import offerings

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Only fill messages.content; skip embedding/memory writes",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    await db.connect()
    try:
        stats = await offerings.backfill_offering_memories_and_content(
            limit=args.limit,
            dry_run=args.dry_run,
            content_only=args.content_only,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
