"""One-time seed: migrate `app/data/holidays_cn.py` hardcoded data to the
`holidays` DB table as `source="manual"` rows.

Run after the migration is applied but before the runtime is switched to
read from DB. Safe to re-run (uses upsert).

Usage:
    .venv/bin/python scripts/seed_holidays_from_hardcoded.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date as date_cls
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import connect_db, db  # noqa: E402
from app.data.holidays_cn import _RAW, WORKDAY_SWAPS  # noqa: E402
from app.services.schedule_domain.holiday_repo import (  # noqa: E402
    SOURCE_MANUAL,
    HolidayEntry,
    upsert_many,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    await connect_db()

    entries: list[HolidayEntry] = []

    for row in _RAW:
        try:
            d = date_cls.fromisoformat(row["date"])
        except ValueError:
            logger.warning(f"Skipping invalid date: {row}")
            continue
        entries.append(HolidayEntry(
            date=d,
            name=row["name"],
            type=row["type"],
            country_code="CN" if row["type"] != "international" else "INTL",
            is_workday_swap=False,
            source=SOURCE_MANUAL,
        ))

    for swap_date in WORKDAY_SWAPS:
        try:
            d = date_cls.fromisoformat(swap_date)
        except ValueError:
            continue
        entries.append(HolidayEntry(
            date=d,
            name="调休上班",
            type="custom",
            country_code="CN",
            is_workday_swap=True,
            source=SOURCE_MANUAL,
        ))

    logger.info(f"Seeding {len(entries)} holiday rows from legacy hardcoded data...")
    stats = await upsert_many(entries, allow_overwrite_manual=True)
    logger.info(f"Done. stats={stats}")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
