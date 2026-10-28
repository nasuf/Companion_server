"""Backfill / recompute the beta statistics derived tables (UXM01.02).

user_daily_facts and metrics_daily are versioned caches; every day can be
recomputed from raw data, so this script is safe to re-run. By default it only
computes days that are missing, computed under an older definition version or
an older internal-account set, or still unsettled after their windows closed.

Usage (from Companion_server/, inside the server container in production):
  # dry-run: list the days that would be computed
  python scripts/backfill_daily_facts.py
  # compute those days (from the earliest raw data, capped at 400 days)
  python scripts/backfill_daily_facts.py --apply
  # force a range, e.g. after a definition change
  python scripts/backfill_daily_facts.py --start 2026-10-01 --end 2026-10-27 --all --apply

Prints only dates and counts; never message text or user identifiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db  # noqa: E402
from app.services.effect import daily_facts  # noqa: E402
from app.services.effect.definitions import local_today  # noqa: E402


def _day(value: str) -> date:
    return date.fromisoformat(value)


async def plan(start: date | None, end: date | None, *, every_day: bool) -> list[date]:
    last = min(end or local_today(), local_today())
    first = start or await daily_facts.earliest_data_day()
    if first is None:
        return []
    first = max(first, last - timedelta(days=daily_facts.MAX_BACKFILL_DAYS - 1))
    if every_day:
        return [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return await daily_facts.days_needing_compute(first, last)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=_day)
    parser.add_argument("--end", type=_day)
    parser.add_argument("--all", action="store_true", help="recompute every day in range, not only missing/stale")
    parser.add_argument("--apply", action="store_true", help="write; without it only the plan is printed")
    args = parser.parse_args(argv)
    if args.start and args.end and args.start > args.end:
        parser.error("--start must not be after --end")
    await db.connect()
    try:
        days = await plan(args.start, args.end, every_day=args.all)
        summary = {"days": len(days), "first": days[0].isoformat() if days else None,
                   "last": days[-1].isoformat() if days else None, "applied": args.apply}
        if args.apply and days:
            result = await daily_facts.compute_days(days)
            summary.update(computed=len(result["computed"]), failed=result["failed"])
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary.get("failed") else 0
    finally:
        await db.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
