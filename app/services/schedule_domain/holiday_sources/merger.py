"""Combine chinesecalendar + nager.date + un_observed into a unified candidate list.

Called by the admin preview endpoint (to show candidates for a given year
before user selection) and by the weekly refresh cron (to upsert).

Dedup precedence by `(date, country_code, name)`:
  chinesecalendar > nager > un_observed
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.services.schedule_domain.holiday_repo import HolidayEntry
from app.services.schedule_domain.holiday_sources.chinesecalendar import (
    fetch_cn_year,
)
from app.services.schedule_domain.holiday_sources.nager import fetch_intl_year
from app.services.schedule_domain.holiday_sources.un_observed import (
    compute_un_observed,
)

logger = logging.getLogger(__name__)

Status = Literal["ok", "unavailable", "failed"]


@dataclass
class SourceStatus:
    """Per-source availability report for admin UI."""

    chinesecalendar: Status
    chinesecalendar_error: str | None
    nager: Status
    nager_error: str | None
    un_observed: Status


async def collect_candidates(
    year: int,
    *,
    include_international: bool = True,
) -> tuple[list[HolidayEntry], SourceStatus]:
    """Fetch from all sources in parallel, dedup, return sorted candidates."""

    cc_result = await asyncio.to_thread(fetch_cn_year, year)

    if include_international:
        nager_task = fetch_intl_year(year)
        un_entries = compute_un_observed(year)
        nager_result = await nager_task
    else:
        nager_result = {"available": False, "entries": [], "error": "skipped"}
        un_entries = []

    status = SourceStatus(
        chinesecalendar="ok" if cc_result["available"] else "unavailable",
        chinesecalendar_error=cc_result.get("error"),
        nager=_nager_status(nager_result, include_international),
        nager_error=nager_result.get("error"),
        un_observed="ok" if include_international else "unavailable",
    )

    merged: dict[tuple[str, str, str], HolidayEntry] = {}
    # Lower-precedence first so higher-precedence overwrites.
    for entry in un_entries:
        merged[entry.unique_key()] = entry
    for entry in nager_result.get("entries", []):
        merged[entry.unique_key()] = entry
    for entry in cc_result.get("entries", []):
        merged[entry.unique_key()] = entry

    ordered = sorted(
        merged.values(),
        key=lambda e: (e.date, e.country_code, e.name),
    )
    logger.info(
        f"Holiday candidates for {year}: "
        f"cc={len(cc_result.get('entries', []))} "
        f"nager={len(nager_result.get('entries', []))} "
        f"un_observed={len(un_entries)} "
        f"merged={len(ordered)}"
    )
    return ordered, status


def _nager_status(result: dict, include_international: bool) -> Status:
    if not include_international:
        return "unavailable"
    if result.get("available"):
        return "ok"
    return "failed" if result.get("error") else "unavailable"
