"""合并 chinesecalendar + nager.date + 本地源 成为统一候选列表.

Called by the admin preview endpoint (to show candidates for a given year
before user selection) and by the manual refresh path (to upsert).

按 `(date, country_code, name)` 去重，优先级: chinesecalendar > nager > local.

Sources 参数允许精细控制要查哪几个源; 默认全查.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.services.schedule_domain.holiday_repo import (
    SOURCE_CHINESE_CALENDAR,
    SOURCE_LOCAL,
    SOURCE_NAGER,
    HolidayEntry,
)
from app.services.schedule_domain.holiday_sources.chinesecalendar import (
    fetch_cn_year,
)
from app.services.schedule_domain.holiday_sources.local import compute_local
from app.services.schedule_domain.holiday_sources.nager import fetch_intl_year

logger = logging.getLogger(__name__)

Status = Literal["ok", "unavailable", "failed"]

ALL_SOURCES = frozenset({SOURCE_CHINESE_CALENDAR, SOURCE_NAGER, SOURCE_LOCAL})


@dataclass
class SourceStatus:
    """Per-source availability report for admin UI."""

    chinesecalendar: Status
    chinesecalendar_error: str | None
    nager: Status
    nager_error: str | None
    local: Status

    def to_response_dict(self) -> dict[str, dict[str, str | None]]:
        """Serialize for the /preview response. Extend here when new
        sources are added instead of editing the admin endpoint."""
        return {
            "chinesecalendar": {
                "status": self.chinesecalendar,
                "error": self.chinesecalendar_error,
            },
            "nager": {"status": self.nager, "error": self.nager_error},
            "local": {"status": self.local, "error": None},
        }


async def collect_candidates(
    year: int,
    *,
    sources: set[str] | frozenset[str] | None = None,
) -> tuple[list[HolidayEntry], SourceStatus]:
    """Fetch requested sources in parallel, dedup, return sorted candidates.

    `sources` 控制查询范围 (任一子集 of `{chinesecalendar, nager, local}`).
    None 等同于 ALL_SOURCES. 未被请求的源 status='unavailable', entries 为空.
    """
    sources = frozenset(sources) if sources is not None else ALL_SOURCES

    # CC (同步库, 放线程池) + nager (async HTTP) 并发跑: nager HTTP 0.5-2s,
    # CC CPU ~0.1s, 串行会白等一个. compute_local 是纯同步瞬时, 直接算.
    cc_coro = (
        asyncio.to_thread(fetch_cn_year, year)
        if SOURCE_CHINESE_CALENDAR in sources
        else _skipped_result()
    )
    nager_coro = (
        fetch_intl_year(year)
        if SOURCE_NAGER in sources
        else _skipped_result()
    )
    cc_result, nager_result = await asyncio.gather(cc_coro, nager_coro)
    local_entries = compute_local(year) if SOURCE_LOCAL in sources else []

    # chinesecalendar 的 "no data for year" 是年份未覆盖, 语义上是 unavailable
    # 不是 failed (不像 nager 的 HTTPError 那样区分网络错误). 所以 CC 只分
    # requested/ok/unavailable 三态, 不引入 failed.
    status = SourceStatus(
        chinesecalendar=(
            "ok"
            if SOURCE_CHINESE_CALENDAR in sources and cc_result.get("available")
            else "unavailable"
        ),
        chinesecalendar_error=cc_result.get("error"),
        nager=_nager_status(nager_result, SOURCE_NAGER in sources),
        nager_error=nager_result.get("error"),
        local="ok" if SOURCE_LOCAL in sources else "unavailable",
    )

    merged: dict[tuple[str, str, str], HolidayEntry] = {}
    # Lower-precedence first so higher-precedence overwrites.
    for entry in local_entries:
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
        f"local={len(local_entries)} "
        f"merged={len(ordered)}"
    )
    return ordered, status


async def _skipped_result() -> dict:
    """占位协程: 该源未被请求时, 给 gather 一个 awaitable 填位."""
    return {"available": False, "entries": [], "error": None}


def _nager_status(result: dict, requested: bool) -> Status:
    if not requested:
        return "unavailable"
    if result.get("available"):
        return "ok"
    return "failed" if result.get("error") else "unavailable"
