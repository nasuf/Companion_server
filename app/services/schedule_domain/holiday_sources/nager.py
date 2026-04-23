"""Fetch international holidays from nager.date.

We anchor on US (the broadest coverage of Western holidays relevant to
Chinese consumers: Valentine's, Mother's/Father's Day, Halloween,
Thanksgiving, Christmas). UN-observed days (妇女节, 儿童节) and
algorithmic holidays are handled in `un_observed.py` — nager doesn't
return them as US federal holidays.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from app.services.schedule_domain.holiday_repo import SOURCE_NAGER, HolidayEntry

logger = logging.getLogger(__name__)

_ENDPOINT = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
_TIMEOUT_S = 10.0
_DEFAULT_COUNTRY = "US"

# nager 返回的英文 localName/name 映射到中文。未知条目保留英文名。
_NAME_MAP = {
    "Valentine's Day": "情人节",
    "Mother's Day": "母亲节",
    "Father's Day": "父亲节",
    "Halloween": "万圣节",
    "Thanksgiving": "感恩节",
    "Christmas Eve": "平安夜",
    "Christmas Day": "圣诞节",
    "New Year's Day": "元旦",
    "Independence Day": "美国独立日",
    "Memorial Day": "阵亡将士纪念日",
    "Labor Day": "美国劳动节",
    "Martin Luther King, Jr. Day": "马丁·路德·金纪念日",
    "Washington's Birthday": "华盛顿诞辰",
    "Columbus Day": "哥伦布日",
    "Veterans Day": "退伍军人节",
    "Juneteenth": "六月节",
}


async def fetch_intl_year(
    year: int,
    *,
    country: str = _DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """Return `{available, entries, error}`.

    entries: list[HolidayEntry] with country_code='INTL' (treat as non-CN
    international holidays regardless of origin country).
    """
    url = _ENDPOINT.format(year=year, country=country)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except httpx.HTTPError as e:
        logger.warning(f"nager.date fetch failed for {year}/{country}: {e}")
        return {"available": False, "entries": [], "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        logger.warning(f"nager.date unexpected error for {year}/{country}: {e}")
        return {"available": False, "entries": [], "error": str(e)}

    entries: list[HolidayEntry] = []
    for row in raw:
        try:
            d = date.fromisoformat(row["date"])
        except (KeyError, ValueError):
            continue
        local = str(row.get("localName") or row.get("name") or "").strip()
        english = str(row.get("name") or "").strip()
        name = _NAME_MAP.get(english, _NAME_MAP.get(local, local or english))
        if not name:
            continue
        entries.append(HolidayEntry(
            date=d,
            name=name,
            type="international",
            country_code="INTL",
            is_workday_swap=False,
            source=SOURCE_NAGER,
            metadata={
                "anchor_country": country,
                "english_name": english,
                "local_name": local,
                "types": row.get("types"),
            },
        ))
    return {"available": True, "entries": entries, "error": None}
