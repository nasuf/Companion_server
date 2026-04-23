"""Holiday CRUD repository.

Backs `time_service.is_holiday()` and the admin console. Data is populated
by (a) one-time seed from legacy `app/data/holidays_cn.py`, (b) admin-driven
preview → bulk_save flow, (c) weekly refresh cron (chinesecalendar + nager).

`source="manual"` rows are treated as authoritative and never overwritten
by the refresh cron — admin wins by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from prisma import Json

from app.db import db

logger = logging.getLogger(__name__)

SOURCE_CHINESE_CALENDAR = "chinesecalendar"
SOURCE_NAGER = "nager"
SOURCE_LOCAL = "local"          # 本地源: 算法生成的国际纪念日 + 母亲/父亲节
SOURCE_MANUAL = "manual"

# Admin 手动刷新 / refresh cron 可覆盖的 source — 'manual' 永远受保护.
REFRESHABLE_SOURCES = frozenset({SOURCE_CHINESE_CALENDAR, SOURCE_NAGER, SOURCE_LOCAL})

VALID_TYPES = frozenset({"legal", "traditional", "international", "custom"})


@dataclass
class HolidayEntry:
    """In-memory representation; mirrors the Holiday model."""

    date: date
    name: str
    type: str
    country_code: str = "CN"
    is_workday_swap: bool = False
    source: str = SOURCE_MANUAL
    metadata: dict[str, Any] | None = None
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def unique_key(self) -> tuple[str, str, str]:
        return (self.date.isoformat(), self.country_code, self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "name": self.name,
            "type": self.type,
            "country_code": self.country_code,
            "is_workday_swap": self.is_workday_swap,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _to_dt(d: date) -> datetime:
    """Prisma Python 的 JSON 序列化器不认 `datetime.date`, 只认 `datetime`.
    所有传给 db.holiday.* 的 date 参数 (where / data) 都必须过这一层.
    对 DB 里的 @db.Date 字段来说, 传零时分秒 + 无时区的 datetime 是等价的.
    """
    return datetime(d.year, d.month, d.day)


def _row_to_entry(row: Any) -> HolidayEntry:
    """Convert a Prisma Holiday row to a HolidayEntry."""
    d = row.date
    if isinstance(d, datetime):
        d = d.date()
    return HolidayEntry(
        id=row.id,
        date=d,
        name=row.name,
        type=row.type,
        country_code=row.countryCode,
        is_workday_swap=row.isWorkdaySwap,
        source=row.source,
        metadata=dict(row.metadata) if row.metadata else None,
        created_at=row.createdAt,
        updated_at=row.updatedAt,
    )


async def list_holidays(
    *,
    year: int | None = None,
    country_code: str | None = None,
) -> list[HolidayEntry]:
    """List holidays matching the filters (sorted by date ASC)."""
    where: dict[str, Any] = {}
    if country_code:
        where["countryCode"] = country_code
    if year is not None:
        where["date"] = {
            "gte": _to_dt(date(year, 1, 1)),
            "lt": _to_dt(date(year + 1, 1, 1)),
        }
    rows = await db.holiday.find_many(where=where, order={"date": "asc"})
    return [_row_to_entry(r) for r in rows]


async def get_holiday_on(d: date) -> HolidayEntry | None:
    """Return the first matching Holiday for a date (any country)."""
    rows = await db.holiday.find_many(where={"date": _to_dt(d)}, take=1)
    return _row_to_entry(rows[0]) if rows else None


async def find_by_name(name: str) -> list[HolidayEntry]:
    """Return all rows with the given name — used by time_parser for
    "春节" → list of actual ISO dates lookup.
    """
    rows = await db.holiday.find_many(where={"name": name}, order={"date": "asc"})
    return [_row_to_entry(r) for r in rows]


async def upsert_many(
    entries: list[HolidayEntry],
    *,
    allow_overwrite_manual: bool = False,
) -> dict[str, int]:
    """Bulk upsert. Returns `{inserted, updated, skipped}`.

    When `allow_overwrite_manual=False` (default), rows whose existing
    `source == "manual"` are skipped — refresh-cron path sets this to
    preserve admin edits.
    """
    inserted = 0
    updated = 0
    skipped = 0
    for entry in entries:
        if entry.type not in VALID_TYPES:
            raise ValueError(f"invalid holiday type: {entry.type}")

        entry_dt = _to_dt(entry.date)
        existing = await db.holiday.find_first(
            where={
                "date": entry_dt,
                "countryCode": entry.country_code,
                "name": entry.name,
            }
        )
        if existing is None:
            create_data: dict[str, Any] = {
                "date": entry_dt,
                "name": entry.name,
                "type": entry.type,
                "countryCode": entry.country_code,
                "isWorkdaySwap": entry.is_workday_swap,
                "source": entry.source,
            }
            # Prisma Python 对 Json? 字段:
            #   - 收到 None 会抛 "A value is required but not set"
            #   - 收到原始 dict 会抛 "should be of type Json" 拒绝
            # 非空 dict 必须用 prisma.Json() 包装, None 则干脆不放 key.
            if entry.metadata:
                create_data["metadata"] = Json(entry.metadata)
            await db.holiday.create(data=create_data)  # type: ignore[arg-type]
            inserted += 1
            continue

        if not allow_overwrite_manual and existing.source == SOURCE_MANUAL:
            skipped += 1
            continue

        update_data: dict[str, Any] = {
            "type": entry.type,
            "isWorkdaySwap": entry.is_workday_swap,
            "source": entry.source,
        }
        if entry.metadata:
            update_data["metadata"] = Json(entry.metadata)
        await db.holiday.update(where={"id": existing.id}, data=update_data)  # type: ignore[arg-type]
        updated += 1

    stats = {"inserted": inserted, "updated": updated, "skipped": skipped}
    if any(stats.values()):
        logger.info(f"Holiday upsert complete: {stats}")
    return stats


async def delete_by_id(holiday_id: str) -> bool:
    try:
        await db.holiday.delete(where={"id": holiday_id})
        return True
    except Exception as e:
        logger.warning(f"Holiday delete failed ({holiday_id}): {e}")
        return False


async def count_by_source(*, year: int | None = None) -> dict[str, int]:
    """Useful for admin UI to show how many rows each source contributed."""
    entries = await list_holidays(year=year)
    stats: dict[str, int] = {}
    for e in entries:
        stats[e.source] = stats.get(e.source, 0) + 1
    return stats
