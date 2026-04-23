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
from datetime import date, datetime, time
from typing import Any, Literal, get_args

from prisma import Json

from app.db import db

logger = logging.getLogger(__name__)

SOURCE_CHINESE_CALENDAR = "chinesecalendar"
SOURCE_NAGER = "nager"
SOURCE_LOCAL = "local"          # 本地源: 算法生成的国际纪念日 + 母亲/父亲节
SOURCE_MANUAL = "manual"

HolidayType = Literal["legal", "traditional", "international", "custom"]
VALID_TYPES = frozenset(get_args(HolidayType))


@dataclass
class HolidayEntry:
    """In-memory representation; mirrors the Holiday model."""

    date: date
    name: str
    type: HolidayType
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
    """Prisma Python 的 JSON 序列化器不认 `date`, 只认 `datetime`.
    所有传给 db.holiday.* 的 date 参数 (where / data) 都必须过这一层.
    对 DB 里的 @db.Date 字段来说, 零时分秒的 naive datetime 等价于 date.
    """
    return datetime.combine(d, time.min)


def _build_data(entry: "HolidayEntry", *, is_create: bool) -> dict[str, Any]:
    """构造 create/update 共享的 data dict.

    is_create=True: 用于 create (带上 date/name/countryCode 不可变键).
    is_create=False: 用于 update (只带可变字段, 避免 Prisma 参数噪声).

    metadata 为 None 则不放 key (走 DB DEFAULT NULL); 非空字典必须用
    `prisma.Json()` 包装. 空字典 `{}` 走有效 Json([]) 入库 (不特判).
    """
    data: dict[str, Any] = {
        "type": entry.type,
        "isWorkdaySwap": entry.is_workday_swap,
        "source": entry.source,
    }
    if is_create:
        data["date"] = _to_dt(entry.date)
        data["name"] = entry.name
        data["countryCode"] = entry.country_code
    if entry.metadata is not None:
        data["metadata"] = Json(entry.metadata)
    return data


def _row_to_entry(row: Any) -> HolidayEntry:
    """Convert a Prisma Holiday row to a HolidayEntry.

    Runtime trust: `row.type` is a DB str; upsert_many gatekeeps writes
    so values are always in VALID_TYPES.
    """
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
        metadata=row.metadata or None,
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


async def upsert_many(
    entries: list[HolidayEntry],
    *,
    allow_overwrite_manual: bool = False,
) -> dict[str, int]:
    """Bulk upsert. Returns `{inserted, updated, skipped}`.

    When `allow_overwrite_manual=False` (default), rows whose existing
    `source == "manual"` are skipped — refresh path sets this to preserve
    admin edits.

    Performance: one batched `find_many` for existence lookup (not 1 per
    entry), then N parallel `create`/`update` calls as needed.
    """
    if not entries:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    for entry in entries:
        if entry.type not in VALID_TYPES:
            raise ValueError(f"invalid holiday type: {entry.type}")

    # 批量查已存行, 避免 N 次 find_first 串行 roundtrip
    min_date = min(e.date for e in entries)
    max_date = max(e.date for e in entries)
    countries = list({e.country_code for e in entries})
    existing_rows = await db.holiday.find_many(
        where={
            "date": {"gte": _to_dt(min_date), "lte": _to_dt(max_date)},
            "countryCode": {"in": countries},
        },
    )
    existing_by_key: dict[tuple[date, str, str], Any] = {}
    for row in existing_rows:
        row_date = row.date.date() if isinstance(row.date, datetime) else row.date
        existing_by_key[(row_date, row.countryCode, row.name)] = row

    inserted = 0
    updated = 0
    skipped = 0
    for entry in entries:
        existing = existing_by_key.get(
            (entry.date, entry.country_code, entry.name)
        )
        if existing is None:
            await db.holiday.create(  # type: ignore[arg-type]
                data=_build_data(entry, is_create=True),
            )
            inserted += 1
            continue

        if not allow_overwrite_manual and existing.source == SOURCE_MANUAL:
            skipped += 1
            continue

        await db.holiday.update(  # type: ignore[arg-type]
            where={"id": existing.id},
            data=_build_data(entry, is_create=False),
        )
        updated += 1

    stats = {"inserted": inserted, "updated": updated, "skipped": skipped}
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
