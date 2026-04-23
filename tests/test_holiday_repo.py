"""Unit tests for holiday_repo upsert + list + delete + source-protection."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.schedule_domain.holiday_repo import (
    SOURCE_CHINESE_CALENDAR,
    SOURCE_MANUAL,
    HolidayEntry,
    _row_to_entry,
    list_holidays,
    upsert_many,
)


def _fake_row(
    *,
    id: str = "h1",
    d: date = date(2026, 1, 1),
    name: str = "元旦",
    type: str = "legal",
    country: str = "CN",
    swap: bool = False,
    source: str = SOURCE_MANUAL,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        date=d,
        name=name,
        type=type,
        countryCode=country,
        isWorkdaySwap=swap,
        source=source,
        metadata=None,
        createdAt=datetime(2026, 4, 23, tzinfo=timezone.utc),
        updatedAt=datetime(2026, 4, 23, tzinfo=timezone.utc),
    )


class TestRowConversion:
    def test_row_to_entry_basic(self):
        row = _fake_row(name="春节", d=date(2026, 2, 17))
        entry = _row_to_entry(row)
        assert entry.name == "春节"
        assert entry.date == date(2026, 2, 17)
        assert entry.country_code == "CN"

    def test_row_to_entry_handles_datetime_date(self):
        # Prisma may return DateTime even when schema says Date
        row = _fake_row(d=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))  # type: ignore[arg-type]
        entry = _row_to_entry(row)
        assert entry.date == date(2026, 1, 1)


class TestListHolidays:
    @pytest.mark.asyncio
    async def test_year_filter_builds_range(self):
        mock_find = AsyncMock(return_value=[_fake_row()])
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_many = mock_find
            await list_holidays(year=2026)

        _, kwargs = mock_find.call_args
        where = kwargs["where"]
        # Prisma 序列化器不认 date, 所以 where 里必须是 datetime (零时零分零秒)
        assert where["date"]["gte"] == datetime(2026, 1, 1)
        assert where["date"]["lt"] == datetime(2027, 1, 1)

    @pytest.mark.asyncio
    async def test_country_filter_applied(self):
        mock_find = AsyncMock(return_value=[])
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_many = mock_find
            await list_holidays(country_code="INTL")

        _, kwargs = mock_find.call_args
        assert kwargs["where"]["countryCode"] == "INTL"


class TestUpsert:
    @pytest.mark.asyncio
    async def test_insert_when_no_existing_row(self):
        mock_create = AsyncMock()
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_first = AsyncMock(return_value=None)
            mock_db.holiday.create = mock_create
            mock_db.holiday.update = AsyncMock()

            stats = await upsert_many([
                HolidayEntry(date=date(2027, 1, 1), name="元旦", type="legal"),
            ])

        assert stats == {"inserted": 1, "updated": 0, "skipped": 0}
        assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_update_when_existing_non_manual(self):
        existing = _fake_row(source=SOURCE_CHINESE_CALENDAR)
        mock_update = AsyncMock()
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_first = AsyncMock(return_value=existing)
            mock_db.holiday.create = AsyncMock()
            mock_db.holiday.update = mock_update

            stats = await upsert_many([
                HolidayEntry(
                    date=date(2026, 1, 1),
                    name="元旦",
                    type="legal",
                    source=SOURCE_CHINESE_CALENDAR,
                ),
            ])

        assert stats == {"inserted": 0, "updated": 1, "skipped": 0}
        assert mock_update.await_count == 1

    @pytest.mark.asyncio
    async def test_manual_rows_are_protected_by_default(self):
        """Refresh-cron path (allow_overwrite_manual=False) must not overwrite
        admin-curated rows.
        """
        existing = _fake_row(source=SOURCE_MANUAL)
        mock_update = AsyncMock()
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_first = AsyncMock(return_value=existing)
            mock_db.holiday.create = AsyncMock()
            mock_db.holiday.update = mock_update

            stats = await upsert_many([
                HolidayEntry(
                    date=date(2026, 1, 1),
                    name="元旦",
                    type="legal",
                    source=SOURCE_CHINESE_CALENDAR,
                ),
            ], allow_overwrite_manual=False)

        assert stats == {"inserted": 0, "updated": 0, "skipped": 1}
        assert mock_update.await_count == 0

    @pytest.mark.asyncio
    async def test_manual_rows_overwritten_when_allowed(self):
        """Admin-driven bulk_save path should be able to overwrite manual rows."""
        existing = _fake_row(source=SOURCE_MANUAL)
        mock_update = AsyncMock()
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_first = AsyncMock(return_value=existing)
            mock_db.holiday.create = AsyncMock()
            mock_db.holiday.update = mock_update

            stats = await upsert_many([
                HolidayEntry(
                    date=date(2026, 1, 1),
                    name="元旦",
                    type="legal",
                    source=SOURCE_MANUAL,
                ),
            ], allow_overwrite_manual=True)

        assert stats == {"inserted": 0, "updated": 1, "skipped": 0}

    @pytest.mark.asyncio
    async def test_invalid_type_raises(self):
        with patch("app.services.schedule_domain.holiday_repo.db") as mock_db:
            mock_db.holiday = MagicMock()
            mock_db.holiday.find_first = AsyncMock()
            with pytest.raises(ValueError, match="invalid holiday type"):
                await upsert_many([
                    HolidayEntry(date=date(2026, 1, 1), name="x", type="bogus"),
                ])
