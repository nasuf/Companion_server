"""Tests for in-process holiday cache + integration with time_service/time_parser."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from app.services.schedule_domain import holiday_cache
from app.services.schedule_domain.holiday_repo import HolidayEntry


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with an empty cache; fresh state on teardown."""
    holiday_cache.reset_for_testing()
    yield
    holiday_cache.reset_for_testing()


SAMPLE_ENTRIES = [
    HolidayEntry(date=date(2026, 1, 1), name="元旦", type="legal", country_code="CN"),
    HolidayEntry(date=date(2026, 2, 17), name="春节", type="legal", country_code="CN"),
    HolidayEntry(
        date=date(2026, 2, 18),
        name="春节假期",
        type="legal",
        country_code="CN",
        metadata={"canonical_name": "春节"},
    ),
    HolidayEntry(date=date(2026, 2, 15), name="调休上班", type="custom", is_workday_swap=True),
    HolidayEntry(date=date(2026, 12, 25), name="圣诞节", type="international", country_code="INTL"),
]


class TestReload:
    @pytest.mark.asyncio
    async def test_db_error_leaves_empty_cache_without_crashing(self):
        """启动时 holidays 表不存在 / 连接池耗尽时, reload 不应抛出,
        应返回零 stats + 标记 _loaded=True, 供启动继续推进。
        """
        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            side_effect=RuntimeError("relation \"holidays\" does not exist"),
        ):
            stats = await holiday_cache.reload()

        assert stats == {"total": 0, "unique_dates": 0, "unique_names": 0, "workday_swaps": 0}
        assert holiday_cache.is_loaded() is True
        assert holiday_cache.get_by_date(date(2026, 1, 1)) is None

    @pytest.mark.asyncio
    async def test_populates_cache(self):
        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=SAMPLE_ENTRIES,
        ):
            stats = await holiday_cache.reload()

        assert stats["total"] == 5
        assert holiday_cache.is_loaded() is True
        assert holiday_cache.get_by_date(date(2026, 1, 1)).name == "元旦"  # type: ignore[union-attr]
        assert holiday_cache.get_by_date(date(2026, 12, 25)).name == "圣诞节"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_prefers_non_swap_when_same_date(self):
        entries = [
            HolidayEntry(date=date(2026, 2, 15), name="调休上班", type="custom", is_workday_swap=True),
            HolidayEntry(date=date(2026, 2, 15), name="特殊节日", type="legal"),
        ]
        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=entries,
        ):
            await holiday_cache.reload()

        got = holiday_cache.get_by_date(date(2026, 2, 15))
        assert got is not None
        assert got.name == "特殊节日"

    @pytest.mark.asyncio
    async def test_canonical_name_aliases_to_same_dates(self):
        """"春节假期" (metadata.canonical_name='春节') 让 "春节" 也能解析到所有假期日."""
        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=SAMPLE_ENTRIES,
        ):
            await holiday_cache.reload()

        dates_for_spring = holiday_cache.list_dates_for_name("春节")
        assert date(2026, 2, 17) in dates_for_spring  # 主日
        assert date(2026, 2, 18) in dates_for_spring  # 假期日通过 canonical alias


class TestWorkdaySwap:
    @pytest.mark.asyncio
    async def test_flagged_correctly(self):
        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=SAMPLE_ENTRIES,
        ):
            await holiday_cache.reload()

        assert holiday_cache.is_workday_swap(date(2026, 2, 15)) is True
        assert holiday_cache.is_workday_swap(date(2026, 1, 1)) is False


class TestTimeServiceIntegration:
    """`is_holiday()` 应当优先读 cache, cache miss 时走 lunardate."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_holiday_info(self):
        from app.services.schedule_domain.time_service import is_holiday

        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=SAMPLE_ENTRIES,
        ):
            await holiday_cache.reload()

        info = is_holiday(date(2026, 1, 1))
        assert info is not None
        assert info.name == "元旦"
        assert info.type == "legal"
        assert info.days_away == 0

    @pytest.mark.asyncio
    async def test_cache_miss_falls_back_to_lunardate(self):
        """清 cache, 查某个 lunardate 能算出的日期 (如 2025 春节 = 农历正月初一)."""
        from app.services.schedule_domain.time_service import is_holiday

        # 留 cache 为空
        info = is_holiday(date(2025, 1, 29))  # 2025 农历正月初一
        assert info is not None
        assert info.name == "春节"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self):
        from app.services.schedule_domain.time_service import is_holiday
        # 随机非节日
        info = is_holiday(date(2026, 7, 4))  # 既不在 sample, 也不是农历节日
        assert info is None


class TestTimeParserIntegration:
    @pytest.mark.asyncio
    async def test_spring_festival_mention_resolves_to_actual_date(self):
        from datetime import datetime as dt_cls
        from app.services.schedule_domain.time_parser import parse_time_expressions
        from app.services.schedule_domain.time_service import _TZ

        with patch(
            "app.services.schedule_domain.holiday_cache.list_holidays",
            return_value=SAMPLE_ENTRIES,
        ):
            await holiday_cache.reload()

        results = parse_time_expressions(
            "春节前后我们聚一下",
            now=dt_cls(2026, 1, 10, 10, 0, tzinfo=_TZ),
        )
        names = [r.original_text for r in results]
        assert "春节" in names
