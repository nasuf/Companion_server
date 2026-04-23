"""Tests for holiday source clients + merger dedup logic."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from app.services.schedule_domain.holiday_repo import (
    SOURCE_CHINESE_CALENDAR,
    SOURCE_NAGER,
    SOURCE_UN_OBSERVED,
    HolidayEntry,
)
from app.services.schedule_domain.holiday_sources.chinesecalendar import (
    fetch_cn_year,
    _normalize_cn_name,
)
from app.services.schedule_domain.holiday_sources.merger import collect_candidates
from app.services.schedule_domain.holiday_sources.un_observed import (
    compute_un_observed,
    _nth_weekday,
)


class TestChineseCalendar:
    def test_fetch_2025_returns_populated(self):
        result = fetch_cn_year(2025)
        assert result["available"] is True
        # 2025 should have many entries (元旦, 春节, 清明, 劳动, 端午, 中秋, 国庆)
        assert len(result["entries"]) > 20
        names = {e.name for e in result["entries"]}
        assert "元旦" in names
        assert "春节" in names
        assert "国庆节" in names

    def test_fetch_future_year_degrades(self):
        # 2030 is way beyond any chinesecalendar release
        result = fetch_cn_year(2030)
        assert result["available"] is False
        assert "no data" in result["error"].lower() or "implemented" in result["error"].lower()

    def test_consecutive_days_get_suffix(self):
        result = fetch_cn_year(2026)
        assert result["available"]
        # 春节 should appear once; 春节假期 multiple times
        main_days = [e for e in result["entries"] if e.name == "春节"]
        holiday_days = [e for e in result["entries"] if e.name == "春节假期"]
        assert len(main_days) == 1
        assert len(holiday_days) >= 3  # 春节假期至少 3 天

    def test_workday_swap_flagged(self):
        result = fetch_cn_year(2025)
        assert result["available"]
        swaps = [e for e in result["entries"] if e.is_workday_swap]
        assert len(swaps) >= 3  # 2025 has 5 workday swaps
        for swap in swaps:
            assert swap.name == "调休上班"
            assert swap.type == "custom"

    def test_normalize_name_case_insensitive(self):
        assert _normalize_cn_name("Tomb-sweeping Day") == "清明节"
        assert _normalize_cn_name("TOMB-SWEEPING DAY") == "清明节"
        assert _normalize_cn_name("Unknown Festival") == "Unknown Festival"


class TestUnObserved:
    def test_fixed_dates(self):
        entries = compute_un_observed(2026)
        names = {e.name: e.date for e in entries}
        assert names["妇女节"] == date(2026, 3, 8)
        assert names["儿童节"] == date(2026, 6, 1)
        assert names["国际劳动节"] == date(2026, 5, 1)

    def test_mothers_day_2025_is_may_11(self):
        # 母亲节 = 5 月第 2 个周日. 2025-05-11 是周日.
        entries = compute_un_observed(2025)
        mothers = [e for e in entries if e.name == "母亲节"][0]
        assert mothers.date == date(2025, 5, 11)

    def test_fathers_day_2025_is_june_15(self):
        # 父亲节 = 6 月第 3 个周日. 2025-06-15 是周日.
        entries = compute_un_observed(2025)
        fathers = [e for e in entries if e.name == "父亲节"][0]
        assert fathers.date == date(2025, 6, 15)

    def test_nth_weekday_helper(self):
        # 2025-05 第 2 个周日 = 11
        assert _nth_weekday(2025, 5, 6, 2) == date(2025, 5, 11)
        # 2025-11 第 4 个周四 = 27 (Thanksgiving)
        assert _nth_weekday(2025, 11, 3, 4) == date(2025, 11, 27)

    def test_all_marked_international_intl_country(self):
        entries = compute_un_observed(2026)
        for e in entries:
            assert e.type == "international"
            assert e.country_code == "INTL"
            assert e.source == SOURCE_UN_OBSERVED


class TestMerger:
    @pytest.mark.asyncio
    async def test_cc_unavailable_still_returns_intl_entries(self):
        """即使 chinesecalendar 空 (2030), 国际源仍可用 → 返回 nager + un_observed."""
        cc_stub = {"available": False, "entries": [], "error": "no data for 2030"}
        nager_stub = {
            "available": True,
            "entries": [HolidayEntry(
                date=date(2030, 12, 25),
                name="圣诞节",
                type="international",
                country_code="INTL",
                source=SOURCE_NAGER,
            )],
            "error": None,
        }
        with patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_cn_year",
            return_value=cc_stub,
        ), patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_intl_year",
            new=AsyncMock(return_value=nager_stub),
        ):
            entries, status = await collect_candidates(2030, include_international=True)

        assert status.chinesecalendar == "unavailable"
        assert status.nager == "ok"
        assert status.un_observed == "ok"
        names = {e.name for e in entries}
        assert "圣诞节" in names
        assert "妇女节" in names  # from un_observed

    @pytest.mark.asyncio
    async def test_cc_overrides_un_observed_on_same_key(self):
        """chinesecalendar > un_observed when (date, country, name) collides."""
        cc_stub = {
            "available": True,
            "entries": [HolidayEntry(
                date=date(2026, 5, 1),
                name="劳动节",
                type="legal",
                country_code="CN",
                source=SOURCE_CHINESE_CALENDAR,
            )],
            "error": None,
        }
        nager_stub = {"available": True, "entries": [], "error": None}
        with patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_cn_year",
            return_value=cc_stub,
        ), patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_intl_year",
            new=AsyncMock(return_value=nager_stub),
        ):
            entries, _ = await collect_candidates(2026, include_international=True)

        labor_cn = [e for e in entries if e.country_code == "CN" and e.name == "劳动节"]
        assert len(labor_cn) == 1
        assert labor_cn[0].source == SOURCE_CHINESE_CALENDAR

    @pytest.mark.asyncio
    async def test_skip_international_when_flag_false(self):
        cc_stub = {
            "available": True,
            "entries": [HolidayEntry(
                date=date(2026, 1, 1),
                name="元旦",
                type="legal",
                country_code="CN",
                source=SOURCE_CHINESE_CALENDAR,
            )],
            "error": None,
        }
        with patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_cn_year",
            return_value=cc_stub,
        ):
            entries, status = await collect_candidates(2026, include_international=False)

        assert status.nager == "unavailable"
        assert status.un_observed == "unavailable"
        # 只剩 CN 条目
        assert all(e.country_code == "CN" for e in entries)

    @pytest.mark.asyncio
    async def test_nager_failure_reported_as_failed(self):
        cc_stub = {"available": True, "entries": [], "error": None}
        nager_stub = {"available": False, "entries": [], "error": "HTTPError: timeout"}
        with patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_cn_year",
            return_value=cc_stub,
        ), patch(
            "app.services.schedule_domain.holiday_sources.merger.fetch_intl_year",
            new=AsyncMock(return_value=nager_stub),
        ):
            _, status = await collect_candidates(2026, include_international=True)

        assert status.nager == "failed"
        assert status.nager_error is not None
        assert "HTTPError" in status.nager_error
