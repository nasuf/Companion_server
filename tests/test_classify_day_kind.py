"""Spec Part 5 §3.2: classify_day_kind 的 4 类输出."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from app.services.schedule_domain.time_service import HolidayInfo, classify_day_kind


def _info(name: str, d: date) -> HolidayInfo:
    return HolidayInfo(name=name, date=d, type="legal", days_away=0)


class TestClassifyDayKind:
    def test_holiday_wins_over_everything(self):
        """节假日优先: 即便落在周末也返回 '节假日·X'."""
        d = date(2026, 10, 3)  # 假设国庆在周六
        kind = classify_day_kind(d, holiday_info=_info("国庆节", d))
        assert kind == "节假日·国庆节"

    def test_workday_swap(self):
        """调休日 > 周末分支."""
        d = date(2026, 10, 11)  # 任意周日, mock 成调休
        with patch(
            "app.services.schedule_domain.time_service.is_holiday", return_value=None,
        ), patch(
            "app.services.schedule_domain.time_service.is_workday_swap", return_value=True,
        ):
            assert classify_day_kind(d) == "调休上班日"

    def test_weekend(self):
        d = date(2026, 4, 25)  # Saturday
        with patch(
            "app.services.schedule_domain.time_service.is_holiday", return_value=None,
        ), patch(
            "app.services.schedule_domain.time_service.is_workday_swap", return_value=False,
        ):
            assert classify_day_kind(d) == "周末"

    def test_weekday(self):
        d = date(2026, 4, 23)  # Thursday
        with patch(
            "app.services.schedule_domain.time_service.is_holiday", return_value=None,
        ), patch(
            "app.services.schedule_domain.time_service.is_workday_swap", return_value=False,
        ):
            assert classify_day_kind(d) == "工作日"

    def test_self_lookup_when_info_omitted(self):
        """不传 holiday_info 时应该自查 is_holiday."""
        d = date(2026, 10, 1)
        with patch(
            "app.services.schedule_domain.time_service.is_holiday",
            return_value=_info("国庆节", d),
        ) as mock_is_holiday:
            kind = classify_day_kind(d)
        mock_is_holiday.assert_called_once_with(d)
        assert kind == "节假日·国庆节"

    def test_prefetched_info_skips_lookup(self):
        """传了 holiday_info 时不应再查 is_holiday (避免热路径重复 cache hit)."""
        d = date(2026, 10, 1)
        with patch(
            "app.services.schedule_domain.time_service.is_holiday",
        ) as mock_is_holiday:
            classify_day_kind(d, holiday_info=_info("国庆节", d))
        mock_is_holiday.assert_not_called()
