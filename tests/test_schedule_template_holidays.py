"""作息模板兜底的节假日/调休行为。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.schedule_domain.schedule import _personalize_template


def _slot(schedule: list[dict], activity: str) -> dict:
    return next(slot for slot in schedule if slot["activity"] == activity)


def test_weekend_workday_swap_uses_workday_template():
    """周末调休上班日不应被周末分支改成休闲作息。"""
    schedule = _personalize_template(
        None,
        datetime(2026, 2, 15, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        holiday=None,
        workday_swap=True,
    )

    wake = _slot(schedule, "起床洗漱")
    breakfast = _slot(schedule, "吃早餐")
    work_slots = [slot for slot in schedule if slot["type"] == "work"]

    assert wake["start"] == "07:00"
    assert wake["end"] == "08:00"
    assert breakfast["start"] == "08:00"
    assert breakfast["end"] == "09:00"
    assert len(work_slots) == 2
    assert all(slot["activity"] == "工作/学习" for slot in work_slots)
