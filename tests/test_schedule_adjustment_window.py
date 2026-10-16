"""作息调整 (spec §3.4.2) 写回作息表: 只改写调整涉及的那一截, 其余保持原样.

旧的 update_schedule_slot 只写 type/activity, 而 LLM 作息的状态读 event/status ——
AI 嘴上答应「陪你不睡了」, 顶部仍是睡眠, 下一句「在干嘛」还说在睡觉; 对模板作息又
把命中的整段 (常常是 23:00-07:00 一整夜) 改成「和用户聊天/空闲」, 用户走了 AI 一整夜
都显示空闲; 当前段按字符串比较匹配, 凌晨落在跨午夜段里时根本匹配不到。

这里守的是: 调整窗口里是新状态, 窗口外还是原来的作息。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.schedule_domain import schedule as schedule_mod
from app.services.schedule_domain.schedule import (
    _BASE_SCHEDULE_TEMPLATE,
    get_current_status,
    update_schedule_slot,
)
from tests.conftest import FakeRedis

_AGENT = "agent-1"

# spec 字段 + 生产常见的「次日 07:30」写法
_LLM_DAY = [
    {"start": "07:30", "end": "08:15", "event": "起床洗漱", "status": "空闲"},
    {"start": "09:00", "end": "18:00", "event": "在车间盯发酵", "status": "很忙碌"},
    {"start": "21:00", "end": "22:30", "event": "泡个热水澡", "status": "空闲"},
    {"start": "22:30", "end": "次日 07:30", "event": "进入梦乡", "status": "睡眠"},
]
# 夜猫子: 01:00 才睡
_NIGHT_OWL_DAY = [
    {"start": "09:00", "end": "18:00", "event": "上班", "status": "忙碌"},
    {"start": "21:00", "end": "01:00", "event": "打游戏", "status": "空闲"},
    {"start": "01:00", "end": "09:00", "event": "睡觉", "status": "睡眠"},
]
# 生成指令「不含凌晨时段」的产物: 夜里一段都没有
_NO_NIGHT_DAY = [
    {"start": "07:30", "end": "08:15", "event": "日常活动", "status": "空闲"},
    {"start": "09:00", "end": "18:00", "event": "上班", "status": "忙碌"},
    {"start": "19:00", "end": "23:00", "event": "追剧", "status": "空闲"},
]


def _at(hhmm: str, day: str = "2026-10-15") -> datetime:
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")


def _template() -> list[dict]:
    return [slot.copy() for slot in _BASE_SCHEDULE_TEMPLATE]


@pytest.fixture
def store(monkeypatch) -> FakeRedis:
    redis = FakeRedis()
    monkeypatch.setattr(schedule_mod, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(schedule_mod, "db", SimpleNamespace(
        aidailyschedule=SimpleNamespace(upsert=AsyncMock()),
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
    ))
    return redis


def _stored(redis: FakeRedis, day: str) -> list[dict] | None:
    raw = redis._store.get(f"schedule:{_AGENT}:{day.replace('-', '')}")
    return json.loads(raw) if raw else None


def _status(schedule: list[dict] | None, hhmm: str, day: str = "2026-10-15") -> tuple[str, str]:
    assert schedule is not None
    status = get_current_status(schedule, _at(hhmm, day))
    return status["status"], status["activity"]


class TestLlmSchedule:
    @pytest.mark.asyncio
    async def test_stay_up_until_named_time(self, store):
        """LLM 作息: 写 event/status, 窗口里醒着, 窗口外还是原来的作息。"""
        updated = await update_schedule_slot(
            _AGENT, _LLM_DAY, "从22:30睡觉调整为次日00:30睡觉", now=_at("22:40"),
        )

        status = get_current_status(updated, _at("23:00"))
        assert (status["status"], status["activity"]) == ("idle", "聊天")
        assert status["status_source"] == "adjustment"
        # 答应之前那 10 分钟仍是睡着的, 白天和晚上的安排都不动
        assert _status(updated, "22:35") == ("sleep", "进入梦乡")
        assert _status(updated, "21:30") == ("idle", "泡个热水澡")
        assert _status(updated, "10:00") == ("very_busy", "在车间盯发酵")
        # 窗口之后的后半夜还是那一觉
        assert _status(updated, "03:00") == ("sleep", "进入梦乡")

        window = [slot for slot in updated if slot.get("source") == "user_adjustment"]
        assert window == [{
            "event": "聊天", "activity": "聊天", "status": "空闲", "type": "leisure",
            "start": "22:40", "end": "00:30", "source": "user_adjustment",
        }]
        assert _stored(store, "2026-10-15") == updated
        assert _LLM_DAY[-1]["end"] == "次日 07:30", "调用方传进来的作息不能被就地改掉"

    @pytest.mark.asyncio
    async def test_status_is_what_the_next_turn_reads(self, store):
        """下一句「你在干嘛」读的就是写回去的这份。"""
        await update_schedule_slot(_AGENT, _LLM_DAY, "从22:30睡觉调整为陪你聊到次日01:00", now=_at("23:10"))
        stored = _stored(store, "2026-10-15")
        assert _status(stored, "23:40") == ("idle", "聊天")


class TestTemplateSchedule:
    @pytest.mark.asyncio
    async def test_only_the_window_turns_idle_not_the_whole_night(self, store):
        """模板作息旧实现把 23:00-07:00 整段改成空闲; 现在只改说好的那一小时。"""
        updated = await update_schedule_slot(
            _AGENT, _template(), "从23:00睡觉调整为再陪你一会儿", now=_at("23:30"),
        )

        assert _status(updated, "23:45") == ("idle", "聊天")
        assert _status(updated, "23:10") == ("sleep", "睡觉")
        assert _status(updated, "03:00") == ("sleep", "睡觉")
        assert _status(updated, "10:00") == ("very_busy", "工作/学习")
        idle_minutes = [
            minute for minute in range(23 * 60, 31 * 60, 5)
            if get_current_status(updated, _at("00:00") + timedelta(minutes=minute))["status"] == "idle"
        ]
        assert (idle_minutes[0], idle_minutes[-1]) == (23 * 60 + 30, 24 * 60 + 25), "默认只改 60 分钟"

        # 切开的那几截保留原来的旧字段
        night = [slot for slot in updated if slot.get("type") == "sleep" and not slot.get("source")]
        assert {(slot["start"], slot["end"]) for slot in night} == {("23:00", "23:30"), ("00:30", "07:00")}
        assert all(slot["activity"] == "睡觉" for slot in night)

    @pytest.mark.asyncio
    async def test_duration_in_adjustment_sets_window(self, store):
        updated = await update_schedule_slot(
            _AGENT, _template(), "从工作调整为陪你聊半小时", now=_at("10:00"),
        )
        assert _status(updated, "10:20") == ("idle", "聊天")
        assert _status(updated, "10:40") == ("very_busy", "工作/学习")


class TestCrossMidnight:
    @pytest.mark.asyncio
    async def test_early_morning_inside_overnight_slot(self, store):
        """01:00 落在 23:00-07:00 的凌晨那一截: 旧实现字符串比较匹配不到, 调整无效。"""
        updated = await update_schedule_slot(
            _AGENT, _template(), "从睡觉调整为再陪你聊半小时", now=_at("01:00"),
        )
        assert _status(updated, "01:15") == ("idle", "聊天")
        assert _status(updated, "01:45") == ("sleep", "睡觉")
        # 今晚那一觉照旧
        assert _status(updated, "23:30") == ("sleep", "睡觉")

    @pytest.mark.asyncio
    async def test_early_morning_window_keeps_night_inference(self, store):
        """作息不含夜里时, 凌晨陪聊结束后还是睡着的, 不能变成「两段之间的空档」。"""
        updated = await update_schedule_slot(
            _AGENT, _NO_NIGHT_DAY, "从睡觉调整为陪你聊一会儿", now=_at("01:00"),
        )
        assert _status(updated, "01:30") == ("idle", "聊天")
        status = get_current_status(updated, _at("02:30"))
        assert (status["status"], status["status_source"]) == ("sleep", "unscheduled_night")
        assert _status(updated, "23:30")[0] == "sleep"

    @pytest.mark.asyncio
    async def test_past_midnight_part_goes_to_tomorrows_schedule(self, store, monkeypatch):
        """答应陪到 1 点: 过了 0 点今天的作息才生成, 那一截要补进去, 否则 AI 又睡着了。"""
        await update_schedule_slot(
            _AGENT, _LLM_DAY, "从22:30睡觉调整为次日01:00睡觉", now=_at("23:30"),
        )
        assert _stored(store, "2026-10-16") is None

        from app.services import runtime_config
        monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock())
        tomorrow = await schedule_mod.generate_daily_schedule(
            _AGENT, "小满", None, life_overview=None, date=_at("00:05", "2026-10-16"),
        )

        assert _status(tomorrow, "00:30", "2026-10-16") == ("idle", "聊天")
        assert _status(tomorrow, "01:30", "2026-10-16") == ("sleep", "睡觉")
        assert _stored(store, "2026-10-16") == tomorrow

    @pytest.mark.asyncio
    async def test_past_midnight_part_patches_existing_tomorrow(self, store):
        store._store[f"schedule:{_AGENT}:20261016"] = json.dumps(_template(), ensure_ascii=False)
        await update_schedule_slot(
            _AGENT, _LLM_DAY, "从22:30睡觉调整为次日01:00睡觉", now=_at("23:30"),
        )
        tomorrow = _stored(store, "2026-10-16")
        assert _status(tomorrow, "00:30", "2026-10-16") == ("idle", "聊天")
        assert _status(tomorrow, "01:30", "2026-10-16") == ("sleep", "睡觉")
        assert _status(tomorrow, "23:30", "2026-10-16") == ("sleep", "睡觉")

    @pytest.mark.asyncio
    async def test_later_adjustment_is_not_overwritten_by_carry(self, store, monkeypatch):
        """顺延只在生成时补一次; 之后同一天再调整, 不能被它盖回去。"""
        await update_schedule_slot(_AGENT, _template(), "从23:00睡觉调整为次日01:00睡觉", now=_at("23:30"))
        from app.services import runtime_config
        monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock())
        tomorrow = await schedule_mod.generate_daily_schedule(
            _AGENT, "小满", None, life_overview=None, date=_at("00:05", "2026-10-16"),
        )
        updated = await update_schedule_slot(
            _AGENT, tomorrow, "从聊天调整为现在就去睡觉", now=_at("00:20", "2026-10-16"),
        )
        assert _status(updated, "00:40", "2026-10-16") == ("sleep", "睡觉")
        assert _status(_stored(store, "2026-10-16"), "00:40", "2026-10-16")[0] == "sleep"


class TestSleepDirection:
    @pytest.mark.asyncio
    async def test_earlier_bedtime_sleeps_from_then_until_planned_sleep(self, store):
        updated = await update_schedule_slot(
            _AGENT, _NIGHT_OWL_DAY, "从01:00睡觉调整为23:00睡觉", now=_at("22:00"),
        )
        assert _status(updated, "22:30") == ("idle", "打游戏")
        assert _status(updated, "23:30") == ("sleep", "睡觉")
        assert _status(updated, "00:30")[0] == "sleep"

    @pytest.mark.asyncio
    async def test_go_to_sleep_now_until_planned_sleep(self, store):
        updated = await update_schedule_slot(
            _AGENT, _template(), "从看剧调整为现在就去睡觉", now=_at("21:30"),
        )
        assert _status(updated, "21:20") == ("idle", "看剧/看书")
        assert _status(updated, "22:30") == ("sleep", "睡觉")
        assert _status(updated, "23:30") == ("sleep", "睡觉")

    @pytest.mark.asyncio
    async def test_postponed_bedtime_keeps_what_comes_before(self, store):
        """「次日01:00睡觉」是推迟入睡: 原定睡着的那截改成醒着, 睡前的安排不动。"""
        updated = await update_schedule_slot(
            _AGENT, _template(), "从23:00睡觉调整为次日01:00睡觉", now=_at("22:30"),
        )
        assert _status(updated, "22:45") == ("busy", "准备睡觉")
        assert _status(updated, "23:30") == ("idle", "聊天")
        assert _status(updated, "01:30") == ("sleep", "睡觉")

    @pytest.mark.asyncio
    async def test_sleep_words_that_mean_staying_up(self, store):
        updated = await update_schedule_slot(
            _AGENT, _template(), "从23:00睡觉调整为晚点睡陪你", now=_at("23:30"),
        )
        assert _status(updated, "23:45") == ("idle", "聊天")


class TestBoundsAndUnparsed:
    @pytest.mark.asyncio
    async def test_window_is_capped_at_three_hours(self, store):
        updated = await update_schedule_slot(
            _AGENT, _template(), "从23:00睡觉调整为陪你聊到早上06:00", now=_at("23:00"),
        )
        assert _status(updated, "01:50") == ("idle", "聊天")
        assert _status(updated, "02:10") == ("sleep", "睡觉")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("adjustment, when", [
        # 时刻刚过去 (多半是 LLM 照抄了原定时刻), 不当成明天
        ("从23:00睡觉调整为陪你到23:00", "23:10"),
        # 已经在睡还说去睡, 什么都不用改
        ("从看剧调整为睡觉", "23:30"),
    ])
    async def test_unusable_adjustment_leaves_schedule_alone(self, store, adjustment, when):
        original = _template()
        updated = await update_schedule_slot(_AGENT, original, adjustment, now=_at(when))
        assert updated is original
        assert _stored(store, "2026-10-15") is None

    @pytest.mark.parametrize("text, expected", [
        ("次日01:00睡觉", [110]),
        ("1点睡", [110]),
        ("凌晨1点半", [140]),
        # 刚过去的 23:00, 下一次在明晚; 由 _ADJUST_CLOCK_HORIZON 挡掉, 不在这里过滤
        ("晚上11点", [1430]),
        ("晚一点睡", None),
    ])
    def test_clock_offsets(self, text, expected):
        assert schedule_mod._clock_offsets(text, 23 * 60 + 10) == expected

    @pytest.mark.parametrize("text, expected", [
        ("再陪你半小时", 30),
        ("推迟一个半小时", 90),
        ("多聊20分钟", 20),
        ("两个小时", 120),
        ("一会儿", None),
    ])
    def test_duration_minutes(self, text, expected):
        assert schedule_mod._duration_minutes(text) == expected
