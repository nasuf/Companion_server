"""作息表没写到的时刻, AI 的状态从哪来.

2026-10-14 生产会话: 23:40 回「刚醒」(作息里是 22:30-次日 07:30 睡眠), 34 分钟
后过了午夜再问, 改口「刚摸鱼呢」「上班偷偷刷手机」「没睡呀」。原因是新一天的作息
按生成指令「不含凌晨时段」从 07:30 才开始, 00:14 一段都没命中, 回落成「自由时间/
空闲」, 而且兜底返回值缺 activity 键 —— 提示词里成了「你现在在，有空聊天。」,
主回复的状态约束段因为活动为空整段跳过, 模型只能现编。顶部状态同一个函数算,
一起显示「空闲」, 或者在当天作息生成前干脆消失。

这里守的是: 夜里永远是睡着的, 状态永远带着活动, 「交流状态」只决定回得快不快。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.schedule_domain.schedule import (
    format_schedule_context,
    get_current_status,
)

# 生产 agent 2e643f0a 2026-10-14 的作息 (节选, 保留原始时刻写法)
_DAY_WITH_CIRI = [
    {"start": "07:30", "end": "08:15", "event": "日常活动", "status": "空闲"},
    {"start": "08:30", "end": "09:00", "event": "日常活动", "status": "忙碌"},
    {"start": "20:30", "end": "22:00", "event": "泡个热水澡", "status": "空闲"},
    {"start": "22:00", "end": "22:30", "event": "躺在床上刷手机", "status": "空闲"},
    {"start": "22:30", "end": "次日 07:30", "event": "进入梦乡，梦里全是发酵的气泡声。", "status": "睡眠"},
]
# 2026-10-15 那份: 07:30 才开始, 23:00 就结束, 夜里一段都没有
_DAY_WITHOUT_NIGHT = [
    {"start": "07:30", "end": "08:15", "event": "日常活动", "status": "空闲"},
    {"start": "09:00", "end": "11:30", "event": "监控发酵温度", "status": "很忙碌"},
    {"start": "22:30", "end": "23:00", "event": "检查明日发酵设备状态，准备入睡", "status": "空闲"},
]


def _at(hhmm: str, day: str = "2026-10-15") -> datetime:
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")


class TestNightIsAsleep:
    @pytest.mark.parametrize("when", ["00:14", "03:00", "07:29"])
    def test_before_the_first_slot_is_sleep(self, when):
        status = get_current_status(_DAY_WITHOUT_NIGHT, _at(when))
        assert status["status"] == "sleep"
        assert status["status_source"] == "unscheduled_night"

    @pytest.mark.parametrize("when", ["23:00", "23:30", "23:59"])
    def test_after_the_last_slot_is_sleep(self, when):
        assert get_current_status(_DAY_WITHOUT_NIGHT, _at(when))["status"] == "sleep"

    def test_first_slot_still_wins_once_it_starts(self):
        status = get_current_status(_DAY_WITHOUT_NIGHT, _at("07:30"))
        assert (status["status"], status["status_source"]) == ("idle", "slot")

    def test_gap_between_daytime_slots_is_free_time(self):
        """两段之间漏掉的空档才是自由时间, 不能也判成睡着。"""
        status = get_current_status(_DAY_WITHOUT_NIGHT, _at("08:30"))
        assert status["status"] == "idle"
        assert status["status_source"] == "unscheduled_gap"
        assert status["activity"] == "自由时间"

    def test_upcoming_survives_inference(self):
        """调用方直接读 status["upcoming"]; 推断出来的状态也得带上这个键。"""
        status = get_current_status(_DAY_WITHOUT_NIGHT, _at("00:14"))
        assert status["upcoming"].startswith("07:30")


class TestNextDayClockNotation:
    def test_ciri_end_counts_as_tonight(self):
        status = get_current_status(_DAY_WITH_CIRI, _at("23:40", "2026-10-14"))
        assert status["status"] == "sleep"
        assert status["activity"] == "进入梦乡，梦里全是发酵的气泡声。"

    def test_ciri_tail_covers_early_morning(self):
        """「次日 07:30」按时刻解析后, 凌晨那一截也命中 (与 23:00-07:00 的旧语义一致)。"""
        status = get_current_status(_DAY_WITH_CIRI, _at("00:14"))
        assert (status["status"], status["status_source"]) == ("sleep", "slot")
        assert get_current_status(_DAY_WITH_CIRI, _at("07:30"))["status"] == "idle"

    @pytest.mark.parametrize("end", ["次日07:30", "次日 07:30", "第二天07:30", "07:30", "次日 7：30"])
    def test_end_spellings_seen_in_production(self, end):
        schedule = [
            {"start": "08:00", "end": "22:00", "event": "上班", "status": "忙碌"},
            {"start": "22:00", "end": end, "event": "睡觉", "status": "睡眠"},
        ]
        assert get_current_status(schedule, _at("05:00"))["status"] == "sleep"
        assert get_current_status(schedule, _at("23:00"))["status"] == "sleep"

    @pytest.mark.parametrize("end", ["24:00", "00:00", "23:59"])
    def test_midnight_spellings_end_the_evening_slot(self, end):
        schedule = [
            {"start": "08:00", "end": "23:00", "event": "上班", "status": "忙碌"},
            {"start": "23:00", "end": end, "event": "追剧", "status": "空闲"},
        ]
        status = get_current_status(schedule, _at("23:30"))
        assert (status["status"], status["activity"]) == ("idle", "追剧")

    @pytest.mark.parametrize("bad", ["25:00", "24:30", "07:61", "", None])
    def test_impossible_clock_values_are_rejected(self, bad):
        from app.services.schedule_domain.schedule import _clock_minutes

        assert _clock_minutes(bad) is None

    def test_unparseable_slots_are_skipped_not_fatal(self):
        schedule = [
            {"start": "上午", "end": "中午", "event": "???", "status": "忙碌"},
            {"start": "09:00", "end": "18:00", "event": "上班", "status": "忙碌"},
        ]
        assert get_current_status(schedule, _at("10:00"))["activity"] == "上班"
        assert get_current_status(schedule, _at("20:00"))["status"] == "sleep"


class TestActivityIsNeverEmpty:
    def test_inferred_statuses_carry_activity(self):
        for when in ("00:14", "08:30", "23:30"):
            status = get_current_status(_DAY_WITHOUT_NIGHT, _at(when))
            assert status["activity"] and status["event"] == status["activity"]

    def test_slot_without_event_gets_a_status_default(self):
        schedule = [{"start": "09:00", "end": "18:00", "event": "", "status": "忙碌"}]
        status = get_current_status(schedule, _at("10:00"))
        assert status["activity"] == "忙手头的事"

    @pytest.mark.parametrize("status", [
        {"status": "idle", "event": "自由时间"},
        {"status": "idle"},
        {"status": "busy", "activity": ""},
        {"status": "very_busy"},
    ])
    def test_context_never_renders_an_empty_activity(self, status):
        rendered = format_schedule_context(status)
        assert "在，" not in rendered and "正在，" not in rendered and "忙，" not in rendered

    def test_inferred_sleep_is_not_rendered_twice(self):
        status = get_current_status(_DAY_WITHOUT_NIGHT, _at("00:14"))
        assert format_schedule_context(status) == "你现在正在睡觉，可能会延迟回复。"

    def test_scheduled_sleep_keeps_its_detail(self):
        status = get_current_status(_DAY_WITH_CIRI, _at("23:40", "2026-10-14"))
        assert "发酵的气泡声" in format_schedule_context(status)


class TestConversationModeDoesNotOverrideSchedule:
    """「交流状态」(距上次回复 <30min) 只换延迟分布, 不能把作息改写成空闲。

    顶部状态与回复里的作息都来自 received_status / get_current_status, 交流状态
    要是哪天开始改写它, 就会重现「明明在睡觉却说在摸鱼」。
    """

    @pytest.mark.asyncio
    async def test_received_status_passes_through_conversation_mode(self, monkeypatch):
        from app.services import runtime_config
        from app.services.interaction import reply_context

        monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
        monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
        monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
            "replyDelayEnabled": True, "replyDelayMaxSeconds": 3600,
        })
        now = datetime.now(timezone.utc)
        monkeypatch.setattr(
            reply_context, "get_last_reply_timestamp",
            AsyncMock(return_value=now - timedelta(minutes=3)),
        )
        sleeping = get_current_status(_DAY_WITHOUT_NIGHT, _at("00:14"))
        ctx = await reply_context.build_reply_timing_context(
            agent_id="a", user_id="u", received_status=sleeping, received_at=now,
        )
        assert ctx["interaction_mode"] == "conversation_mode"
        assert ctx["delay_seconds"] <= 5
        assert ctx["received_status"]["status"] == "sleep"
        assert ctx["received_status"]["activity"] == "睡觉"


class TestHeaderStatusBeforeTodaysScheduleExists:
    """今天的作息要等 03:30 任务或第一条消息才生成; 这之前顶部不能消失, 也不能
    拿昨天的白天安排冒充今天。"""

    @staticmethod
    def _freeze(monkeypatch, api, hhmm: str):
        tz = ZoneInfo("Asia/Shanghai")
        frozen = datetime.strptime(f"2026-10-15 {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen.astimezone(tz) if tz else frozen.replace(tzinfo=None)

        monkeypatch.setattr(api, "datetime", _Frozen)

    @staticmethod
    def _schedules(monkeypatch, api, by_date: dict[str, list[dict] | None]):
        async def cached(agent_id, day):
            return by_date.get(day.strftime("%Y-%m-%d"))

        monkeypatch.setattr(api, "get_cached_schedule", cached)
        monkeypatch.setattr(api, "db", SimpleNamespace(
            aidailyschedule=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
            aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
        ))

    @pytest.mark.asyncio
    async def test_last_nights_sleep_shows_until_today_is_generated(self, monkeypatch):
        from app.api.public import conversations as api

        self._freeze(monkeypatch, api, "00:14")
        self._schedules(monkeypatch, api, {"2026-10-14": _DAY_WITH_CIRI})
        result = await api._current_ai_status("agent")
        assert result is not None
        assert result["ai_status"] == "sleep"
        assert result["ai_status_label"] == "睡眠"

    @pytest.mark.asyncio
    async def test_yesterday_daytime_is_not_reused(self, monkeypatch):
        from app.api.public import conversations as api

        self._freeze(monkeypatch, api, "10:00")
        self._schedules(monkeypatch, api, {"2026-10-14": _DAY_WITH_CIRI})
        assert await api._current_ai_status("agent") is None

    @pytest.mark.asyncio
    async def test_todays_schedule_wins_when_present(self, monkeypatch):
        from app.api.public import conversations as api

        self._freeze(monkeypatch, api, "00:14")
        self._schedules(monkeypatch, api, {
            "2026-10-14": _DAY_WITH_CIRI, "2026-10-15": _DAY_WITHOUT_NIGHT,
        })
        result = await api._current_ai_status("agent")
        assert result is not None
        assert (result["ai_status"], result["ai_activity"]) == ("sleep", "睡觉")


class TestDailyCronKeepsSameDaySchedule:
    @pytest.mark.asyncio
    async def test_existence_check_reads_the_day_key_only(self, monkeypatch):
        from app.services.schedule_domain import schedule

        redis = SimpleNamespace(exists=AsyncMock(return_value=1))
        monkeypatch.setattr(schedule, "get_redis", AsyncMock(return_value=redis))
        assert await schedule.has_cached_schedule("a", _at("03:30")) is True
        redis.exists.assert_awaited_once_with("schedule:a:20261015")


    @pytest.mark.asyncio
    async def test_cron_skips_agents_that_already_have_today(self, monkeypatch):
        """午夜后聊天时现场生成的作息, AI 已经照着它说过话; 03:30 不能整份换掉。"""
        from jobs import scheduler

        captured = {}

        async def run_all(fn, **_kwargs):
            captured["gen"] = fn

        async def run_job(_name, _ttl, body):
            await body()

        generate = AsyncMock()
        monkeypatch.setattr(scheduler, "_run_for_all_agents", run_all)
        monkeypatch.setattr(scheduler, "_run_distributed_job", run_job)
        monkeypatch.setattr(scheduler, "scan_special_dates_today", AsyncMock())
        monkeypatch.setattr(scheduler, "generate_daily_schedule", generate)
        monkeypatch.setattr(scheduler, "get_life_overview", AsyncMock(return_value="画像"))
        await scheduler._run_daily_schedules()

        agent = SimpleNamespace(id="a", name="盛星茉", userId="u", mbti=None, currentMbti=None)
        monkeypatch.setattr(scheduler, "has_cached_schedule", AsyncMock(return_value=True))
        await captured["gen"](agent)
        generate.assert_not_awaited()

        monkeypatch.setattr(scheduler, "has_cached_schedule", AsyncMock(return_value=False))
        await captured["gen"](agent)
        generate.assert_awaited_once()
