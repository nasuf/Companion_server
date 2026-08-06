"""完局伴聊的作息染色.

这组测试守的是一个**静默失效**的 bug: `_agent_state_text` 曾经把 agent_id 当作息
表传给 `get_current_status(schedule: list[dict])`, 于是函数遍历字符串的字符、
`'x'.get()` 抛 AttributeError, 又被 `except: return ""` 吞掉 —— 线上表现是所有
agent 的染色都是空串, 不报错、不告警, 三个真实 agent 实测全空。

字段名也曾错成 activity/state (实际是 event/status)。`dict.get` 拿不到只返回
None, 同样不报错。这两类错误都只有测试能拦。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.games.native import _STATE_MOOD, _agent_state_text

pytestmark = pytest.mark.asyncio


def _patch(monkeypatch, schedule):
    monkeypatch.setattr(
        "app.services.schedule_domain.schedule.get_cached_schedule",
        AsyncMock(return_value=schedule),
    )


_WORKDAY = [
    {"start": "00:00", "end": "23:59", "event": "在工位改稿", "status": "busy"},
]


class TestColouring:
    async def test_event_and_mood_are_both_used(self, monkeypatch):
        _patch(monkeypatch, _WORKDAY)
        assert await _agent_state_text("a1") == "在工位改稿，手头还有事没忙完"

    async def test_idle_contributes_no_mood(self, monkeypatch):
        """空闲是默认状态, 说"我很闲"反而奇怪 —— 只报在干什么."""
        _patch(monkeypatch, [{"start": "00:00", "end": "23:59",
                              "event": "在阳台发呆", "status": "idle"}])
        assert await _agent_state_text("a1") == "在阳台发呆"

    async def test_sleep_reads_as_a_situation_not_a_refusal(self, monkeypatch):
        """作息只染色不设卡: 用户点开游戏就是现在想玩, 让他等是最糟的失败.

        所以睡眠给的是"本来已经准备睡了"这种处境, 由模型说成"陪你下一盘再睡",
        而不是任何形式的拒绝。
        """
        _patch(monkeypatch, [{"start": "00:00", "end": "23:59",
                              "event": "躺床上刷手机", "status": "sleep"}])
        text = await _agent_state_text("a1")
        assert "准备睡" in text
        for refusal in ("不能", "等", "明天", "改天"):
            assert refusal not in text

    async def test_mood_map_covers_every_non_idle_status(self):
        """作息表的状态枚举是 idle/busy/very_busy/sleep, 少一个就静默丢染色."""
        assert set(_STATE_MOOD) == {"busy", "very_busy", "sleep"}


class TestDegradation:
    async def test_no_schedule_yields_empty(self, monkeypatch):
        """实测 13 个玩过游戏的 agent 里今天只有 3 个有作息 —— 没有是常态, 不是错误."""
        _patch(monkeypatch, None)
        assert await _agent_state_text("a1") == ""

    async def test_missing_agent_id_yields_empty(self):
        assert await _agent_state_text(None) == ""

    async def test_schedule_lookup_failure_does_not_break_the_reply(self, monkeypatch):
        """染色是锦上添花, 挂了也不能让完局回复发不出去."""
        monkeypatch.setattr(
            "app.services.schedule_domain.schedule.get_cached_schedule",
            AsyncMock(side_effect=RuntimeError("redis down")),
        )
        assert await _agent_state_text("a1") == ""


class TestContractWithScheduleModule:
    async def test_schedule_is_fetched_before_being_read(self, monkeypatch):
        """回归守卫: 曾经把 agent_id 直接当 schedule 传进 get_current_status.

        那个签名收的是 list[dict], 传字符串会遍历出字符再 .get() —— 报错被吞,
        线上静默返回空串。这里断言"先取表, 再拿表去查状态"。
        """
        seen: dict = {}
        monkeypatch.setattr(
            "app.services.schedule_domain.schedule.get_cached_schedule",
            AsyncMock(return_value=_WORKDAY),
        )

        def _spy(schedule, *a, **k):
            seen["arg"] = schedule
            return {"event": "x", "status": "idle"}

        monkeypatch.setattr(
            "app.services.schedule_domain.schedule.get_current_status", _spy,
        )
        await _agent_state_text("a1")
        assert seen["arg"] == _WORKDAY, "传进去的必须是作息表, 不是 agent_id"

    async def test_reads_the_keys_get_current_status_actually_returns(self, monkeypatch):
        """读的必须是 event/status —— 曾经错写成 activity/state.

        `dict.get` 拿不到只返回 None 不报错, 所以这种错只能靠断言输出来抓。
        这里直接给一个只有 event/status 的返回值: 如果实现改回读 activity/state,
        结果会变成空串。
        """
        _patch(monkeypatch, _WORKDAY)
        monkeypatch.setattr(
            "app.services.schedule_domain.schedule.get_current_status",
            lambda *a, **k: {"event": "在改稿", "status": "very_busy"},
        )
        assert await _agent_state_text("a1") == "在改稿，正忙得脚不沾地"
