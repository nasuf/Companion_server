"""AI 状态约束段的守卫.

作息进主回复提示词的方式有过一次失败: 早期把它当「参考信息」注入, 结果 §4 主回复
开始满嘴讲自己在干嘛, 跟 §3.4.3 状态查询分支撞车, 于是整段被移除 (commit 631188f)。

现在的做法是注入但框成**约束** —— 差别不在注不注入, 在于说"不得与之矛盾"还是
"这些可以聊"。业界做法也支持注入: Generative Agents 每条对话生成都带
`{agent}'s status:`, LangChain 的 GenerativeAgent 模板同样无条件带。

所以这个文件守的是"框架没有退回成素材", 而不是"有没有注入"。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.services.prompting.defaults import CHAT_AI_STATE_CONSTRAINT_PROMPT
from app.services.schedule_domain.schedule import (
    format_upcoming,
    get_current_status,
    get_upcoming_slots,
)

_SCHEDULE = [
    {"start": "07:00", "end": "08:00", "event": "起床洗漱", "status": "忙碌"},
    {"start": "09:00", "end": "12:00", "event": "在工作室做皮具", "status": "很忙碌"},
    {"start": "12:00", "end": "13:30", "event": "午饭+午休", "status": "空闲"},
    {"start": "14:00", "end": "18:00", "event": "继续打磨边缘", "status": "忙碌"},
    {"start": "23:00", "end": "07:00", "event": "睡觉", "status": "睡眠"},
]


def _at(hhmm: str) -> datetime:
    return datetime.strptime(f"2026-07-29 {hhmm}", "%Y-%m-%d %H:%M")


class TestFramingStaysAConstraint:
    def test_prompt_forbids_volunteering_the_state(self):
        """这一句是整段的关键。少了它, 模型会把状态当话题, 重现当年的撞车。"""
        assert "不要主动展开" in CHAT_AI_STATE_CONSTRAINT_PROMPT
        assert "除非用户明确询问" in CHAT_AI_STATE_CONSTRAINT_PROMPT

    def test_prompt_states_the_non_contradiction_rule(self):
        assert "不可与此矛盾" in CHAT_AI_STATE_CONSTRAINT_PROMPT

    def test_prompt_covers_future_tense_not_just_now(self):
        """只约束当下挡不住「等下一起看电影吧」这类未来时态的编造。"""
        assert "之后要做什么" in CHAT_AI_STATE_CONSTRAINT_PROMPT
        assert "不要临时编一个" in CHAT_AI_STATE_CONSTRAINT_PROMPT

    def test_prompt_does_not_invite_the_model_to_report_status(self):
        """反向检查: 不该出现鼓励讲述的措辞。"""
        for inviting in ("可以聊聊", "分享一下你的", "告诉用户你在"):
            assert inviting not in CHAT_AI_STATE_CONSTRAINT_PROMPT


class TestUpcomingSlots:
    def test_only_future_slots(self):
        slots = get_upcoming_slots(_SCHEDULE, _at("10:30"))
        assert [s["start"] for s in slots] == ["12:00", "14:00"]

    def test_is_bounded(self):
        """给全天会让这段太长, 远端安排对当前这轮也没用。"""
        assert len(get_upcoming_slots(_SCHEDULE, _at("00:30"), limit=2)) == 2

    def test_empty_late_at_night(self):
        """当天最后一段之后没有"接下来" —— 此时该整段略过, 而不是渲染个空壳。"""
        assert get_upcoming_slots(_SCHEDULE, _at("23:30")) == []
        assert format_upcoming([]) == ""

    def test_format_is_readable(self):
        rendered = format_upcoming(get_upcoming_slots(_SCHEDULE, _at("10:30")))
        assert rendered == "12:00 午饭+午休；14:00 继续打磨边缘"

    def test_tolerates_legacy_slot_shape(self):
        """旧缓存用 activity 字段而不是 event, 2 天 TTL 内会同时存在。"""
        legacy = [{"start": "15:00", "end": "16:00", "activity": "遛狗", "type": "leisure"}]
        assert format_upcoming(get_upcoming_slots(legacy, _at("10:00"))) == "15:00 遛狗"


class TestStatusCarriesUpcoming:
    """收口在 get_current_status 里 —— 聊天路径有两处构建 ai_status, 分开算的话
    下次新增调用点会漏, 而漏掉的表现是 AI 谈起未来时又开始现编, 不报错。"""

    @pytest.mark.parametrize("when,expected_event", [
        ("10:30", "在工作室做皮具"),
        ("12:30", "午饭+午休"),
        ("23:30", "睡觉"),
    ])
    def test_current_event_is_unchanged(self, when, expected_event):
        assert get_current_status(_SCHEDULE, _at(when))["event"] == expected_event

    def test_upcoming_is_always_present_as_a_key(self):
        """调用方直接读 status["upcoming"], 缺键会 KeyError 打崩热路径。"""
        for when in ("06:00", "10:30", "23:30"):
            assert "upcoming" in get_current_status(_SCHEDULE, _at(when))

    def test_upcoming_reflects_the_time_of_day(self):
        assert "12:00" in get_current_status(_SCHEDULE, _at("10:30"))["upcoming"]
        assert get_current_status(_SCHEDULE, _at("23:30"))["upcoming"] == ""

    def test_gap_between_slots_still_reports_upcoming(self):
        """18:00-23:00 是空档, 当前活动落到「自由时间」兜底, 但接下来仍该有。"""
        status = get_current_status(_SCHEDULE, _at("19:00"))
        assert status["event"] == "自由时间"
        assert "23:00" in status["upcoming"]


class TestSectionRendering:
    @pytest.mark.asyncio
    async def test_section_omits_the_upcoming_line_when_there_is_none(self):
        """深夜没有后续安排时不该渲染出一个空的「接下来：」。"""
        from app.services.prompting.utils import render_template

        rendered = render_template(
            CHAT_AI_STATE_CONSTRAINT_PROMPT,
            {"activity": "睡觉", "status": "sleep", "upcoming_line": ""},
        )
        assert "接下来" not in rendered
        assert "睡觉" in rendered

    @pytest.mark.asyncio
    async def test_section_includes_upcoming_when_present(self):
        from app.services.prompting.utils import render_template

        rendered = render_template(
            CHAT_AI_STATE_CONSTRAINT_PROMPT,
            {
                "activity": "做皮具", "status": "very_busy",
                "upcoming_line": "接下来：12:00 午饭。",
            },
        )
        assert "12:00 午饭" in rendered

    def test_prompt_builder_passes_upcoming_through(self):
        import inspect

        from app.services.chat import prompt_builder

        source = inspect.getsource(prompt_builder.build_system_prompt)
        assert '"upcoming_line"' in source
        assert 'ai_status.get("upcoming")' in source
