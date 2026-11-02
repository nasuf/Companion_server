"""TIME12 固定作息在回复侧: 主回复「你的作息」段、时间核验对 AI 自己作息的处理。"""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.schedule_domain import schedule as schedule_mod
from app.services.schedule_domain.routine import AgentRoutine, describe_now
from app.services.time_domain import guard as temporal_guard
from app.services.time_domain.context import scope_event_memories

_TZ = ZoneInfo("Asia/Shanghai")
_NOW = datetime(2026, 10, 30, 19, 53, tzinfo=_TZ)
_ROUTINE = AgentRoutine(7 * 60 + 30, 8 * 60, 21 * 60, 23 * 60, frozenset({6}), False, {})


def _persona_event() -> ClassifiedMemory:
    # 人设种子里的经历带着随机的过去日期 (life_story._LIFE_EVENT_TIME_RANGE)
    return ClassifiedMemory("没来得及和要离开的熟客好好告别，感到遗憾", "strong", .8,
                            main_category="生活", occur_time="2025-09-10T01:22:00Z", source="ai")


def _user_event() -> ClassifiedMemory:
    return ClassifiedMemory("上周五加班到很晚", "strong", .8, main_category="生活",
                            occur_time="2026-10-23T13:00:00Z", source="user")


# ── 问法范围 ─────────────────────────────────────────────────────────

def test_habitual_question_is_not_scoped_to_one_day():
    rows = [_persona_event(), _user_event()]
    diag: dict = {}
    assert scope_event_memories(rows, "今晚加班了，平时没这么晚。你呢？", now=_NOW, diagnostics=diag) == rows
    assert diag["temporal_context"]["out_of_window_event_count"] == 0


def test_ai_and_user_out_of_window_events_are_counted_apart():
    diag: dict = {}
    kept = scope_event_memories([_persona_event(), _user_event()], "你今晚几点下班？",
                                now=_NOW, diagnostics=diag)
    assert kept == []
    state = diag["temporal_context"]
    assert (state["out_of_window_event_count"], state["out_of_window_ai_event_count"]) == (2, 1)


# ── 时间核验 ─────────────────────────────────────────────────────────

def _guard_env(monkeypatch, verdict, fallback="哎这个我想不起来了"):
    texts = {"chat.temporal_check": "{date} {question} {evidence} {reply}", "chat.temporal_fallback": fallback}
    monkeypatch.setattr(temporal_guard, "get_prompt_text", AsyncMock(side_effect=lambda key: texts[key]))
    monkeypatch.setattr(temporal_guard, "get_utility_model", lambda: None)
    check = AsyncMock(return_value=verdict)
    monkeypatch.setattr(temporal_guard, "invoke_json", check)
    return check


@pytest.mark.asyncio
async def test_agent_only_gap_with_schedule_evidence_is_not_rechecked(monkeypatch):
    check = _guard_env(monkeypatch, {"grounded": False})
    diag = {"temporal_context": {"out_of_window_event_count": 1, "out_of_window_ai_event_count": 1,
                                 "in_window_event_count": 0, "requested_event_date": "2026-10-30"}}
    reply = await temporal_guard.guard_temporal_reply(
        "我九点下班", question="你今晚几点下班", evidence="", diagnostics=diag,
        agent_evidence="平时：21:00下班", agent_day_in_prompt=True)
    assert reply == "我九点下班"
    assert diag["temporal_context"]["event_check_status"] == "agent_schedule_evidence"
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_past_day_is_still_checked_with_that_days_schedule(monkeypatch):
    check = _guard_env(monkeypatch, {"grounded": True})
    diag = {"temporal_context": {"out_of_window_event_count": 1, "out_of_window_ai_event_count": 1,
                                 "in_window_event_count": 0, "requested_event_date": "2026-10-28"}}
    reply = await temporal_guard.guard_temporal_reply(
        "那天店里盘点到很晚", question="你前天干嘛了", evidence="", diagnostics=diag,
        agent_evidence="2026-10-28 你的安排：09:00-18:00 店里盘点")
    assert reply == "那天店里盘点到很晚"
    assert "2026-10-28 你的安排" in check.await_args.args[1]


@pytest.mark.asyncio
async def test_user_gap_still_checked_with_agent_evidence_attached(monkeypatch):
    check = _guard_env(monkeypatch, {"grounded": False})
    diag = {"temporal_context": {"out_of_window_event_count": 2, "out_of_window_ai_event_count": 1,
                                 "in_window_event_count": 0, "requested_event_date": "2026-10-30"}}
    reply = await temporal_guard.guard_temporal_reply(
        "你今天加班到九点", question="我今天几点下班的", evidence="旧事件", diagnostics=diag,
        agent_evidence="平时：晚上21:00下班")
    assert reply == "哎这个我想不起来了"
    prompt = check.await_args.args[1]
    assert "【角色作息】" in prompt and "晚上21:00下班" in prompt


@pytest.mark.asyncio
async def test_fallback_survives_unreadable_prompt(monkeypatch):
    _guard_env(monkeypatch, {"grounded": False})

    async def text(key):
        if key == "chat.temporal_fallback":
            raise RuntimeError("redis down")
        return "{date} {question} {evidence} {reply}"
    monkeypatch.setattr(temporal_guard, "get_prompt_text", AsyncMock(side_effect=text))
    diag = {"temporal_context": {"out_of_window_event_count": 1, "in_window_event_count": 0,
                                 "requested_event_date": "2026-10-30"}}
    assert await temporal_guard.guard_temporal_reply(
        "你昨天去了西湖", question="我昨天去哪了", evidence="", diagnostics=diag) == temporal_guard._FALLBACK
    assert "。" not in temporal_guard._FALLBACK


# ── AI 作息证据 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_day_evidence_today_past_and_out_of_range(monkeypatch):
    monkeypatch.setattr(schedule_mod, "_local_now", lambda: _NOW)
    today = [{"start": "08:00", "end": "21:00", "event": "上班", "status": "很忙碌", "kind": "工作"}]
    past = [{"start": "09:00", "end": "18:00", "event": "店里盘点", "status": "忙碌"}]
    query = AsyncMock(return_value=[{"schedule_data": json.dumps(past, ensure_ascii=False)}])
    monkeypatch.setattr(schedule_mod, "db", SimpleNamespace(query_raw=query))
    view = describe_now(_ROUTINE, today, _NOW)

    text = await schedule_mod.agent_day_evidence(
        "a", routine_view=view, target_date="2026-10-30", today_schedule=today)
    assert "平时：" in text and "2026-10-30 你的安排：08:00-21:00 上班" in text
    query.assert_not_awaited()

    text = await schedule_mod.agent_day_evidence("a", routine_view=None, target_date="2026-10-28")
    assert text == "2026-10-28 你的安排：09:00-18:00 店里盘点"

    assert await schedule_mod.agent_day_evidence("a", routine_view=None, target_date="2026-01-01") == ""
    assert await schedule_mod.agent_day_evidence("a", routine_view=None, target_date="2026-11-05") == ""
    assert await schedule_mod.agent_day_evidence("a", routine_view=None, target_date="bad") == ""


# ── 主回复「你的作息」段 ─────────────────────────────────────────────

async def _prompt_text(key: str, **_kwargs) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


async def _build(ai_status, monkeypatch, *, disabled: bool = False):
    from app.services.chat import prompt_builder

    async def optional(key):
        if disabled and key == "chat.agent_routine_section":
            return None
        return await _prompt_text(key)

    diagnostics: dict = {}
    with (
        patch("app.services.chat.prompt_builder.get_prompt_text", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
        patch.object(prompt_builder, "_get_optional_prompt", AsyncMock(side_effect=optional)),
    ):
        prompt = await prompt_builder.build_system_prompt(
            agent=SimpleNamespace(name="阿恺", values={"gender": "male"}),
            memories=None, memory_relevance="weak", reply_count=1, reply_total=150,
            ai_status=ai_status, diagnostics=diagnostics,
        )
    return prompt, diagnostics


@pytest.mark.asyncio
async def test_routine_section_is_rendered_in_the_turn_part(monkeypatch):
    from app.services.chat.prompt_builder import split_system_prompt

    status = {"activity": "上班", "status": "very_busy",
              "routine": describe_now(_ROUTINE, None, datetime(2026, 10, 30, 19, 55, tzinfo=_TZ))}
    prompt, diag = await _build(status, monkeypatch)
    lead, turn = split_system_prompt(prompt)
    prompt = str(prompt)
    assert "## 你的作息\n平时：早上7:30左右起床，8:00开始上班，21:00下班" in prompt
    assert "现在19:55，正在上班，离下班还有1小时5分钟" in prompt
    assert "不编“提前溜了”" in prompt
    assert "## 你的作息" in turn and "## 你的作息" not in lead
    assert diag["agent_routine"]["status"] == "injected"
    assert diag["agent_routine"]["phase"] == "at_work"


@pytest.mark.asyncio
async def test_routine_section_absent_or_disabled_is_explicit(monkeypatch):
    prompt, diag = await _build({"activity": "看书", "status": "idle"}, monkeypatch)
    assert "## 你的作息" not in prompt
    assert diag["agent_routine"] == {"status": "not_derived"}
    assert "你的作息" in diag["empty_prompt_sections_removed"]

    status = {"activity": "上班", "status": "busy", "routine": describe_now(_ROUTINE, None, _NOW)}
    prompt, diag = await _build(status, monkeypatch, disabled=True)
    assert "## 你的作息" not in prompt and diag["agent_routine"]["status"] == "disabled"
