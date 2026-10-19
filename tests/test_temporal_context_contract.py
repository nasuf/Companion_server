"""Anonymized regressions from TIME01: long gaps, midnight, old events, opt-out."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.chat.temporal_context import build_turn_clock, dated_dialogue, prepare_reply_time
from app.services.chat.followup_policy import derive_followup_policy, guard_optional_questions
from app.services.chat.prompt_builder import build_chat_messages, compute_reengagement_gap_seconds
from app.services.memory.retrieval.context_selector import (
    ClassifiedMemory, format_memory_time, select_context, split_by_source,
)
from app.services.memory.retrieval.hybrid import _memory_to_cache_dict, _memory_from_cache_dict
from app.services.temporal import parse_time

NOW = datetime(2026, 10, 19, 11, 54, tzinfo=timezone.utc)


def msg(role, text, *, ago=0, ident=None, proactive=False):
    return {"id": ident, "role": role, "content": text,
            "createdAt": (NOW - timedelta(seconds=ago)).isoformat(),
            "metadata": {"proactive": proactive}}


@pytest.mark.parametrize("gap", [60, 3600, 9 * 86400])
def test_proactive_output_does_not_erase_user_absence(gap):
    rows = [msg("user", "旧消息", ago=gap), msg("assistant", "主动消息", ago=30, proactive=True),
            msg("user", "你在干嘛", ident="current")]
    clock = build_turn_clock(rows, now=NOW, exclude_ids={"current"})
    assert clock.gap(clock.last_user_at) == gap
    assert clock.gap(clock.last_dialogue_at) == gap
    assert clock.gap(clock.last_outbound_at) == 30
    assert compute_reengagement_gap_seconds(rows, {"current"}, NOW) == gap


def test_missing_future_and_naive_timestamps_are_not_zero_gap():
    rows = [{"role": "user", "createdAt": "invalid"},
            msg("user", "future", ago=-60)]
    clock = build_turn_clock(rows, now=NOW)
    assert clock.diagnostics()["user_gap_seconds"] is None
    assert clock.unknown_timestamps == 2
    assert parse_time("2026-10-19T11:54:00") == NOW
    assert parse_time("2026-10-19T19:54:00+08:00") == NOW


def test_dated_context_preserves_midnight_and_old_illness():
    rows = [msg("user", "肚子难受", ago=9 * 86400),
            {"role": "user", "content": "没睡", "createdAt": "2026-10-17T17:45:00Z"},
            {"role": "user", "content": "你在干嘛", "createdAt": "2026-10-18T02:18:00Z"}]
    text = dated_dialogue(rows)
    assert "2026-10-10" in text
    assert "2026-10-18 01:45" in text and "2026-10-18 10:18" in text
    assert len(dated_dialogue(rows * 200, max_chars=500)) <= 500


def test_history_cutoff_keeps_new_bubble_after_old_same_role():
    rows = [msg("assistant", "上周旧事", ago=7 * 86400),
            msg("assistant", "今天的新消息", ago=30), msg("user", "你好")]
    output = build_chat_messages("system", rows, drop_older_than_seconds=10800)
    assert "今天的新消息" in str(output)
    assert "上周旧事" not in str(output)


@pytest.mark.parametrize("source", ["ai", "user"])
def test_event_time_survives_selection_cache_and_tier_render(source):
    selected = select_context([{"id": "m1", "content": "午饭吃了三明治", "source": source,
        "importance": .7, "similarity": .9, "rank_score": .8, "main_category": "生活",
        "occur_time": "2026-09-08T04:00:00Z", "statement_time": "2026-09-08T04:05:00Z",
        "last_accessed_at": NOW.isoformat()}])
    assert len(selected) == 1
    memory = _memory_from_cache_dict(_memory_to_cache_dict(selected[0]))
    assert memory.occur_time == "2026-09-08T04:00:00Z"
    text = str(split_by_source([memory]))
    assert "事件时间 2026-09-08 12:00" in text
    assert "2026-10-19" not in text


def test_statement_and_recorded_clocks_do_not_claim_event_date():
    memory = ClassifiedMemory(text="昨天很累", relevance="medium", score=.7,
                              statement_time=NOW, main_category="情绪")
    assert "提及时间" in format_memory_time(memory)
    assert "事件时间未知" in format_memory_time(memory)
    old_cache = {"text": "以前吃过面", "main_category": "生活", "last_accessed_at": NOW}
    assert format_memory_time(_memory_from_cache_dict(old_cache)) == "[事件时间未知] 以前吃过面"


@pytest.mark.parametrize("text", ["能不能不要一直问了", "别老问", "不要再追问我", "不用问我了"])
def test_current_optout_and_later_question_preserve_answer(text):
    rows = [msg("user", text, ago=240), msg("assistant", "好，不问了", ago=235),
            msg("user", "你中午吃了什么")]
    policy = derive_followup_policy(rows, now=NOW)
    assert not policy.optional_questions_allowed
    out, n = guard_optional_questions(["中午吃了面，你还烦不烦呀？", "我这会儿在做饭呢。"], policy)
    assert out == ["中午吃了面", "我这会儿在做饭呢。"]
    assert n == 1


def test_optout_expires_or_user_explicitly_reopens():
    assert derive_followup_policy([msg("user", "别老问", ago=10801)], now=NOW).optional_questions_allowed
    policy = derive_followup_policy([msg("user", "别老问", ago=60)], now=NOW,
                                    current_message="你现在可以问我了")
    assert policy.optional_questions_allowed and policy.reason == "user_reopened"


@pytest.mark.asyncio
async def test_prompt_render_and_read_failure_are_observable(monkeypatch):
    from app.services.prompting import store
    from app.services.prompting.defaults import CHAT_TEMPORAL_CONTEXT_PROMPT
    read = AsyncMock(return_value=CHAT_TEMPORAL_CONTEXT_PROMPT)
    monkeypatch.setattr(store, "get_prompt_text", read)
    diagnostics = {}
    _, policy, text = await prepare_reply_time([msg("user", "没睡", ago=8 * 3600)],
        exclude_ids=set(), current_message="不要一直问", diagnostics=diagnostics, now=NOW)
    assert "2026-10-19 19:54" in text and "optional_questions_allowed=false" in text
    assert not policy.optional_questions_allowed
    assert diagnostics["temporal_context"]["prompt_status"] == "enabled"
    read.side_effect = RuntimeError("unavailable")
    await prepare_reply_time([], exclude_ids=set(), current_message="你好", diagnostics=diagnostics, now=NOW)
    assert diagnostics["temporal_context"]["prompt_status"] == "unavailable"
    assert diagnostics["temporal_context"]["user_gap_seconds"] is None


@pytest.mark.asyncio
async def test_proactive_does_not_query_or_send_after_user_optout(monkeypatch):
    from app.services.proactive import followup
    from types import SimpleNamespace
    state = SimpleNamespace(conversation_id="synthetic", metadata={})
    turns = [followup._Turn("u", "user", "不要一直问", False, NOW - timedelta(minutes=10)),
             followup._Turn("a", "assistant", "好，不问啦", False, NOW - timedelta(minutes=9))]
    monkeypatch.setattr(followup, "_load_recent_turns", AsyncMock(return_value=turns))
    advance = AsyncMock()
    judge = AsyncMock()
    monkeypatch.setattr(followup, "_enter_a_mode", advance)
    monkeypatch.setattr(followup, "judge_topic_completion", judge)
    await followup._judge_and_maybe_follow_up(state, NOW, trace_id=None)
    advance.assert_awaited_once_with(state, NOW, "user_opted_out")
    judge.assert_not_awaited()


@pytest.mark.parametrize("text", ['他说别再问我', '我没有说别问我', '她说“别问我了”', '解释一下“别问我了”的意思'])
def test_reported_optout_does_not_change_user_policy(text):
    assert derive_followup_policy([msg("user", text)], now=NOW, current_message=text).optional_questions_allowed


def test_dated_evidence_scoping_preserves_history_identity_and_unknown_times():
    from app.services.chat.temporal_context import scope_event_memories
    old = ClassifiedMemory("午饭吃了面", "strong", .8, main_category="生活", occur_time="2026-09-08T04:00:00Z")
    today = ClassifiedMemory("午饭吃了饭", "strong", .8, main_category="生活", occur_time=NOW)
    unknown = ClassifiedMemory("肚子不舒服", "strong", .8, main_category="情绪")
    identity = ClassifiedMemory("住在杭州", "strong", .8, main_category="身份", occur_time=old.occur_time)
    ongoing = ClassifiedMemory("一直在杭州工作", "strong", .8, main_category="生活", occur_time=old.occur_time)
    rows = [old, today, unknown, identity, ongoing]
    diagnostic = {}
    assert scope_event_memories(rows, "你今天中午吃了什么", now=NOW, diagnostics=diagnostic) == rows[1:]
    assert diagnostic["temporal_context"]["out_of_window_event_count"] == 1
    assert scope_event_memories(rows, "9月8日吃了什么", now=NOW, diagnostics={}) == [old, unknown, identity, ongoing]
    for query in ("今天跟以前有什么不同", "今天想聊9月8日，你记得吗", "吃了什么", "今天好热"):
        assert scope_event_memories(rows, query, now=NOW, diagnostics={}) == rows
    assert scope_event_memories([], "今天吃什么", now=NOW, diagnostics={}) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
@pytest.mark.parametrize("route", ["ordinary", "crisis_care", "contradiction", "persona_correction", "current_state"])
async def test_actual_executor_optout_and_protected_questions(monkeypatch, executor, route):
    import json
    from types import SimpleNamespace
    from app.config import settings
    from app.services.chat import orchestrator as chat, intent_handlers
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from app.services import persona_grounding
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.now = NOW
    monkeypatch.setattr(settings, "chat_executor", executor)
    io.db.message.find_many.side_effect = lambda **_: [SimpleNamespace(
        id="old", role="user", content="不要一直问我", createdAt=NOW-timedelta(minutes=3), metadata={})]
    monkeypatch.setattr(persona_grounding, "guard_reply", AsyncMock(return_value=("在整理东西，你在干嘛？", route == "persona_correction")))
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["在整理东西，你在干嘛？"], "", False, None)))
    if route == "crisis_care":
        io.decision.crisis_care_turn = True
        from unittest.mock import Mock
        from app.services.chat import graph_phases
        forbidden = Mock(side_effect=AssertionError("Crisis evidence must not be date-filtered"))
        monkeypatch.setattr(chat, "scope_event_memories", forbidden)
        monkeypatch.setattr(graph_phases, "scope_event_memories", forbidden)
    if route == "contradiction":
        chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance="strong")
        chat.detect_l1_contradiction.return_value = {"conflict_description": "城市变更"}
        monkeypatch.setattr(chat, "generate_contradiction_inquiry", AsyncMock(return_value="你现在搬到杭州了吗？"))
        monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["你现在搬到杭州了吗？"], "", False, None)))
    if route == "current_state":
        monkeypatch.setattr(intent_handlers, "current_state_reply", AsyncMock(return_value="在整理东西，你在干嘛？"))
    events = [event async for event in chat.stream_chat_response("c-1", "你在干嘛" if route == "current_state" else "我到杭州了",
        io.agent, "u-1", forced_intent=IntentType.CURRENT_STATE if route == "current_state" else IntentType.NONE,
        reply_context={"received_at": NOW.isoformat()})]
    answer = " ".join(json.loads(e["data"]).get("text", "") for e in events if e["event"] == "reply")
    assert answer
    assert ("？" in answer) == (route in {"crisis_care", "contradiction"})
    assert sum(e["event"] == "done" for e in events) == 1


@pytest.mark.asyncio
async def test_real_summary_anchors_source_dates(monkeypatch):
    from app.services.chat import session_recap
    monkeypatch.setattr(session_recap, "get_prompt_text", AsyncMock(return_value="概括：{conversation}"))
    model = AsyncMock(return_value="用户当时在等面试结果")
    monkeypatch.setattr(session_recap, "invoke_text", model)
    monkeypatch.setattr(session_recap, "get_utility_model", lambda: None)
    result = await session_recap._summarize([msg("user", "今天等面试结果", ago=9*86400)])
    assert "2026-10-10" in result
    assert "2026-10-10" in model.await_args.args[1]
    result = await session_recap._summarize([{"role":"user", "content":"以前去过"}])
    assert "历史对话 时间未知" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,status,keeps", [
    ({"grounded": True}, "grounded", True), ({"grounded": False}, "rejected", False),
    ({"grounded": "yes"}, "unavailable", False), (RuntimeError("offline"), "unavailable", False),
    (TimeoutError(), "unavailable", False),
])
async def test_date_guard_fails_closed_only_for_evidence_gap(monkeypatch, verdict, status, keeps):
    from app.services.chat import temporal_guard
    monkeypatch.setattr(temporal_guard, "get_prompt_text", AsyncMock(return_value="{date} {evidence} {reply}"))
    monkeypatch.setattr(temporal_guard, "get_utility_model", lambda: None)
    check = AsyncMock(side_effect=verdict) if isinstance(verdict, Exception) else AsyncMock(return_value=verdict)
    monkeypatch.setattr(temporal_guard, "invoke_json", check)
    diag = {"temporal_context": {"out_of_window_event_count":1,"in_window_event_count":0,"requested_event_date":"2026-10-19"}}
    reply = await temporal_guard.guard_temporal_reply("今天中午吃了牛肉面", question="你今天吃什么", evidence="旧事件", diagnostics=diag)
    assert (reply == "今天中午吃了牛肉面") is keeps
    assert diag["temporal_context"]["event_check_status"] == status
    check.reset_mock()
    assert await temporal_guard.guard_temporal_reply("正常回答", question="你好", evidence="", diagnostics={}) == "正常回答"
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_temporal_prompts_disabled_or_failed_are_explicit(monkeypatch):
    from app.services.chat import temporal_guard, temporal_context
    from app.services.prompting.store import PromptDisabledError
    monkeypatch.setattr(temporal_guard, "get_prompt_text", AsyncMock(side_effect=PromptDisabledError("chat.temporal_check")))
    diag={"temporal_context":{"out_of_window_event_count":1,"requested_event_date":"2026-10-19"}}
    assert await temporal_guard.guard_temporal_reply("回答", question="问题", evidence="", diagnostics=diag) == "回答"
    assert diag["temporal_context"]["event_check_status"] == "disabled"
    from app.services.prompting import store
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(side_effect=RuntimeError("offline")))
    assert await temporal_context.render_event_scope(diag) == ""
    assert diag["temporal_context"]["event_prompt_status"] == "unavailable"
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(side_effect=PromptDisabledError("chat.temporal_context")))
    await prepare_reply_time([], exclude_ids=set(), current_message="你好", diagnostics=diag, now=NOW)
    assert diag["temporal_context"]["prompt_status"] == "disabled"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_actual_date_guard_rejects_unbacked_today_claim(monkeypatch, executor):
    import json
    from app.config import settings
    from app.services.chat import orchestrator as chat, temporal_guard
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services import persona_grounding
    from tests.g02_harness_support import configure_pair
    io=configure_pair(monkeypatch);io.now=NOW
    monkeypatch.setattr(settings, "chat_executor", executor)
    old=ClassifiedMemory("旧午饭吃了三明治", "strong", .8, main_category="生活", occur_time="2026-09-08T04:00:00Z")
    chat.fetch_parallel_context.return_value=FetchedContext(memory_relevance="strong", classified_memories=[old])
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["今天中午吃了牛肉面"], "", False, None)))
    monkeypatch.setattr(persona_grounding, "guard_reply", AsyncMock(return_value=("今天中午吃了牛肉面", False)))
    monkeypatch.setattr(temporal_guard, "get_utility_model", lambda: None)
    check=AsyncMock(return_value={"grounded":False});monkeypatch.setattr(temporal_guard, "invoke_json", check)
    events=[e async for e in chat.stream_chat_response("c-1", "你今天中午吃了什么", io.agent, "u-1", reply_context={"received_at": NOW.isoformat()})]
    answer=" ".join(json.loads(e["data"]).get("text", "") for e in events if e["event"]=="reply")
    assert "牛肉面" not in answer and "记不太清" in answer
    check.assert_awaited_once()
    assert sum(e["event"]=="done" for e in events)==1


def test_current_state_context_drops_old_transient_state():
    from app.services.chat.temporal_context import dated_active_dialogue
    rows=[msg("user","昨晚还没睡",ago=8*3600),msg("user","刚刚回家",ago=120)]
    text=dated_active_dialogue(rows,now=NOW,exclude_ids=set())
    assert "还没睡" not in text and "刚刚回家" in text


def test_repeated_question_guard_keeps_unrelated_questions():
    from app.services.chat.followup_policy import FollowupPolicy, repeats_recent_question
    policy=FollowupPolicy(recent_questions=("你今天工作忙完了吗？",))
    assert repeats_recent_question("你今天工作忙完了吗", policy)
    assert not repeats_recent_question("要听听我刚听的歌吗？", policy)


@pytest.mark.asyncio
async def test_tier_time_suffix_preserves_custom_prompt_editing_and_single_step_replay(monkeypatch):
    from app.services.chat import intent_replies, trace_enrich
    from app.services.prompting import utils
    from app.services.prompting.trace_components import (
        ManagedPromptText, start_prompt_render_trace, reset_prompt_render_trace,
        snapshot_prompt_render_traces,
    )
    custom = ManagedPromptText.compose([
        ("chat.response_instruction", "自定义回复规则"),
        ("memory.weak_reply", "定制正文：{message}"),
    ], "memory.weak_reply")
    monkeypatch.setattr(utils, "get_prompt_text", AsyncMock(return_value=custom))
    model = AsyncMock(return_value="回复")
    monkeypatch.setattr(intent_replies, "invoke_text", model)
    monkeypatch.setattr(intent_replies, "get_chat_model", lambda: None)
    token = start_prompt_render_trace()
    try:
        await intent_replies._render_tier_reply("memory.weak_reply", {"message":"你好"},
            max_total=100, temporal_instructions="回合日期", temporal_evidence="日期证据")
        prompt = model.await_args.args[1]
        from tests.test_trace_enrich import _fake_llm_step
        step = _fake_llm_step(prompt)
        trace_enrich.apply_prompt_render_traces([step], snapshot_prompt_render_traces())
        assert step["prompt_key"] == "memory.weak_reply"
        parts = {c["prompt_key"]:prompt[c["start"]:c["end"]] for c in step["prompt_components"]}
        assert parts == {"chat.response_instruction":"自定义回复规则", "memory.weak_reply":"定制正文：你好",
                         "chat.temporal_context":"回合日期", "chat.temporal_evidence":"日期证据"}
    finally:
        reset_prompt_render_trace(token)
