"""Short-window opt-out decisions must follow time and explicit permission."""
from datetime import datetime, timedelta, timezone
from itertools import permutations

import pytest

from app.services.chat.followup_policy import derive_followup_policy

NOW = datetime(2026, 10, 31, 0, 5, tzinfo=timezone.utc)


def message(text, seconds_ago=0, role="user"):
    return {"role": role, "content": text, "createdAt": NOW - timedelta(seconds=seconds_ago)}


@pytest.mark.parametrize("reverse_decision", [False, True])
def test_latest_dated_instruction_wins_independently_of_input_order(reverse_decision):
    texts = ["别再追问我了", "你现在可以问我了"]
    if reverse_decision:
        texts.reverse()
    rows = [message(texts[0], 180), message("好，听你的", 120, "assistant"), message(texts[1], 60)]
    for ordered in permutations(rows):
        original = list(ordered)
        policy = derive_followup_policy(original, now=NOW)
        assert policy.optional_questions_allowed is not reverse_decision
        assert original == list(ordered), "The caller's message order must not be mutated"


@pytest.mark.parametrize("text", [
    "不允许你问我", "我不允许你问我", "你现在不可以问我了", "你还不能继续问我",
    "不是让你继续问我", "我没说你可以问我", "如果你可以问我", "为什么你可以问我",
    "你想问什么是什么意思", "他说你可以问我", "她说“你可以问我”",
    "解释一下‘你可以问我’的意思", "不可以继续问我", "别问我吧",
])
def test_denied_reported_conditional_or_discussed_permission_cannot_reopen(text):
    policy = derive_followup_policy([message("别再追问我了", 90)], now=NOW, current_message=text)
    assert not policy.optional_questions_allowed
    assert policy.reason == "user_opted_out"


@pytest.mark.parametrize("text", [
    "你现在可以问我了", "你可以继续问我", "你继续问我吧", "可以继续问", "允许你问我",
    "问我吧", "你想问什么", "好了，你现在可以问我了", "那你现在可以问我了",
    "你现在可以问我了😊", "你继续问我吧 ❤️ ", "你现在可以问我了👩‍💻",
])
def test_explicit_current_invitation_reopens(text):
    policy = derive_followup_policy([message("别再追问我了", 90)], now=NOW, current_message=text)
    assert policy.optional_questions_allowed and policy.reason == "user_reopened"


@pytest.mark.parametrize("text,allowed", [
    ("先别再追问我了。现在你可以问我了", True),
    ("你现在可以问我了，不过别再追问我了", False),
    ("别再追问我了；你继续问我吧", True),
    ("你继续问我吧；别再追问我了", False),
])
@pytest.mark.parametrize("use_current", [False, True])
def test_last_explicit_clause_wins(text, allowed, use_current):
    rows = [] if use_current else [message(text)]
    policy = derive_followup_policy(rows, now=NOW, current_message=text if use_current else "")
    assert policy.optional_questions_allowed is allowed


def test_mixed_timezone_offsets_are_sorted_by_instant():
    rows = [
        {"role": "user", "content": "别再追问我了", "createdAt": "2026-10-30T17:04:00-07:00"},
        {"role": "user", "content": "你可以问我了", "createdAt": "2026-10-31T08:02:00+08:00"},
    ]
    assert not derive_followup_policy(rows, now=NOW).optional_questions_allowed


def test_invalid_future_or_out_of_window_permission_does_not_override_stop():
    rows = [message("别再追问我了", 60), message("你可以问我了", -1),
            message("你可以问我了", 10801),
            {"role": "user", "content": "你可以问我了", "createdAt": "invalid"}]
    assert not derive_followup_policy(rows, now=NOW).optional_questions_allowed


def test_current_request_wins_over_dated_history():
    assert not derive_followup_policy([message("你可以问我了")], now=NOW,
                                      current_message="不要再追问我了").optional_questions_allowed


def test_equal_timestamps_keep_input_order():
    rows = [message("别再追问我了"), message("你可以问我了")]
    assert derive_followup_policy(rows, now=NOW).optional_questions_allowed
    assert not derive_followup_policy(rows[::-1], now=NOW).optional_questions_allowed


def test_recent_questions_follow_time_not_input_order():
    rows = [message(f"你吃了{i}个吗？", 10-i, "assistant") for i in range(8)]
    policy = derive_followup_policy(rows[::-1], now=NOW)
    assert policy.recent_questions == tuple(f"你吃了{i}个吗？" for i in range(2, 8))


def test_session_expiry_remains_explicitly_outside_persistent_preference_scope():
    assert derive_followup_policy([message("别再追问我了", 10801)], now=NOW).optional_questions_allowed
    assert not derive_followup_policy([message("别再追问我了", 10800)], now=NOW).optional_questions_allowed


@pytest.mark.parametrize("text", ["不允许你问我", "我不允许你问我", "你现在不可以问我了", "你还不能继续问我"])
def test_explicit_denial_stops_questions_without_an_earlier_optout(text):
    policy = derive_followup_policy([], now=NOW, current_message=text)
    assert not policy.optional_questions_allowed and policy.reason == "user_opted_out"


@pytest.mark.parametrize("text", [
    '解释一下“别问我了”的意思', "解释一下‘别问我了’的意思", "解释一下'别问我了'的意思",
    '他说“别问我了”，她说“你可以问我”', '“你可以\n问我了”是什么意思',
    "不是不允许你问我", "我不是让你别问我", "我没有说别问我", "如果你不能问我",
])
def test_quoted_reported_or_negated_denial_does_not_create_optout(text):
    assert derive_followup_policy([], now=NOW, current_message=text).optional_questions_allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("newest", ["别再追问我了", "不允许你问我", "如果你可以问我"])
async def test_proactive_judge_never_generates_after_latest_stop_or_ambiguous_resume(monkeypatch, newest):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.proactive import followup

    # Deliberately reverse the database order; a valid older invitation cannot
    # undo the newer instruction. The ambiguous case follows a real stop.
    turns = [followup._Turn("a", "assistant", "好，听你的", False, NOW - timedelta(seconds=1)),
             followup._Turn("u3", "user", newest, False, NOW - timedelta(seconds=2)),
             followup._Turn("u2", "user", "别再追问我了", False, NOW - timedelta(seconds=3)),
             followup._Turn("u1", "user", "你可以问我了", False, NOW - timedelta(seconds=4))]
    monkeypatch.setattr(followup, "_load_recent_turns", AsyncMock(return_value=turns))
    advance, judge, send = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(followup, "_enter_a_mode", advance)
    monkeypatch.setattr(followup, "judge_topic_completion", judge)
    monkeypatch.setattr(followup, "_send_followup", send)
    state = SimpleNamespace(conversation_id="synthetic", metadata={})
    await followup._judge_and_maybe_follow_up(state, NOW, trace_id=None)
    advance.assert_awaited_once_with(state, NOW, "user_opted_out")
    judge.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_reply_diagnostics_keep_stop_after_denied_permission(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.chat.temporal_context import prepare_reply_time
    from app.services.prompting import store
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(return_value="{followup_policy}"))
    diagnostics = {}
    _, policy, _ = await prepare_reply_time(
        [message("别再追问我了", 60)], exclude_ids=set(), current_message="不允许你问我",
        diagnostics=diagnostics, now=NOW)
    assert not policy.optional_questions_allowed
    assert diagnostics["temporal_context"]["optional_questions_allowed"] is False
    assert diagnostics["temporal_context"]["followup_reason"] == "user_opted_out"


@pytest.mark.asyncio
@pytest.mark.parametrize("crisis", [False, True])
async def test_actual_graph_respects_denied_permission_without_blocking_crisis(monkeypatch, crisis):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.chat import orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentType
    from app.services import persona_grounding
    from tests.g02_harness_support import configure_pair

    io = configure_pair(monkeypatch)
    io.now = NOW
    io.decision.crisis_care_turn = crisis
    io.db.message.find_many.side_effect = lambda **_: [SimpleNamespace(
        id="optout", role="user", content="不要一直问我", createdAt=NOW-timedelta(minutes=3), metadata={})]
    reply = "我在整理东西，你在干嘛？"
    monkeypatch.setattr(persona_grounding, "guard_reply", AsyncMock(return_value=(reply, False)))
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=([reply], "", False, None)))
    events = [e async for e in chat.stream_chat_response(
        "c-1", "不允许你问我", io.agent, "u-1", forced_intent=IntentType.NONE,
        reply_context={"received_at": NOW.isoformat()})]
    answer = " ".join(json.loads(e["data"]).get("text", "") for e in events if e["event"] == "reply")
    assert "我在整理东西" in answer
    assert ("？" in answer) is crisis
    assert sum(e["event"] == "done" for e in events) == 1
