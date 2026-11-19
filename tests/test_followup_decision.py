"""Shared optional follow-up decision: same event, response, cooldown, silence."""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.time_domain.followup_decision import (
    ANSWERED_COOLDOWN,
    UNANSWERED_COOLDOWN,
    decide_followup,
    decide_reply,
    may_be_event_followup,
    outcome_event,
    same_event,
)

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def h(*rows):
    return [{"id": f"m{i}", "role": role, "content": text,
             "created_at": (NOW - timedelta(minutes=minutes)).isoformat()}
            for i, (minutes, role, text) in enumerate(rows)]


DAY = 24 * 60


def test_rephrased_unanswered_ask_from_yesterday_is_suppressed():
    # Roadmap example: asked yesterday, no answer; no "有消息了吗" today.
    history = h((DAY + 60, "user", "我明天下午有个面试"), (DAY, "assistant", "面试结果出来了吗？"))
    decision = decide_followup("后来收到录用通知了吗？", history, now=NOW)
    assert not decision.allowed and decision.reason == "unanswered_cooldown"
    assert decision.next_eligible_at == NOW - timedelta(minutes=DAY) + UNANSWERED_COOLDOWN


def test_progress_and_result_wordings_are_the_same_event():
    a = outcome_event("那件事进展怎么样？", ["上午面了一家公司"])
    b = outcome_event("后来有结果没？", ["上午面了一家公司"])
    assert a is not None and b is not None and same_event(a, b)


@pytest.mark.parametrize("question", ["今天的新面试顺利吗？", "二面的通知来了吗？", "另一家公司的面试怎么样？"])
def test_a_new_instance_of_the_same_kind_is_a_different_event(question):
    history = h((DAY, "user", "周一有个面试"), (DAY - 5, "assistant", "面试有结果了吗？"))
    assert decide_followup(question, history, now=NOW).allowed


def test_sharing_a_generic_word_is_not_event_identity():
    history = h((120, "user", "最近工作好忙"), (100, "assistant", "工作上还顺利吗？"))
    assert decide_followup("工作的事情后来解决了吗？", history, now=NOW).reason == "not_an_outcome_followup"
    assert not may_be_event_followup("今天过得怎么样？")


def test_user_raising_the_event_reopens_it():
    history = h((3 * DAY, "user", "下周要答辩"), (2 * DAY, "assistant", "答辩准备得怎么样？"),
                (DAY, "assistant", "答辩顺利吗？"))
    decision = decide_followup("答辩顺利吗？", history, now=NOW, current_user_message="答辩今天终于结束了")
    assert decision.allowed and decision.reason == "user_discussing_event"


def test_resolved_event_without_new_information_is_not_asked_again():
    history = h((300, "user", "下午去复查"), (200, "assistant", "复查结果怎么样？"),
                (190, "user", "复查结果出来了，一切正常"))
    assert decide_followup("检查结果怎么样了？", history, now=NOW).reason == "event_resolved_no_new_info"


def test_answered_still_waiting_cools_down_from_the_answer():
    history = h((300, "user", "交了签证材料"), (200, "assistant", "签证有消息吗？"), (190, "user", "还没消息，在等"))
    decision = decide_followup("签证批下来了吗？", history, now=NOW)
    assert decision.reason == "answered_recently"
    assert decision.next_eligible_at == NOW - timedelta(minutes=190) + ANSWERED_COOLDOWN


async def test_long_fast_forward_never_cycles_back_into_asking():
    """Two unanswered asks go quiet; 60 days of re-evaluation never re-arm,
    even after the asks leave the 7-day page (the family ledger keeps them)."""
    history = h((3 * DAY, "user", "下周要答辩"), (2 * DAY, "assistant", "答辩结束了吗？"),
                (DAY, "assistant", "答辩顺利吗？"))
    for days in range(0, 58):
        _, decisions = await _reply("答辩结束了吗？", history, now=NOW + timedelta(days=days, hours=3))
        assert not decisions[0].allowed, days
        assert decisions[0].reason == "silent_until_new_evidence"
        assert decisions[0].next_eligible_at is None


def test_evaluation_time_never_moves_the_cooldown_clock():
    history = h((60, "user", "周末相亲"), (30, "assistant", "相亲感觉怎么样？"))
    first = decide_followup("相亲聊得来吗？", history, now=NOW)
    again = decide_followup("相亲聊得来吗？", history, now=NOW + timedelta(hours=5))
    assert first.next_eligible_at == again.next_eligible_at


def test_truncated_history_cannot_prove_an_unanswered_count_below_the_cap():
    history = h((3 * DAY, "user", "最近在准备比赛"), (3 * DAY - 30, "assistant", "比赛顺利吗？"))
    assert decide_followup("比赛有结果了吗？", history, now=NOW).allowed
    blocked = decide_followup("比赛有结果了吗？", history, now=NOW, history_truncated=True)
    assert not blocked.allowed and blocked.reason == "history_truncated"


class FakeDB:
    """In-memory stand-in for the two bounded reads (dense page + family ledger)."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def query_raw(self, sql, _user, _workspace, now, days, limit, pattern=None):
        import re as _re

        self.calls.append("ledger" if pattern else "recent")
        cutoff = datetime.fromisoformat(now) - timedelta(days=days)
        rows = [r for r in self.rows if cutoff < datetime.fromisoformat(r["created_at"])
                <= datetime.fromisoformat(now) and (pattern is None or _re.search(pattern, r["content"]))]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]


async def _reply(text, rows, now=NOW, user=""):
    out, decisions, _ = await decide_reply(text, database=FakeDB(rows), user_id="u", workspace_id="w",
                                           now=now, current_user_message=user)
    return out, decisions


async def test_only_the_repeated_clause_is_removed_and_layout_is_kept():
    history = h((DAY + 60, "user", "我明天下午有个面试"), (DAY, "assistant", "面试结果出来了吗？"))
    text, decisions = await _reply("今天降温了，记得加衣服。\n录用通知来了没？", history)
    assert text == "今天降温了，记得加衣服。" and len(decisions) == 1


async def test_unaffected_bubble_is_returned_unchanged_without_any_read():
    db = FakeDB([])
    text, decisions, _ = await decide_reply("哈哈哈\n后来呢？|你继续说", database=db, user_id="u",
                                            workspace_id="w", now=NOW)
    assert text == "哈哈哈\n后来呢？|你继续说" and decisions == [] and db.calls == []


async def test_live_discussion_questions_are_never_counted_as_follow_ups():
    # Evening: user brings up the interview, the agent asks a detail. Next day
    # the first real outcome follow-up is allowed.
    history = h((20 * 60, "user", "明天上午面试，有点紧张"), (20 * 60 - 1, "assistant", "是什么公司的面试呀？"),
                (20 * 60 - 2, "user", "字节"), (20 * 60 - 3, "assistant", "面试前要不要早点睡？"))
    text, decisions = await _reply("面试怎么样了？", history, user="我回来啦")
    assert text == "面试怎么样了？" and decisions[0].reason == "first_ask"


@pytest.mark.parametrize("question", ["后来呢？", "今天工作顺利吗？", "你今天出来玩了吗？", "有什么好消息吗？"])
async def test_generic_questions_never_borrow_an_old_event(question):
    history = h((2 * DAY, "user", "面试过了"), (2 * DAY - 5, "assistant", "恭喜！"))
    text, _ = await _reply(question, history, user="我在看电影，主角刚出场")
    assert text == question


async def test_an_ignored_ask_outside_the_dense_page_still_counts():
    # The ask is 10 days old: outside the 7-day page, inside the family ledger.
    history = h((12 * DAY, "user", "下周要答辩"), (10 * DAY, "assistant", "答辩顺利吗？"))
    text, decisions = await _reply("答辩结束了吗？", history)
    assert text == "" and decisions[0].reason == "silent_until_new_evidence"


async def test_bare_casual_reply_is_not_an_event_resolution():
    history = h((300, "user", "下午去复查"), (200, "assistant", "复查结果怎么样？"), (199, "user", "我回来了"))
    _, decisions = await _reply("检查结果出来了吗？", history, now=NOW + timedelta(days=2))
    assert decisions[0].reason == "answered_cooldown_elapsed"


def test_one_answer_to_several_questions_counts_as_a_response():
    history = h((200, "user", "今天体检了"), (100, "assistant", "体检累不累？报告什么时候出？"),
                (95, "user", "还好，不累"))
    decision = decide_followup("体检报告出来了吗？", history, now=NOW)
    assert decision.reason == "answered_recently"


def test_numbered_exam_subjects_are_different_events():
    history = h((DAY, "user", "昨天考了科目一"), (DAY - 30, "assistant", "科目一过了吗？"))
    assert decide_followup("科目二什么时候考？", history, now=NOW).allowed
    assert not decide_followup("科目一成绩出来了吗？", history, now=NOW).allowed


async def test_preparation_questions_the_night_before_do_not_block_the_result_question():
    history = h((18 * 60, "user", "明天下午面试"), (18 * 60 - 1, "assistant", "面试准备好了吗？"),
                (18 * 60 - 2, "user", "差不多了"))
    for question in ("面试怎么样？", "面试顺利吗？", "今天面试怎么样？"):
        text, decisions = await _reply(question, history)
        assert text == question and decisions[0].reason == "first_ask"


async def test_second_live_question_is_not_cut():
    history = h((20, "user", "明天要去面试"), (19, "assistant", "面试准备得怎么样了？"),
                (18, "user", "还行吧，就是有点慌"))
    text, _ = await _reply("你觉得面试会顺利吗？", history, user="还行吧，就是有点慌")
    assert text == "你觉得面试会顺利吗？"


@pytest.mark.parametrize("question", ["你说的那家火锅怎么样？", "上次那部电影怎么样？"])
async def test_reference_with_its_own_object_borrows_nothing(question):
    history = h((60, "user", "下午去体检了"), (50, "assistant", "体检结果出来了吗？"),
                (40, "user", "还没，晚上吃了顿火锅"))
    text, _ = await _reply(question, history)
    assert text == question


@pytest.mark.parametrize("statement", ["面试准备好了，明天上午去", "拿到面试通知了，明天去"])
async def test_future_or_preparation_statements_are_not_results(statement):
    history = h((DAY + 30, "user", "下周一有个面试"), (DAY, "assistant", "面试有消息了吗？"),
                (DAY - 1, "user", statement))
    _, decisions = await _reply("面试怎么样？结果出来了吗？", history)
    assert all(d.reason != "event_resolved_no_new_info" for d in decisions)


async def test_unrelated_question_in_the_same_bubble_is_not_a_follow_up():
    history = h((DAY, "user", "下周一有个面试"), (DAY - 1, "assistant", "面试加油！晚饭吃好了吗？"))
    text, decisions = await _reply("面试怎么样了？", history)
    assert text == "面试怎么样了？" and decisions[0].reason == "first_ask"


async def test_full_ledger_reaching_past_a_week_still_allows_a_first_ask():
    busy = [(10 * DAY - i * 300, "user", f"在改第{i}版简历") for i in range(45)]
    text, decisions = await _reply("申请有结果了吗？", h(*busy))
    assert text and decisions[0].reason == "first_ask"
    recent = [(3 * DAY - i * 60, "user", f"在改第{i}版简历") for i in range(45)]
    _, decisions = await _reply("申请有结果了吗？", h(*recent))
    assert decisions[0].reason == "history_truncated"


async def test_slow_history_read_only_drops_the_optional_question(monkeypatch):
    import asyncio
    from app.services.time_domain import followup_decision as fd

    class SlowDB:
        async def query_raw(self, *_args):
            await asyncio.sleep(5)

    monkeypatch.setattr(fd, "REPLY_READ_TIMEOUT_S", 0.05)
    replies, decision = await fd.guard_reply_followups(
        ["今天降温了。面试结果出来了吗？"], user_id="u", workspace_id="w", now=NOW,
        current_user_message="好冷", database=SlowDB())
    assert replies == ["今天降温了。"]
    assert decision["status"] == "unavailable" and decision["suppressed_count"] == 1


# M02.20.04: ignored event questions, told results, shared-noun and 后来+result borrowing.

def test_an_ignored_pre_event_question_cools_down_a_rephrased_outcome_ask():
    history = h((1200, "user", "下周有个比赛"), (600, "assistant", "比赛准备得怎么样？"))
    decision = decide_followup("比赛有进展吗？", history, now=NOW)
    assert decision.reason == "unanswered_cooldown" and decision.evidence["ignored_event_questions"] == 1


def test_an_answered_pre_event_question_never_counts():
    history = h((1500, "user", "明天上午有个面试"), (1490, "assistant", "面试几点呀？"), (1488, "user", "十点"))
    assert decide_followup("面试结果怎么样？", history, now=NOW).reason == "first_ask"


def test_a_live_exchange_ending_on_a_question_is_not_ignoring_it():
    history = h((600, "user", "下周要答辩"), (598, "assistant", "答辩是哪天？"), (597, "user", "周三"),
                (596, "assistant", "答辩前紧张吗？"))
    assert decide_followup("答辩顺利吗？", history, now=NOW + timedelta(days=1)).reason == "first_ask"


def test_two_ignored_event_questions_go_quiet():
    history = h((5760, "user", "下个月出差去上海"), (4320, "assistant", "出差的行程定了吗？"),
                (2880, "assistant", "出差准备得怎么样啦？"))
    assert decide_followup("上海的出差顺利吗？", history, now=NOW).reason == "silent_until_new_evidence"


def test_a_result_the_user_already_told_ends_follow_ups_without_a_prior_ask():
    history = h((300, "user", "今天去拔牙"), (240, "assistant", "拔牙疼不疼？"), (235, "user", "拔完了，搞定了"))
    assert decide_followup("牙医那边怎么样了？", history, now=NOW).reason == "event_resolved_no_new_info"
    pending = h((300, "user", "今天去拔牙"), (235, "user", "拔完了，搞定了"), (100, "user", "拔牙的地方还在等复查结果"))
    assert decide_followup("牙医那边怎么样了？", pending, now=NOW).reason != "event_resolved_no_new_info"


def test_a_shared_concrete_noun_borrows_the_named_event():
    ref = outcome_event("奖学金批下来没？", ["在申请奖学金"])
    assert ref is not None and ref.family == "application" and ref.resolved_from == "subject"
    assert may_be_event_followup("奖学金批下来没？")
    # A generic word, or a noun the event message does not share, borrows nothing.
    assert outcome_event("工作批下来没？", ["工作上在申请调岗"]) is None
    assert outcome_event("驾照批下来没？", ["在申请奖学金"]) is None
    # Weak wording never borrows by noun alone.
    assert outcome_event("奖学金怎么样？", ["在申请奖学金"]) is None


def test_later_plus_a_result_noun_borrows_the_event():
    history = h((600, "user", "上午去做了体检"), (480, "assistant", "体检还顺利吗？"))
    assert decide_followup("后来报告怎么说？", history, now=NOW).reason == "unanswered_cooldown"


def test_a_question_the_user_is_answering_right_now_is_not_ignored():
    history = h((300, "user", "下周有个比赛"), (60, "assistant", "比赛准备得怎么样？"))
    decision = decide_followup("比赛有结果了吗？", history, now=NOW, current_user_message="还行吧")
    assert decision.reason == "first_ask"


def test_a_just_told_result_still_allows_live_detail_questions():
    history = h((2, "user", "面试通过了！"))
    assert decide_followup("面试结果怎么通知你的？", history, now=NOW).reason == "user_discussing_event"


async def test_unavailable_history_keeps_bare_strong_word_questions():
    from app.services.time_domain.followup_decision import guard_reply_followups

    replies, diag = await guard_reply_followups(
        ["你吃好了吗？", "面试结果出来了吗？", "  原样，"], user_id="u", workspace_id=None,
        now=NOW, current_user_message="")
    assert replies == ["你吃好了吗？", "  原样，"] and diag["status"] == "scope_unavailable"


def test_event_questions_outside_the_dense_page_are_never_ignored_touches():
    # Dense page starts 2 days ago; the pre-event question 10 days ago sits only
    # in the family ledger, where an unnamed answer would be invisible.
    dense = h((2 * DAY, "user", "下周要答辩"))
    ledger = h((10 * DAY, "assistant", "答辩紧张吗？"))
    decision = decide_followup("答辩结束了吗？", dense, ledger=ledger, now=NOW)
    assert decision.reason == "first_ask"
