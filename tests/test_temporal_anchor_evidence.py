"""Original-message time evidence must cover the whole turn, not a subset."""
from datetime import datetime, timezone

import pytest

from app.services.chat.temporal_context import build_turn_clock
from app.services.temporal_query import resolve_temporal_query

NOW = datetime(2026, 10, 22, 1, tzinfo=timezone.utc)


def row(ident, stamp="2026-10-20T01:00:00Z", **changes):
    return {"id": ident, "role": "user", "createdAt": stamp, **changes}


@pytest.mark.parametrize("other", [None, row("b", None), row("b", "bad"),
    row("b", "2099-01-01T00:00:00Z"), row("b", role="assistant"),
    row("b", synthetic_current=True)])
def test_partial_turn_cannot_borrow_one_messages_time(other):
    rows = [row("a")] + ([other] if other else [])
    clock = build_turn_clock(rows, exclude_ids={"a", "b"}, now=NOW, received_at=NOW)
    assert clock.statement_at is None


@pytest.mark.parametrize("other", [row("a", "2026-10-21T01:00:00Z"),
    row("a", None), row("a", role="assistant")])
def test_conflicting_duplicate_cannot_choose_a_convenient_time(other):
    clock = build_turn_clock([row("a"), other], exclude_ids={"a"}, now=NOW)
    assert clock.statement_at is None


def test_multi_message_turn_outside_history_cannot_fallback_to_one_receipt():
    clock = build_turn_clock([], exclude_ids={"a", "b"}, now=NOW, received_at=NOW)
    assert clock.statement_at is None


def test_single_invalid_persisted_clock_cannot_be_replaced_with_processing_time():
    clock = build_turn_clock([row("a", "bad")], exclude_ids={"a"}, now=NOW, received_at=NOW)
    assert clock.statement_at is None


@pytest.mark.parametrize("reverse", [False, True])
def test_same_instant_duplicates_and_timezone_representations_are_not_conflicts(reverse):
    rows = [row("a"), row("a", "2026-10-20T09:00:00+08:00"), row("b", "2026-10-20T02:00:00Z")]
    clock = build_turn_clock(rows[::-1] if reverse else rows, exclude_ids={"a", "b"}, now=NOW)
    evidence = clock.diagnostics()["statement_evidence"]
    assert evidence == {"version": 1, "reason": "persisted_message", "expected_messages": 2,
        "verified_messages": 2, "missing_messages": 0, "invalid_messages": 0, "conflicting_messages": 0,
        "earliest_at": "2026-10-20T01:00:00+00:00", "latest_at": "2026-10-20T02:00:00+00:00",
        "processing_delay_seconds": 172800.0}
    assert clock.statement_end_at.hour == 2


@pytest.mark.parametrize("rows,ids,reason,counts", [
    ([row("a")], {"a", "b"}, "incomplete_message_times", (1, 1, 0, 0)),
    ([row("a"), row("b", "bad")], {"a", "b"}, "invalid_message_times", (1, 0, 1, 0)),
    ([row("a"), row("a", "bad")], {"a"}, "conflicting_message_times", (0, 0, 0, 1)),
    ([row("a")], {str(i) for i in range(121)}, "message_count_limit", (0, 121, 0, 0)),
    ([], set(), "unknown", (0, 0, 0, 0)),
])
def test_incomplete_evidence_has_no_fabricated_processing_delay(rows, ids, reason, counts):
    clock = build_turn_clock(rows, exclude_ids=ids, now=NOW)
    d = clock.diagnostics()["statement_evidence"]
    assert d["reason"] == reason
    assert tuple(d[k] for k in ("verified_messages", "missing_messages", "invalid_messages", "conflicting_messages")) == counts
    assert sum(counts) == d["expected_messages"]
    assert d["earliest_at"] is d["latest_at"] is d["processing_delay_seconds"] is None


@pytest.mark.parametrize("ids", [set(), {"a"}])
def test_receipt_fallback_is_explicit_without_forging_persisted_evidence(ids):
    clock = build_turn_clock([row("a", synthetic_current=True)], exclude_ids=ids, now=NOW,
        received_at="2026-10-20T01:00:00Z")
    d = clock.diagnostics()["statement_evidence"]
    assert clock.anchor_source == "server_received"
    assert d["verified_messages"] == 0 and d["expected_messages"] == len(ids)
    assert d["processing_delay_seconds"] == 172800


def test_history_window_does_not_hide_missing_anchor_evidence():
    rows = [row("old")] + [row(f"h{i}", role="assistant") for i in range(120)]
    clock = build_turn_clock(rows, exclude_ids={"old", "h119"}, now=NOW, received_at=NOW)
    assert clock.statement_at is None and clock.inspected_messages == 120
    assert clock.statement_evidence.missing == 1 and clock.statement_evidence.invalid == 1


def test_cross_date_evidence_keeps_window_but_disables_single_anchor():
    clock = build_turn_clock([row("a", "2026-10-20T15:59:59Z"), row("b", "2026-10-20T16:00:00Z")],
        exclude_ids={"a", "b"}, now=NOW)
    assert clock.statement_at is None and clock.statement_end_at is None
    d = clock.diagnostics()["statement_evidence"]
    assert d["reason"] == "multiple_message_dates" and d["verified_messages"] == 2
    assert d["earliest_at"] and d["latest_at"]


def test_invalid_timezone_does_not_crash_clock_construction(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "schedule_timezone", "bad/zone")
    clock = build_turn_clock([row("a")], exclude_ids={"a"}, now=NOW)
    assert clock.statement_at is None and clock.anchor_source == "invalid_timezone"


@pytest.mark.parametrize("text", ["1小时前去了哪里", "十五分钟之前", "半个小时前", "半天后"])
def test_elapsed_expressions_cannot_borrow_another_fragment_clock(text):
    q = resolve_temporal_query(text, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message",
        anchor_end="2026-10-20T02:00:00Z")
    assert q.reason == "multiple_statement_instants" and q.status == "ambiguous"
    assert not q.parsed_times() and q.retrieval_range() is None


@pytest.mark.parametrize("text", ["昨天", "两天前", "2026年10月17日", "上午9点30分"])
def test_calendar_queries_still_work_for_complete_same_day_aggregation(text):
    q = resolve_temporal_query(text, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message",
        anchor_end="2026-10-20T02:00:00Z")
    assert q.status == "resolved" and len(q.expressions) == 1


@pytest.mark.parametrize("end,reason", [("bad", "invalid_anchor_window"),
    ("2026-10-19T01:00:00Z", "invalid_anchor_window"),
    ("2026-10-21T01:00:00Z", "multiple_statement_dates")])
def test_invalid_or_cross_day_window_cannot_date_filter(end, reason):
    q = resolve_temporal_query("昨天", anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message", anchor_end=end)
    assert q.reason == reason and q.retrieval_range() is None


def test_identical_instants_keep_elapsed_query_and_cache_contract():
    args = dict(anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    old = resolve_temporal_query("一小时前", **args)
    new = resolve_temporal_query("一小时前", **args, anchor_end="2026-10-20T09:00:00+08:00")
    assert old == new and old.cache_token() == new.cache_token()


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["langgraph"])
@pytest.mark.parametrize("kind", ["missing", "elapsed", "calendar", "conflict", "cross_day"])
async def test_real_executors_use_complete_original_time_evidence(monkeypatch, executor, kind):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.now = NOW
    message = "一小时前去了哪里" if kind == "elapsed" else "昨天去了哪里"
    rows = [row("a")]
    if kind != "missing":
        rows.append(row("a" if kind == "conflict" else "b",
            "2026-10-20T16:01:00Z" if kind == "cross_day" else "2026-10-20T02:00:00Z"))
    rows = [SimpleNamespace(**{**r, "createdAt": datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")),
        "content": message, "metadata": {}}) for r in rows]
    io.db.message.find_many.side_effect = lambda **_: list(reversed(rows))
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["我记得你提过"], "", False, None)))
    events = [e async for e in chat.stream_chat_response("c-1", message, io.agent, "u-1",
        save_user_message=False, user_message_id="a", forced_intent=IntentType.NONE,
        reply_context={"turn_message_ids": ["a", "b"], "received_at": NOW.isoformat()})]
    q = chat.fetch_parallel_context.await_args.kwargs["temporal_query"]
    if kind == "calendar":
        assert q.expressions[0].start.date().isoformat() == "2026-10-19"
    else:
        assert not q.parsed_times() and q.retrieval_range() is None
    assert sum(e["event"] == "done" for e in events) == 1
