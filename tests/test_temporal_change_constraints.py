from datetime import datetime, timezone

import pytest

from app.services.time_domain.change_constraints import historical_change_filter
from app.services.time_domain.query import resolve_temporal_query


def period(day):
    query = resolve_temporal_query(
        f"2026年10月{day}日当时我住哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    assert query.status == "resolved"
    return query.expressions[0]


def change(**overrides):
    return {
        "id": "change-1",
        "old_memory_id": "old",
        "new_memory_id": "new",
        "kind": "transition",
        "source_available": True,
        "semantics": {
            "statement_at": "2026-11-01T00:00:00+08:00",
            "boundary": {
                "start": "2026-10-20T00:00:00+08:00",
                "end_exclusive": "2026-10-21T00:00:00+08:00",
            },
        },
        **overrides,
    }


@pytest.mark.parametrize(
    "memory_id,day,allowed",
    [
        ("old", 19, True),
        ("old", 20, False),
        ("old", 21, False),
        ("new", 19, False),
        ("new", 20, False),
        ("new", 21, True),
        ("unrelated", 20, True),
    ],
)
def test_day_precision_move_preserves_uncertain_day(memory_id, day, allowed):
    assert historical_change_filter(memory_id, period(day), [change()])[0] is allowed


@pytest.mark.parametrize("available", [True, False])
def test_retracted_mistake_is_never_a_past_truth_even_when_source_is_gone(available):
    allowed, reason, ids = historical_change_filter(
        "old", period(1), [change(kind="correction", source_available=available)]
    )
    assert not allowed and reason == "retrospective_correction" and ids == ("change-1",)


def test_invalid_source_does_not_authorize_transition_dates():
    assert (
        historical_change_filter("old", period(1), [change(source_available=False)])[0]
        is False
    )
    assert (
        historical_change_filter("new", period(25), [change(source_available=False)])[0]
        is False
    )


def test_future_plan_does_not_close_current_fact():
    assert (
        historical_change_filter("old", period(25), [change(kind="planned")])[0] is True
    )


def test_unknown_change_date_does_not_create_an_exact_endpoint():
    unknown = change(
        semantics={"statement_at": "2026-10-20T00:00:00+08:00", "boundary": {}}
    )
    assert historical_change_filter("old", period(19), [unknown])[0] is True
    assert historical_change_filter("old", period(20), [unknown])[0] is False
