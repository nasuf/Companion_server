from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.interaction_streak import (
    MAKEUP_LOOKBACK_DAYS,
    MakeupError,
    SOURCE_MAKEUP,
    apply_makeup,
    get_current_streak,
    is_makeup_eligible,
    record_user_message_day,
    streak_from_dates,
)


def test_streak_counts_back_from_today_when_today_is_marked():
    today = date(2026, 9, 16)
    marked = {today - timedelta(days=i) for i in range(5)}
    assert streak_from_dates(marked, today) == 5


def test_streak_uses_yesterday_when_today_is_still_open():
    today = date(2026, 9, 16)
    marked = {today - timedelta(days=i) for i in range(1, 4)}
    assert streak_from_dates(marked, today) == 3


def test_streak_is_zero_when_yesterday_and_today_are_empty():
    today = date(2026, 9, 16)
    marked = {today - timedelta(days=3), today - timedelta(days=4)}
    assert streak_from_dates(marked, today) == 0


def test_streak_stops_at_the_first_gap():
    today = date(2026, 9, 16)
    marked = {today, today - timedelta(days=1), today - timedelta(days=3)}
    assert streak_from_dates(marked, today) == 2


def test_makeup_eligible_rejects_today_future_and_old_days():
    today = date(2026, 9, 16)
    created = date(2026, 8, 1)
    assert is_makeup_eligible(
        today, today=today, workspace_created_on=created, already_marked=False
    ) is False
    assert is_makeup_eligible(
        today + timedelta(days=1),
        today=today,
        workspace_created_on=created,
        already_marked=False,
    ) is False
    assert is_makeup_eligible(
        today - timedelta(days=MAKEUP_LOOKBACK_DAYS + 1),
        today=today,
        workspace_created_on=created,
        already_marked=False,
    ) is False
    assert is_makeup_eligible(
        date(2026, 7, 31),
        today=today,
        workspace_created_on=created,
        already_marked=False,
    ) is False
    assert is_makeup_eligible(
        today - timedelta(days=1),
        today=today,
        workspace_created_on=created,
        already_marked=True,
    ) is False
    assert is_makeup_eligible(
        today - timedelta(days=1),
        today=today,
        workspace_created_on=created,
        already_marked=False,
    ) is True


@pytest.mark.asyncio
async def test_get_current_streak_returns_zero_without_workspace():
    assert await get_current_streak(None) == 0


@pytest.mark.asyncio
async def test_record_user_message_day_is_idempotent(monkeypatch):
    redis = SimpleNamespace(set=AsyncMock(return_value=True), delete=AsyncMock())
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[{"id": "row-1"}]))
    monkeypatch.setattr("app.services.interaction_streak.get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr("app.services.interaction_streak.db", db)

    today = date(2026, 9, 16)
    with patch(
        "app.services.interaction_streak.local_activity_date",
        return_value=today,
    ):
        assert await record_user_message_day("ws-1", "user-1") is True
        redis.set.return_value = False
        assert await record_user_message_day("ws-1", "user-1") is False
    assert db.query_raw.await_count == 1


class _FakeTx:
    def __init__(self, insert_rows):
        self.insert_rows = insert_rows
        self.query_calls = []

    async def query_raw(self, query, *args):
        self.query_calls.append((query, args))
        return self.insert_rows


class _TxContext:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_apply_makeup_consumes_a_card_in_the_same_transaction(monkeypatch):
    today = date(2026, 9, 16)
    day = today - timedelta(days=1)
    tx = _FakeTx([{"id": "row-1"}])
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 1, 1),
                )
            )
        ),
        tx=lambda: _TxContext(tx),
        query_raw=AsyncMock(side_effect=[
            [{"localDate": day, "source": SOURCE_MAKEUP}],
            [{"localDate": today, "source": "user_message"}],
            [{"ok": 1}],
        ]),
    )
    consume = AsyncMock(return_value=1)
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    monkeypatch.setattr("app.services.interaction_streak.consume_batch_units", consume)
    monkeypatch.setattr(
        "app.services.interaction_streak.batch_summary",
        AsyncMock(return_value={"quantity": 1}),
    )

    result = await apply_makeup("ws-1", "user-1", day, today=today)

    assert result["day"]["source"] == SOURCE_MAKEUP
    assert result["makeup_cards"] == 1
    consume.assert_awaited_once()
    assert consume.await_args.kwargs["client"] is tx


@pytest.mark.asyncio
async def test_apply_makeup_rejects_today(monkeypatch):
    today = date(2026, 9, 16)
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 1, 1),
                )
            )
        )
    )
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    with pytest.raises(MakeupError, match="cannot_makeup_today"):
        await apply_makeup("ws-1", "user-1", today, today=today)


@pytest.mark.asyncio
async def test_apply_makeup_rolls_back_when_the_day_is_already_marked(monkeypatch):
    today = date(2026, 9, 16)
    day = today - timedelta(days=1)
    tx = _FakeTx([])
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 1, 1),
                )
            )
        ),
        tx=lambda: _TxContext(tx),
    )
    consume = AsyncMock()
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    monkeypatch.setattr("app.services.interaction_streak.consume_batch_units", consume)
    with pytest.raises(MakeupError, match="already_marked"):
        await apply_makeup("ws-1", "user-1", day, today=today)
    consume.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_makeup_rejects_days_outside_the_lookback_window(monkeypatch):
    today = date(2026, 9, 16)
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 1, 1),
                )
            )
        )
    )
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    with pytest.raises(MakeupError, match="outside_lookback"):
        await apply_makeup(
            "ws-1",
            "user-1",
            today - timedelta(days=MAKEUP_LOOKBACK_DAYS + 1),
            today=today,
        )


@pytest.mark.asyncio
async def test_apply_makeup_rejects_insufficient_cards(monkeypatch):
    today = date(2026, 9, 16)
    day = today - timedelta(days=1)
    tx = _FakeTx([{"id": "row-1"}])
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 1, 1),
                )
            )
        ),
        tx=lambda: _TxContext(tx),
    )
    consume = AsyncMock(side_effect=ValueError("insufficient_inventory"))
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    monkeypatch.setattr("app.services.interaction_streak.consume_batch_units", consume)
    with pytest.raises(MakeupError, match="insufficient_inventory"):
        await apply_makeup("ws-1", "user-1", day, today=today)


@pytest.mark.asyncio
async def test_apply_makeup_rejects_archived_workspace(monkeypatch):
    today = date(2026, 9, 16)
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-old",
                    status="archived",
                    createdAt=date(2026, 1, 1),
                )
            )
        )
    )
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    with pytest.raises(MakeupError, match="workspace_not_found"):
        await apply_makeup("ws-old", "user-1", today - timedelta(days=1), today=today)


@pytest.mark.asyncio
async def test_get_current_streak_empty_ledger_is_zero(monkeypatch):
    fake_db = SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    today = date(2026, 9, 16)
    assert await get_current_streak("ws-new", today=today) == 0


@pytest.mark.asyncio
async def test_get_interaction_overview_flags_makeup_gaps(monkeypatch):
    from app.services.interaction_streak import get_interaction_overview

    today = date(2026, 9, 16)
    yesterday = today - timedelta(days=1)
    fake_db = SimpleNamespace(
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    id="ws-1",
                    status="active",
                    createdAt=date(2026, 9, 1),
                )
            )
        ),
        query_raw=AsyncMock(
            return_value=[{"localDate": today, "source": "user_message"}]
        ),
    )
    monkeypatch.setattr("app.services.interaction_streak.db", fake_db)
    monkeypatch.setattr(
        "app.services.interaction_streak.batch_summary",
        AsyncMock(return_value={"quantity": 3}),
    )
    result = await get_interaction_overview(
        "ws-1", "user-1", year=2026, month=9, today=today
    )
    assert result["current_streak"] == 1
    by_date = {row["date"]: row for row in result["days"]}
    assert by_date[yesterday.isoformat()]["makeup_eligible"] is True
    assert by_date[today.isoformat()]["makeup_eligible"] is False
    assert by_date[today.isoformat()]["source"] == "user_message"


@pytest.mark.asyncio
async def test_record_user_message_day_fail_open_when_db_errors(monkeypatch):
    redis = SimpleNamespace(set=AsyncMock(return_value=True), delete=AsyncMock())
    db = SimpleNamespace(query_raw=AsyncMock(side_effect=RuntimeError("db down")))
    monkeypatch.setattr("app.services.interaction_streak.get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr("app.services.interaction_streak.db", db)
    today = date(2026, 9, 16)
    with patch(
        "app.services.interaction_streak.local_activity_date",
        return_value=today,
    ):
        assert await record_user_message_day("ws-1", "user-1") is False
    redis.delete.assert_awaited_once()
