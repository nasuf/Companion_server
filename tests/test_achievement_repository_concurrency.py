from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import repository


@pytest.mark.asyncio
async def test_concurrent_unlock_attempts_create_only_one_unlock():
    class _AtomicInsertDb:
        def __init__(self):
            self.lock = asyncio.Lock()
            self.inserted = False
            self.query_count = 0

        async def query_raw(self, _query: str, *_args):
            self.query_count += 1
            await asyncio.sleep(0)
            async with self.lock:
                if self.inserted:
                    return []
                self.inserted = True
                return [{
                    "id": "unlock-1",
                    "unlocked_at": datetime(2026, 6, 1, tzinfo=UTC),
                }]

    fake_db = _AtomicInsertDb()

    with (
        patch.object(repository, "_is_unlock_cached", AsyncMock(return_value=False)),
        patch.object(repository, "_cache_unlocked_achievements", AsyncMock()),
        patch.object(repository, "db", fake_db),
    ):
        results = await asyncio.gather(
            repository.unlock_achievement(
                user_id="u1",
                agent_id="a1",
                achievement_id=1,
                notify=False,
            ),
            repository.unlock_achievement(
                user_id="u1",
                agent_id="a1",
                achievement_id=1,
                notify=False,
            ),
        )

    assert sorted(results) == [False, True]
    assert fake_db.query_count == 2


@pytest.mark.asyncio
async def test_unlock_scope_keeps_different_agents_independent():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[
        [{"id": "unlock-a1", "unlocked_at": datetime(2026, 6, 1, tzinfo=UTC)}],
        [{"id": "unlock-a2", "unlocked_at": datetime(2026, 6, 1, tzinfo=UTC)}],
    ])

    with (
        patch.object(repository, "_is_unlock_cached", AsyncMock(return_value=False)),
        patch.object(repository, "_cache_unlocked_achievements", AsyncMock()),
        patch.object(repository, "db", fake_db),
    ):
        first = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            achievement_id=1,
            notify=False,
        )
        second = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a2",
            achievement_id=1,
            notify=False,
        )

    assert first is True
    assert second is True
    assert fake_db.query_raw.await_args_list[0].args[2] == "a1"
    assert fake_db.query_raw.await_args_list[1].args[2] == "a2"


@pytest.mark.asyncio
async def test_duplicate_event_source_is_counted_once_by_repository_contract():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"id": "event-1"}], []])

    with patch.object(repository, "db", fake_db):
        first = await repository.record_event(
            user_id="u1",
            agent_id="a1",
            event_type="aggregation_window_completed",
            source_id="m2",
        )
        duplicate = await repository.record_event(
            user_id="u1",
            agent_id="a1",
            event_type="aggregation_window_completed",
            source_id="m2",
        )

    assert first is True
    assert duplicate is False
