"""Chat evidence is validated separately from the achievement rule catalogue."""

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import pytest
from app.services.achievements import engine, repository, source
from app.services.achievements.events import (
    UserMessageAchievementEvent,
    AssistantMessageAchievementEvent,
    IntentAchievementEvent,
    AggregationAchievementEvent,
    AssistantTurnAchievementEvent,
)

AT = datetime(2026, 10, 15, 15, 59, 59, tzinfo=timezone.utc)


@pytest.fixture
def stored(monkeypatch):
    row = dict(
        id="m",
        content="原消息",
        metadata={},
        created_at=AT,
        user_id="u",
        agent_id="a",
        workspace_id="w",
    )
    query = AsyncMock(return_value=[row])
    monkeypatch.setattr(source.db, "query_raw", query)
    return row, query


def user(**changes):
    return replace(
        UserMessageAchievementEvent("u", "a", "w", "c", "m", "伪造回调"), **changes
    )


@pytest.mark.asyncio
async def test_content_and_clock_are_loaded_from_original_message(stored):
    row, query = stored
    row["metadata"] = {"component_card": {"type": "location"}}
    result, identity = await source.resolve_message_event(
        user(occurred_at=datetime.now(timezone.utc))
    )
    assert result.text == row["content"] and result.occurred_at == AT
    assert result.component_card == row["metadata"]["component_card"]
    assert identity.message_id == "m" and identity.role == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"user_id": "other"},
        {"agent_id": "other"},
        {"workspace_id": "other"},
        {"message_id": ""},
    ],
)
async def test_untrusted_scope_is_rejected(stored, change):
    with pytest.raises(source.AchievementSourceUnavailable):
        await source.resolve_message_event(user(**change))


@pytest.mark.asyncio
async def test_missing_source_is_distinct_from_database_failure(stored):
    _, query = stored
    query.return_value = []
    with pytest.raises(source.AchievementSourceUnavailable):
        await source.resolve_message_event(user())
    query.side_effect = RuntimeError("synthetic outage")
    with pytest.raises(RuntimeError, match="synthetic outage"):
        await source.resolve_message_event(user())


@pytest.mark.asyncio
async def test_assistant_metadata_and_intent_aggregation_time(stored):
    row, _ = stored
    row["metadata"] = {"proactive": True}
    result, identity = await source.resolve_message_event(
        AssistantMessageAchievementEvent("c", "m", "fake", {"sticker_url": "fake"})
    )
    assert result.metadata == {"proactive": True} and result.text == "原消息"
    assert identity.role == "assistant"
    for event in [
        IntentAchievementEvent("schedule_adjust", "u", "a", None, "c", "m"),
        AggregationAchievementEvent("u", "a", "c", "m", 2),
    ]:
        result, identity = await source.resolve_message_event(event)
        assert result.occurred_at == AT and result.workspace_id == "w"


@pytest.mark.asyncio
async def test_task_source_resets_after_evaluation_failure(stored, monkeypatch):
    monkeypatch.setattr(
        engine, "achievement_evaluation_enabled", AsyncMock(return_value=True)
    )
    evaluate = AsyncMock(side_effect=RuntimeError("rule failed"))
    monkeypatch.setattr(engine, "evaluate_user_message", evaluate)
    with pytest.raises(RuntimeError, match="rule failed"):
        await engine.handle_achievement_event(user())
    assert source.current_source.get() is None


@pytest.mark.asyncio
async def test_missing_source_skips_rules(stored, monkeypatch):
    stored[1].return_value = []
    monkeypatch.setattr(
        engine, "achievement_evaluation_enabled", AsyncMock(return_value=True)
    )
    evaluate = AsyncMock()
    monkeypatch.setattr(engine, "evaluate_user_message", evaluate)
    await engine.handle_achievement_event(user())
    evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_failures_are_not_duplicates(monkeypatch):
    monkeypatch.setattr(
        repository.db, "query_raw", AsyncMock(side_effect=RuntimeError("db failed"))
    )
    monkeypatch.setattr(
        repository, "achievement_evaluation_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(repository, "_is_unlock_cached", AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="db failed"):
        await repository.record_event(
            user_id="u", agent_id="a", event_type="user_message", source_id="m"
        )
    with pytest.raises(RuntimeError, match="db failed"):
        await repository.unlock_achievement(user_id="u", agent_id="a", achievement_id=1)


@pytest.mark.asyncio
async def test_derived_event_uses_task_clock_and_scope(stored, monkeypatch):
    _, identity = await source.resolve_message_event(user())
    query = AsyncMock(return_value=[{"id": "event"}])
    monkeypatch.setattr(repository.db, "query_raw", query)
    token = source.current_source.set(identity)
    try:
        await repository.record_event(
            user_id="u",
            agent_id="a",
            workspace_id="w",
            conversation_id="c",
            event_type="derived",
            source_id="m",
        )
        assert query.await_args.args[10] == AT
        assert "source_message_id" in query.await_args.args[9]
        with pytest.raises(source.AchievementSourceUnavailable):
            await repository.record_event(
                user_id="u", agent_id="other", event_type="derived"
            )
    finally:
        source.current_source.reset(token)


@pytest.mark.asyncio
async def test_memory_events_keep_their_own_source_contract(stored):
    from app.services.achievements.events import MemoryChangelogAchievementEvent

    event = MemoryChangelogAchievementEvent("u", "memory", "create", "w")
    assert await source.resolve_message_event(event) == (event, None)
    stored[1].assert_not_awaited()


@pytest.mark.asyncio
async def test_bounds_missing_turn_and_streak_failure(stored, monkeypatch):
    with pytest.raises(ValueError, match="invalid_source_sample_limit"):
        await source.recent_message_sources("u", "a", 51)
    stored[0]["metadata"] = {"achievement_turn_id": "turn"}
    event = AssistantTurnAchievementEvent("c", "m", ["fake"], ["u-message"], None)
    with pytest.raises(source.AchievementSourceUnavailable, match="incomplete_turn"):
        await source.resolve_message_event(event)
    from app.services import interaction_streak as streak

    query = AsyncMock(side_effect=RuntimeError("db outage"))
    monkeypatch.setattr(streak.db, "query_raw", query)
    await streak.record_user_message_day_for_conversation("c")
    query.assert_not_awaited()
    with pytest.raises(RuntimeError, match="db outage"):
        await streak.record_user_message_day_for_conversation("c", message_id="m")


@pytest.mark.asyncio
async def test_oversized_assistant_turn_never_uses_callback_fallback(
    stored, monkeypatch
):
    from app.services.achievements.rules import assistant_message_rules as rules

    identity = source.MessageSource("m", "c", "u", "a", "w", "assistant", AT)
    token = source.current_source.set(identity)
    try:
        monkeypatch.setattr(
            rules.db, "query_raw", AsyncMock(return_value=[{"content": "x"}] * 101)
        )
        unlock = AsyncMock()
        monkeypatch.setattr(rules, "unlock_achievement", unlock)
        await rules._check_turn_pair_100(
            user_id="u",
            agent_id="a",
            workspace_id="w",
            conversation_id="c",
            at=AT,
            assistant_texts=["x" * 100],
            user_message_ids=["user"],
            turn_id="turn",
        )
        unlock.assert_not_awaited()
    finally:
        source.current_source.reset(token)
