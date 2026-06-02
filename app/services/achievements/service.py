"""Stable public entrypoints for the achievement system."""

from __future__ import annotations

from app.services.achievements.engine import (
    handle_achievement_event,
    handle_assistant_message_event,
    handle_assistant_turn_event,
    handle_intent_event,
    handle_memory_changelog_event,
    handle_user_message_event,
)
from app.services.achievements.events import (
    AssistantMessageAchievementEvent,
    AssistantTurnAchievementEvent,
    IntentAchievementEvent,
    MemoryChangelogAchievementEvent,
    UserMessageAchievementEvent,
)
from app.services.achievements.repository import list_achievements
from app.services.achievements.rules.daily_rollup_rules import run_daily_rollup
from app.services.achievements.utils import count_chars

__all__ = [
    "AssistantMessageAchievementEvent",
    "AssistantTurnAchievementEvent",
    "IntentAchievementEvent",
    "MemoryChangelogAchievementEvent",
    "UserMessageAchievementEvent",
    "count_chars",
    "handle_achievement_event",
    "handle_assistant_message_event",
    "handle_assistant_turn_event",
    "handle_intent_event",
    "handle_memory_changelog_event",
    "handle_user_message_event",
    "list_achievements",
    "run_daily_rollup",
]
