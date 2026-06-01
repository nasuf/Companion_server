"""Stable public entrypoints for the achievement system."""

from __future__ import annotations

from app.services.achievements.assistant_messages import process_assistant_message
from app.services.achievements.memory_events import process_memory_changelog
from app.services.achievements.repository import list_achievements, record_event, unlock_achievement
from app.services.achievements.rollup import run_daily_rollup
from app.services.achievements.user_messages import process_user_message
from app.services.achievements.utils import count_chars

__all__ = [
    "count_chars",
    "list_achievements",
    "process_assistant_message",
    "process_memory_changelog",
    "process_user_message",
    "record_event",
    "run_daily_rollup",
    "unlock_achievement",
]
