"""Typed event payloads consumed by the achievement engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class UserMessageAchievementEvent:
    user_id: str
    agent_id: str
    workspace_id: str | None
    conversation_id: str
    message_id: str
    text: str
    agent_name: str | None = None
    reply_context: dict | None = None
    aggregation_route: str | None = None
    component_card: dict | None = None
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AssistantMessageAchievementEvent:
    conversation_id: str
    message_id: str
    text: str
    metadata: dict | None = None
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AssistantTurnAchievementEvent:
    conversation_id: str
    message_id: str
    assistant_texts: list[str]
    user_message_ids: list[str]
    turn_id: str | None = None
    metadata: dict | None = None
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryChangelogAchievementEvent:
    user_id: str
    memory_id: str
    operation: str
    workspace_id: str | None = None


@dataclass(frozen=True, slots=True)
class IntentAchievementEvent:
    intent: str
    user_id: str
    agent_id: str
    workspace_id: str | None
    conversation_id: str
    message_id: str | None
    metadata: dict | None = None
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AggregationAchievementEvent:
    user_id: str
    agent_id: str
    conversation_id: str
    source_id: str
    part_count: int
    workspace_id: str | None = None
    occurred_at: datetime | None = None
