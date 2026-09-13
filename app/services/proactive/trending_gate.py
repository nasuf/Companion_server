"""Eligibility gate for proactive trending / hot-news enrichment.

Product scope (2026-09): only these trigger types may attach live web context:
  - silence_wakeup: natural "what's going on lately" opener
  - scheduled_scene: AI sharing while in an activity slot
  - special_date: holiday/birthday greeting can lightly reference current events

memory_proactive is intentionally excluded — those messages are anchored on
private memories, not public headlines.

Runtime knobs (enabled / probabilities) live in SystemConfig and are read via
runtime_config.resolve_config_sync — not env settings directly.
"""

from __future__ import annotations

import random

from app.config import settings
from app.services.runtime_config import resolve_config_sync

TRENDING_ELIGIBLE_TRIGGER_TYPES = frozenset({
    "silence_wakeup",
    "scheduled_scene",
    "special_date",
})


def is_trending_eligible_trigger(trigger_type: str) -> bool:
    return (trigger_type or "").strip() in TRENDING_ELIGIBLE_TRIGGER_TYPES


def should_attach_trending(
    trigger_type: str,
    *,
    random_value: float | None = None,
) -> bool:
    """Return True when this send attempt should fetch/inject trending context."""
    cfg = resolve_config_sync(agent_id=None)
    if not cfg.proactive_trending_enabled:
        return False
    if not is_trending_eligible_trigger(trigger_type):
        return False
    probability = cfg.proactive_trending_probability
    if probability <= 0:
        return False
    roll = random.random() if random_value is None else random_value
    return roll < probability


def should_attach_trending_link_card(
    *,
    trending_attached: bool,
    random_value: float | None = None,
) -> bool:
    """Second-stage gate: link card only after trending context was selected."""
    if not trending_attached:
        return False
    if not settings.proactive_link_recommendation_enabled:
        return False
    cfg = resolve_config_sync(agent_id=None)
    probability = cfg.proactive_trending_link_probability
    if probability <= 0:
        return False
    roll = random.random() if random_value is None else random_value
    return roll < probability
