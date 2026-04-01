"""主动聊天策略层。"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from app.services.proactive_state import ProactiveStateRecord, PROACTIVE_WINDOWS, is_same_local_day

UTC = timezone.utc

STAGE_HIT_MULTIPLIER = {
    "cold_start": 0.85,
    "warming": 1.0,
    "intimate": 1.15,
}


def should_hit_window(state: ProactiveStateRecord) -> tuple[bool, float]:
    base_rate = 0.0
    for window in PROACTIVE_WINDOWS:
        if int(window["index"]) == int(state.current_window_index or 0):
            base_rate = float(window["hit_rate"])
            break

    multiplier = STAGE_HIT_MULTIPLIER.get(state.stage, 1.0)
    silence_penalty = max(0.5, 1.0 - 0.08 * max(0, state.silence_level_n))
    final_rate = max(0.0, min(1.0, base_rate * multiplier * silence_penalty))
    return random.random() < final_rate, final_rate


def scene_candidate_available(
    state: ProactiveStateRecord,
    schedule_status: dict[str, Any] | None,
    now: datetime | None = None,
) -> bool:
    now_ts = now or datetime.now(UTC)
    status = str((schedule_status or {}).get("status") or "idle")
    if status == "sleep":
        return False
    if is_same_local_day(state.daily_scene_triggered_at, now_ts):
        return False
    return True


def select_trigger_type(
    state: ProactiveStateRecord,
    *,
    scene_available: bool,
) -> str:
    weights: list[tuple[str, float]]
    if state.stage == "cold_start":
        weights = [
            ("silence_wakeup", 0.55),
            ("memory_proactive", 0.35),
            ("scheduled_scene", 0.10 if scene_available else 0.0),
        ]
    elif state.stage == "warming":
        weights = [
            ("silence_wakeup", 0.30),
            ("memory_proactive", 0.45),
            ("scheduled_scene", 0.25 if scene_available else 0.0),
        ]
    else:
        weights = [
            ("silence_wakeup", 0.20),
            ("memory_proactive", 0.45),
            ("scheduled_scene", 0.35 if scene_available else 0.0),
        ]

    filtered = [(name, weight) for name, weight in weights if weight > 0]
    total = sum(weight for _, weight in filtered)
    if total <= 0:
        return random.choice(["silence_wakeup", "memory_proactive"])

    r = random.random() * total
    acc = 0.0
    for name, weight in filtered:
        acc += weight
        if r <= acc:
            return name
    return filtered[-1][0]


def fallback_trigger_type() -> str:
    return random.choice(["silence_wakeup", "memory_proactive"])
