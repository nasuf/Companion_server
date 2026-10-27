"""Server-side turn timing; never infer missing observations as zero.

The unit is one logical turn, anchored to its LAST user message. Aggregation
wait is measured inside the aggregation queue; planned delay is a policy value,
not elapsed time and must not be added to actual waits. Generation covers the
whole graph (reads, models and formatting) up to the first bubble ready to emit
or save. Streaming paths emit before persistence; short circuits save first.
`first_bubble_at` is server readiness, NOT a commit or client delivery receipt.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any

VERSION = "server-turn-timing-v1"
CONTEXT_KEY = "_turn_timing"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


def _stamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (ValueError, OverflowError):
        return None


def _elapsed(start: Any, end: datetime) -> float | None:
    stamp = _stamp(start)
    if stamp is None or end < stamp:
        return None
    return round((end - stamp).total_seconds() * 1000, 3)


def initial_timing(received_at: datetime) -> dict:
    return {"received_at": received_at.isoformat(), "aggregation_wait_ms": 0.0}


def mark_queue(context: dict | None, queue: str, *, leaving: bool = False) -> dict | None:
    """Copy only; a failed enqueue must not mutate the immediate fallback path."""
    if not isinstance(context, dict):
        return context
    result = dict(context)
    source = context.get(CONTEXT_KEY)
    timing = dict(source) if isinstance(source, dict) else {}
    now = utc_now()
    if leaving:
        timing["aggregation_wait_ms"] = _elapsed(timing.get("aggregation_entered_at"), now)
    else:
        timing[f"{queue}_entered_at"] = now.isoformat()
    result[CONTEXT_KEY] = timing
    return result


@dataclass
class TurnTiming:
    """Invocation-owned monotonic clock; shared across explicit intent fragments."""

    started_at: datetime = field(default_factory=utc_now)
    started_clock: float = field(default_factory=time.perf_counter)
    first_bubble_at: datetime | None = None
    generation_ms: float | None = None

    def first_bubble(self, context: dict | None) -> dict:
        if self.first_bubble_at is None:
            self.first_bubble_at = utc_now()
            self.generation_ms = round(max(0.0, time.perf_counter() - self.started_clock) * 1000, 3)
        ctx = context if isinstance(context, dict) else {}
        raw = ctx.get(CONTEXT_KEY)
        timing = raw if isinstance(raw, dict) else {}
        received = _stamp(timing.get("received_at"))
        delay = _number(ctx.get("delay_seconds"))
        reason = ctx.get("delay_reason")
        return {
            "timing_definition_version": VERSION,
            "turn_received_at": received.isoformat() if received else None,
            "aggregation_wait_ms": _number(timing.get("aggregation_wait_ms")),
            "planned_delay_ms": round(delay * 1000, 3) if delay is not None else None,
            "delay_reason": reason if isinstance(reason, str) and reason else None,
            "generation_started_at": self.started_at.isoformat(),
            "generation_ms": self.generation_ms,
            "first_bubble_at": self.first_bubble_at.isoformat(),
            "first_bubble_clock": "server_bubble_ready",
            "pre_generation_wait_ms": _elapsed(timing.get("received_at"), self.started_at),
            "delayed_queue_wait_ms": _elapsed(timing.get("delayed_entered_at"), self.started_at),
        }
