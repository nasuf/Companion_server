"""Notification debounce shared by SQL-serialized monitors.

Measurement level stays truthful during recovery; only the notification state
is latched. Unknown samples and gaps cannot provide recovery evidence.
"""

from datetime import UTC, datetime

from app.config import settings


def timestamp(value) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def advance(
    level: str, state: dict, now: datetime, *, interval: int
) -> tuple[bool, dict]:
    """Return (recovered, state). Called once per committed sample, under lock."""
    state = dict(state)
    if level != "normal":
        state.pop("recovery", None)
        return False, state
    if state.get("level", "normal") == "normal":
        return False, {"level": "normal", "checks": []}
    previous = state.get("recovery") or {}
    continuous = (
        previous
        and 0 < (now - timestamp(previous["last_at"])).total_seconds() <= interval * 2
    )
    recovery = {
        "since": previous["since"] if continuous else now.isoformat(),
        "samples": int(previous.get("samples", 0)) + 1 if continuous else 1,
        "last_at": now.isoformat(),
    }
    # Duplicate/out-of-order observations must not advance the streak.
    if previous and now <= timestamp(previous["last_at"]):
        return False, state
    if (
        recovery["samples"] >= settings.ops_recovery_samples
        and (now - timestamp(recovery["since"])).total_seconds()
        >= settings.ops_recovery_seconds
    ):
        return True, {"level": "normal", "checks": []}
    return False, {**state, "recovery": recovery}
