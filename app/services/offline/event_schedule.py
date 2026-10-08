"""One presentation format for event cards/details, without fake midnight slots."""

from datetime import datetime


def schedule_label(event: dict) -> str | None:
    if not event:
        return None
    try:
        start, end = [
            datetime.fromisoformat(event[k].replace("Z", "+00:00"))
            for k in ("starts_at", "ends_at")
        ]
        day = start.strftime("%Y/%m/%d")
        if event.get("time_precision") == "datetime":
            label = day + start.strftime(" %H:%M")
            if event.get("end_time_known", True):
                label += "—" + end.strftime(
                    "%H:%M" if start.date() == end.date() else "%m/%d %H:%M"
                )
        else:
            label = day + (
                ("—" + end.strftime("%m/%d")) if start.date() != end.date() else ""
            )
        if event.get("daily_hours"):
            label += " 每日" + "—".join(event["daily_hours"])
        return label
    except (ValueError, TypeError, KeyError):
        return None
