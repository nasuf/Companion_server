from datetime import timezone

from app.services.achievements.utils import _aware


def test_aware_accepts_iso_datetime_string():
    parsed = _aware("2026-06-14T07:12:36.123456")

    assert parsed.tzinfo == timezone.utc
    assert parsed.year == 2026


def test_aware_accepts_zulu_datetime_string():
    parsed = _aware("2026-06-14T07:12:36Z")

    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
