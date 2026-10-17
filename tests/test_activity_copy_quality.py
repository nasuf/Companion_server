"""Admin copy-quality view: bounded, and missing status is never success."""
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from app.api.admin import stats
from app.services.offline import copy_quality


def row(i, **fields):
    return {"id": f"a{i}", "created_at": datetime(2026, 10, 15, tzinfo=UTC), "title": "标题" * 40,
            "source": "cleversee", "status": "pending", **fields}


async def test_counts_statuses_and_reports_missing_ones_as_not_recorded(monkeypatch):
    rows = [
        row(1, card_copy="verified", detail="fallback_rejected", detail_issues=json.dumps(["x" * 200, "y"]),
            message="verified", invite="verified_no_evidence"),
        row(2, card_copy=None, detail=None, detail_issues="not json", message="fallback_invalid",
            message_issues=["明年不符"], invite=None),
    ]
    query = AsyncMock(return_value=rows)
    monkeypatch.setattr(copy_quality.db, "query_raw", query)
    report = await copy_quality.collect_copy_quality(7)
    assert report["fields"]["card_copy"] == {"verified": 1, "not_recorded": 1}
    assert report["fields"]["detail"] == {"fallback_rejected": 1, "not_recorded": 1}
    assert report["fields"]["invite"] == {"verified_no_evidence": 1, "not_recorded": 1}
    first, second = report["recent"]
    assert first["issues"]["detail"] == ["x" * 120, "y"] and len(first["title"]) == 60
    assert second["issues"]["message"] == ["明年不符"] and second["relevance"] == "not_recorded"
    assert second["issues"]["detail"] == []  # Unreadable diagnostics are dropped, not shown raw.
    assert report["total"] == 2 and report["truncated"] is False
    assert query.await_args.args[1:] == (7, copy_quality.ROW_LIMIT + 1)


async def test_window_and_rows_are_bounded(monkeypatch):
    query = AsyncMock(return_value=[row(i) for i in range(copy_quality.ROW_LIMIT + 1)])
    monkeypatch.setattr(copy_quality.db, "query_raw", query)
    report = await copy_quality.collect_copy_quality(999)
    assert report["window_days"] == copy_quality.MAX_WINDOW_DAYS
    assert report["truncated"] is True and report["total"] == copy_quality.ROW_LIMIT
    assert len(report["recent"]) == copy_quality.RECENT_LIMIT
    sql = query.await_args.args[0]
    assert "make_interval(days => $1::int)" in sql and "LIMIT $2::int" in sql
    assert "messages" not in sql  # Statuses come from the activity row only.


def test_endpoint_is_admin_only():
    route = next(r for r in stats.router.routes if r.path.endswith("/activity-copy-quality"))
    dependencies = {d.call for d in route.dependant.dependencies}
    assert stats.require_admin_jwt in dependencies


async def test_endpoint_returns_the_bounded_report(monkeypatch):
    collect = AsyncMock(return_value={"total": 0})
    monkeypatch.setattr(copy_quality, "collect_copy_quality", collect)
    assert await stats.activity_copy_quality(days=3, _={}) == {"total": 0}
    collect.assert_awaited_once_with(3)
