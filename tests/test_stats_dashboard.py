"""Window summaries and administrator facts: no invented zeros or daily-p90 mean."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.effect import (
    dashboard as d,
    daily_facts as facts,
    definitions as defs,
)
from tests.test_beta_daily_facts_postgres import pg, _seed, _user, _metrics, _now, DAY


def point(value=None, *, numerator=None, denominator=None, state="ok", settled=True):
    return dict(
        value=value,
        numerator=numerator,
        denominator=denominator,
        state=state,
        settled=settled,
    )


def test_weighted_rates_unique_users_and_unknowns():
    rows = [point(numerator=1, denominator=20), point(numerator=40, denominator=80)]
    assert d.aggregate_points("continuation", rows)["value"] == 0.41  # not (.05+.5)/2
    assert d.aggregate_points("feedback_per_100_turns", rows)["value"] == 41
    assert (
        d.aggregate_points("chat_active_users", [point(4), point(4)], distinct=5)[
            "value"
        ]
        == 5
    )
    for state in ("failed", "not_computed", "not_collected", "stale"):
        out = d.aggregate_points("turns", [point(0), point(None, state=state)])
        assert out["state"] == state and out["value"] is None
    assert d.aggregate_points("turns", [point(0)])["value"] == 0
    assert (
        d.aggregate_points("continuation", [point(numerator=1, denominator=5)])["state"]
        == "insufficient_sample"
    )
    assert (
        d.aggregate_points(
            "continuation",
            [point(numerator=1, denominator=40, state="unsettled", settled=False)],
        )["value"]
        is None
    )
    assert (
        d.aggregate_points("cost_per_turn", [point(numerator=0.5, denominator=2)])[
            "value"
        ]
        == 0.25
    )


def test_quantiles_are_window_values_and_costs_have_distinct_definitions():
    rows = [point(1), point(100)]
    result = d.aggregate_points(
        "reply_first_bubble_ms_p90", rows, quantile={"first_n": 100, "first_p90": 80}
    )
    assert result["value"] == 80 and result["sample_size"] == 100
    assert d.aggregate_points("reply_first_bubble_ms_p90", rows)["state"] == "failed"
    assert "TTS" in defs.METRICS["cost_per_active_user"]["definition"]
    assert "不是逐句" in defs.METRICS["feedback_per_100_turns"]["definition"]


def test_csv_prevents_formula_and_omits_sensitive_fields():
    text = d.users_csv(
        [
            {
                "id": "x",
                "username": ' \t=HYPERLINK("x")',
                "content": "private",
                "cost_cny": 0,
            }
        ]
    )
    assert "' \t=HYPERLINK" in text and "private" not in text and "content" not in text
    assert text.startswith("\ufeffid,username,")


@pytest.mark.parametrize(
    "days,end", [(0, None), (93, None), (2, "x"), (2, "2999-01-01"), (2, "2025-01-01")]
)
def test_window_bounds(days, end):
    with pytest.raises(ValueError):
        d.window(days, end)


async def test_users_journey_are_derived_and_filters_work(pg):
    ids = await _seed(pg)
    await facts.compute_day(DAY, now=_now(DAY + timedelta(days=5)))
    out = await d.users(DAY, DAY, cohort="external", q=ids.external)
    assert out["total"] == 1
    row = out["items"][0]
    assert row["id"] == ids.external and row["status"] == "new" and row["state"] == "ok"
    assert (
        row["user_messages"],
        row["turns"],
        row["feedback_count"],
        row["cost_cny"],
    ) == (6, 1, 1, 0.5)
    assert (await d.users(DAY, DAY, cohort="external", q=ids.admin))["total"] == 0
    assert (await d.users(DAY, DAY, cohort="all", q=ids.admin))["total"] == 1
    # Arbitrary SQL in q is bound as a value.
    assert (await d.users(DAY, DAY, cohort="all", q="' OR true --"))["total"] == 0
    assert (await d.users(DAY, DAY, cohort="all", q=ids.external, page=2, page_size=1))[
        "items"
    ] == []
    for kwargs in (
        {"sort": "id;DROP TABLE users"},
        {"order": "asc NULLS FIRST;"},
        {"status": "weird"},
        {"page_size": 101},
    ):
        with pytest.raises(ValueError):
            await d.users(DAY, DAY, cohort="external", **kwargs)
    journey = await d.journey(ids.external, DAY, DAY + timedelta(days=1))
    assert journey["items"][0]["turns"] == 1
    assert journey["items"][1] == {
        "day": str(DAY + timedelta(days=1)),
        "state": "not_computed",
    }
    assert await d.journey("missing", DAY, DAY) is None
    assert (await d.journey(ids.admin, DAY, DAY))["user"]["internal"] is True
    await pg.db.execute_raw(
        "UPDATE users SET status='deleting' WHERE id=$1", ids.external
    )
    assert await d.journey(ids.external, DAY, DAY) is None
    assert (await d.users(DAY, DAY, cohort="all", q=ids.external))["items"] == []


async def test_new_metrics_and_missing_marker_do_not_turn_into_zero(pg):
    ids = await _seed(pg)
    await facts.compute_day(DAY, now=_now(DAY + timedelta(days=5)))
    metrics = await _metrics(pg, DAY, "external")
    assert metrics["feedback_per_100_turns"]["value"] == 100
    assert metrics["cost_per_turn"]["value"] == 0.5
    await pg.db.execute_raw(
        "DELETE FROM metrics_daily WHERE local_date=$1::date", str(DAY)
    )
    row = (await d.users(DAY, DAY, cohort="external", q=ids.external))["items"][0]
    assert (
        row["state"] == "not_computed"
        and row["status"] == "unknown"
        and row["user_messages"] is None
    )


@pytest.mark.parametrize(
    "path", ["users", "users/u/journey", "daily-metrics?summary=true"]
)
def test_new_api_requires_admin(api_client, auth_header, path):
    response = api_client.get("/admin-api/stats/" + path, headers=auth_header("u-1"))
    assert response.status_code in (401, 403)


def test_api_bounds_and_csv_download(api_client):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        with patch.object(
            d,
            "users",
            AsyncMock(
                return_value={
                    "items": [{"id": "x", "username": "safe", "user_messages": 1}],
                    "truncated": False,
                }
            ),
        ):
            response = api_client.get("/admin-api/stats/users?format=csv")
            assert (
                response.status_code == 200
                and "text/csv" in response.headers["content-type"]
            )
            assert "no-store" == response.headers["cache-control"]
            assert response.headers["x-export-truncated"] == "false"
        for query in (
            "days=93",
            "days=0",
            "end=no",
            "page_size=101",
            "cohort=secret",
            "format=raw",
        ):
            assert api_client.get("/admin-api/stats/users?" + query).status_code == 422
        with patch.object(d, "journey", AsyncMock(return_value=None)):
            assert (
                api_client.get("/admin-api/stats/users/none/journey").status_code == 404
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
