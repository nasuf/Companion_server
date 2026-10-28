"""Daily compatibility cards, cost totals and owned drill-downs on real PostgreSQL."""

from datetime import timedelta
from unittest.mock import AsyncMock, patch
import json
import pytest
from app.services.effect import (
    details,
    costs,
    retention,
    daily_facts as facts,
    definitions as defs,
    dashboard,
)
from app.services.effect.context_summary import summarize, merge
from tests.test_beta_daily_facts_postgres import pg, _seed, _now, _at, _msg, DAY


def test_context_merge_recomputes_cache_ratio_and_does_not_duplicate_multi_model_cost():
    start, end = defs.local_day_bounds(DAY)
    now = _now(DAY)
    a = summarize(
        [
            dict(
                rounds="5",
                main_tokens="4000",
                tokens_by_model={
                    "ark/m": {"input": 100, "cached_input": 20, "output": 2},
                    "other/m": {"input": 50, "cached_input": 0, "output": 1},
                },
                cost_cny=0.2,
                web_search_calls=2,
            )
        ],
        start,
        end,
        now,
    )
    b = summarize(
        [
            dict(
                rounds="5",
                main_tokens="4000",
                tokens_by_model={
                    "ark/m": {"input": 900, "cached_input": 900, "output": 8}
                },
                cost_cny=0.3,
                web_search_calls=0,
            )
        ],
        start,
        end,
        now,
    )
    result = merge([a, b], start, end, now.isoformat())
    assert result["rows"] == 2
    assert result["groups"][0]["cache_hit_rate"] == 0.92
    assert result["groups"][1]["cache_hit_rate"] is None
    assert result["groups"][1]["cache_unknown_turns"] == 1
    assert result["costs"][0]["recorded_token_cost_cny"] == 0.5
    assert result["costs"][0]["normalized_search_cost_cny"] == 0.008
    assert merge([], start, end, now.isoformat())["status"] == "empty"


async def test_cards_and_hourly_read_only_derived_rows_and_respect_cohort(pg):
    ids = await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    await facts.compute_day(DAY, now=now)
    # Break access to raw message rows after computation: all compatible cards
    # still work from the derived tables and retain exact weighted counts.
    external = await details.read(DAY, DAY, "reply_effect", cohort="external")
    all_rows = await details.read(DAY, DAY, "reply_effect", cohort="all")
    assert external["data"]["summary"]["turns"] < all_rows["data"]["summary"]["turns"]
    hourly = await details.read(DAY, DAY, "hourly", cohort="external")
    assert sum(r["count"] for r in hourly["data"]) == 6
    context = await details.read(DAY, DAY, "context", cohort="all")
    assert context["data"]["rows"] >= 1
    copied = await details.read(DAY, DAY, "copy_quality", cohort="all")
    assert copied["data"]["total"] == 0 and not copied["data"]["truncated"]
    assert (await details.read(DAY, DAY, "health", cohort="all"))["data"] == {}
    await pg.db.execute_raw(
        "UPDATE users SET is_internal=true WHERE id=$1", ids.external
    )
    assert (await details.read(DAY, DAY, "reply_effect", cohort="external"))[
        "coverage"
    ]["state"] == "not_computed"
    assert (await details.read(DAY, DAY, "context", cohort="external"))["data"][
        "rows"
    ] == 0
    with pytest.raises(ValueError):
        await details.read(DAY, DAY, "arbitrary_table")
    assert (await details.read(DAY, DAY + timedelta(days=1), "context"))["data"] is None


async def test_costs_preserve_media_units_missing_price_and_quality_eval_separation(
    pg, monkeypatch
):
    ids = await _seed(pg)
    await facts.compute_day(DAY, now=_now(DAY + timedelta(days=5)))
    # Seed derived ledger: query contract must not use cost=0 as evidence of price.
    await pg.db.execute_raw(
        "DELETE FROM stats_cost_daily WHERE local_date=$1::date", str(DAY)
    )
    rows = [
        (
            "model",
            "chat",
            dict(
                input_tokens=100,
                output_tokens=10,
                cost_cny=0.5,
                call_count=1,
                models={
                    "ark/test": {
                        "input": 100,
                        "output": 10,
                        "cached_input": 20,
                        "cost_cny": 0.5,
                    }
                },
            ),
        ),
        ("model", "quality_eval", dict(cost_cny=2, call_count=2)),
        (
            "tts",
            "speech_output",
            dict(request_count=3, units=40, duration_ms=800, bytes=1200, cost_cny=0.2),
        ),
        ("media", "image", dict(request_count=2, bytes=500)),
        ("asr", "text", dict(request_count=1, units=10, cost_cny=0.1)),
    ]
    for n, (category, scope, values) in enumerate(rows):
        await pg.db.execute_raw(
            """INSERT INTO stats_cost_daily
          (id,local_date,user_id,category,scope,data,definition_version,computed_at)
          VALUES($1,$2::date,$3,$4,$5,$6::jsonb,$7,NOW())""",
            f"{ids.external}-{n}",
            str(DAY),
            ids.external,
            category,
            scope,
            json.dumps({**costs.blank(), **values}),
            defs.DEFINITION_VERSION,
        )
    monkeypatch.setattr(costs.settings, "web_search_price_cny_per_k", 4)
    monkeypatch.setattr(costs.settings, "asr_price_cny_per_second", 0.01)
    result = await costs.read_costs(DAY, DAY, cohort="all")
    assert result["total"]["cost_cny"] == 0.8
    assert result["evaluation"]["cost_cny"] == 2
    assert result["categories"]["tts"]["duration_ms"] == 800
    assert result["media"]["image"]["bytes"] == 500  # not double counted
    assert result["media_users"][0]["usage"]["tts"]["bytes"] == 1200
    assert result["categories"]["model"]["models"]["ark/test"]["cost_cny"] == 0.5
    assert (await costs.read_costs(DAY, DAY, cohort="all", page=2))["media_users"] == []
    monkeypatch.setattr(costs.settings, "asr_price_cny_per_second", 0)
    assert (await costs.read_costs(DAY, DAY, cohort="all"))["total"]["cost_cny"] is None
    await pg.db.execute_raw(
        "DELETE FROM metrics_daily WHERE local_date=$1::date", str(DAY)
    )
    assert (await costs.read_costs(DAY, DAY, cohort="all"))["total"][
        "state"
    ] == "not_computed"
    for kwargs in ({"agent": "x" * 129}, {"page": 0}, {"page_size": 101}):
        with pytest.raises(ValueError):
            await costs.read_costs(DAY, DAY, cohort="all", **kwargs)


async def test_retention_filters_and_chat_target_ownership(pg):
    ids = await _seed(pg)
    await facts.compute_day(DAY, now=_now(DAY + timedelta(days=5)))
    view = await retention.read_retention(DAY, DAY, cohort="external")
    assert sum(r["count"] for r in view["hourly"]) == 6
    assert sum(r["count"] for r in view["registrations"]) == 1
    assert (
        await retention.read_retention(DAY, DAY, cohort="all", platform="nonexistent")
    )["hourly"] == []
    target = await dashboard.chat_target(ids.external, day=str(DAY))
    assert target and target["rank"] > 0
    assert (
        await dashboard.chat_target(
            ids.admin, conversation_id=target["conversation_id"]
        )
        is None
    )
    assert await dashboard.chat_target(ids.external, trace_id="missing") is None
    assert (
        await dashboard.chat_target(
            ids.external, conversation_id=target["conversation_id"]
        )
        == target
    )
    for kwargs in ({}, {"trace_id": "x" * 129}, {"day": "invalid"}):
        with pytest.raises(ValueError):
            await dashboard.chat_target(ids.external, **kwargs)


@pytest.mark.parametrize(
    "endpoint",
    [
        "detail-metrics/context",
        "detail-metrics/health",
        "system-health-summary",
        "costs",
        "retention",
        "review-items",
        "reply-timing",
        "token-reconciliation",
        "proactive-funnel",
        "release-markers",
        "weekly-review",
        "users/u/chat-target",
        "message-distribution",
    ],
)
def test_admin_only_for_all_new_reads(api_client, auth_header, endpoint):
    assert api_client.get(
        "/admin-api/stats/" + endpoint, headers=auth_header("not-admin")
    ).status_code in (401, 403, 422)


async def test_weekly_report_queue_idempotence_and_distribution(pg, monkeypatch):
    from app.services.effect import weekly

    ids = await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    monkeypatch.setattr(defs, "local_today", lambda now=None: DAY + timedelta(days=5))
    monkeypatch.setattr(facts, "local_today", defs.local_today)
    for n in range(14):
        await facts.compute_day(DAY - timedelta(days=n), now=now)
    report = await weekly.report(DAY, "external")
    assert report["coverage"]["state"] == "ok"
    assert "合成" not in report["markdown"]
    assert len(report["current"]["summary"]) == 8
    assert report["worst"] and all("content" not in row for row in report["worst"])
    first = await weekly.enqueue(DAY - timedelta(days=6), DAY, "external", ids.admin)
    again = await weekly.enqueue(DAY - timedelta(days=6), DAY, "external", ids.admin)
    assert first["added"] == first["selected"] > 0 and again["added"] == 0
    rows = await pg.db.query_raw(
        "SELECT counts FROM stats_review_queue WHERE user_id=$1", ids.external
    )
    assert rows and any("no_reply" in r["counts"] for r in rows)
    dist = await dashboard.message_distribution(DAY, DAY, "external")
    assert sum(b["count"] for b in dist["buckets"]) >= 1
    assert dist["buckets"][1]["count"] == 1
    view = await retention.read_retention(
        DAY - timedelta(days=6), DAY, cohort="external"
    )
    assert len(view["registrations"]) == len(view["activity"]) == 7
    assert view["gap_quantiles"] == {"p50": None, "p90": None}
    await pg.db.execute_raw(
        "DELETE FROM metrics_daily WHERE local_date=$1::date", str(DAY)
    )
    with pytest.raises(ValueError):
        await weekly.enqueue(DAY, DAY, "external", ids.admin)
    assert (await dashboard.message_distribution(DAY, DAY, "external"))["buckets"] == []
    # The extra fourteen-day aggregate fixture has no owner FK; remove it explicitly.
    for table in ("metrics_daily", "stats_detail_daily"):
        await pg.db.execute_raw(
            f"DELETE FROM {table} WHERE local_date BETWEEN $1::date AND $2::date",
            str(DAY - timedelta(days=13)),
            str(DAY),
        )


def test_weekly_markers_keep_ids_and_all_commits():
    from app.services.effect.weekly import markers
    from datetime import date

    rows = markers(date(2026, 10, 1), date(2026, 10, 28))
    assert rows and len({r["id"] for r in rows}) == len(rows)
    assert all(isinstance(r["commits"], list) for r in rows)
