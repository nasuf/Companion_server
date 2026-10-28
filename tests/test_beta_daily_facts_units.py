"""UXM01.02 派生表的调度、选日、接口校验和状态判定 (不需要数据库的部分)."""
from __future__ import annotations

import ast
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.effect import daily_facts, definitions
from app.services.effect.definitions import DEFINITION_VERSION, METRICS, local_day_bounds

DAY = date(2027, 3, 12)


def _row(**over):
    return {"value": 5.0, "numerator": 5.0, "denominator": 40.0, "sample_size": 40, "settled": True,
            "definition_version": DEFINITION_VERSION, "population_fingerprint": "fp", **over}


class TestPointState:
    def test_collection_start_marks_earlier_days_not_collected(self):
        since = date.fromisoformat(METRICS["opened_users"]["collected_since"])
        assert daily_facts.point_state("opened_users", since - timedelta(days=1), _row(),
                                       expected_fingerprint="fp") == ("not_collected", None)
        assert daily_facts.point_state("opened_users", since, _row(), expected_fingerprint="fp")[0] == "ok"

    def test_cost_before_llm_usage_existed_is_not_collected(self):
        assert daily_facts.point_state("cost_per_active_user", date(2026, 4, 1), _row(value=0.0),
                                       expected_fingerprint="fp") == ("not_collected", None)

    def test_missing_and_outdated_rows_never_show_a_value(self):
        assert daily_facts.point_state("turns", DAY, None, expected_fingerprint="fp") == ("not_computed", None)
        assert daily_facts.point_state("turns", DAY, _row(definition_version="old"),
                                       expected_fingerprint="fp") == ("stale", None)
        assert daily_facts.point_state("turns", DAY, _row(), expected_fingerprint="other") == ("stale", None)

    @pytest.mark.parametrize("over,state,value", [
        ({"denominator": 0.0}, "no_sample", None),
        ({"settled": False}, "unsettled", None),
        ({"denominator": 7.0}, "insufficient_sample", None),
        ({"value": 0.0, "numerator": 0.0}, "ok", 0.0),
    ])
    def test_rates(self, over, state, value):
        assert daily_facts.point_state("no_reply_10m", DAY, _row(**over), expected_fingerprint="fp") == (state, value)

    def test_means_report_small_samples_but_not_empty_ones(self):
        assert daily_facts.point_state("turns_per_active_user", DAY, _row(denominator=3.0, value=2.0),
                                       expected_fingerprint="fp") == ("ok", 2.0)
        assert daily_facts.point_state("turns_per_active_user", DAY, _row(denominator=0.0),
                                       expected_fingerprint="fp") == ("no_sample", None)

    def test_quantiles_need_enough_samples_and_counts_can_be_unsettled(self):
        metric = "reply_first_bubble_ms_p90"
        late = date.fromisoformat(METRICS[metric]["collected_since"]) + timedelta(days=1)
        assert daily_facts.point_state(metric, late, _row(sample_size=0), expected_fingerprint="fp")[0] == "no_sample"
        assert daily_facts.point_state(metric, late, _row(sample_size=5), expected_fingerprint="fp")[0] == (
            "insufficient_sample")
        assert daily_facts.point_state(metric, late, _row(sample_size=20, value=900.0),
                                       expected_fingerprint="fp") == ("ok", 900.0)
        assert daily_facts.point_state("turns", DAY, _row(settled=False), expected_fingerprint="fp") == (
            "unsettled", 5.0)


class TestSettlement:
    def test_each_metric_settles_after_its_own_window(self):
        end = local_day_bounds(DAY)[1].replace(tzinfo=timezone.utc)
        assert not daily_facts.is_settled("turns", DAY, end + timedelta(minutes=9))
        assert daily_facts.is_settled("turns", DAY, end + timedelta(minutes=10))
        assert not daily_facts.is_settled("next_day_return", DAY, end + timedelta(hours=23))
        assert daily_facts.is_settled("next_day_return", DAY, (end + timedelta(hours=24)).replace(tzinfo=None))
        assert not daily_facts.facts_settled(DAY, end + timedelta(hours=5))
        assert daily_facts.facts_settled(DAY, end + timedelta(hours=6))

    def test_every_metric_declares_kind_unit_and_window(self):
        for key, spec in METRICS.items():
            assert spec["kind"] in {"count", "rate", "mean", "quantile"}, key
            assert spec["unit"] and spec["settle_after_s"] >= 0, key
        assert set(definitions.describe(("turns",))["metrics"]) == {"turns"}


class TestDaySelection:
    async def test_nightly_recomputes_recent_days_today_and_heals(self):
        now = datetime(2027, 3, 12, 20, 40, tzinfo=timezone.utc)  # 北京时间 03-13 04:40
        today = date(2027, 3, 13)
        heal = [today - timedelta(days=40)]
        with patch.object(daily_facts, "days_needing_compute", AsyncMock(return_value=heal)) as need, \
             patch.object(daily_facts, "compute_days", AsyncMock(return_value={"computed": [], "failed": []})) as run:
            await daily_facts.run_nightly(now=now)
        assert need.await_args.args == (today - timedelta(days=daily_facts.HEAL_LOOKBACK_DAYS), today)
        # 从近往远: 今天和昨天先就绪
        assert run.await_args.args[0] == sorted(
            heal + [today - timedelta(days=i) for i in range(3, -1, -1)], reverse=True)

    async def test_nightly_failure_is_reported(self):
        with patch.object(daily_facts, "days_needing_compute", AsyncMock(return_value=[])), \
             patch.object(daily_facts, "compute_days", AsyncMock(return_value={"computed": [], "failed": ["d"]})):
            with pytest.raises(RuntimeError, match="1 day"):
                await daily_facts.run_nightly()

    async def test_catchup_starts_at_earliest_data_and_is_capped(self):
        now = datetime(2027, 3, 12, 12, tzinfo=timezone.utc)
        today = date(2027, 3, 12)
        with patch.object(daily_facts, "earliest_data_day", AsyncMock(return_value=date(2020, 1, 1))), \
             patch.object(daily_facts, "days_needing_compute", AsyncMock(return_value=[today])) as need, \
             patch.object(daily_facts, "compute_days", AsyncMock(return_value={"computed": ["x"], "failed": []})):
            assert (await daily_facts.run_catchup(now=now))["computed"] == ["x"]
        assert need.await_args.args == (today - timedelta(days=daily_facts.MAX_BACKFILL_DAYS - 1), today)
        with patch.object(daily_facts, "earliest_data_day", AsyncMock(return_value=None)):
            assert await daily_facts.run_catchup(now=now) == {"computed": [], "failed": []}
        with patch.object(daily_facts, "earliest_data_day", AsyncMock(return_value=today)), \
             patch.object(daily_facts, "days_needing_compute", AsyncMock(return_value=[today])), \
             patch.object(daily_facts, "compute_days", AsyncMock(return_value={"computed": [], "failed": ["x"]})):
            with pytest.raises(RuntimeError, match="catch-up"):
                await daily_facts.run_catchup(now=now)

    async def test_one_failed_day_does_not_stop_the_rest(self):
        with patch.object(daily_facts, "compute_day", AsyncMock(side_effect=[RuntimeError("x"), {"day": "b"}])):
            result = await daily_facts.compute_days([DAY, DAY + timedelta(days=1)])
        assert result == {"computed": [(DAY + timedelta(days=1)).isoformat()], "failed": [DAY.isoformat()]}

    async def test_empty_range_needs_nothing(self):
        assert await daily_facts.days_needing_compute(DAY, DAY - timedelta(days=1)) == []


class TestBackgroundHeal:
    async def test_runs_once_and_never_raises(self):
        @asynccontextmanager
        async def acquired(*_a, **_k):
            yield True

        @asynccontextmanager
        async def busy(*_a, **_k):
            from app.services.runtime.distributed_lock import DistributedLockNotAcquired
            raise DistributedLockNotAcquired("busy")
            yield

        with patch("app.services.runtime.distributed_lock.distributed_lock", acquired), \
             patch.object(daily_facts, "compute_days", AsyncMock()) as run:
            await daily_facts._heal_in_background([DAY])
        run.assert_awaited_once_with([DAY])
        with patch("app.services.runtime.distributed_lock.distributed_lock", busy), \
             patch.object(daily_facts, "compute_days", AsyncMock()) as run:
            await daily_facts._heal_in_background([DAY])
        run.assert_not_awaited()
        with patch("app.services.runtime.distributed_lock.distributed_lock", acquired), \
             patch.object(daily_facts, "compute_days", AsyncMock(side_effect=RuntimeError("db"))):
            await daily_facts._heal_in_background([DAY])  # 只记日志


async def test_read_series_survives_a_failed_on_demand_refresh(monkeypatch):
    class Db:
        async def query_raw(self, *_a):
            return []

    monkeypatch.setattr("app.db.db", Db())
    now = datetime(2027, 3, 12, 6, tzinfo=timezone.utc)
    with patch.object(daily_facts, "compute_day", AsyncMock(side_effect=RuntimeError("down"))), \
         patch.object(daily_facts, "population_fingerprint", AsyncMock(return_value="fp")), \
         patch.object(daily_facts, "days_needing_compute", AsyncMock(return_value=[])):
        out = await daily_facts.read_series(date(2027, 3, 11), date(2027, 3, 12), cohort="external",
                                            metrics=["turns"], now=now)
    assert [p["state"] for p in out["series"]["turns"]] == ["not_computed", "not_computed"]
    assert out["computed_at"] is None and out["recomputing"] == []


class TestScheduler:
    async def test_nightly_and_startup_share_one_lock_but_report_apart(self):
        from jobs import scheduler

        calls = []

        async def fake_job(name, ttl, body, *, health_name=None):
            calls.append((name, health_name))
            await body()

        with patch.object(scheduler, "_run_distributed_job", fake_job), \
             patch("app.services.effect.daily_facts.run_nightly", AsyncMock()) as nightly, \
             patch("app.services.effect.daily_facts.run_catchup", AsyncMock()) as catchup:
            await scheduler._run_daily_facts()
            await scheduler._run_daily_facts("daily_facts_startup_catchup")
        assert calls == [("daily_facts", "daily_facts"), ("daily_facts", "daily_facts_startup_catchup")]
        nightly.assert_awaited_once()
        catchup.assert_awaited_once()

        with patch.object(scheduler, "_run_distributed_job", fake_job), \
             patch("app.services.effect.daily_facts.run_nightly", AsyncMock(side_effect=RuntimeError("x"))), \
             patch.object(scheduler, "_job_failed") as failed:
            await scheduler._run_daily_facts()
        failed.assert_called_once()

    def test_cron_time_matches_the_constant_the_invariant_uses(self):
        source = (Path(__file__).resolve().parents[1] / "jobs" / "scheduler.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_job":
                kw = {k.arg: k.value for k in node.keywords}
                if isinstance(kw.get("id"), ast.Constant) and kw["id"].value == "daily_facts":
                    assert (kw["hour"].value, kw["minute"].value) == (
                        daily_facts.NIGHTLY_HOUR, daily_facts.NIGHTLY_MINUTE)
                    return
        raise AssertionError("daily_facts cron job not registered")

    def test_both_triggers_are_registered(self):
        from jobs import scheduler

        jobs = {job.id for job in scheduler.scheduler_job_definitions()}
        assert "daily_facts" in jobs


def _admin(api_client):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


@pytest.mark.parametrize("params,detail", [
    ({"metrics": "turns,bogus"}, "unknown metrics"),
    ({"dimension": "trigger_type", "metrics": "turns"}, "supports"),
    ({"end": "2999-01-01"}, "future"),
    ({"end": "0001-01-01"}, "on or after"),
    ({"end": "not-a-date"}, "YYYY-MM-DD"),
])
def test_daily_metrics_endpoint_rejects_bad_requests(api_client, params, detail):
    app, dep = _admin(api_client)
    try:
        with patch("app.services.effect.daily_facts.read_series", AsyncMock()) as read:
            r = api_client.get("/admin-api/stats/daily-metrics", params=params)
        assert r.status_code == 422 and detail in r.text
        read.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(dep, None)


def test_daily_metrics_endpoint_bounds_window_and_returns_definitions(api_client):
    app, dep = _admin(api_client)
    try:
        with patch("app.services.effect.daily_facts.read_series", AsyncMock(return_value={"series": {}})) as read:
            ok = api_client.get("/admin-api/stats/daily-metrics",
                                params={"days": 7, "end": "2026-10-20", "cohort": "all",
                                        "metrics": "proactive_events", "dimension": "trigger_type"})
            too_long = api_client.get("/admin-api/stats/daily-metrics", params={"days": 400})
            default = api_client.get("/admin-api/stats/daily-metrics")
        assert ok.status_code == 200 and too_long.status_code == 422 and default.status_code == 200
        args, kwargs = read.await_args_list[0]
        assert args == (date(2026, 10, 14), date(2026, 10, 20))
        assert kwargs == {"cohort": "all", "metrics": ["proactive_events"], "dimension": "trigger_type"}
        assert set(ok.json()["definitions"]["metrics"]) == {"proactive_events"}
        assert read.await_args_list[1].kwargs["metrics"] == list(METRICS)
        with patch("app.services.effect.daily_facts.read_series", AsyncMock(return_value={"series": {}})) as read:
            by_path = api_client.get("/admin-api/stats/daily-metrics", params={"dimension": "reply_path"})
        assert by_path.status_code == 200  # 只给维度时默认取该维度支持的指标
        assert read.await_args.kwargs["metrics"] == [
            "reply_first_bubble_ms_p50", "reply_first_bubble_ms_p90",
            "reply_processing_ms_p50", "reply_processing_ms_p90"]
    finally:
        app.dependency_overrides.pop(dep, None)


def test_daily_metrics_endpoint_requires_admin(api_client, auth_header):
    r = api_client.get("/admin-api/stats/daily-metrics", headers=auth_header("u-1"))
    assert r.status_code in (401, 403)
