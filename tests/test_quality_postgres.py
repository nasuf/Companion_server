"""Quality accounting and sampling against real isolated PostgreSQL."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from prisma import Json

from tests.test_service_health_postgres import world
from app.services.quality import budget, jobs, daily, sampling, review
from app.services.quality.policy import local_day


@pytest.fixture
async def quality(world, monkeypatch):
    database, conv, space, user, admin = world
    for module in (budget, jobs, daily, sampling, review):
        monkeypatch.setattr(module, "db", database)
    monkeypatch.setattr(jobs, "notify", AsyncMock())
    monkeypatch.setattr(jobs, "notify_review_pending", AsyncMock())
    await database.execute_raw(
        "UPDATE quality_executor SET job_id=NULL,lease_token=NULL,lease_until=NULL"
    )
    original = set(
        r["id"] for r in await database.query_raw("SELECT id FROM quality_jobs")
    )
    try:
        yield world
    finally:
        await database.execute_raw(
            "UPDATE quality_executor SET job_id=NULL,lease_token=NULL,lease_until=NULL"
        )
        ids = [
            r["id"]
            for r in await database.query_raw("SELECT id FROM quality_jobs")
            if r["id"] not in original
        ]
        await database.execute_raw(
            "DELETE FROM llm_usage WHERE id IN (SELECT 'quality-'||id FROM quality_charges WHERE job_id=ANY($1::text[]))",
            ids,
        )
        await database.execute_raw(
            "UPDATE quality_budget_months m SET committed_cny=committed_cny-s.removed FROM (SELECT month,sum(COALESCE(cost_cny,reserved_cny)) AS removed FROM quality_charges WHERE job_id=ANY($1::text[]) GROUP BY month) s WHERE m.month=s.month",
            ids,
        )
        await database.execute_raw(
            "DELETE FROM quality_charges WHERE job_id=ANY($1::text[])", ids
        )
        await database.execute_raw(
            "DELETE FROM quality_jobs WHERE id=ANY($1::text[])", ids
        )


async def running(kind="daily"):
    ident = await jobs.enqueue(kind, "test-" + uuid4().hex, uuid4().hex, {})
    return await jobs.claim(job_id=ident)


async def test_budget_concurrent_reservations_and_idempotent_settlement(
    quality, monkeypatch
):
    database, *_ = quality

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 4, 12, tzinfo=UTC)

    monkeypatch.setattr(budget, "datetime", FixedClock)
    job = await running()
    results = await asyncio.gather(
        *(
            budget.reserve(
                job["id"],
                "dashscope/qwen3.5-flash",
                0.2,
                lease_token=job["lease_token"],
            )
            for _ in range(8)
        ),
        return_exceptions=True,
    )
    ids = [v for v in results if isinstance(v, str)]
    assert (
        len(ids) == 5
        and sum(isinstance(v, budget.BudgetExhausted) for v in results) == 3
    )
    usage = {"input": 100, "output": 10, "cached_input": 0}
    await asyncio.gather(*(budget.settle(ids[0], 0.01, usage) for _ in range(4)))
    rows = await database.query_raw(
        "SELECT * FROM llm_usage WHERE id=$1", "quality-" + ids[0]
    )
    assert (
        len(rows) == 1
        and rows[0]["user_id"] is None
        and rows[0]["scope"] == "quality_eval"
    )
    assert rows[0]["conversation_id"] is None and rows[0]["cost_cny"] == 0.01
    with pytest.raises(ValueError, match="settlement_conflict"):
        await budget.settle(ids[0], 0.02, usage)
    await budget.uncertain(ids[1])
    with pytest.raises(budget.BudgetExhausted):
        await budget.reserve(
            job["id"], "dashscope/qwen3.5-flash", 0.2, lease_token=job["lease_token"]
        )
    with pytest.raises(jobs.LostLease):
        await budget.reserve(
            job["id"], "dashscope/qwen3.5-flash", 0.01, lease_token="wrong"
        )


async def test_cluster_executor_expired_owner_and_prompt_debounce(quality):
    database, *_ = quality
    key = uuid4().hex
    first = await jobs.enqueue("prompt", key, "v1", {}, delay=600)
    last = await jobs.enqueue("prompt", key, "v2", {}, delay=600)
    assert (
        await database.query_raw("SELECT status FROM quality_jobs WHERE id=$1", first)
    )[0]["status"] == "skipped"
    assert await jobs.claim(job_id=last) is None
    job = await running()
    other = await jobs.enqueue("routing", uuid4().hex, "v1", {})
    assert await jobs.claim(job_id=other) is None
    await database.execute_raw(
        "UPDATE quality_executor SET lease_until=NOW()-INTERVAL '1 second' WHERE id='primary'"
    )
    new = await jobs.claim(job_id=other)
    assert new and new["id"] == other
    assert (
        await database.query_raw(
            "SELECT status FROM quality_jobs WHERE id=$1", job["id"]
        )
    )[0]["status"] == "interrupted"
    with pytest.raises(jobs.LostLease):
        await jobs.finish(job, "completed", {})


async def test_daily_fixed_sample_reuses_bursts_and_review_is_owned(
    quality, monkeypatch
):
    database, conv, space, user, admin = quality
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    # A unique old day avoids unrelated synthetic rows in the shared database.
    base = now.replace(tzinfo=None) - timedelta(hours=1)

    async def msg(role, text, second, meta=None):
        return await database.message.create(
            data={
                "conversationId": conv,
                "role": role,
                "content": text,
                "createdAt": base + timedelta(seconds=second),
                "metadata": Json(meta or {}),
            }
        )

    await msg("user", "我住在南京", 0)
    target = await msg("user", "不是苏州", 1)
    reply = await msg("assistant", "好，是南京", 3)
    await msg("assistant", "我记住啦", 4)
    await msg("user", "👌", 10)
    await msg("assistant", "好", 11)
    await msg("user", "系统自动卡片", 20, {"auto_card": True})
    await msg("assistant", "好", 21)
    await msg("user", "人工接管内容", 30, {"human_takeover": True})
    await msg("assistant", "人工回复", 31, {"human_takeover": True})
    count, selected = await sampling.select_day(local_day(now), now)
    assert count == 1 and selected[0]["message_id"] == target.id
    assert (await sampling.select_day(local_day(now), now))[1] == selected
    ident = await jobs.enqueue("daily", uuid4().hex, "sample", {}, now=now)
    job = await jobs.claim(job_id=ident)
    await sampling.initialize_samples(job, now)
    sample = (
        await database.query_raw("SELECT * FROM quality_samples WHERE job_id=$1", ident)
    )[0]
    detail = await sampling.context(sample["id"])
    assert detail["history"][-1]["content"] == "我住在南京\n不是苏州"
    assert detail["reply"] == "好，是南京\n我记住啦"
    result = await review.save(
        sample["id"],
        admin,
        0,
        "confirmed",
        "hallucination",
        "测试人工核实的具体错误证据",
    )
    assert result["revision"] == 1
    with pytest.raises(review.ReviewConflict):
        await review.save(sample["id"], admin, 0, "false_positive", None, "")
    cases = await database.query_raw(
        "SELECT * FROM quality_cases WHERE sample_id=$1", sample["id"]
    )
    assert (
        len(cases) == 1 and cases[0]["case_data"]["source"]["type"] == "quality_review"
    )
    await review.save(sample["id"], admin, 1, "false_positive", None, "人工更正")
    assert not (
        await database.query_raw(
            "SELECT enabled FROM quality_cases WHERE sample_id=$1", sample["id"]
        )
    )[0]["enabled"]
    await database.message.delete(where={"id": reply.id})
    assert await sampling.context(sample["id"]) is None
    assert not await database.query_raw(
        "SELECT id FROM quality_cases WHERE sample_id=$1", sample["id"]
    )


async def test_stale_executor_cannot_write_result_and_recovery_notifies(
    quality, monkeypatch
):
    from app.services.quality import regression

    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    job = await running("routing")
    sink = regression.LocalSink(job)
    await sink.prepare({"corpus_hash": "synthetic"})
    await database.execute_raw(
        "UPDATE quality_executor SET lease_until=NOW()-INTERVAL '1 second' WHERE id='primary'"
    )
    with pytest.raises(jobs.LostLease):
        await sink.result({"case_id": "x", "repetition": 0, "pass": True})
    next_id = await jobs.enqueue("routing", uuid4().hex, "v1", {})
    next_job = await jobs.claim(job_id=next_id)
    jobs.notify.assert_awaited_once_with(
        __import__("unittest").mock.ANY, job["id"], "interrupted"
    )
    assert next_job and not await database.query_raw(
        "SELECT * FROM quality_results WHERE job_id=$1", job["id"]
    )


async def test_quality_admin_api_denies_nonadmin_and_invalid_dates(monkeypatch):
    from app.api.admin import quality as api
    from app.api.jwt_auth import require_user
    from fastapi import FastAPI
    import httpx

    app = FastAPI()
    app.include_router(api.router)

    async def ordinary():
        return {"sub": "synthetic", "role": "user"}

    app.dependency_overrides[require_user] = ordinary
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in (
            "",
            "/jobs/x/samples",
            "/samples/x",
            "/routing-history",
            "/jobs/x/results",
            "/jobs/x/report",
        ):
            assert (await client.get("/admin-api/quality" + path)).status_code == 403

        async def admin():
            return {"sub": "synthetic", "role": "admin"}

        app.dependency_overrides[require_user] = admin
        assert (
            await client.get("/admin-api/quality?day=1900-01-01")
        ).status_code == 400
        assert (
            await client.get("/admin-api/quality/jobs/x/samples?limit=500")
        ).status_code == 422


async def test_regression_report_groups_baseline_not_run_and_filters(quality, monkeypatch):
    """O01.08.V01: summary -> failing group detail, versions as digests only."""
    from app.api.admin import quality as api
    from app.api.jwt_auth import require_user
    from fastapi import FastAPI
    import httpx

    database = quality[0]
    monkeypatch.setattr(api, "db", database)
    snapshot = {"hash": "snap-digest", "prompts": [{"key": "chat.system_base", "content": "PROMPT-BODY-SECRET"}],
                "route": ["ark", "model-x"], "prices": {"secret": 1}}

    async def regression_job(results, report):
        ident = await jobs.enqueue("nightly", "test-" + uuid4().hex, uuid4().hex, snapshot)
        await database.execute_raw(
            "UPDATE quality_jobs SET status='completed',report=$2::jsonb WHERE id=$1", ident, json.dumps(report))
        for case_id, repetition, suite, critical, passed in results:
            row = dict(case_id=case_id, repetition=repetition, suite=suite, critical=critical, turns=1)
            row["pass"] = passed
            await database.execute_raw(
                "INSERT INTO quality_results(job_id,case_id,repetition,result) VALUES($1,$2,$3,$4::jsonb)",
                ident, case_id, repetition, json.dumps(row))
        return ident

    common = dict(corpus_version="v9", corpus_hash="corpus-a", repeats=3)
    base = await regression_job(
        [("persona:a", r, "persona", True, True) for r in range(3)] + [("reply:b", 0, "reply", False, True)],
        dict(common, gate_status="baseline_recorded"))
    candidate = await regression_job(
        [("persona:a", 0, "persona", True, True), ("persona:a", 1, "persona", True, False),
         ("persona:a", 2, "persona", True, None), ("reply:b", 0, "reply", False, True),
         ("human_confirmed:c", 0, "human_confirmed", False, False)],
        dict(common, baseline_job_id=base, expected=7, evaluated=5,
             gate={"status": "incomplete", "reason": "missing_scenarios"},
             expected_by_group={"persona|critical": 3, "reply|other": 1, "human_confirmed|other": 1,
                                "never|other": 2},
             errored_scenarios=[{"case_id": "persona:a", "repetition": 2}]))
    daily_job = await jobs.enqueue("daily", "test-" + uuid4().hex, uuid4().hex, {})

    app = FastAPI()
    app.include_router(api.router)

    async def admin():
        return {"sub": "synthetic", "role": "admin"}

    app.dependency_overrides[require_user] = admin
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/admin-api/quality/jobs/{candidate}/report")
        assert response.status_code == 200, response.text[:300]
        assert "PROMPT-BODY-SECRET" not in response.text and "secret" not in response.text
        body = response.json()
        assert (await client.get(f"/admin-api/quality/jobs/{daily_job}/report")).status_code == 404
        assert (await client.get("/admin-api/quality/jobs/missing/report")).status_code == 404
        failing = (await client.get(f"/admin-api/quality/jobs/{candidate}/results",
                                    params={"suite": "persona", "critical": "true", "failed_only": "true"})).json()
        other = (await client.get(f"/admin-api/quality/jobs/{candidate}/results",
                                  params={"suite": "reply", "failed_only": "true"})).json()

    assert body["state"] == "completed"
    assert body["versions"]["code"] and body["versions"]["prompt_version"] is None
    assert body["versions"] == dict(body["versions"], prompt_snapshot="snap-digest", prompt_count=1,
                                    route=["ark", "model-x"], corpus_version="v9", corpus_hash="corpus-a", repeats=3)
    assert body["totals"] == dict(expected=7, evaluated=5, not_run=2, unjudged=1, errored=1,
                                  expected_by_group="recorded")
    groups = {g["key"]: g for g in body["groups"]}
    persona = groups["persona|critical"]
    assert (persona["passed"], persona["failed"], persona["unjudged"], persona["judged"]) == (1, 1, 1, 2)
    assert persona["pass_rate"] == 0.5 and persona["small_sample"] is True
    assert 0 < persona["ci95"][0] < 0.5 < persona["ci95"][1] < 1
    assert persona["origin"] == "synthetic" and persona["not_run"] == 0
    assert groups["human_confirmed|other"]["origin"] == "production_confirmed"
    never = groups["never|other"]
    assert (never["samples"], never["not_run"], never["ci95"], never["pass_rate"]) == (0, 2, None, None)
    assert body["baseline"]["id"] == base and body["baseline"]["corpus_matches"] is True
    assert body["baseline"]["groups"]["persona|critical"]["pass_rate"] == 1.0
    assert body["gate"]["reason"] == "missing_scenarios"
    assert [(r["case_id"], r["repetition"]) for r in failing["items"]] == [("persona:a", 1), ("persona:a", 2)]
    assert other["items"] == []


def test_wilson_interval_bounds():
    from app.api.admin.quality import wilson

    assert wilson(0, 0) is None
    low, high = wilson(10, 10)
    assert high == 1.0 and 0.6 < low < 0.75
    low, high = wilson(0, 10)
    assert low == 0.0 and 0.25 < high < 0.35


def test_report_state_separates_running_stale_not_run_and_failures():
    from app.api.admin.quality import report_state

    assert report_state(dict(status="running", lease_expired=False)) == "running"
    assert report_state(dict(status="running", lease_expired=True)) == "stale"
    assert report_state(dict(status="pending")) == "not_run"
    assert report_state(dict(status="completed")) == "completed"
    assert report_state(dict(status="skipped")) == "skipped"
    assert report_state(dict(status="budget_exhausted")) == "failed"


async def test_regression_report_tolerates_legacy_and_malformed_reports(quality, monkeypatch):
    """Old reports lack per-group plans; malformed fields must not become numbers."""
    from app.api.admin import quality as api

    database = quality[0]
    monkeypatch.setattr(api, "db", database)
    ident = await jobs.enqueue("nightly", "test-" + uuid4().hex, uuid4().hex, {"prompts": "bad"})
    other = await jobs.enqueue("nightly", "test-" + uuid4().hex, uuid4().hex, {})
    await database.execute_raw(
        "UPDATE quality_jobs SET status='completed',report=$2::jsonb WHERE id=$1",
        other, json.dumps({"corpus_hash": "other-corpus"}))
    await database.execute_raw(
        "UPDATE quality_jobs SET status='running',lease_until=NOW()-interval '1 minute',report=$2::jsonb WHERE id=$1",
        ident, json.dumps({
            "corpus_hash": "corpus-b", "baseline_job_id": other, "errored_scenarios": "bad",
            "gate": {"status": "incomplete", "noise_rate": "bad"}, "error_type": "LostLease",
            "expected_by_group": {"no-separator": 3, "x|other": "bad"},
        }))
    await database.execute_raw(
        "INSERT INTO quality_results(job_id,case_id,repetition,result) VALUES($1,'x:a',0,$2::jsonb)",
        ident, json.dumps(dict(case_id="x:a", repetition=0, suite="x", critical=False, turns=1, **{"pass": True})))
    body = await api.regression_report(ident)
    assert body["state"] == "stale"
    assert body["versions"]["prompt_count"] is None and body["versions"]["route"] is None
    assert body["totals"]["expected"] is None and body["totals"]["not_run"] is None
    assert body["totals"]["evaluated"] == 1 and body["totals"]["errored"] == 0
    assert body["thresholds"]["noise_rate"] == api.settings.quality_noise_rate
    assert body["error"] == {"type": "LostLease", "case_id": None}
    assert [g["key"] for g in body["groups"]] == ["x|other"] and body["groups"][0]["not_run"] is None
    assert body["baseline"]["corpus_matches"] is False and body["baseline"]["groups"] == {}

    # Baseline job since deleted: no comparison rather than an empty one.
    await database.execute_raw(
        "UPDATE quality_jobs SET report='{\"baseline_job_id\":\"deleted-job\"}'::jsonb WHERE id=$1", ident)
    legacy = await api.regression_report(ident)
    assert legacy["totals"]["expected_by_group"] == "not_collected"
    assert legacy["baseline"] is None and legacy["gate"] is None and legacy["error"] is None
    await database.execute_raw("UPDATE quality_jobs SET report='{}'::jsonb WHERE id=$1", ident)
    assert (await api.regression_report(ident))["baseline"] is None
