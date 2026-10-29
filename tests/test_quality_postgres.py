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
    await database.execute_raw(
        "UPDATE quality_executor SET job_id=NULL,lease_token=NULL,lease_until=NULL WHERE id='primary'"
    )
    original = set(
        r["id"] for r in await database.query_raw("SELECT id FROM quality_jobs")
    )
    try:
        yield world
    finally:
        await database.execute_raw(
            "UPDATE quality_executor SET job_id=NULL,lease_token=NULL,lease_until=NULL WHERE id='primary'"
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
