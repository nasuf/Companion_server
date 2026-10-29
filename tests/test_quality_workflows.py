"""Lifecycle tests use real SQL; only provider calls and clock-driven inputs fake.

These assert business-table isolation, terminal accounting, review visibility,
lease ownership and repeatable regression baselines rather than implementation.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from tests.test_quality_postgres import quality, running
from tests.test_service_health_postgres import message, world
from app.services.quality import (
    budget,
    daily,
    jobs,
    regression,
    sampling,
    transport,
    judge,
    public_results,
    snapshots,
    triggers,
    worker,
)
from app.config import settings

REAL_NOTIFY = jobs.notify


def verdict(*, failure=False):
    return json.dumps(
        dict(
            categories={
                c: "fail" if failure and c == "hallucination" else "pass"
                for c in judge.CATEGORIES
            },
            critical_kinds=["speaker_confusion"] if failure else [],
            confidence=0.9,
            reason="合成判定：主体归属已核实",
        )
    )


async def sample_rows(database, job, conv, user, n=3):
    rows = []
    for i in range(n):
        u = await message(database, conv, "user", 20 - i * 3)
        r = await message(database, conv, "assistant", 19 - i * 3)
        sid = uuid4().hex
        await database.execute_raw(
            "INSERT INTO quality_samples(id,job_id,user_id,conversation_id,message_id,reply_id,internal) VALUES($1,$2,$3,$4,$5,$6,false)",
            sid,
            job["id"],
            user,
            conv,
            u.id,
            r.id,
        )
        rows.append(sid)
    job["report"] = {"definition_version": "quality-review-v1"}
    await database.execute_raw(
        "UPDATE quality_jobs SET report=$2::jsonb WHERE id=$1",
        job["id"],
        json.dumps(job["report"]),
    )
    return rows


@pytest.mark.parametrize(
    "mode", ["ok", "budget", "missing_key", "invalid_judge", "lost_lease", "archived"]
)
async def test_daily_lifecycle_failure_states_and_no_business_writes(
    quality, monkeypatch, mode
):
    database, conv, space, user, admin = quality
    job = await running()
    await sample_rows(database, job, conv, user)
    if mode == "archived":
        await database.execute_raw(
            "UPDATE conversations SET archived_at=NOW() WHERE id=$1", conv
        )
    before = await database.query_raw(
        "SELECT id,content,metadata FROM messages WHERE conversation_id=$1 ORDER BY id",
        conv,
    )
    fake = AsyncMock(return_value=(verdict(failure=True), {}))
    if mode == "budget":
        fake.side_effect = budget.BudgetExhausted("daily")
    elif mode == "missing_key":
        fake.side_effect = transport.NotConfigured("missing")
    elif mode == "invalid_judge":
        fake.return_value = ("{}", {})
    elif mode == "lost_lease":
        fake.side_effect = jobs.LostLease("lost")
    monkeypatch.setattr(transport, "complete", fake)
    if mode == "lost_lease":
        with pytest.raises(jobs.LostLease):
            await daily.run(job)
    else:
        await daily.run(job)
    after = await database.query_raw(
        "SELECT id,content,metadata FROM messages WHERE conversation_id=$1 ORDER BY id",
        conv,
    )
    assert before == after
    counts = await daily.counts(job["id"])
    if mode == "ok":
        assert counts["judged_count"] == 3 and counts["review_pending_count"] == 3
    elif mode == "budget":
        assert counts["budget_unjudged_count"] == 3 and counts["judged_count"] == 0
    elif mode != "lost_lease":
        assert counts["unknown_count"] == 3 and counts["review_pending_count"] == 0
    status = (
        await database.query_raw(
            "SELECT status FROM quality_jobs WHERE id=$1", job["id"]
        )
    )[0]["status"]
    assert (
        status
        == {
            "ok": "completed",
            "budget": "budget_exhausted",
            "missing_key": "not_configured",
            "invalid_judge": "completed",
            "lost_lease": "running",
            "archived": "completed",
        }[mode]
    )
    if mode == "archived":
        fake.assert_not_awaited()


@pytest.mark.parametrize("mode", ["normal", "budget", "config", "renderer", "unknown"])
async def test_regression_exact_baseline_and_incomplete_never_pass(
    quality, monkeypatch, mode
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    snap = {
        "prompts": [{"key": "test", "content": "fixture", "is_enabled": True}],
        "route": ["ark", "model"],
    }
    cases = [
        dict(
            id="case",
            suite="test",
            critical=True,
            history=[],
            route="main",
            now="2026-10-29T04:00:00Z",
            turns=[dict(message="你好", criterion="礼貌回应")],
        )
    ]
    fake = AsyncMock(return_value={"judge": {"pass": True}, "reply": "你好"})
    monkeypatch.setattr(regression.replay, "replay_turn", fake)

    async def one():
        ident = await jobs.enqueue("routing", uuid4().hex, "v1", snap)
        job = await jobs.claim(job_id=ident)
        await regression.run(job, cases=cases)
        return (
            await database.query_raw(
                "SELECT status,report FROM quality_jobs WHERE id=$1", ident
            )
        )[0]

    initial = await one()
    assert initial["report"]["gate_status"] == "baseline_recorded"
    if mode == "budget":
        fake.side_effect = budget.BudgetExhausted("test")
    elif mode == "config":
        fake.side_effect = transport.NotConfigured("test")
    elif mode == "renderer":
        fake.side_effect = regression.replay.RendererError(
            {"error_type": "Synthetic", "frames": []}
        )
    elif mode == "unknown":
        fake.return_value = {"judge": {"pass": None}, "reply": "你好"}
    result = await one()
    assert result["report"]["gate_status"] == (
        "passed" if mode == "normal" else "incomplete"
    )
    if mode == "renderer":
        assert result["report"]["renderer_diagnostic"]["error_type"] == "Synthetic"
    if mode == "normal":
        fake.return_value = {"judge": {"pass": False}, "reply": "错误"}
        failed = await one()
        assert (
            failed["status"] == "failed" and failed["report"]["gate_status"] == "failed"
        )


async def test_missing_snapshot_is_not_configured_and_no_provider_call(
    quality, monkeypatch
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    job = await running("routing")
    provider = AsyncMock()
    monkeypatch.setattr(regression.replay, "replay_turn", provider)
    await regression.run(job, cases=[])
    provider.assert_not_awaited()
    assert (
        await database.query_raw(
            "SELECT status FROM quality_jobs WHERE id=$1", job["id"]
        )
    )[0]["status"] == "not_configured"


async def test_api_lists_results_and_review_conflict_use_real_database(
    quality, monkeypatch
):
    from app.api.admin import quality as api
    from app.api.jwt_auth import require_admin_jwt

    database, conv, space, user, admin = quality
    monkeypatch.setattr(api, "db", database)
    monkeypatch.setattr(public_results, "db", database)
    job = await running()
    ids = await sample_rows(database, job, conv, user)
    from app.services.quality import budget

    charge = await budget.reserve(
        job["id"], "synthetic/model", 0.01, lease_token=job["lease_token"]
    )
    await budget.settle(charge, 0.005, dict(input=100, output=10, cached_input=0))
    await database.execute_raw(
        "UPDATE quality_samples SET review_queue=true WHERE job_id=$1", job["id"]
    )
    await database.execute_raw(
        "INSERT INTO quality_results(job_id,case_id,repetition,result) VALUES($1,'test',0,'{\"pass\":false}'::jsonb)",
        job["id"],
    )
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[require_admin_jwt] = lambda: {
        "sub": admin,
        "role": "admin",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        data = (await c.get("/admin-api/quality")).json()
        assert data["sampled_at"] and data["policy"]["monthly_cny"] == 60
        item = next(j for j in data["jobs"] if j["id"] == job["id"])
        assert (
            float(item["settled_cny"]) == 0.005
            and float(item["reserved_or_unknown_cny"]) == 0
        )
        base = "/admin-api/quality/jobs/" + job["id"]
        page = (await c.get(base + "/samples?limit=1")).json()
        assert len(page["items"]) == 1 and page["next_cursor"]
        assert (await c.get(base + "/results")).json()["items"][0]["case_id"] == "test"
        sid = ids[0]
        assert (await c.get("/admin-api/quality/samples/" + sid)).json()["history"]
        assert (await c.get("/admin-api/quality/samples/nope")).status_code == 404
        body = {
            "revision": 0,
            "decision": "confirmed",
            "category": "hallucination",
            "note": "合成来源证据显示主体错误",
        }
        url = "/admin-api/quality/samples/" + sid + "/review"
        assert (await c.put(url, json=body)).status_code == 200
        assert (await c.put(url, json=body)).status_code == 409
        body.update(revision=1, note="短")
        assert (await c.put(url, json=body)).status_code == 400
        assert (
            await c.put(
                "/admin-api/quality/samples/nope/review",
                json={"revision": 0, "decision": "uncertain"},
            )
        ).status_code == 404
        remaining = (await c.get(base + "/samples")).json()["items"]
        assert sid not in {x["id"] for x in remaining}
        assert (await c.get("/admin-api/quality/routing-history")).status_code == 200


async def test_notifications_reuse_admin_channel_and_dedupe(quality, monkeypatch):
    from app.services.notifications import service

    database, conv, space, user, admin = quality
    # Restore the actual helper replaced by the isolation fixture.
    monkeypatch.setattr(
        settings, "redis_memory_alert_user_ids", admin + "," + user + "," + admin
    )
    async with database.tx() as tx:
        await REAL_NOTIFY(tx, "synthetic-job", "budget_exhausted")
        await REAL_NOTIFY(tx, "synthetic-job", "budget_exhausted")
    rows = await database.query_raw(
        "SELECT user_id,type FROM notification_events WHERE dedupe_key='quality:synthetic-job:budget_exhausted'"
    )
    assert rows == [{"user_id": admin, "type": "admin_service_health"}]


async def test_regression_has_no_future_turns_and_preserves_historical_clock(
    quality, monkeypatch
):
    from copy import deepcopy

    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    seen = []

    async def fake(job, case, *args, **kwargs):
        seen.append(deepcopy(case))
        return {"reply": "合成回应", "judge": {"pass": True}}

    monkeypatch.setattr(regression.replay, "replay_turn", fake)
    job = await running("routing")
    job["snapshot"] = {"prompts": [{"key": "fixture"}], "route": ["ark", "fixture"]}
    case = dict(
        id="two-days",
        suite="fixture",
        critical=False,
        route="main",
        now="2026-10-01T04:00:00+00:00",
        history=[["user", "旧事"], ["assistant", "好"]],
        gap=120,
        turns=[
            dict(message="第一天", criterion="第一天"),
            dict(message="第七天", criterion="第七天", now="2026-10-07T04:00:00+00:00"),
        ],
    )
    await regression.run(job, cases=[case])
    assert all("turns" not in c for c in seen)
    assert (
        seen[1]["history_rows"][0]["createdAt"]
        == seen[0]["history_rows"][0]["createdAt"]
    )
    assert seen[1]["history_rows"][-2]["createdAt"] == seen[0]["now"]
    assert seen[1]["now"] == "2026-10-07T04:00:00+00:00"
    assert seen[1]["history"][-1] == ["assistant", "合成回应"]


async def test_sample_initialization_cannot_write_after_lease_loss(
    quality, monkeypatch
):
    database, *_ = quality
    job = await running()
    monkeypatch.setattr(sampling, "select_day", AsyncMock(return_value=(0, [])))
    await database.execute_raw(
        "UPDATE quality_executor SET lease_token='another-worker' WHERE id='primary'"
    )
    with pytest.raises(jobs.LostLease):
        await sampling.initialize_samples(job, datetime.now(UTC))
    assert (
        await database.query_raw(
            "SELECT report FROM quality_jobs WHERE id=$1", job["id"]
        )
    )[0]["report"] == {}


async def test_human_regression_uses_exact_source_or_unknown(quality, monkeypatch):
    from app.services.quality import review

    database, conv, space, user, admin = quality
    job = await running()
    ids = await sample_rows(database, job, conv, user, 1)
    await review.save(
        ids[0], admin, 0, "confirmed", "hallucination", "合成记忆事实不应捏造"
    )
    monkeypatch.setattr(regression, "db", database)
    cases = await regression.corpus({"kind": "prompt"})
    human = next(c for c in cases if c["suite"] == "human_confirmed")
    assert human["route"] == "recorded_prompt" and human["recorded_messages"] is None
    fake = AsyncMock()
    monkeypatch.setattr(transport, "complete", fake)
    result = await regression.replay.replay_turn(
        job, {**human, **human["turns"][0]}, [], ["ark", "fixture"]
    )
    assert result["judge"]["pass"] is None
    fake.assert_not_awaited()


def test_daily_bound_and_inconsistent_judgement():
    value = {
        "history": [
            {"role": "user", "content": "old" * 5000},
            {"role": "user", "content": "current"},
        ],
        "prompt_evidence": ["evidence" * 5000],
        "reply": "hi",
    }
    bounded = daily.bounded_context(value, 1000)
    assert bounded["history"] == [value["history"][-1]] and bounded["truncated"]
    with pytest.raises(ValueError):
        daily.bounded_context({**value, "reply": "x" * 30000}, 1000)
    v = json.loads(verdict())
    v["categories"].pop("temporal")
    with pytest.raises(ValueError, match="missing_judge"):
        judge.parse(json.dumps(v))
    v = json.loads(verdict())
    v["critical_kinds"] = ["hallucination"]
    with pytest.raises(ValueError, match="inconsistent"):
        judge.parse(json.dumps(v))


def test_historical_prompt_requires_exact_unique_trace_and_valid_roles():
    from app.services.quality.recorded import reply_messages

    trace = {
        "inputs_json": {
            "messages": [
                [
                    {"kwargs": {"type": "system", "content": "人设"}},
                    {"kwargs": {"type": "human", "content": "你好"}},
                ]
            ]
        },
        "outputs_json": {"generations": [[{"text": "第一句||第二句[EMO:中性/20]"}]]},
    }
    found = reply_messages([trace], "第一句")
    assert found == [
        {"role": "system", "content": "人设"},
        {"role": "user", "content": "你好"},
    ]
    assert reply_messages([trace, trace], "第一句") is None
    assert reply_messages([trace], "别的回复") is None
    assert (
        reply_messages(
            [{}, {"inputs_json": {}}, {**trace, "outputs_json": {"generations": [{}]}}],
            "第一句",
        )
        is None
    )
    for messages in (
        [],
        [None],
        [{"role": "tool", "content": "bad"}],
        [{"role": "user", "content": []}],
        [{"role": "assistant", "content": "no-user"}],
    ):
        assert (
            reply_messages([{**trace, "inputs_json": {"messages": messages}}], "第一句")
            is None
        )


async def test_recorded_model_replay_retains_real_persona_not_synthetic(monkeypatch):
    from app.services.quality import replay

    messages = [
        {"role": "system", "content": "你是普洱的编辑小伊"},
        {"role": "user", "content": "你住哪里"},
    ]
    fake = AsyncMock(side_effect=[("我住普洱", {}), ('{"pass":true}', {})])
    monkeypatch.setattr(transport, "complete", fake)
    case = {
        "route": "recorded_prompt",
        "recorded_messages": messages,
        "criterion": "不混淆用户位置",
    }
    result = await replay.replay_turn({}, case, [], ["ark", "model"])
    assert result["judge"]["pass"] is True
    assert fake.await_args_list[0].args[3] == messages
    fake.side_effect = [("我住普洱", {}), ('{"other":true}', {})]
    assert (await replay.replay_turn({}, case, [], ["ark", "model"]))["judge"][
        "pass"
    ] is None
