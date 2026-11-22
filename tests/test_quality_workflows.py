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
REAL_REVIEW_NOTIFY = jobs.notify_review_pending


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
    # 只有成功跑完且留下待评审样本时才提醒评审；失败态走既有的失败通知。
    if mode == "ok":
        jobs.notify_review_pending.assert_awaited_once_with(
            __import__("unittest").mock.ANY, job["id"], 3
        )
    else:
        jobs.notify_review_pending.assert_not_awaited()


async def repetition_score(database, reply_id, user, conv, label):
    await database.execute_raw(
        """INSERT INTO reply_repetition_scores(turn_message_id,local_date,user_id,conversation_id,
          occurred_at,situation,bubbles,candidate_bubbles,judged_bubbles,label,repeated,
          scorer_version,computed_at) VALUES($1,CURRENT_DATE,$2,$3,NOW(),'normal',1,1,1,$4,$5,
          'repetition-v1',NOW())""",
        reply_id,
        user,
        conv,
        label,
        label in ("verbatim", "paraphrase"),
    )


async def test_review_queue_skips_repetition_fail_cleared_by_metric(
    quality, monkeypatch
):
    database, conv, space, user, admin = quality
    job = await running()
    labels = ["not_repeat", "none", "legit", "paraphrase", "verbatim", "failed", "unjudgeable"]
    ids = await sample_rows(database, job, conv, user, n=len(labels) + 1)
    # Cleared samples sort first, so the cap would admit them if the metric
    # were applied after LIMIT instead of before.
    for i, sid in enumerate(ids[:3]):
        ids[i] = f"0{i}{sid[2:]}"
        await database.execute_raw(
            "UPDATE quality_samples SET id=$2 WHERE id=$1", sid, ids[i]
        )
    replies = {
        r["id"]: r["reply_id"]
        for r in await database.query_raw(
            "SELECT id,reply_id FROM quality_samples WHERE job_id=$1", job["id"]
        )
    }
    # Metric verdict per sample; the last has no metric row at all.
    for sid, label in zip(ids, labels):
        await repetition_score(database, replies[sid], user, conv, label)
    monkeypatch.setattr(settings, "quality_review_limit", 5)
    repetition_only = json.dumps(
        dict(
            categories={
                c: "fail" if c == "repetition" else "pass" for c in judge.CATEGORIES
            },
            critical_kinds=[],
            confidence=0.8,
            reason="合成判定：与上文表述相近",
        )
    )
    monkeypatch.setattr(
        transport, "complete", AsyncMock(return_value=(repetition_only, {}))
    )
    await daily.run(job)
    queued = {
        r["id"]
        for r in await database.query_raw(
            "SELECT id FROM quality_samples WHERE job_id=$1 AND review_queue", job["id"]
        )
    }
    assert queued == set(ids[3:])

    # A cleared reply with another failing category still enters the queue.
    job = await running()
    ids = await sample_rows(database, job, conv, user, n=1)
    reply = (
        await database.query_raw(
            "SELECT reply_id FROM quality_samples WHERE id=$1", ids[0]
        )
    )[0]["reply_id"]
    await repetition_score(database, reply, user, conv, "legit")
    both = json.loads(repetition_only)
    both["categories"]["hallucination"] = "fail"
    transport.complete.return_value = (json.dumps(both), {})
    await daily.run(job)
    assert (
        await database.query_raw(
            "SELECT review_queue FROM quality_samples WHERE id=$1", ids[0]
        )
    )[0]["review_queue"]


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
        assert data["sampled_at"] and data["policy"]["monthly_cny"] == 120
        item = next(j for j in data["jobs"] if j["id"] == job["id"])
        assert (
            float(item["settled_cny"]) == 0.005
            and float(item["reserved_or_unknown_cny"]) == 0
        )
        base = "/admin-api/quality/jobs/" + job["id"]
        page = (await c.get(base + "/samples?limit=1")).json()
        assert len(page["items"]) == 1 and page["next_cursor"]
        assert page["items"][0]["repetition_label"] is None
        reply = page["items"][0]["reply_id"]
        await repetition_score(database, reply, user, conv, "paraphrase")
        page = (await c.get(base + "/samples?limit=1")).json()
        assert page["items"][0]["repetition_label"] == "paraphrase"
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


async def test_review_pending_notification_reaches_admins_once_per_job(quality, monkeypatch):
    database, conv, space, user, admin = quality
    monkeypatch.setattr(settings, "redis_memory_alert_user_ids", admin + "," + user)
    async with database.tx() as tx:
        await REAL_REVIEW_NOTIFY(tx, "synthetic-daily", 4)
        await REAL_REVIEW_NOTIFY(tx, "synthetic-daily", 4)
    rows = await database.query_raw(
        "SELECT user_id,title,body FROM notification_events WHERE dedupe_key='quality:synthetic-daily:review_pending'"
    )
    assert [r["user_id"] for r in rows] == [admin]
    assert rows[0]["title"] == "质量抽检有待评审样本" and "4 条" in rows[0]["body"]


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
        "outputs_json": {"generations": [[{"text": "合成第一句||第二句[EMO:中性/20]"}]]},
    }
    found = reply_messages([trace], "合成第一句")
    assert found == [
        {"role": "system", "content": "人设"},
        {"role": "user", "content": "你好"},
    ]
    assert reply_messages([trace, trace], "合成第一句") is None
    assert reply_messages([trace], "别的回复") is None
    assert (
        reply_messages(
            [{}, {"inputs_json": {}}, {**trace, "outputs_json": {"generations": [{}]}}],
            "合成第一句",
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
            reply_messages([{**trace, "inputs_json": {"messages": messages}}], "合成第一句")
            is None
        )


def test_historical_prompt_matches_delivered_reply_after_emoji_and_bubble_join():
    """2026-11-20: both first confirmed cases were stored without their input
    because delivery appends an emoji and joins split bubbles with a space."""
    from app.services.quality.recorded import reply_messages

    def call(content, text):
        return {
            "inputs_json": {"messages": [[{"kwargs": {"type": "human", "content": content}}]]},
            "outputs_json": {"generations": [[{"text": text}]]},
        }

    main = call("主回复", "啊 会不会很着急呀||我也帮不上忙\n[EMO:中性/30]")
    assert reply_messages([main], "啊 会不会很着急呀👌") == [{"role": "user", "content": "主回复"}]
    assert reply_messages([main], "啊 会不会很着急呀 我也帮不上忙")
    assert reply_messages([main], "会不会很开心呀") is None
    assert reply_messages([main], "👌") is None
    # Short bubbles are too common across helper outputs to identify a call.
    assert reply_messages([main], "我也") is None

    # A topic-continuation line comes first and has its own call; the longest
    # bubble identifies the main reply instead.
    bridge = call("承接", "对了 上次说的快递")
    assert reply_messages([bridge, main], ["对了 上次说的快递", "啊 会不会很着急呀"]) == [
        {"role": "user", "content": "主回复"}
    ]
    # A later bubble from another reply fragment means this call did not produce
    # the reviewed reply; unknown is better than replaying the wrong input.
    assert reply_messages([main], ["啊 会不会很着急呀", "原来是拼多多大老远寄过去呀"]) is None
    long_history = call("x", "啊 会不会很着急呀")
    long_history["inputs_json"]["messages"][0] *= 103
    assert len(reply_messages([long_history], "啊 会不会很着急呀")) == 103
    # Ambiguity counts calls whose input cannot be parsed, so a helper whose
    # output repeats the text is never taken as the single match.
    image = {**main, "inputs_json": {"messages": [[{"kwargs": {"type": "human", "content": [{"type": "image"}]}}]]}}
    helper = call("分类", "回复是：啊 会不会很着急呀")
    assert reply_messages([image, helper], "啊 会不会很着急呀") is None


async def test_confirmed_case_keeps_large_reply_call_and_backfills_old_cases(quality, monkeypatch):
    from app.services.quality import review

    database, conv, space, user, admin = quality
    job = await running()
    [sid] = await sample_rows(database, job, conv, user, n=1)
    trace = "trace-" + uuid4().hex
    reply_id, reply_at = (await database.query_raw(
        "SELECT r.id,r.created_at FROM quality_samples s JOIN messages r ON r.id=s.reply_id WHERE s.id=$1", sid))[0].values()
    await database.execute_raw("UPDATE quality_samples SET trace_id=$2 WHERE id=$1", sid, trace)
    await database.execute_raw("UPDATE messages SET content='合成回复第一句🙂' WHERE id=$1", reply_id)
    main = _main_prompt(("你的身份", "你叫小伴"), ("核心规则", "填充" * 10000))  # ~60KB
    calls = [(main, "合成回复第一句||第二句")] + [
        ({"messages": [[{"kwargs": {"type": "human", "content": f"后处理{i}"}}]]}, "{}")
        for i in range(4)  # four later small calls push it out of the latest three
    ]
    try:
        await database.execute_raw(
            """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at)
            VALUES($1,$1,'chat_request','chain','success',$2::timestamp - interval '20 seconds')""",
            trace, str(reply_at),
        )
        for i, (inputs, text) in enumerate(calls):
            await database.execute_raw(
                """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at,inputs_json,outputs_json)
                VALUES($1,$2,'ChatOpenAI','llm','success',$3::timestamp - make_interval(secs=>$4),$5::jsonb,$6::jsonb)""",
                f"{trace}-{i}", trace, str(reply_at), 10 - i, json.dumps(inputs, ensure_ascii=False),
                json.dumps({"generations": [[{"text": text}]]}, ensure_ascii=False),
            )
        detail = await sampling.context(sid, replay_input=True)
        assert detail["recorded_messages"][1] == {"role": "user", "content": "你在干嘛"}

        await review.save(sid, admin, 0, "confirmed", "hallucination", "合成回复不应捏造事实")
        stored = (await database.query_raw("SELECT case_data FROM quality_cases WHERE sample_id=$1", sid))[0]
        assert stored["case_data"]["recorded_messages"] == detail["recorded_messages"]

        # Cases saved before this fix carry null; the next corpus load resolves them once.
        await database.execute_raw(
            "UPDATE quality_cases SET case_data=jsonb_set(case_data,'{recorded_messages}','null') WHERE sample_id=$1", sid)
        monkeypatch.setattr(regression, "db", database)
        monkeypatch.setattr(sampling, "db", database)
        human = lambda cases: next(c for c in cases if c["suite"] == "human_confirmed")
        # The quick prompt gate stays read-only.
        assert human(await regression.corpus({"kind": "prompt"}))["recorded_messages"] is None
        assert human(await regression.corpus({"kind": "nightly"}))["recorded_messages"] == detail["recorded_messages"]
        stored = (await database.query_raw("SELECT case_data FROM quality_cases WHERE sample_id=$1", sid))[0]
        assert stored["case_data"]["recorded_messages"] == detail["recorded_messages"]

        # Unresolvable input is marked once; a lookup failure never fails the manifest.
        await database.execute_raw(
            "UPDATE quality_cases SET case_data=jsonb_set(case_data,'{recorded_messages}','null') WHERE sample_id=$1", sid)
        await database.execute_raw("DELETE FROM trace_runs WHERE trace_id=$1 AND run_id<>$1", trace)
        assert human(await regression.corpus({"kind": "nightly"}))["recorded_messages"] is None
        stored = (await database.query_raw("SELECT case_data FROM quality_cases WHERE sample_id=$1", sid))[0]
        assert stored["case_data"]["recorded_input_unresolved"] is True
        lookups = AsyncMock(side_effect=AssertionError("marked cases are not looked up again"))
        monkeypatch.setattr(sampling, "context", lookups)
        assert human(await regression.corpus({"kind": "nightly"}))["recorded_messages"] is None
        await database.execute_raw(
            "UPDATE quality_cases SET case_data=case_data-'recorded_input_unresolved' WHERE sample_id=$1", sid)
        monkeypatch.setattr(sampling, "context", AsyncMock(side_effect=ConnectionError))
        assert human(await regression.corpus({"kind": "nightly"}))["recorded_messages"] is None
    finally:
        await database.execute_raw("DELETE FROM quality_cases WHERE sample_id=$1", sid)
        await database.execute_raw("DELETE FROM trace_runs WHERE trace_id=$1 AND run_id<>$1", trace)
        await database.execute_raw("DELETE FROM trace_runs WHERE run_id=$1", trace)


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


def _regression_case(case_id):
    return dict(
        id=case_id,
        suite="test",
        critical=False,
        history=[],
        route="main",
        now="2026-10-29T04:00:00Z",
        turns=[dict(message="你好", criterion="礼貌回应")],
    )


async def _run_regression(database, cases, concurrency=1):
    snap = {
        "prompts": [{"key": "test", "content": "fixture", "is_enabled": True}],
        "route": ["ark", "model"],
    }
    ident = await jobs.enqueue("routing", uuid4().hex, "v1", snap)
    job = await jobs.claim(job_id=ident)
    await regression.run(job, cases=cases, concurrency=concurrency)
    return (
        await database.query_raw(
            "SELECT status,report FROM quality_jobs WHERE id=$1", ident
        )
    )[0]


class _ControlBlip(RuntimeError):
    transient = True  # mirrors evals.quality.nightly.ControlRequestError


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectTimeout("synthetic"),
        ValueError("evaluation_response_incomplete"),
        _ControlBlip("http_error"),
    ],
    ids=["network", "truncated_output", "control_5xx"],
)
async def test_regression_retries_transient_scenario_errors(
    quality, monkeypatch, error
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    monkeypatch.setattr(regression, "RETRY_DELAYS_SECONDS", (0, 0))
    calls = []

    async def flaky(job, current, *args, **kwargs):
        calls.append(current["id"])
        if current["id"] == "b" and calls.count("b") == 1:
            raise error
        return {"judge": {"pass": True}, "reply": "你好"}

    monkeypatch.setattr(regression.replay, "replay_turn", flaky)
    result = await _run_regression(database, [_regression_case("a"), _regression_case("b")])
    report = result["report"]
    assert calls == ["a", "b", "b"]
    assert report["evaluated"] == report["expected"] == 2
    assert report["errored_scenarios"] == []
    assert report["gate_status"] == "baseline_recorded"
    assert "error_type" not in report


async def test_regression_persistent_scenario_error_does_not_abort_run(
    quality, monkeypatch
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    monkeypatch.setattr(regression, "RETRY_DELAYS_SECONDS", (0, 0))
    calls = []

    async def broken(job, current, *args, **kwargs):
        calls.append(current["id"])
        if current["id"] == "a":
            # Unknown validation text must never reach the report verbatim.
            raise ValueError("模型原文 should not leak")
        if current["id"] == "b":
            raise httpx.ConnectTimeout("synthetic")
        return {"judge": {"pass": True}, "reply": "你好"}

    monkeypatch.setattr(regression.replay, "replay_turn", broken)
    cases = [_regression_case(c) for c in ("a", "b", "c")]
    result = await _run_regression(database, cases)
    report = result["report"]
    # Non-transient: one attempt. Transient: initial try plus both retries.
    assert calls == ["a", "b", "b", "b", "c"]
    assert report["evaluated"] == report["expected"] == 3
    assert report["errored_scenarios"] == [
        dict(case_id="a", repetition=0, error="ValueError", attempts=1),
        dict(case_id="b", repetition=0, error="ConnectTimeout", attempts=3),
    ]
    assert report["error_type"] == "ValueError" and report["error_case_id"] == "a"
    assert report["gate_status"] == "incomplete"
    assert result["status"] == "failed"
    assert "模型原文" not in json.dumps(report, ensure_ascii=False)


async def test_regression_boundary_history_usage_and_transcript_scoring(
    quality, monkeypatch
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    seen = []

    async def scripted(job, current, *args, **kwargs):
        seen.append(current)
        return {
            "judge": {"pass": True},
            "reply": "我在呢",
            "calls": [dict(route="chat.completions", input=120, output=8, cost_cny=0.01),
                      dict(input=300, web_search_calls=1)],
        }

    monkeypatch.setattr(regression.replay, "replay_turn", scripted)
    case = dict(
        _regression_case("boundary"),
        route="boundary",
        boundary={"turns": [{"patience": 40}, {"patience": 20}]},
        turns=[dict(message="你好", criterion="礼貌"),
               dict(message="再见", criterion="礼貌", now="2026-10-30T04:00:00Z")],
        history_rows=[dict(role="user", content="早", createdAt="2026-10-28T04:00:00+00:00")],
        transcript_scorer="long_companion_sim",
    )
    report = (await _run_regression(database, [case]))["report"]
    assert [c["boundary_turn"] for c in seen] == [{"patience": 40}, {"patience": 20}]
    assert all("turns" not in c and "boundary" not in c for c in seen)
    assert seen[1]["now"] == "2026-10-30T04:00:00Z"
    assert seen[1]["history_rows"][0]["content"] == "早" and len(seen[1]["history_rows"]) == 3
    usage = report["route_usage"]["chat.completions"]
    assert usage["calls"] == 4 and usage["input"] == 840 and usage["max_input"] == 300
    assert usage["search_calls"] == 2
    assert report["evaluated"] == 1 and report["errored_scenarios"] == []


async def test_regression_infrastructure_error_outside_scenarios_fails_run(
    quality, monkeypatch
):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    monkeypatch.setattr(
        regression.replay, "replay_turn",
        AsyncMock(return_value={"judge": {"pass": True}, "reply": "你好"}),
    )
    monkeypatch.setattr(regression.LocalSink, "result", AsyncMock(side_effect=RuntimeError("sink")))
    result = await _run_regression(database, [_regression_case("a")])
    assert result["status"] == "failed"
    assert result["report"]["error_type"] == "RuntimeError"
    assert result["report"]["error_case_id"] == "a"
    assert result["report"]["gate_status"] == "incomplete"


async def test_regression_runs_scenarios_in_parallel_and_records_all(quality, monkeypatch):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    running = peak = 0

    async def slow(job, current, *args, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.2)  # long enough for every worker's lease renewal to land
        running -= 1
        return {"judge": {"pass": True}, "reply": "你好"}

    monkeypatch.setattr(regression.replay, "replay_turn", slow)
    cases = [_regression_case(c) for c in "abcdefgh"]
    result = await _run_regression(database, cases, concurrency=4)
    assert peak == 4
    report = result["report"]
    assert report["evaluated"] == report["expected"] == 8
    assert report["expected_by_group"] == {"test|other": 8}
    assert report["gate_status"] == "baseline_recorded" and result["status"] == "completed"


async def test_regression_parallel_budget_exhaustion_stops_every_scenario(quality, monkeypatch):
    database, *_ = quality
    monkeypatch.setattr(regression, "db", database)
    started = []

    async def spend(job, current, *args, **kwargs):
        started.append(current["id"])
        if current["id"] == "b":
            raise budget.BudgetExhausted("run")
        await asyncio.sleep(0.05)
        return {"judge": {"pass": True}, "reply": "你好"}

    monkeypatch.setattr(regression.replay, "replay_turn", spend)
    cases = [_regression_case(c) for c in "abcdefgh"]
    result = await _run_regression(database, cases, concurrency=2)
    assert result["status"] == "budget_exhausted"
    assert result["report"]["gate_status"] == "incomplete"
    assert started == ["a", "b"]  # "a" was cancelled mid-run, nothing else started
    assert result["report"]["evaluated"] == 0


def test_parallel_failure_priority_lease_then_budget_then_config():
    """When several scenarios fail together, the outcome follows the sequential rules:
    a lost lease is never reported as a mere failure."""
    other, lease = RuntimeError("sink"), jobs.LostLease("gone")
    spent, missing = budget.BudgetExhausted("run"), regression.transport.NotConfigured()
    group = BaseExceptionGroup("x", [other, ExceptionGroup("y", [missing, spent]), lease])
    assert regression._primary_failure(group) is lease
    assert regression._primary_failure(BaseExceptionGroup("x", [other, missing, spent])) is spent
    assert regression._primary_failure(BaseExceptionGroup("x", [other, missing])) is missing
    assert regression._primary_failure(BaseExceptionGroup("x", [other])) is other


@pytest.mark.parametrize(
    "exc, transient",
    [
        (httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://t"),
                               response=httpx.Response(503)), True),
        (httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://t"),
                               response=httpx.Response(429)), True),
        (httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://t"),
                               response=httpx.Response(400)), False),
        (asyncio.TimeoutError(), True),
        (ValueError("evaluation_usage_out_of_bounds"), False),
        (KeyError("judge"), False),
        # A malformed judge verdict inside the renderer clears on a fresh attempt.
        (regression.replay.RendererError({"error_type": "ValueError", "code": "invalid_grounding_judgement"}), True),
        (regression.replay.RendererError({"error_type": "ValueError", "code": "memory_extraction_incomplete"}), False),
        (regression.replay.RendererError({"error_type": "ValueError", "code": "Bad: 用户原文"}), False),
        (regression.replay.RendererError({"error_type": "RuntimeError", "frames": []}), False),
    ],
)
def test_transient_classification(exc, transient):
    assert regression.is_transient(exc) is transient


def test_error_codes_never_carry_free_text():
    control = RuntimeError("http_error")
    control.code = "transport_error"
    assert regression.error_code(control) == "transport_error"
    assert regression.error_code(ValueError("missing_judge_categories")) == "missing_judge_categories"
    assert regression.error_code(ValueError("Bad Thing: 用户原文")) == "ValueError"
    assert regression.error_code(KeyError("judge")) == "KeyError"


def _main_prompt(*sections, extra_system=()):
    system = "## 回复要求\n像真人一样聊天\n" + "".join(f"## {t}\n{b}\n" for t, b in sections)
    return {"messages": [[
        {"kwargs": {"type": "system", "content": system}},
        {"kwargs": {"type": "human", "content": "你在干嘛"}},
        *({"kwargs": {"type": "system", "content": c}} for c in extra_system),
    ]]}


def test_persona_sections_keep_only_grounding_sections_whole_and_bounded():
    inputs = _main_prompt(
        ("你的身份", "你叫小伴，在杭州做陶艺"),
        ("反幻觉硬约束", "不要编造"),
        ("你记得的事情", "x" * (sampling.SECTION_BYTES + 1)),
        extra_system=["## 你的作息\n15:00 开始整理日落数据\n## 情绪标记\n输出 EMO"],
    )
    assert sampling.persona_sections(inputs) == {
        "你的身份": "你叫小伴，在杭州做陶艺",
        "你的作息": "15:00 开始整理日落数据",
    }
    # The per-turn sections follow the whole history, often past 100 messages.
    long = _main_prompt(("你的身份", "你叫小伴"))
    long["messages"][0][1:1] = [{"kwargs": {"type": "human", "content": "嗯"}}] * 150
    long["messages"][0].append({"kwargs": {"type": "system", "content": "## 你的作息\n15:00 上班"}})
    assert sampling.persona_sections(long) == {"你的身份": "你叫小伴", "你的作息": "15:00 上班"}
    # Whole sections only, within the aggregate bound.
    big = _main_prompt(*[(t, "y" * 4000) for t in sampling.PERSONA_SECTIONS])
    kept = sampling.persona_sections(big)
    assert sum(len(v.encode()) for v in kept.values()) <= sampling.PERSONA_BYTES
    assert all(v == "y" * 4000 for v in kept.values()) and len(kept) == 3
    for bad in (None, {}, {"messages": "x"}, {"messages": [[None, {"kwargs": {"type": "system", "content": 3}}]]}):
        assert sampling.persona_sections(bad) == {}


def test_bounded_context_drops_persona_last_and_marks_it_missing():
    value = {
        "history": [{"role": "user", "content": "current"}],
        "prompt_evidence": [],
        "persona_evidence": {"你的身份": "z" * 3000},
        "persona_evidence_status": "available",
        "reply": "hi",
    }
    kept = daily.bounded_context(value, 4000)
    assert kept["persona_evidence"] == value["persona_evidence"]
    assert kept["persona_evidence_status"] == "available"
    dropped = daily.bounded_context(value, 1000)
    assert dropped["persona_evidence"] == {} and dropped["persona_evidence_status"] == "not_collected"
    assert dropped["truncated"] and value["persona_evidence"]


async def test_sample_context_carries_main_prompt_persona_even_when_the_prompt_is_large(quality):
    database, conv, space, user, admin = quality
    job = await running()
    [sid] = await sample_rows(database, job, conv, user, n=1)
    trace = "trace-" + uuid4().hex
    await database.execute_raw("UPDATE quality_samples SET trace_id=$2 WHERE id=$1", sid, trace)
    reply_at = (await database.query_raw(
        "SELECT r.created_at FROM quality_samples s JOIN messages r ON r.id=s.reply_id WHERE s.id=$1", sid))[0]["created_at"]
    padding = "填充" * 20000  # ~120KB: above the 40KB prompt-evidence bound
    try:
        await database.execute_raw(
            """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at)
            VALUES($1,$1,'chat_request','chain','success',$2::timestamp - interval '20 seconds')""",
            trace, str(reply_at),
        )
        for i, inputs in enumerate([
            _main_prompt(("你的身份", "你叫小伴"), ("你的作息", "15:00 整理日落数据"), ("核心规则", padding)),
            {"messages": [[{"kwargs": {"type": "human", "content": "判定"}}]]},
        ]):
            await database.execute_raw(
                """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at,inputs_json,outputs_json)
                VALUES($1,$2,'ChatOpenAI','llm','success',$3::timestamp - make_interval(secs=>$4),$5::jsonb,'{}'::jsonb)""",
                f"{trace}-{i}", trace, str(reply_at), 10 - i, json.dumps(inputs, ensure_ascii=False),
            )
        detail = await sampling.context(sid)
        assert detail["persona_evidence"] == {"你的身份": "你叫小伴", "你的作息": "15:00 整理日落数据"}
        assert detail["persona_evidence_status"] == "available"
        assert all(padding not in json.dumps(e, ensure_ascii=False) for e in detail["prompt_evidence"])
        await database.execute_raw("UPDATE quality_samples SET trace_id=NULL WHERE id=$1", sid)
        missing = await sampling.context(sid)
        assert missing["persona_evidence"] == {} and missing["persona_evidence_status"] == "not_collected"
    finally:
        await database.execute_raw("DELETE FROM trace_runs WHERE trace_id=$1 AND run_id<>$1", trace)
        await database.execute_raw("DELETE FROM trace_runs WHERE run_id=$1", trace)
