"""Real PostgreSQL daily-review pipeline; only models/embeddings are synthetic."""

import asyncio
from datetime import datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
from zoneinfo import ZoneInfo

from prisma import Json
import pytest

from app.services.schedule_domain import (
    schedule,
    review_inputs as inputs,
    review_write as writer,
)
from app.services.memory import evidence_read, input_evidence
from app.services.memory.lifecycle import capacity
from app.services.memory.storage import repo
from app.services.workspace import workspaces
from app.services.llm import models
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_input_write_api_postgres import VECTOR, reject_write

NOW = datetime(2026, 10, 13, 4, tzinfo=ZoneInfo("Asia/Shanghai"))
END = NOW.replace(hour=0)
START = END - timedelta(days=1)


@pytest.fixture
async def daily(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    monkeypatch.setattr(schedule, "db", db)
    monkeypatch.setattr(schedule, "_local_now", lambda: NOW)
    workspace = await db.chatworkspace.find_unique(where={"id": wid})
    monkeypatch.setattr(
        workspaces, "get_active_workspace", AsyncMock(return_value=workspace)
    )
    row = await db.aidailyschedule.create(
        data={
            "agentId": agents[0],
            "date": START.replace(tzinfo=None),
            "scheduleData": Json(
                [{"start": "09:00", "end": "10:00", "event": "整理旧票根"}]
            ),
        }
    )
    await db.scheduleadjustlog.create(
        data={
            "agentId": agents[0],
            "adjustType": "event",
            "reason": "合成调整",
            "createdAt": START + timedelta(hours=2),
        }
    )
    await db.proactivechatlog.create(
        data={
            "agentId": agents[0],
            "userId": uid,
            "workspaceId": wid,
            "message": "合成生成话术",
            "createdAt": START + timedelta(hours=3),
        }
    )
    await db.proactivechatlog.create(
        data={
            "agentId": agents[1],
            "userId": uid,
            "workspaceId": spaces[1],
            "message": "PRIVATE_OTHER_SCOPE",
            "createdAt": START + timedelta(hours=3),
        }
    )
    await db.proactivechatlog.create(
        data={
            "agentId": agents[0],
            "userId": uid,
            "workspaceId": wid,
            "message": "PRIVATE_TODAY",
            "createdAt": END,
        }
    )
    game = await db.gamesession.create(
        data={
            "userId": uid,
            "agentId": agents[0],
            "workspaceId": wid,
            "roomId": uuid4().hex,
            "userPlayer": Json({}),
            "aiPlayer": Json({}),
            "status": "settled",
            "result": Json({"user_outcome": "win", "game_title": "合成棋"}),
            "createdAt": START + timedelta(hours=4),
            "endedAt": START + timedelta(hours=5),
        }
    )

    async def prompt(key):
        return {
            "schedule.daily_summary": "{name}\n{schedule_text}\n{adjustments_text}\n{proactive_text}\n{chat_summary_text}\n{games_text}",
            "schedule.daily_summary_memories": "拆分：{summary_text}",
            "persona.grounding_context": "设定：{facts}",
        }[key]

    monkeypatch.setattr(schedule, "get_prompt_text", prompt)
    monkeypatch.setattr(
        schedule,
        "get_utility_model",
        lambda: SimpleNamespace(model_name="synthetic-model"),
    )
    monkeypatch.setattr(
        schedule, "_ground_schedule", AsyncMock(side_effect=lambda agent, slots: slots)
    )
    text = AsyncMock(return_value="整理了旧票根，有点开心")
    classification = AsyncMock(
        return_value={
            "memories": [
                {"type": "生活", "content": "整理了旧票根", "score": 99},
                {"type": "情绪", "content": "有点开心", "score": 55},
            ]
        }
    )
    monkeypatch.setattr(models, "invoke_text", text)
    monkeypatch.setattr(schedule, "invoke_json", classification)
    second = [0.0, 1.0] + [0.0] * 1022
    embedding = AsyncMock(
        side_effect=lambda content: VECTOR if "票根" in content else second
    )
    monkeypatch.setattr(writer, "generate_embedding", embedding)
    monkeypatch.setattr(writer, "after_memory_changelog", AsyncMock())

    async def run():
        return await schedule.review_daily_schedule(agents[0], uid)

    f = SimpleNamespace(
        db=db,
        uid=uid,
        wid=wid,
        agents=agents,
        spaces=spaces,
        convs=convs,
        row=row,
        game=game,
        text=text,
        classification=classification,
        embedding=embedding,
        run=run,
    )
    yield f
    await db.gamesession.delete_many(where={"userId": {"in": users}})
    await db.proactivechatlog.delete_many(where={"userId": {"in": users}})
    await db.scheduleadjustlog.delete_many(where={"agentId": {"in": agents}})
    await db.aidailyschedule.delete_many(where={"agentId": {"in": agents}})


async def counts(f):
    return (
        await f.db.query_raw(
            """SELECT
        (SELECT count(*) FROM memories_ai WHERE workspace_id=$1 AND provenance='daily_summary') AS memories,
        (SELECT count(*) FROM memory_input_origins WHERE workspace_id=$1 AND source_status='daily_summary') AS origins,
        (SELECT count(*) FROM memory_evidence_links WHERE workspace_id=$1 AND extractor_version='daily-review-v1') AS links""",
            f.wid,
        )
    )[0]


async def review_row(f):
    return (
        await f.db.query_raw(
            "SELECT * FROM memory_daily_reviews WHERE workspace_id=$1", f.wid
        )
    )[0]


async def test_full_daily_review_freezes_actual_inputs_and_links_atomic_outputs(daily):
    f = daily
    ids = await f.run()
    assert len(ids) == 2
    assert await counts(f) == {"memories": 2, "origins": 1, "links": 2}
    receipt = (
        await f.db.query_raw(
            "SELECT payload_text FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
    )[0]
    payload = json.loads(receipt["payload_text"])["input"]
    assert (
        payload["model"] == "synthetic-model" and payload["local_date"] == "2026-10-12"
    )
    assert payload["calls"][0]["prompt"] == f.text.await_args.args[1]
    assert payload["calls"][1]["output"] == f.classification.return_value
    assert "PRIVATE_" not in receipt["payload_text"]
    assert "计划（不代表活动已实际完成）" in payload["calls"][0]["prompt"]
    assert "不证明现实经历或用户已收到" in payload["calls"][0]["prompt"]
    assert "用户赢1局" in payload["calls"][0]["prompt"]
    assert "《合成棋》" in payload["calls"][0]["prompt"]
    assert "未采集聊天回顾" in payload["calls"][0]["prompt"]
    for mid in ids:
        row = await f.db.aimemory.find_unique(where={"id": mid})
        assert (
            row.level == 3
            and row.importance == 0.49
            and row.provenance == "daily_summary"
        )
        data = await evidence_read.memory_evidence_detail(
            user_id=f.uid, workspace_id=f.wid, side="ai", memory_id=mid
        )
        item = data["items"][0]
        assert data["state"] == "linked" and item["source_status"] == "daily_summary"
        assert item["daily_summary"] == {**payload["summary"]}
        assert item["input"]["kind"] == "daily_summary" and item["current_content"]
        assert "合成生成话术" not in json.dumps(data, ensure_ascii=False, default=str)
    logs = await f.db.memorychangelog.find_many(
        where={"workspaceId": f.wid, "memoryId": {"in": ids}}
    )
    assert {r.operation for r in logs} == {"insert"}
    assert {r.newValue for r in logs} == {"整理了旧票根", "有点开心"}
    assert (await review_row(f))["status"] == "completed"
    assert await f.run() == [] and f.text.await_count == 1


async def test_concurrent_natural_invocations_claim_exactly_once(daily):
    f = daily
    results = await asyncio.gather(*(f.run() for _ in range(6)))
    assert sum(bool(r) for r in results) == 1 and f.text.await_count == 1
    assert await counts(f) == {"memories": 2, "origins": 1, "links": 2}


@pytest.mark.parametrize(
    "step", ["memory", "vector", "origin", "link", "audit", "completion"]
)
async def test_failure_at_each_commit_step_leaves_no_partial_summary(daily, step):
    f = daily
    table, predicate, operation = {
        "memory": ("memories_ai", "NEW.provenance='daily_summary'", "INSERT"),
        "vector": ("memory_embeddings", "true", "INSERT"),
        "origin": (
            "memory_input_origins",
            "NEW.source_status='daily_summary'",
            "INSERT",
        ),
        "link": (
            "memory_evidence_links",
            "NEW.extractor_version='daily-review-v1'",
            "INSERT",
        ),
        "audit": ("memory_changelogs", "NEW.operation='insert'", "INSERT"),
        "completion": ("memory_daily_reviews", "NEW.status='completed'", "UPDATE"),
    }[step]
    async with reject_write(f.db, table, predicate, operation=operation):
        with pytest.raises(Exception, match="synthetic input failure"):
            await f.run()
    assert await counts(f) == {"memories": 0, "origins": 0, "links": 0}
    assert (await review_row(f))["status"] == "failed"
    assert await f.run() == []


@pytest.mark.parametrize(
    "change",
    [
        "schedule",
        "adjustment",
        "proactive",
        "game",
        "agent",
        "generation",
        "takeover",
        "archive",
    ],
)
async def test_input_and_scope_changes_during_generation_fail_closed(daily, change):
    f = daily

    async def classify(*args, **kwargs):
        if change == "schedule":
            await f.db.aidailyschedule.update(
                where={"id": f.row.id}, data={"scheduleData": Json([])}
            )
        elif change == "adjustment":
            await f.db.scheduleadjustlog.update_many(
                where={"agentId": f.agents[0]}, data={"reason": "changed"}
            )
        elif change == "proactive":
            await f.db.proactivechatlog.update_many(
                where={"workspaceId": f.wid}, data={"message": "changed"}
            )
        elif change == "game":
            await f.db.gamesession.delete(where={"id": f.game.id})
        elif change == "agent":
            await f.db.aiagent.update(where={"id": f.agents[0]}, data={"city": "乙城"})
        elif change == "generation":
            await f.db.execute_raw(
                "UPDATE chat_workspaces SET execution_generation='changed-generation' WHERE id=$1",
                f.wid,
            )
        elif change == "takeover":
            await f.db.execute_raw(
                "INSERT INTO human_takeovers(id,user_id,status,updated_at) VALUES($1,$2,'active',NOW())",
                uuid4().hex,
                f.uid,
            )
        else:
            await f.db.chatworkspace.update(
                where={"id": f.wid}, data={"status": "archived"}
            )
        return {"memories": [{"content": "整理了旧票根", "score": 70}]}

    f.classification.side_effect = classify
    with pytest.raises(Exception, match="changed|takeover|inactive"):
        await f.run()
    assert await counts(f) == {"memories": 0, "origins": 0, "links": 0}
    assert (await review_row(f))["status"] == "failed"


@pytest.mark.parametrize("stage", ["summary", "classification", "embedding"])
async def test_provider_failures_are_not_empty_success(daily, stage):
    f = daily
    getattr(
        f,
        {
            "summary": "text",
            "classification": "classification",
            "embedding": "embedding",
        }[stage],
    ).side_effect = RuntimeError("synthetic provider failure")
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await f.run()
    assert (await review_row(f))["status"] == "failed" and (await counts(f))[
        "memories"
    ] == 0


async def test_empty_schedule_records_zero_input_without_model_calls(daily):
    f = daily
    await f.db.aidailyschedule.delete(where={"id": f.row.id})
    assert await f.run() == []
    f.text.assert_not_awaited()
    assert (await review_row(f))["status"] == "completed"
    assert await counts(f) == {"memories": 0, "origins": 1, "links": 0}


async def test_snapshot_survives_result_deletion_without_recreating(daily):
    f = daily
    ids = await f.run()
    await f.db.aimemory.delete_many(where={"id": {"in": ids}})
    await f.db.execute_raw(
        "DELETE FROM memory_daily_reviews WHERE workspace_id=$1", f.wid
    )
    assert await f.run() == []
    assert await counts(f) == {"memories": 0, "origins": 1, "links": 0}


async def test_existing_fact_is_not_updated_or_given_generated_evidence(daily):
    f = daily
    old = await repo.create(
        source="ai",
        userId=f.uid,
        workspaceId=f.wid,
        content="整理了旧票根",
        level=1,
        importance=0.95,
        provenance="profile_seed",
    )
    await f.db.execute_raw(
        "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
        old.id,
        "[" + ",".join(map(str, VECTOR)) + "]",
    )
    assert len(await f.run()) == 1
    assert (await f.db.aimemory.find_unique(where={"id": old.id})).importance == 0.95
    assert not await f.db.query_raw(
        "SELECT id FROM memory_evidence_links WHERE memory_id=$1", old.id
    )


async def test_capacity_counts_archived_originals(daily, monkeypatch):
    f = daily
    monkeypatch.setattr(capacity, "DAILY_SUMMARY_ROW_BUDGET", 1)
    await repo.create(
        source="ai",
        userId=f.uid,
        workspaceId=f.wid,
        content="旧总结",
        level=3,
        provenance="daily_summary",
        isArchived=True,
    )
    assert await f.run() == []
    f.text.assert_not_awaited()
    assert (await review_row(f))["status"] == "capacity_skipped"


async def test_postcommit_hook_failure_does_not_turn_committed_review_into_failure(
    daily, monkeypatch
):
    monkeypatch.setattr(
        writer, "after_memory_changelog", AsyncMock(side_effect=RuntimeError("hook"))
    )
    assert (
        len(await daily.run()) == 2
        and (await review_row(daily))["status"] == "completed"
    )


async def test_sample_and_game_status_are_explicit(daily):
    f = daily
    for i in range(6):
        await f.db.proactivechatlog.create(
            data={
                "userId": f.uid,
                "agentId": f.agents[0],
                "workspaceId": f.wid,
                "message": f"生成{i}",
                "createdAt": START + timedelta(hours=6, minutes=i),
            }
        )
    await f.db.gamesession.update(where={"id": f.game.id}, data={"status": "started"})
    await f.run()
    assert "未结算1局" in f.text.await_args.args[1]
    payload = json.loads(
        (
            await f.db.query_raw(
                "SELECT payload_text FROM memory_input_origins WHERE workspace_id=$1",
                f.wid,
            )
        )[0]["payload_text"]
    )["input"]
    assert (
        payload["summary"]["proactive_truncated"] is True
        and payload["summary"]["proactive_messages"] == 5
    )


async def test_cross_owner_or_agent_cannot_supply_inputs(daily):
    f = daily
    with pytest.raises(Exception, match="scope_changed"):
        await inputs.collect_inputs(
            f.db,
            user_id=f.uid,
            workspace_id=f.wid,
            agent_id=f.agents[1],
            start=START,
            end=END,
        )


async def test_summary_cannot_write_stable_persona_even_mislabeled_as_life(daily):
    f = daily
    f.classification.return_value = {
        "memories": [
            {"type": "身份", "content": "我的职业是医生", "score": 99},
            {"type": "生活", "content": "我住在甲城", "score": 99},
            {"type": "偏好", "content": "我喜欢咖啡", "score": 90},
        ]
    }
    assert await f.run() == [] and (await review_row(f))["status"] == "completed"
    assert (await counts(f))["memories"] == 0


async def test_empty_model_summary_has_recorded_input_but_no_classification(daily):
    f = daily
    f.text.return_value = " "
    assert await f.run() == []
    f.classification.assert_not_awaited()
    assert (await review_row(f))["status"] == "completed"
    assert (await counts(f))["origins"] == 1


@pytest.mark.parametrize("change", ["claim", "capacity"])
async def test_changed_claim_or_capacity_cannot_commit(daily, monkeypatch, change):
    f = daily

    async def classify(*args, **kwargs):
        if change == "claim":
            await f.db.execute_raw(
                "UPDATE memory_daily_reviews SET status='failed' WHERE workspace_id=$1",
                f.wid,
            )
        else:
            monkeypatch.setattr(capacity, "DAILY_SUMMARY_ROW_BUDGET", 0)
        return {"memories": [{"content": "整理了旧票根", "score": 70}]}

    f.classification.side_effect = classify
    with pytest.raises(Exception, match="changed"):
        await f.run()
    assert (await counts(f))["memories"] == 0


async def test_read_failure_is_failed_not_empty_source(daily, monkeypatch):
    f = daily
    monkeypatch.setattr(
        inputs,
        "collect_inputs",
        AsyncMock(side_effect=RuntimeError("source unavailable")),
    )
    with pytest.raises(RuntimeError, match="source unavailable"):
        await f.run()
    f.text.assert_not_awaited()
    assert (await review_row(f))["status"] == "failed"


async def test_daily_input_cannot_be_bound_as_user_fact(daily):
    from app.services.memory.evidence import EvidenceSource, bind_memory_evidence

    f = daily
    await f.run()
    origin = (
        await f.db.query_raw(
            "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
    )[0]["id"]
    target = await repo.create(
        userId=f.uid, workspaceId=f.wid, content="不应接受AI总结的用户事实"
    )
    version = (
        await f.db.query_raw(
            "SELECT source_version FROM memory_input_origins WHERE id=$1", origin
        )
    )[0]["source_version"]
    with pytest.raises(Exception, match="daily_review_invalid_target"):
        await bind_memory_evidence(
            memory_id=target.id,
            side="user",
            user_id=f.uid,
            workspace_id=f.wid,
            sources=(EvidenceSource("input", origin, expected_version=version),),
            extractor_version=writer.EXTRACTOR,
        )


async def test_deleted_input_details_are_unavailable_not_zero(daily):
    f = daily
    ids = await f.run()
    await f.db.execute_raw(
        "DELETE FROM memory_input_origins WHERE workspace_id=$1", f.wid
    )
    data = await evidence_read.memory_evidence_detail(
        user_id=f.uid, workspace_id=f.wid, side="ai", memory_id=ids[0]
    )
    assert data["items"][0]["availability"] == "deleted"
    assert (
        data["items"][0]["daily_summary"] is None
        and data["items"][0]["source_ref"] is None
    )
