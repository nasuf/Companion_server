"""Actual Prisma/PG transactions, Redis tickets and preflight; only LLM/embedding stubbed."""

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Json
from redis.asyncio import Redis

from app.services.chat import preflight
from app.services.memory import evidence, evidence_read, repair_queue
from app.services.memory.interaction import (
    contradiction as c,
    contradiction_state as state,
    contradiction_write as writer,
)
from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.storage import repo
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_input_write_api_postgres import VECTOR, reject_write


@pytest.fixture
async def confirming(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    for module in (state, writer, repair_queue):
        monkeypatch.setattr(module, "db", db)
    embedding = AsyncMock(return_value=VECTOR)
    monkeypatch.setattr(writer, "generate_embedding", embedding)
    monkeypatch.setattr(writer, "after_memory_changelog", AsyncMock())
    redis = Redis.from_url(os.environ["PROACTIVE_E2E_REDIS_URL"], decode_responses=True)
    monkeypatch.setattr(c, "get_redis", AsyncMock(return_value=redis))
    old = await repo.create(
        userId=uid,
        workspaceId=wid,
        content="用户住在甲城",
        level=1,
        importance=0.95,
        type="identity",
        mainCategory="身份",
        subCategory="现居地",
        provenance="user_stated",
    )
    first = await db.message.create(
        data={"conversationId": convs[0], "role": "user", "content": "我现在搬到乙城了"}
    )
    second = await db.message.create(
        data={
            "conversationId": convs[0],
            "role": "user",
            "content": "是的，我搬到乙城了",
        }
    )
    scope = dict(
        conversation_id=convs[0], user_id=uid, workspace_id=wid, agent_id=agents[0]
    )
    conflict = {
        "has_conflict": True,
        "conflicting_memory_id": old.id,
        "old_content": old.content,
        "new_info": first.content,
    }
    await c.save_pending_contradiction(
        convs[0],
        conflict,
        **{k: v for k, v in scope.items() if k != "conversation_id"},
        message_ids=[first.id],
        user_message=first.content,
    )
    pending = await c.load_pending_contradiction(convs[0])
    sources = await state.prepare_confirmation(
        pending, **scope, message_ids=[second.id], user_message=second.content
    )
    analysis = {"change_type": "变化", "updated_memory": "用户住在乙城"}

    async def apply(**overrides):
        return await writer.apply_confirmed_change(
            pending,
            overrides.pop("analysis", analysis),
            **{**scope, "confirmation_sources": sources, **overrides},
        )

    yield SimpleNamespace(
        db=db,
        uid=uid,
        wid=wid,
        agents=agents,
        users=users,
        spaces=spaces,
        convs=convs,
        old=old,
        first=first,
        second=second,
        scope=scope,
        pending=pending,
        sources=sources,
        analysis=analysis,
        apply=apply,
        embedding=embedding,
        redis=redis,
    )
    await redis.delete(c._PENDING_KEY_PREFIX + convs[0])
    await redis.aclose()


@pytest.mark.parametrize(
    "change,level,importance,archived",
    [("变化", 1, 0.9, True), ("错误", 2, 0.7, True), ("新增", 1, 0.85, False)],
)
async def test_confirmed_change_preserves_policy_and_scoped_evidence(
    confirming, change, level, importance, archived
):
    f = confirming
    a = {**f.analysis, "change_type": change}
    if change == "新增":
        a.update(
            new_memory="用户喜欢散步",
            new_memory_main_category="生活",
            new_memory_sub_category="其他",
        )
    result = await f.apply(analysis=a)
    assert result["status"] == "applied"
    new = await f.db.usermemory.find_unique(where={"id": result["memory_id"]})
    assert new.level == level and new.importance == pytest.approx(importance)
    assert new.provenance == "user_stated" and new.statementTime and new.valueUpdatedAt
    assert (
        await f.db.usermemory.find_unique(where={"id": f.old.id})
    ).isArchived == archived
    data = await evidence_read.memory_evidence_detail(
        user_id=f.uid, workspace_id=f.wid, side="user", memory_id=new.id
    )
    assert data["state"] == "linked" and len(data["items"]) == 4
    source = next(r for r in data["items"] if r["source_kind"] == "input")
    assert (
        source["input"]["kind"] == "contradiction_confirmation"
        and source["confirmation_change"] == change
    )
    parent = next(r for r in data["items"] if r["source_kind"] == "memory")
    assert parent["source_archived"] == archived
    assert {
        r["source_ref"] for r in data["items"] if r["source_kind"] == "message"
    } == {f.first.id, f.second.id}
    assert all(r["source_version"] and r["current_content"] for r in data["items"])
    assert "用户住在" not in json.dumps(data, default=str, ensure_ascii=False)
    assert await f.db.query_raw(
        "SELECT memory_id FROM memory_embeddings WHERE memory_id=$1", new.id
    )
    assert (await f.apply(analysis=a))["status"] == "already_applied"


@pytest.mark.parametrize(
    "step", ["memory", "vector", "origin", "link", "archive", "audit"]
)
async def test_each_failure_rolls_back_without_compensating_writes(confirming, step):
    f = confirming
    table, predicate, operation = {
        "memory": ("memories_user", "NEW.content='用户住在乙城'", "INSERT"),
        "vector": ("memory_embeddings", "true", "INSERT"),
        "origin": (
            "memory_input_origins",
            "NEW.source_status='contradiction_confirmation'",
            "INSERT",
        ),
        "link": (
            "memory_evidence_links",
            "NEW.extractor_version='contradiction-confirmation-v1'",
            "INSERT",
        ),
        "archive": ("memories_user", "NEW.is_archived", "UPDATE"),
        "audit": ("memory_changelogs", "NEW.operation='contradiction_new'", "INSERT"),
    }[step]
    count = await f.db.usermemory.count(where={"workspaceId": f.wid})
    async with reject_write(f.db, table, predicate, operation=operation):
        with pytest.raises(Exception, match="synthetic input failure"):
            await f.apply()
    assert await f.db.usermemory.count(where={"workspaceId": f.wid}) == count
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived
    assert (
        await f.db.query_raw(
            "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
        == []
    )
    assert await f.db.memorychangelog.count(where={"workspaceId": f.wid}) == 0
    assert (await f.apply())["status"] == "applied"


@pytest.mark.parametrize(
    "change",
    [
        "memory",
        "message",
        "deleted_message",
        "archived_memory",
        "generation",
        "conversation",
        "takeover",
    ],
)
async def test_change_during_model_or_embedding_cannot_override_newer_state(
    confirming, change
):
    f = confirming

    async def mutate(_):
        if change == "memory":
            await f.db.usermemory.update(
                where={"id": f.old.id}, data={"content": "用户住在丙城"}
            )
        elif change == "message":
            await f.db.message.update(
                where={"id": f.second.id}, data={"content": "我没搬家"}
            )
        elif change == "deleted_message":
            await f.db.message.delete(where={"id": f.first.id})
        elif change == "archived_memory":
            await f.db.usermemory.update(
                where={"id": f.old.id}, data={"isArchived": True}
            )
        elif change == "generation":
            await f.db.execute_raw(
                "UPDATE chat_workspaces SET execution_generation=md5(execution_generation) WHERE id=$1",
                f.wid,
            )
        elif change == "conversation":
            await f.db.conversation.update(
                where={"id": f.convs[0]}, data={"isDeleted": True}
            )
        else:
            await f.db.message.update(
                where={"id": f.second.id},
                data={"metadata": Json({"human_takeover": True})},
            )
        return VECTOR

    f.embedding.side_effect = mutate
    with pytest.raises((MemoryWriteError, ValueError)):
        await f.apply()
    assert (
        await f.db.query_raw(
            "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
        == []
    )
    assert (
        await f.db.usermemory.count(
            where={"workspaceId": f.wid, "content": "用户住在乙城"}
        )
        == 0
    )


async def test_concurrent_replay_and_deleted_result_never_duplicate_or_resurrect(
    confirming,
):
    f = confirming
    results = await asyncio.gather(f.apply(), f.apply())
    assert sorted(r["status"] for r in results) == ["already_applied", "applied"]
    mid = next(r["memory_id"] for r in results if r["status"] == "applied")
    await f.db.usermemory.delete(where={"id": mid})
    assert (await f.apply())["status"] == "already_applied"
    assert (
        await f.db.usermemory.count(
            where={"workspaceId": f.wid, "content": "用户住在乙城"}
        )
        == 0
    )


@pytest.mark.parametrize(
    "analysis",
    [
        {"change_type": "变化"},
        {"change_type": "错误", "updated_memory": ""},
        {"change_type": "新增"},
        {"change_type": "变化", "new_memory": "用户住在甲城"},
    ],
)
async def test_missing_or_unchanged_replacement_leaves_old_fact_intact(
    confirming, analysis
):
    f = confirming
    assert (await f.apply(analysis=analysis))["status"] == "no_change"
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived
    assert (
        await f.db.query_raw(
            "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
        == []
    )


@pytest.mark.parametrize(
    "change",
    ["user_id", "workspace_id", "agent_id", "conversation_id", "expired", "legacy"],
)
async def test_pending_is_not_scope_authority(confirming, change):
    f = confirming
    if change == "expired":
        f.pending["_confirmation"]["created_at"] = "2020-01-01T00:00:00+00:00"
    elif change == "legacy":
        f.pending.pop("_confirmation")
    with pytest.raises(MemoryWriteError):
        await f.apply(
            **({change: "foreign"} if change not in {"expired", "legacy"} else {})
        )
    f.embedding.assert_not_awaited()
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_pending_clear_cannot_remove_a_new_question(confirming):
    f = confirming
    new = json.loads(json.dumps(f.pending))
    new["_confirmation"]["id"] = "a" * 32
    key = c._PENDING_KEY_PREFIX + f.convs[0]
    await f.redis.set(key, json.dumps(new, ensure_ascii=False))
    await c.clear_pending_contradiction(f.convs[0], expected=f.pending)
    assert (await c.load_pending_contradiction(f.convs[0]))["_confirmation"][
        "id"
    ] == "a" * 32
    await c.clear_pending_contradiction(f.convs[0], expected=new)
    assert await c.load_pending_contradiction(f.convs[0]) is None


async def test_reserved_confirmation_messages_cannot_reenter_generic_extraction(
    confirming,
):
    f = confirming
    with pytest.raises(ValueError, match="scope_or_content"):
        await evidence.snapshot_message_sources(
            user_id=f.uid,
            workspace_id=f.wid,
            side="user",
            message_ids=[f.first.id],
            extraction_input=f.first.content,
        )
    with pytest.raises(ValueError, match="reserved_for_confirmation"):
        await evidence.bind_memory_evidence(
            memory_id=f.old.id,
            side="user",
            user_id=f.uid,
            workspace_id=f.wid,
            sources=(
                evidence.EvidenceSource(
                    "message", f.second.id, relation="extracted_from"
                ),
            ),
            extractor_version="test-v1",
        )


@pytest.mark.parametrize("failed", [False, True])
async def test_real_preflight_never_reports_a_failed_write_as_success(
    confirming, monkeypatch, failed
):
    f = confirming
    calls = []

    async def send(reply, *args, **kwargs):
        calls.append((reply, kwargs))
        return [{"event": "reply", "data": reply}, {"event": "done", "data": "{}"}]

    ctx = preflight.PreflightCtx(
        **f.scope,
        agent=SimpleNamespace(name="合成伴侣"),
        tracer=MagicMock(),
        short_circuit_fn=send,
        message_ids=(f.second.id,),
    )
    monkeypatch.setattr(
        preflight, "analyze_contradiction_response", AsyncMock(return_value=f.analysis)
    )
    generate = AsyncMock(return_value="好的，已经记住新的城市。")
    monkeypatch.setattr(preflight, "generate_contradiction_reply", generate)
    if failed:
        f.embedding.side_effect = RuntimeError("synthetic unavailable")
    events = [
        e async for e in preflight.resolve_pending_contradiction(f.second.content, ctx)
    ]
    assert (
        [e["event"] for e in events] == ["reply", "done"]
        and ctx.stopped
        and ctx.memory_confirmation
    )
    assert calls[0][1]["extra_metadata"]["memory_confirmation_outcome"] == (
        "failed" if failed else "applied"
    )
    if failed:
        generate.assert_not_awaited()
        assert "没能保存成功" in calls[0][0]
        assert not (
            await f.db.usermemory.find_unique(where={"id": f.old.id})
        ).isArchived
    else:
        generate.assert_awaited_once()


@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
@pytest.mark.parametrize("failed", [False, True])
async def test_real_confirmation_through_both_chat_executors(
    confirming, monkeypatch, executor, failed
):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from tests.graph_harness_support import configure_chat

    f = confirming
    io = configure_chat(monkeypatch)
    from app.services.chat import multi_intent

    def fire(coro):
        if (
            getattr(coro, "cr_frame", None)
            and coro.cr_frame.f_locals.get("self") is chat._save_replies
        ):
            try:
                coro.send(None)
            except StopIteration:
                pass
        else:
            coro.close()

    monkeypatch.setattr(chat, "_fire_background", fire)
    monkeypatch.setattr(multi_intent, "_fire_background", fire)
    monkeypatch.setattr(settings, "chat_executor", executor)
    monkeypatch.setattr(settings, "chat_graph_conversation_allowlist", f.convs[0])
    monkeypatch.setattr(chat, "db", f.db)
    monkeypatch.setattr(
        chat, "resolve_pending_contradiction", preflight.resolve_pending_contradiction
    )
    monkeypatch.setattr(
        preflight, "analyze_contradiction_response", AsyncMock(return_value=f.analysis)
    )
    monkeypatch.setattr(
        preflight,
        "generate_contradiction_reply",
        AsyncMock(return_value="新的信息已经保存了。"),
    )
    if failed:
        f.embedding.side_effect = RuntimeError("synthetic unavailable")
    agent = await f.db.aiagent.find_unique(where={"id": f.agents[0]})
    events = [
        e
        async for e in chat.stream_chat_response(
            f.convs[0],
            f.second.content,
            agent,
            f.uid,
            save_user_message=False,
            user_message_id=f.second.id,
        )
    ]
    assert [e["event"] for e in events] == ["reply", "done"]
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    saved = chat._save_replies.await_args.args[1][0]
    assert saved["memory_confirmation"] is True
    assert saved["memory_confirmation_outcome"] == ("failed" if failed else "applied")
    chat._background_post_process.assert_called_once()
    assert chat._background_post_process.call_args.kwargs["skip_memory"] is True
    assert (
        await f.db.usermemory.find_unique(where={"id": f.old.id})
    ).isArchived is not failed


@pytest.mark.parametrize(
    "bad",
    [
        "different_conversation",
        "assistant",
        "same_turn",
        "empty",
        "too_many",
        "foreign_memory",
        "changed_memory",
    ],
)
async def test_pending_and_confirmation_require_real_scoped_user_inputs(
    confirming, bad
):
    f = confirming
    conflict = {
        "has_conflict": True,
        "conflicting_memory_id": f.old.id,
        "old_content": f.old.content,
    }
    args = dict(
        conversation_id=f.convs[0],
        user_id=f.uid,
        workspace_id=f.wid,
        agent_id=f.agents[0],
        message_ids=[f.first.id],
        user_message=f.first.content,
    )
    if bad == "same_turn":
        with pytest.raises(MemoryWriteError, match="new_turn"):
            await state.prepare_confirmation(f.pending, **args)
        return
    if bad == "empty":
        args["message_ids"] = []
    elif bad == "too_many":
        args["message_ids"] = [str(i) for i in range(41)]
    elif bad == "assistant":
        await f.db.message.update(where={"id": f.first.id}, data={"role": "assistant"})
    elif bad == "different_conversation":
        await f.db.message.update(
            where={"id": f.first.id}, data={"conversationId": f.convs[1]}
        )
    elif bad == "foreign_memory":
        foreign = await repo.create(
            userId=f.users[-1], workspaceId=f.spaces[-1], content=f.old.content, level=1
        )
        conflict["conflicting_memory_id"] = foreign.id
    else:
        conflict["old_content"] = "过期内容"
    with pytest.raises(MemoryWriteError):
        await state.prepare_pending(conflict, **args)


async def test_ordinary_usage_does_not_expire_pending_but_semantic_edit_does(
    confirming,
):
    f = confirming
    await f.db.usermemory.update(
        where={"id": f.old.id}, data={"mentionCount": 7, "importance": 0.93}
    )
    result = await f.apply()
    assert result["status"] == "applied"
    assert (
        await f.db.usermemory.find_unique(where={"id": result["memory_id"]})
    ).importance == pytest.approx(0.88)


async def test_different_tickets_competing_for_same_old_fact_apply_only_once(
    confirming,
):
    f = confirming
    other = json.loads(json.dumps(f.pending))
    other["_confirmation"]["id"] = "b" * 32

    async def apply_other():
        return await writer.apply_confirmed_change(
            other, f.analysis, **f.scope, confirmation_sources=f.sources
        )

    results = await asyncio.gather(f.apply(), apply_other(), return_exceptions=True)
    assert sum(isinstance(r, dict) and r["status"] == "applied" for r in results) == 1
    assert sum(isinstance(r, MemoryWriteError) for r in results) == 1
    assert (
        await f.db.usermemory.count(
            where={"workspaceId": f.wid, "content": "用户住在乙城"}
        )
        == 1
    )


async def test_replay_does_not_depend_on_embedding_provider_or_old_row(confirming):
    f = confirming
    assert (await f.apply())["status"] == "applied"
    await f.db.usermemory.delete(where={"id": f.old.id})
    f.embedding.side_effect = RuntimeError("synthetic provider down")
    assert (await f.apply())["status"] == "already_applied"


async def test_reserved_backlog_does_not_starve_later_regular_messages(
    confirming, monkeypatch
):
    from app.services.chat import post_process as post
    from datetime import timedelta

    f = confirming
    monkeypatch.setattr(post, "db", f.db)
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = []
    for i in range(post._WATERMARK_BACKLOG_LIMIT):
        rows.append(
            await f.db.message.create(
                data={
                    "conversationId": f.convs[0],
                    "role": "user",
                    "content": "确认片段",
                    "createdAt": base + timedelta(seconds=i),
                    "metadata": Json({"memory_confirmation": True}),
                }
            )
        )
    later = await f.db.message.create(
        data={
            "conversationId": f.convs[0],
            "role": "user",
            "content": "今天喜欢吃面",
            "createdAt": base + timedelta(seconds=len(rows)),
        }
    )
    wm = base - timedelta(seconds=1)

    async def get(*args):
        return wm

    async def set_wm(*args, message_id, expected):
        nonlocal wm
        assert expected == wm and message_id is None  # completed timestamp groups
        wm = args[-1]

    monkeypatch.setattr(post, "get_watermark", get)
    monkeypatch.setattr(post, "set_watermark", set_wm)
    extract = AsyncMock(return_value=["synthetic"])
    monkeypatch.setattr(post, "process_memory_pipeline", extract)
    outcome = await post._pipeline_with_watermark(
        f.uid, [], f.convs[0], side="user", workspace_id=f.wid
    )
    assert outcome.status == "excluded" and outcome.memory_count == 0 and outcome.progress == "advanced"
    extract.assert_not_awaited()
    assert wm == rows[-1].createdAt
    outcome = await post._pipeline_with_watermark(
        f.uid, [], f.convs[0], side="user", workspace_id=f.wid
    )
    assert outcome.status == "completed" and outcome.memory_count == 1 and outcome.progress == "advanced"
    assert later.id in extract.await_args.kwargs["evidence_message_ids"]
    assert "确认片段" not in extract.await_args.kwargs["new_conversation"]


async def test_concurrent_confirmation_capture_does_not_upgrade_shared_message_locks(
    confirming, monkeypatch
):
    f = confirming
    exclude = state.exclude_messages

    async def slow_exclude(*args):
        await asyncio.sleep(0.08)
        await exclude(*args)

    monkeypatch.setattr(state, "exclude_messages", slow_exclude)

    async def capture():
        return await state.prepare_confirmation(
            f.pending,
            **f.scope,
            message_ids=[f.second.id],
            user_message=f.second.content,
        )

    results = await asyncio.gather(capture(), capture(), return_exceptions=True)
    assert all(
        isinstance(value, list) and value[0]["ref"] == f.second.id for value in results
    ), results


@pytest.mark.parametrize(
    "mode",
    [
        "replay",
        "no_change",
        "reply_unavailable",
        "cleanup_unavailable",
        "hook_unavailable",
    ],
)
async def test_confirmation_reply_matches_durable_outcome(
    confirming, monkeypatch, mode
):
    f = confirming
    if mode == "replay":
        await f.apply()
    analysis = dict(f.analysis)
    if mode == "no_change":
        analysis["updated_memory"] = ""
    monkeypatch.setattr(
        preflight, "analyze_contradiction_response", AsyncMock(return_value=analysis)
    )
    generate = AsyncMock(return_value="更正已经保存。")
    if mode == "reply_unavailable":
        generate.side_effect = RuntimeError("synthetic model outage after commit")
    if mode == "cleanup_unavailable":
        monkeypatch.setattr(
            preflight,
            "clear_pending_contradiction",
            AsyncMock(side_effect=RuntimeError("synthetic Redis outage")),
        )
    if mode == "hook_unavailable":
        monkeypatch.setattr(
            writer,
            "after_memory_changelog",
            AsyncMock(side_effect=RuntimeError("synthetic hook outage")),
        )
    monkeypatch.setattr(preflight, "generate_contradiction_reply", generate)
    send = AsyncMock(return_value=[{"event": "done", "data": "{}"}])
    ctx = preflight.PreflightCtx(
        **f.scope,
        agent=SimpleNamespace(name="合成伴侣"),
        tracer=MagicMock(),
        short_circuit_fn=send,
        message_ids=(f.second.id,),
    )
    events = [
        e async for e in preflight.resolve_pending_contradiction(f.second.content, ctx)
    ]
    outcome = send.await_args.kwargs["extra_metadata"]["memory_confirmation_outcome"]
    assert outcome == {"replay": "already_applied", "no_change": "no_change"}.get(
        mode, "applied"
    )
    assert events == [{"event": "done", "data": "{}"}] and ctx.stopped
    old = await f.db.usermemory.find_unique(where={"id": f.old.id})
    receipts = await writer.confirmation_receipt(
        f.db, f.wid, f.pending["_confirmation"]["id"]
    )
    assert old.isArchived is (mode != "no_change")
    assert len(receipts) == (0 if mode == "no_change" else 1)
    if mode in {"replay", "no_change"}:
        generate.assert_not_awaited()
    if mode == "reply_unavailable":
        assert send.await_args.args[0] == "好的，这条更正已经保存了。"
    if mode == "cleanup_unavailable":
        assert (await f.apply())["status"] == "already_applied"
