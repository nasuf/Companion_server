"""用户遗忘后画像整份重建; 周更不再删除变更日志 (M02.04.03)。"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from app.services import portrait
from app.services.memory.storage import repo
from tests.test_memory_lifecycle_postgres import memory_flow, memory  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401


def _wire(f, monkeypatch, *, reply="新画像"):
    monkeypatch.setattr(portrait, "db", f.db)
    monkeypatch.setattr(repo, "db", f.db)
    monkeypatch.setattr(portrait, "resolve_workspace_id", AsyncMock(return_value=f.ids["workspace"]))
    monkeypatch.setattr(portrait, "get_prompt_text", AsyncMock(return_value="{memories}|{behaviour}"))
    monkeypatch.setattr(portrait, "_behaviour_section", AsyncMock(return_value=""))
    monkeypatch.setattr(portrait, "_refresh_tags_best_effort", AsyncMock())
    invoke = AsyncMock(return_value=reply)
    monkeypatch.setattr(portrait, "invoke_text", invoke)
    monkeypatch.setattr(portrait, "get_utility_model", lambda: object())
    return invoke


async def _portrait(f, content, at):
    await f.db.userportrait.create(data={"user": {"connect": {"id": f.ids["owner"]}},
        "agentId": f.ids["agent"], "version": 1, "content": content, "createdAt": at})


async def _forget_log(f, at, op="user_bulk_delete"):
    await f.db.memorychangelog.create(data={"userId": f.ids["owner"], "workspaceId": f.ids["workspace"],
        "memoryId": "gone", "operation": op, "oldValue": "用户养了一只叫团子的猫", "createdAt": at})


async def _latest(f):
    return await f.db.userportrait.find_first(where={"userId": f.ids["owner"]}, order={"version": "desc"})


async def _cleanup(f):
    await f.db.userportrait.delete_many(where={"userId": f.ids["owner"]})


async def test_forget_after_portrait_rebuilds_from_live_memories(memory_flow, monkeypatch):
    f = memory_flow
    invoke = _wire(f, monkeypatch)
    now = datetime.now(UTC)
    try:
        await memory(f, content="用户喜欢爬山")
        await _portrait(f, "旧画像: 养了猫团子", now - timedelta(days=2))
        await _forget_log(f, now - timedelta(days=1))
        stats = await portrait.rebuild_portraits_after_forget()
        assert stats == {"due": 1, "rebuilt": 1, "emptied": 0, "failed": 0}
        latest = await _latest(f)
        assert latest.version == 2 and latest.content == "新画像"
        prompt = invoke.await_args.args[1]
        assert "用户喜欢爬山" in prompt and "团子" not in prompt
        # 同一次遗忘只重建一次
        assert (await portrait.rebuild_portraits_after_forget())["due"] == 0
    finally:
        await _cleanup(f)


async def test_forget_with_no_live_memory_left_empties_the_portrait(memory_flow, monkeypatch):
    f = memory_flow
    invoke = _wire(f, monkeypatch)
    now = datetime.now(UTC)
    try:
        await _portrait(f, "旧画像: 养了猫团子", now - timedelta(days=2))
        await _forget_log(f, now - timedelta(days=1), op="workspace_wipe")
        stats = await portrait.rebuild_portraits_after_forget()
        assert stats["emptied"] == 1
        assert (await _latest(f)).content == ""
        assert await portrait.get_latest_portrait(f.ids["owner"], f.ids["agent"]) == ""
        invoke.assert_not_awaited()
    finally:
        await _cleanup(f)


async def test_forget_before_the_portrait_needs_no_rebuild(memory_flow, monkeypatch):
    f = memory_flow
    _wire(f, monkeypatch)
    now = datetime.now(UTC)
    try:
        await _forget_log(f, now - timedelta(days=3))
        await _portrait(f, "画像", now - timedelta(days=1))
        assert (await portrait.rebuild_portraits_after_forget())["due"] == 0
    finally:
        await _cleanup(f)


async def test_weekly_update_rebuilds_after_forget_and_keeps_the_audit(memory_flow, monkeypatch):
    f = memory_flow
    invoke = _wire(f, monkeypatch)
    monkeypatch.setattr(portrait, "_collect_behaviour_facts", AsyncMock(return_value=[]))
    now = datetime.now(UTC)
    try:
        await memory(f, content="用户喜欢爬山")
        old = await f.db.memorychangelog.create(data={"userId": f.ids["owner"],
            "workspaceId": f.ids["workspace"], "memoryId": "m-old", "operation": "consolidated_into",
            "newValue": "digest", "createdAt": now - timedelta(days=30)})
        await _portrait(f, "旧画像: 养了猫团子", now - timedelta(days=3))
        await _forget_log(f, now - timedelta(days=1), op="delete")
        assert await portrait.update_portrait_weekly(f.ids["owner"], f.ids["agent"]) == "新画像"
        assert "团子" not in invoke.await_args.args[1]
        assert await f.db.memorychangelog.find_unique(where={"id": old.id}) is not None
    finally:
        await _cleanup(f)


async def test_rebuild_counts_model_failures_and_errors(memory_flow, monkeypatch):
    f = memory_flow
    _wire(f, monkeypatch)
    now = datetime.now(UTC)
    try:
        await memory(f, content="用户喜欢爬山")
        await _portrait(f, "旧画像", now - timedelta(days=2))
        await _forget_log(f, now - timedelta(days=1))
        monkeypatch.setattr(portrait, "generate_portrait", AsyncMock(return_value=None))
        assert (await portrait.rebuild_portraits_after_forget())["failed"] == 1
        monkeypatch.setattr(portrait, "generate_portrait", AsyncMock(side_effect=RuntimeError("x")))
        assert (await portrait.rebuild_portraits_after_forget())["failed"] == 1
        assert (await _latest(f)).version == 1  # 留给下一轮
    finally:
        await _cleanup(f)


async def test_weekly_update_rebuilds_an_emptied_portrait(memory_flow, monkeypatch):
    f = memory_flow
    _wire(f, monkeypatch)
    try:
        await memory(f, content="用户喜欢爬山")
        await _portrait(f, "", datetime.now(UTC) - timedelta(days=1))
        assert await portrait.update_portrait_weekly(f.ids["owner"], f.ids["agent"]) == "新画像"
    finally:
        await _cleanup(f)
