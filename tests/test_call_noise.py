"""VOICE01.04: memories extracted from a call transcript drop call noise; text chat is untouched."""
from unittest.mock import AsyncMock

import pytest

from app.services.chat.post_process import _shared_channel
from app.services.memory.recording import pipeline
from app.services.memory.recording.call_noise import CALL_CHANNEL, call_noise


@pytest.mark.parametrize("side,content,reason", [
    # Synthetic paraphrases of the 2026-11-07 call noise, not real user text.
    ("user", "用户说自己这边听得清", "call_mechanics"),
    ("user", "用户想测一下通话信号", "call_mechanics"),
    ("user", "用户说刚才断线了", "call_mechanics"),
    ("user", "用户说要挂电话了", "call_mechanics"),
    ("user", "用户说下次再聊", "call_mechanics"),
    ("user", "用户只是想找对方聊聊", "call_mechanics"),
    ("user", "用户不想说话", "call_mechanics"),
    ("ai", "我这边信号很好", "call_mechanics"),
    ("user", "用户询问AI是否记得自己养的狗", "question_not_fact"),
    ("user", "用户问对方明天做什么", "question_not_fact"),
    ("ai", "我刚下班回到家", "momentary_state"),
    ("ai", "我这会儿正在做饭", "momentary_state"),
])
def test_call_noise_is_recognised(side, content, reason):
    assert call_noise(side, content) == reason


@pytest.mark.parametrize("side,content", [
    ("user", "用户家里养了一只狗"),
    ("user", "用户喜欢吃牛肉面"),
    ("user", "用户心情不好"),            # emotions stay
    ("user", "用户觉得对方说话很无聊"),
    ("user", "用户带孩子去了公园"),
    ("ai", "我平时在花店上班"),
    ("ai", "我答应用户下次带他去吃面"),
    ("ai", "我询问用户今天累不累"),    # AI-side "询问" is not the user-question rule
    ("user", ""),
])
def test_facts_are_kept(side, content):
    assert call_noise(side, content) is None


def test_shared_channel_only_when_every_source_is_from_the_call():
    voice = {"metadata": {"channel": CALL_CHANNEL}}
    assert _shared_channel([voice, voice]) == CALL_CHANNEL
    assert _shared_channel([voice, {"metadata": {}}]) is None
    assert _shared_channel([voice, {"metadata": None}]) is None
    assert _shared_channel([]) is None


@pytest.fixture
def run_pipeline(monkeypatch):
    stored = []

    async def store(user_id, content, **kwargs):
        stored.append((kwargs.get("source"), content))
        return f"m{len(stored)}"

    monkeypatch.setattr(pipeline, "resolve_workspace_id", AsyncMock(return_value="ws"))
    monkeypatch.setattr(pipeline, "should_extract_memory", lambda _text: True)
    from app.config import settings
    monkeypatch.setattr(settings, "enable_memory_prefilter", False)
    monkeypatch.setattr(pipeline, "store_memory", store)

    async def run(facts, *, side, source_channel):
        monkeypatch.setattr(pipeline, "extract_memories", AsyncMock(return_value={"memories": facts}))
        await pipeline._process_memory_pipeline("u", "user: 测试", side=side, source_channel=source_channel)
        return [content for _, content in stored]

    return run


def _fact(content, main="生活", sub="其他"):
    return {"content": content, "summary": content, "main_category": main, "sub_category": sub,
            "importance": 0.4, "type": "life"}


async def test_call_batches_drop_noise_and_keep_facts(run_pipeline):
    kept = await run_pipeline([_fact("用户说自己这边听得清"), _fact("用户询问AI是否记得自己养的狗"),
                               _fact("用户家里养了一只狗", "身份")], side="user", source_channel=CALL_CHANNEL)
    assert kept == ["用户家里养了一只狗"]


async def test_text_chat_batches_are_untouched(run_pipeline):
    kept = await run_pipeline([_fact("用户询问AI是否记得自己养的狗")], side="user", source_channel=None)
    assert kept == ["用户询问AI是否记得自己养的狗"]


# ── 存量修复脚本: 真 PG ─────────────────────────────────────────────────
from uuid import uuid4  # noqa: E402

from prisma import Json  # noqa: E402

from tests.test_service_health_postgres import world  # noqa: E402,F401


@pytest.fixture
async def call_memories(world):
    database, conv, space, user, _ = world
    made = {"batches": [], "messages": []}

    async def memory(side, content, *, channel, archived=False):
        table = "memories_ai" if side == "ai" else "memories_user"
        mid, msg, batch = str(uuid4()), str(uuid4()), str(uuid4())
        await database.message.create(data={"id": msg, "conversationId": conv, "role": "user" if side == "user" else
                                            "assistant", "content": "synthetic",
                                            "metadata": Json({"channel": channel} if channel else {})})
        await database.execute_raw(
            """INSERT INTO memory_extraction_batches(id,user_id,workspace_id,conversation_id,side,scope,sources,status)
               VALUES($1,$2,$3,$4,$5,'{}'::jsonb,jsonb_build_array($6::text),'completed')""",
            batch, user, space, conv, side, msg)
        await database.execute_raw(
            "INSERT INTO memory_extraction_sources(message_id,side,batch_id) VALUES($1,$2,$3)", msg, side, batch)
        await database.execute_raw(
            f"""INSERT INTO {table}(id,user_id,workspace_id,content,level,importance,main_category,sub_category,
                 is_archived,updated_at) VALUES($1,$2,$3,$4,3,0.4,'生活','其他',$5,CURRENT_TIMESTAMP)""",
            mid, user, space, content, archived)
        await database.execute_raw(
            """INSERT INTO memory_write_receipts(batch_id,fact_index,piece_index,input_hash,status,memory_id)
               VALUES($1,0,0,$3,'written',$2)""", batch, mid, '0' * 64)
        made["batches"].append(batch)
        made["messages"].append(msg)
        return mid

    try:
        yield database, user, memory
    finally:
        await database.execute_raw("DELETE FROM memory_write_receipts WHERE batch_id = ANY($1::text[])", made["batches"])
        await database.execute_raw("DELETE FROM memory_extraction_sources WHERE batch_id = ANY($1::text[])", made["batches"])
        await database.execute_raw("DELETE FROM memory_extraction_batches WHERE id = ANY($1::text[])", made["batches"])
        await database.execute_raw("DELETE FROM memory_changelogs WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_user WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_ai WHERE user_id=$1", user)


async def test_repair_archives_only_call_noise_from_call_batches(call_memories, monkeypatch, tmp_path):
    import json as _json
    from types import SimpleNamespace

    from scripts import repair_call_noise_memories as repair

    database, user, memory = call_memories
    heard = await memory("user", "用户说自己这边听得清", channel=CALL_CHANNEL)
    asked = await memory("user", "用户询问AI是否记得自己养的狗", channel=CALL_CHANNEL)
    moment = await memory("ai", "我这会儿正在做饭", channel=CALL_CHANNEL)
    fact = await memory("user", "用户家里养了一只狗", channel=CALL_CHANNEL)
    texted = await memory("user", "用户询问明天天气", channel=None)  # text chat: never touched
    changed = await memory("user", "用户说要挂电话了", channel=CALL_CHANNEL)
    await memory("user", "用户说下次再聊", channel=CALL_CHANNEL, archived=True)  # already archived

    plan = await repair.build_plan(database)
    mine = {**plan, "archive": [i for i in plan["archive"] if i["user_hash"] == repair.user_hash(user)]}
    assert {i["memory_id"] for i in mine["archive"]} == {heard, asked, moment, changed}
    assert repair.summary(mine)["archive_by_reason"] == {
        "user:call_mechanics": 2, "user:question_not_fact": 1, "ai:momentary_state": 1}

    await database.execute_raw("UPDATE memories_user SET content='用户后来改了这条' WHERE id=$1", changed)
    assert await repair.apply_plan(database, mine) == {"archived": 3, "archive_skipped_changed": 1}
    archived = {r["id"]: r["is_archived"] for r in await database.query_raw(
        "SELECT id, is_archived FROM memories_user WHERE user_id=$1 UNION ALL "
        "SELECT id, is_archived FROM memories_ai WHERE user_id=$1", user)}
    assert archived[heard] and archived[asked] and archived[moment]
    assert not archived[fact] and not archived[texted] and not archived[changed]
    audit = await database.query_raw(
        "SELECT new_value FROM memory_changelogs WHERE memory_id=$1 AND operation='repair_archive'", heard)
    assert _json.loads(audit[0]["new_value"])["task"] == "VOICE01.04"
    with pytest.raises(ValueError):
        await repair.apply_plan(database, {"format": "other"})
    with pytest.raises(ValueError):
        await repair.apply_plan(database, {**mine, "archive": [{**mine["archive"][0], "reason": "x"}]})

    class Client:
        connect = disconnect = AsyncMock()

        def __getattr__(self, name):
            return getattr(database, name)

    monkeypatch.setattr("app.db.db", Client())
    path = tmp_path / "plan.json"
    out = await repair.main(SimpleNamespace(plan=path, apply=None))
    assert out["mode"] == "plan" and oct(path.stat().st_mode & 0o777) == "0o600"
    path.write_text(_json.dumps({"format": repair.FORMAT, "archive": [], "kept": 0}))
    assert await repair.main(SimpleNamespace(plan=None, apply=path)) == {
        "mode": "apply", "archived": 0, "archive_skipped_changed": 0}
