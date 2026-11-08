"""B01.08: AI 寄给用户的礼物的状态记忆随礼物状态改写, 一份礼物一对记忆; 签收前不写礼物名。"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from app.services.chat import post_process
from app.services.memory.storage import embedding
from app.services.offline import gift_memory
from app.services.offline import record_memory as rm
from tests.test_service_health_postgres import world  # noqa: F401

SENT = datetime(2026, 10, 14, 16, 30, tzinfo=UTC)  # 北京时间 10-15 凌晨
DELIVERED = datetime(2026, 10, 18, 4, tzinfo=UTC)
GIFT = {"id": "g1", "gift_name": "空白迷你灵感便签本", "gift_note": "想到小灵感就写下来呀",
        "ordered_at": SENT, "created_at": SENT, "agent_name": "昕昕", "notified_sent": True}


def gift(**over):
    return {**GIFT, "status": "shipping", **over}


@pytest.mark.parametrize("status", ["ordered", "shipping", "out_for_delivery"])
def test_in_transit_is_an_ai_memory_without_the_gift_name(status):
    texts = gift_memory.render(gift(status=status))
    assert texts["user"] is None
    assert texts["ai"] == "2026-10-15 我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么"
    assert "便签本" not in texts["ai"]  # 记忆会注入聊天，签收前不能出现礼物名
    assert "保密" not in texts["ai"]  # 写事实不写指令：指令进了长期记忆，揭晓后还会生效
    assert (texts["level"], texts["importance"]) == (2, gift_memory.EVENT_IMPORTANCE)


def test_delivered_is_remembered_by_both_sides_with_the_reason_and_thanks():
    texts = gift_memory.render(gift(status="delivered", delivered_at=DELIVERED,
                                    reveal_message="其实是便签本||你总有小灵感", thanks_message="好喜欢||正好缺个本子"))
    assert texts["ai"].startswith("2026-10-18 用户签收了我寄的礼物：空白迷你灵感便签本")
    assert "「其实是便签本 你总有小灵感」" in texts["ai"] and "想到小灵感就写下来呀" in texts["ai"]
    assert "用户收到后跟我道谢：「好喜欢 正好缺个本子」" in texts["ai"]
    assert texts["user"].startswith("2026-10-18 我收到并签收了昕昕寄来的礼物：空白迷你灵感便签本")
    assert "「好喜欢 正好缺个本子」" in texts["user"]
    assert texts["importance"] == gift_memory.RECEIVED_IMPORTANCE
    # 签收后不再有「等签收再揭晓」
    assert "揭晓，所以没提前说" not in texts["ai"]


def test_delivered_memory_stays_injectable_when_texts_are_long():
    long = "好" * 200
    texts = gift_memory.render(gift(status="delivered", delivered_at=DELIVERED, agent_name=None,
                                    gift_note=long, reveal_message=long, thanks_message=long))
    # 超过单条注入上限的记忆检索时会被整条跳过：放不下的补充句宁可不加。
    assert not gift_memory.exceeds_injection_limit(texts["ai"])
    assert texts["ai"].count("「") == 1
    assert "对方寄来的礼物" in texts["user"]


def test_cancelled_names_when_the_gift_was_sent_and_never_the_gift():
    texts = gift_memory.render(gift(status="cancelled", updated_at=DELIVERED))
    assert texts["ai"] == "2026-10-18 我寄给用户的那份礼物（2026-10-15寄出）中途取消了，没能送到用户手上"
    assert texts["user"] is None and "便签本" not in texts["ai"]
    same_day = gift_memory.render(gift(status="cancelled", updated_at=SENT))
    assert same_day["ai"] == "2026-10-15 我寄给用户的那份礼物中途取消了，没能送到用户手上"
    undated = gift_memory.render(gift(status="cancelled", ordered_at=None, created_at=None))
    assert undated["ai"] == "我寄给用户的那份礼物中途取消了，没能送到用户手上"


@pytest.mark.parametrize("over", [
    {"notified_sent": False},  # 用户从没收到「礼物在路上」：不记
    {"status": "failed"}, {"status": "skipped"}, {"status": "pending_address"}, {"status": "selecting"},
])
def test_no_memory_for_gifts_the_user_never_heard_of_or_that_never_went_out(over):
    assert gift_memory.render(gift(**over)) is None


def test_dates_are_local_and_tolerate_bad_input():
    assert gift_memory._parse_time("not a date") is None
    assert gift_memory._parse_time(None) is None
    naive = gift_memory._parse_time(datetime(2026, 10, 14, 16, 30))
    assert naive.tzinfo is UTC
    assert gift_memory._parse_time("2026-10-14T16:30:00") == naive
    assert gift_memory._date_prefix(naive) == "2026-10-15 "
    assert gift_memory.render(gift(ordered_at="bad", created_at=None))["ai"].startswith("我给用户寄了")


@pytest.fixture
def background(monkeypatch):
    """记下 thanks 排了哪些后台任务, 同步跑完。"""
    synced, pending = [], []
    monkeypatch.setattr(gift_memory, "schedule_sync", lambda gift_id, user_id: synced.append((gift_id, user_id)))
    monkeypatch.setattr(gift_memory, "fire_background", pending.append)

    async def flush():
        while pending:
            await pending.pop(0)
        return synced

    return flush


@pytest.mark.asyncio
async def test_thanks_rewrites_the_state_memory_and_extracts_with_evidence(background, monkeypatch):
    pipeline = AsyncMock(return_value=[])
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", pipeline)
    gift_memory.remember_gift_thanks(
        "user-1", "ws-1", GIFT, message="好喜欢||正好缺个本子",
        message_id="msg-9", message_content="我收到了「空白迷你灵感便签本」，想说：好喜欢||正好缺个本子",
    )
    assert await background() == [("g1", "user-1")]
    args, kwargs = pipeline.await_args
    assert args == ("user-1", "user: 我收到了「空白迷你灵感便签本」，想说：好喜欢||正好缺个本子")
    assert kwargs["side"] == "user" and kwargs["evidence_message_ids"] == ["msg-9"]
    assert kwargs["context_conversation"] == "user: 我签收了你寄来的「空白迷你灵感便签本」"


@pytest.mark.asyncio
async def test_thanks_without_a_chat_message_only_rewrites_the_state_memory(background, monkeypatch):
    pipeline = AsyncMock()
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", pipeline)
    gift_memory.remember_gift_thanks("user-1", None, GIFT, message="谢谢", message_id=None, message_content=None)
    assert await background() == [("g1", "user-1")]
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_thanks_extraction_failure_is_contained(background, monkeypatch):
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", AsyncMock(side_effect=RuntimeError("llm")))
    gift_memory.remember_gift_thanks("user-1", "ws-1", GIFT, message="谢谢", message_id="m", message_content="谢谢")
    assert await background() == [("g1", "user-1")]


@pytest.fixture
async def scene(world, monkeypatch):
    database, conv, space, user, _ = world
    monkeypatch.setattr(rm, "db", database)
    monkeypatch.setattr(embedding, "generate_embedding", AsyncMock(return_value=[0.01] * 1024))
    agent = (await database.conversation.find_unique(where={"id": conv})).agentId
    ids: list[str] = []

    async def create(status="shipping", notify=True):
        ident = str(uuid4())
        ids.append(ident)
        await database.execute_raw(
            """INSERT INTO real_world_gifts(id,user_id,agent_id,workspace_id,conversation_id,status,gift_name,gift_note,
                 ordered_at,created_at,updated_at)
               VALUES($1,$2,$3,$4,$5,$6,'空白迷你灵感便签本','想到小灵感就写下来呀',$7::timestamptz,$7::timestamptz,
                 $7::timestamptz)""", ident, user, agent, space, conv, status, SENT.isoformat())
        if notify:
            await say(ident, "gift_sent", "有个惊喜在路上")
        return ident

    async def say(ident, trigger, content):
        await database.execute_raw(
            """INSERT INTO messages(id,conversation_id,role,content,metadata,created_at)
               VALUES($1,$2,'assistant',$3,jsonb_build_object('real_world_type','gift','source_id',$4::text,
                 'trigger_type',$5::text),CURRENT_TIMESTAMP)""", str(uuid4()), conv, content, ident, trigger)

    async def set_state(ident, **fields):
        sets, params = [], []
        for key, value in fields.items():
            params.append(value.isoformat() if isinstance(value, datetime) else value)
            sets.append(f"{key}=${len(params) + 1}{'::timestamptz' if isinstance(value, datetime) else ''}")
        await database.execute_raw(f"UPDATE real_world_gifts SET {', '.join(sets)} WHERE id=$1", ident, *params)

    async def memories(ident):
        link = await database.query_raw("SELECT * FROM offline_gift_memories WHERE gift_id=$1", ident)
        if not link:
            return None
        out = {}
        for side, table in (("user", "memories_user"), ("ai", "memories_ai")):
            ref = link[0][f"{side}_memory_id"]
            rows = await database.query_raw(
                f"SELECT content, level, is_archived, provenance, sub_category FROM {table} WHERE id=$1", ref
            ) if ref else []
            out[side] = rows[0] if rows else None
        return out

    try:
        yield database, user, create, say, set_state, memories
    finally:
        await database.execute_raw("DELETE FROM messages WHERE conversation_id=$1", conv)
        await database.execute_raw("DELETE FROM real_world_gifts WHERE id = ANY($1::text[])", ids)
        for table in ("memories_user", "memories_ai"):
            await database.execute_raw(
                f"DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM {table} WHERE user_id=$1)", user)
        await database.execute_raw("DELETE FROM memory_changelogs WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_user WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_ai WHERE user_id=$1", user)


async def test_one_pair_follows_the_gift_from_sent_to_thanked(scene):
    database, user, create, say, set_state, memories = scene
    ident = await create()
    assert await gift_memory.sync(ident, user) == "written"
    sent = await memories(ident)
    assert sent["user"] is None and "等用户签收时再揭晓" in sent["ai"]["content"]
    assert sent["ai"]["provenance"] == "gift_record" and sent["ai"]["sub_category"] == "馈赠"
    assert "便签本" not in sent["ai"]["content"]
    assert await gift_memory.sync(ident, user) == "unchanged"
    first = (await database.query_raw("SELECT ai_memory_id FROM offline_gift_memories WHERE gift_id=$1", ident))[0]

    await set_state(ident, status="out_for_delivery")
    assert await gift_memory.sync(ident, user) == "unchanged"  # 派送中: 记忆文字不变
    await say(ident, "gift_received_reply", "其实是便签本||你总有小灵感")
    await set_state(ident, status="delivered", delivered_at=DELIVERED)
    assert await gift_memory.sync(ident, user) == "written"
    got = await memories(ident)
    assert got["ai"]["content"].startswith("2026-10-18 用户签收了我寄的礼物：空白迷你灵感便签本")
    assert "「其实是便签本 你总有小灵感」" in got["ai"]["content"]
    assert "揭晓，所以没提前说" not in got["ai"]["content"]  # 「等签收再揭晓」不残留
    assert got["user"]["content"] == "2026-10-18 我收到并签收了Synthetic寄来的礼物：空白迷你灵感便签本"

    await set_state(ident, thanks_message="好喜欢")
    assert await gift_memory.sync(ident, user) == "written"
    thanked = await memories(ident)
    assert thanked["user"]["content"].endswith("我跟Synthetic说：「好喜欢」")
    after = (await database.query_raw("SELECT ai_memory_id FROM offline_gift_memories WHERE gift_id=$1", ident))[0]
    assert after["ai_memory_id"] == first["ai_memory_id"]  # 始终是同一条, 不追加
    ops = await database.query_raw(
        "SELECT operation FROM memory_changelogs WHERE memory_id=$1 ORDER BY created_at", after["ai_memory_id"])
    assert [o["operation"] for o in ops] == ["insert", "gift_sync", "gift_sync"]


async def test_a_cancelled_gift_rewrites_the_in_transit_memory(scene):
    _, user, create, _, set_state, memories = scene
    ident = await create()
    await gift_memory.sync(ident, user)
    await set_state(ident, status="cancelled", updated_at=DELIVERED)
    assert await gift_memory.sync(ident, user) == "written"
    got = await memories(ident)
    assert got["ai"]["content"] == "2026-10-18 我寄给用户的那份礼物（2026-10-15寄出）中途取消了，没能送到用户手上"


async def test_gifts_the_user_never_heard_of_get_no_memory_and_clearing_archives(scene, monkeypatch):
    database, user, create, _, set_state, memories = scene
    quiet = await create(notify=False)
    assert await gift_memory.sync(quiet, user) == "unchanged" and await memories(quiet) is None
    told = await create()
    await gift_memory.sync(told, user)
    await set_state(told, status="failed")  # 例如改成失败: 已有的记忆收起来, 不删除
    assert await gift_memory.sync(told, user) == "archived"
    assert (await memories(told))["ai"]["is_archived"]
    await set_state(told, status="shipping")
    assert await gift_memory.sync(told, user) == "written"
    assert await gift_memory.archive_for_user(user) == 1
    assert (await memories(told))["ai"]["is_archived"]
    assert await gift_memory.sync(str(uuid4()), user) == "missing"


async def test_clearing_gifts_archives_their_memories_first(monkeypatch):
    from app.services.offline import gift_service

    order = []
    monkeypatch.setattr(gift_service.gift_memory, "archive_for_user",
                        AsyncMock(side_effect=lambda user_id: order.append("archive") or 1))
    monkeypatch.setattr(gift_service.gift_repo, "clear_user_gifts",
                        AsyncMock(side_effect=lambda user_id: order.append("clear") or {"gifts": 1}))
    assert await gift_service.clear_all_gifts("u1") == {"gifts": 1}
    assert order == ["archive", "clear"]


def test_gift_state_memories_are_never_merged_or_promoted():
    from types import SimpleNamespace

    from app.services.memory.lifecycle.value import l1_promotion_allowed
    from app.services.memory.provenance import COMPRESSION_EXEMPT
    from app.services.memory.storage.reconciliation import _is_write_protected

    row = SimpleNamespace(provenance="gift_record", level=2, mainCategory="生活", subCategory="馈赠")
    assert _is_write_protected(row) and "gift_record" in COMPRESSION_EXEMPT
    assert not l1_promotion_allowed(side="ai", provenance="gift_record", main_category="生活",
                                    sub_category="馈赠", contribution_days=9)


@pytest.mark.parametrize("side,provenance,content,reason", [
    ("ai", "ai_authored", "2026-10-15 我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么", "superseded_sent"),
    ("ai", "ai_authored", "2026-10-18 用户签收了我寄的礼物：便签本，揭晓时我跟用户说：「x」", "superseded_received"),
    ("ai", "ai_authored", "2026-10-19 用户收到我送的便签本后特意跟我道谢：「谢谢」", "superseded_thanks"),
    ("ai", "ai_authored", "2026-10-20 我寄给用户的那份礼物（2026-10-15寄出）中途取消了，没能送到用户手上", "superseded_cancelled"),
    ("user", "user_stated", "2026-10-18 我收到并签收了昕昕寄来的礼物：便签本", "superseded_received"),
    # 用户送 AI 的礼物、用户自己说的话、新写法的状态记忆: 不动
    ("ai", "ai_authored", "2026-10-18 用户送给我礼物：一束花", None),
    ("user", "user_stated", "我送给昕昕的围巾她很喜欢", None),
    ("ai", "gift_record", "2026-10-15 我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么", None),
    ("user", "ai_authored", "2026-10-18 我收到并签收了昕昕寄来的礼物：便签本", None),
])
def test_repair_only_matches_the_old_gift_templates(side, provenance, content, reason):
    from scripts import repair_gift_memories as repair

    assert repair.classify(side, provenance, content) == reason


async def test_repair_plan_archives_old_rows_and_rewrites_from_state(scene, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from scripts import repair_gift_memories as repair

    database, user, create, say, set_state, memories = scene
    delivered = await create()
    await say(delivered, "gift_received_reply", "其实是便签本")
    await set_state(delivered, status="delivered", delivered_at=DELIVERED)
    quiet = await create(notify=False)
    space = (await database.query_raw("SELECT workspace_id FROM real_world_gifts WHERE id=$1", quiet))[0]["workspace_id"]
    old = {
        "sent": await database.aimemory.create(data={
            "userId": user, "workspaceId": space, "content": "2026-10-15 我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么",
            "level": 2, "importance": 0.6, "mainCategory": "生活", "subCategory": "馈赠", "provenance": "ai_authored"}),
        "received": await database.usermemory.create(data={
            "userId": user, "workspaceId": space, "content": "2026-10-18 我收到并签收了Synthetic寄来的礼物：空白迷你灵感便签本",
            "level": 2, "importance": 0.72, "mainCategory": "生活", "subCategory": "馈赠", "provenance": "user_stated"}),
        "changed": await database.aimemory.create(data={
            "userId": user, "workspaceId": space, "content": "2026-10-18 用户签收了我寄的礼物：空白迷你灵感便签本",
            "level": 2, "importance": 0.72, "mainCategory": "生活", "subCategory": "馈赠", "provenance": "ai_authored"}),
        "kept": await database.aimemory.create(data={
            "userId": user, "workspaceId": space, "content": "2026-10-18 用户送给我礼物：一束花",
            "level": 2, "importance": 0.72, "mainCategory": "生活", "subCategory": "馈赠", "provenance": "ai_authored"}),
    }
    plan = await repair.build_plan(database)
    mine = {**plan, "archive": [i for i in plan["archive"] if i["user_hash"] == repair.user_hash(user)],
            "orphan": [], "sync": [i for i in plan["sync"] if i["user_id"] == user]}
    assert {i["memory_id"] for i in mine["archive"]} == {old["sent"].id, old["received"].id, old["changed"].id}
    assert [i["gift_id"] for i in mine["sync"]] == [delivered]  # 用户没被告知的那份不补写
    assert repair.summary(mine)["archive_by_reason"] == {
        "ai:superseded_sent": 1, "ai:superseded_received": 1, "user:superseded_received": 1}

    # 计划生成后内容变过的行: 跳过, 不误伤
    await database.aimemory.update(where={"id": old["changed"].id}, data={"content": "用户后来改了这条"})
    stats = await repair.apply_plan(database, mine)
    assert stats == {"archived": 2, "archive_skipped_changed": 1, "sync": {"written": 1}}
    rows = {r["id"]: r["is_archived"] for r in await database.query_raw(
        "SELECT id, is_archived FROM memories_ai WHERE user_id=$1 UNION ALL "
        "SELECT id, is_archived FROM memories_user WHERE user_id=$1", user)}
    assert rows[old["sent"].id] and rows[old["received"].id]
    assert not rows[old["changed"].id] and not rows[old["kept"].id]
    got = await memories(delivered)
    assert got["ai"]["content"].startswith("2026-10-18 用户签收了我寄的礼物：空白迷你灵感便签本")
    audit = await database.query_raw(
        "SELECT new_value FROM memory_changelogs WHERE memory_id=$1 AND operation='repair_archive'", old["sent"].id)
    assert json.loads(audit[0]["new_value"])["task"] == "B01.08"
    assert await repair.apply_plan(database, mine) == {"archived": 0, "archive_skipped_changed": 3,
                                                      "sync": {"unchanged": 1}}
    with pytest.raises(ValueError):
        await repair.apply_plan(database, {"format": "other"})
    with pytest.raises(ValueError):
        await repair.apply_plan(database, {**mine, "archive": [{**mine["archive"][0], "reason": "x"}]})

    class Client:  # main() 自己连库断库; 这里转给测试库, 不真的断开
        connect = disconnect = AsyncMock()

        def __getattr__(self, name):
            return getattr(database, name)

    monkeypatch.setattr("app.db.db", Client())
    path = tmp_path / "plan.json"
    out = await repair.main(SimpleNamespace(plan=path, apply=None))
    assert out["mode"] == "plan" and oct(path.stat().st_mode & 0o777) == "0o600"
    path.write_text(json.dumps({"format": repair.FORMAT, "archive": [], "orphan": [], "sync": []}))
    assert await repair.main(SimpleNamespace(plan=None, apply=path)) == {
        "mode": "apply", "archived": 0, "archive_skipped_changed": 0, "sync": {}}


async def test_repair_puts_rows_of_users_without_gifts_aside(scene):
    from scripts import repair_gift_memories as repair

    database, user, *_ = scene
    lone = await database.user.create(data={"username": "gift-orphan-" + uuid4().hex})
    try:
        row = await database.aimemory.create(data={
            "userId": lone.id, "content": "2026-10-15 我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么",
            "level": 2, "importance": 0.6, "mainCategory": "生活", "subCategory": "馈赠", "provenance": "ai_authored"})
        plan = await repair.build_plan(database)
        assert row.id in {i["memory_id"] for i in plan["orphan"]}
        assert row.id not in {i["memory_id"] for i in plan["archive"]}
    finally:
        await database.execute_raw("DELETE FROM memories_ai WHERE user_id=$1", lone.id)
        await database.user.delete(where={"id": lone.id})


@pytest.mark.asyncio
@pytest.mark.parametrize("backlog_mode", [False, True])
async def test_gift_chat_messages_are_context_not_extraction_targets(backlog_mode, monkeypatch):
    monkeypatch.setattr("app.services.memory.recording.receipts.record_policy_skips", AsyncMock())
    now = datetime.now(UTC)
    gift_card = {
        "id": "gift-card", "role": "user", "content": "我签收了你寄来的「便签本」",
        "createdAt": now, "metadata": {"real_world_type": "gift", "trigger_type": "gift_received"},
    }
    normal = {"id": "normal", "role": "user", "content": "这个本子我拿来记灵感",
              "createdAt": now + timedelta(seconds=1), "metadata": {}}
    pipeline = AsyncMock(return_value=[])
    set_wm = AsyncMock()
    with patch.object(post_process, "get_watermark", AsyncMock(return_value=now - timedelta(minutes=1) if backlog_mode else None)), \
         patch.object(post_process, "_fetch_new_side_messages", AsyncMock(return_value=[gift_card, normal])), \
         patch.object(post_process, "process_memory_pipeline", pipeline), \
         patch.object(post_process, "set_watermark", set_wm):
        await post_process._pipeline_with_watermark("user-1", [gift_card, normal], "conv", side="user",
                                                     workspace_id="ws-1")

    kwargs = pipeline.await_args.kwargs
    assert kwargs["evidence_message_ids"] == ["normal"]
    assert "签收" not in kwargs["new_conversation"]
    assert "签收" in kwargs["context_conversation"]  # 仍是历史上下文，帮抽取理解「这个本子」
    assert set_wm.await_args.args[2] == normal["createdAt"]

    # 只有礼物消息时不调模型，但水位线照常推进，不会每轮都重新捞到它。
    pipeline.reset_mock()
    with patch.object(post_process, "get_watermark", AsyncMock(return_value=now - timedelta(minutes=1))), \
         patch.object(post_process, "_fetch_new_side_messages", AsyncMock(return_value=[gift_card])), \
         patch.object(post_process, "process_memory_pipeline", pipeline), \
         patch.object(post_process, "set_watermark", set_wm):
        outcome = await post_process._pipeline_with_watermark("user-1", [gift_card], "conv", side="user")
        assert outcome.status == "excluded" and outcome.memory_count == 0 and outcome.progress == "advanced"
    pipeline.assert_not_awaited()
    assert set_wm.await_args.args[2] == now
