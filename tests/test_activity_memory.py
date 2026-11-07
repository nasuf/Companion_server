"""VOICE01.10: 活动的状态记忆随活动状态改写, 一个活动只有一对记忆, 没回应过的推荐不写。"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.memory.recording.chat_sources import excluded_from_chat_extraction
from app.services.memory.storage import embedding
from app.services.offline import activity_memory as am
from tests.test_service_health_postgres import world  # noqa: F401

DAY = datetime(2026, 11, 4, 2, tzinfo=UTC)  # 北京时间 11-04 10:00
ARRIVED = datetime(2026, 11, 5, 4, tzinfo=UTC)


def activity(**over):
    return {"title": "江边自来水厂旧址", "location_name": "自来水厂旧址", "city": "镇江", "status": "pending",
            "reached": False, "created_at": DAY, "updated_at": DAY, **over}


@pytest.mark.parametrize("over,expected", [
    ({}, None),  # 只收到推荐, 用户没回应: 不写
    ({"status": "accepted", "accepted_at": DAY}, "用户2026-11-04接受了，放进了待出行，还没去"),
    ({"status": "accepted", "accepted_at": DAY, "reached": True, "arrival_confirmed_at": ARRIVED},
     "用户2026-11-05到了现场"),
    ({"status": "completed", "accepted_at": DAY, "reached": True, "arrival_confirmed_at": ARRIVED},
     "用户2026-11-05到过现场，这次活动已经完成"),
    ({"status": "completed", "accepted_at": DAY}, "用户这次活动已经完成"),
    ({"status": "ignored", "ignored_at": DAY}, "用户2026-11-04说暂时不考虑"),
    ({"status": "cancelled", "accepted_at": DAY}, "用户接受后又取消了（2026-11-04），没去"),
    ({"status": "cancelled"}, None),  # 没接受就删掉的推荐: 不写
    ({"status": "expired", "ignored_at": DAY}, "用户活动过期了（2026-11-04），没去"),
    ({"status": "expired"}, None),
    ({"status": "accepted", "accepted_at": DAY, "title": " "}, None),
    ({"status": "weird", "accepted_at": DAY}, None),
])
def test_text_follows_the_current_state(over, expected):
    texts = am.render(activity(**over), "阿宁")
    if expected is None:
        assert texts is None
        return
    assert texts["user"] == f"2026-11-04 阿宁给用户推荐了线下活动「江边自来水厂旧址」（镇江），{expected}"
    # AI 侧写清是推荐给用户、在用户那边的, 不会被当成 AI 自己去过
    assert texts["ai"].startswith("2026-11-04 我给用户推荐了用户那边的线下活动「江边自来水厂旧址」")
    assert texts["ai"].endswith(expected)


def test_visited_plans_stay_longer_than_the_ones_not_taken():
    kept = am.render(activity(status="accepted", accepted_at=DAY), None)
    faded = am.render(activity(status="ignored", ignored_at=DAY, location_name="", city=""), None)
    other = am.render(activity(status="ignored", ignored_at=DAY, location_name="西津渡", city="镇江"), None)
    assert "「江边自来水厂旧址」（镇江·西津渡）" in other["user"]
    assert (kept["level"], faded["level"]) == (2, 3) and kept["importance"] > faded["importance"]
    assert faded["user"] == "2026-11-04 对方给用户推荐了线下活动「江边自来水厂旧址」，用户2026-11-04说暂时不考虑"
    assert am._day("bad") == "" and am._day(None) == "" and am._day("2026-11-04T02:00:00Z") == "2026-11-04"
    assert am._day(DAY.replace(tzinfo=None)) == "2026-11-04"


def test_activity_cards_inserted_for_the_user_are_not_extracted():
    card = {"role": "user", "content": "活动卡片：江边", "metadata": {"component_card": {"type": "offline_activity"}}}
    assert excluded_from_chat_extraction(card)
    music = {"role": "user", "content": "x", "metadata": {"component_card": {"type": "music_track"}}}
    assert not excluded_from_chat_extraction(music)


@pytest.fixture
async def scene(world, monkeypatch):
    database, conv, space, user, _ = world
    monkeypatch.setattr(am, "db", database)
    monkeypatch.setattr(embedding, "generate_embedding", AsyncMock(return_value=[0.01] * 1024))
    agent = (await database.conversation.find_unique(where={"id": conv})).agentId
    ids: list[str] = []

    async def create(**fields):
        ident = str(uuid4())
        ids.append(ident)
        await database.execute_raw(
            """INSERT INTO offline_activity_recommendations(id,user_id,agent_id,workspace_id,title,location_name,city,
                 status,created_at,updated_at) VALUES($1,$2,$3,$4,'江边自来水厂旧址','自来水厂旧址','镇江','pending',
                 $5::timestamptz,$5::timestamptz)""", ident, user, agent, space, DAY.isoformat())
        if fields:
            await set_state(ident, **fields)
        return ident

    async def set_state(ident, **fields):
        sets, params = [], []
        for key, value in fields.items():
            if value is None:
                sets.append(f"{key}=NULL")
                continue
            params.append(value.isoformat() if isinstance(value, datetime) else value)
            cast = "::timestamptz" if isinstance(value, datetime) else "::boolean" if isinstance(value, bool) else ""
            sets.append(f"{key}=${len(params) + 1}{cast}")
        await database.execute_raw(
            f"UPDATE offline_activity_recommendations SET {', '.join(sets)} WHERE id=$1", ident, *params)

    async def memories(ident):
        link = await database.query_raw("SELECT * FROM offline_activity_memories WHERE activity_id=$1", ident)
        if not link:
            return None
        out = {}
        for side, table in (("user", "memories_user"), ("ai", "memories_ai")):
            rows = await database.query_raw(
                f"SELECT content, level, is_archived, provenance, sub_category FROM {table} WHERE id=$1",
                link[0][f"{side}_memory_id"])
            out[side] = rows[0] if rows else None
        return out

    try:
        yield database, user, create, set_state, memories
    finally:
        # 合成用户只属于这个测试 (world 夹具创建): 按用户清理, 包括链接被清空的那些记忆
        await database.execute_raw("DELETE FROM offline_activity_recommendations WHERE id = ANY($1::text[])", ids)
        for table in ("memories_user", "memories_ai"):
            await database.execute_raw(
                f"DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM {table} WHERE user_id=$1)", user)
        await database.execute_raw("DELETE FROM memory_changelogs WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_user WHERE user_id=$1", user)
        await database.execute_raw("DELETE FROM memories_ai WHERE user_id=$1", user)


async def test_one_pair_is_rewritten_through_the_whole_lifecycle(scene):
    database, user, create, set_state, memories = scene
    ident = await create()
    assert await am.sync(ident, user) == "unchanged" and await memories(ident) is None
    await set_state(ident, status="accepted", accepted_at=DAY)
    assert await am.sync(ident, user) == "written"
    first = await memories(ident)
    assert first["user"]["content"].endswith("放进了待出行，还没去") and first["user"]["level"] == 2
    assert first["ai"]["provenance"] == "activity_record" and first["ai"]["sub_category"] == "其他特殊事件"
    assert await am.sync(ident, user) == "unchanged"  # 状态没变, 不重写
    links = await database.query_raw("SELECT * FROM offline_activity_memories WHERE activity_id=$1", ident)

    await set_state(ident, reached=True, arrival_confirmed_at=ARRIVED)
    assert await am.sync(ident, user) == "written"
    await set_state(ident, status="completed")
    assert await am.sync(ident, user) == "written"
    done = await memories(ident)
    assert done["user"]["content"].endswith("到过现场，这次活动已经完成")
    after = await database.query_raw("SELECT * FROM offline_activity_memories WHERE activity_id=$1", ident)
    # 始终是同一对记忆, 不追加
    assert (after[0]["user_memory_id"], after[0]["ai_memory_id"]) == (
        links[0]["user_memory_id"], links[0]["ai_memory_id"])
    ops = await database.query_raw(
        "SELECT operation FROM memory_changelogs WHERE memory_id=$1 ORDER BY created_at", after[0]["user_memory_id"])
    assert [o["operation"] for o in ops] == ["insert", "activity_sync", "activity_sync"]
    vector = await database.query_raw("SELECT 1 FROM memory_embeddings WHERE memory_id=$1", after[0]["ai_memory_id"])
    assert vector


async def test_a_cancelled_plan_is_rewritten_not_left_behind(scene):
    _, user, create, set_state, memories = scene
    ident = await create(status="accepted", accepted_at=DAY)
    await am.sync(ident, user)
    await set_state(ident, status="cancelled", updated_at=ARRIVED)
    assert await am.sync(ident, user) == "written"
    pair = await memories(ident)
    assert pair["user"]["content"].endswith("接受后又取消了（2026-11-05），没去")
    assert pair["user"]["level"] == 3 and "放进了待出行" not in pair["ai"]["content"]


async def test_a_memory_that_should_no_longer_exist_is_archived(scene):
    _, user, create, set_state, memories = scene
    ident = await create(status="accepted", accepted_at=DAY)
    await am.sync(ident, user)
    # 例如活动被改回待回应: 已有的记忆收起来, 不删除
    await set_state(ident, status="pending", accepted_at=None)
    assert await am.sync(ident, user) == "archived"
    pair = await memories(ident)
    assert pair["user"]["is_archived"] and pair["ai"]["is_archived"]
    assert await am.sync(ident, user) == "unchanged"
    await set_state(ident, status="accepted", accepted_at=DAY)
    assert await am.sync(ident, user) == "written"  # 再接受: 同一对记忆恢复并改写
    assert not (await memories(ident))["user"]["is_archived"]


async def test_a_memory_the_user_deleted_is_not_written_back(scene):
    database, user, create, set_state, memories = scene
    ident = await create(status="accepted", accepted_at=DAY)
    await am.sync(ident, user)
    link = (await database.query_raw("SELECT * FROM offline_activity_memories WHERE activity_id=$1", ident))[0]
    await database.execute_raw("DELETE FROM memories_user WHERE id=$1", link["user_memory_id"])
    await set_state(ident, status="completed")
    assert await am.sync(ident, user) == "written"
    pair = await memories(ident)
    assert pair["user"] is None and pair["ai"]["content"].endswith("这次活动已经完成")


async def test_changes_during_the_embedding_call_are_not_written_stale(scene, monkeypatch):
    _, user, create, set_state, memories = scene
    ident = await create(status="accepted", accepted_at=DAY)
    calls = []

    async def slow_embedding(text):
        calls.append(text)
        if len(calls) == 1:  # 第一次算向量时, 用户把它取消了
            await set_state(ident, status="cancelled", updated_at=ARRIVED)
        return [0.01] * 1024

    monkeypatch.setattr(embedding, "generate_embedding", slow_embedding)
    assert await am.sync(ident, user) == "written"
    assert (await memories(ident))["user"]["content"].endswith("没去")
    monkeypatch.setattr(am, "_write", AsyncMock(return_value="raced"))
    assert await am.sync(ident, user) == "raced"


async def test_missing_activity_and_clearing_all_activities(scene, monkeypatch):
    _, user, create, _, memories = scene
    assert await am.sync(str(uuid4()), user) == "missing"
    first = await create(status="accepted", accepted_at=DAY)
    second = await create(status="ignored", ignored_at=DAY)
    for ident in (first, second):
        await am.sync(ident, user)
    assert await am.archive_for_user(user) == 4
    pairs = [await memories(ident) for ident in (first, second)]
    assert all(m["is_archived"] for pair in pairs for m in pair.values())
    assert await am.archive_for_user(user) == 0
    monkeypatch.setattr(am, "_load", AsyncMock(return_value=None))
    assert await am._write(first, user, None, {}) == "missing"


async def test_background_sync_logs_and_never_raises(monkeypatch):
    scheduled = []
    monkeypatch.setattr(am, "fire_background", lambda coro: scheduled.append(coro))
    am.schedule_sync(None, "u")
    am.schedule_sync("a", "u")
    assert len(scheduled) == 1
    monkeypatch.setattr(am, "sync", AsyncMock(side_effect=RuntimeError("db down")))
    await scheduled[0]  # 失败只记日志
    monkeypatch.setattr(am, "sync", AsyncMock(return_value="written"))
    await am._sync_logged("a", "u")


@pytest.mark.parametrize("side,provenance,content,reason", [
    ("user", "user_stated", "用户提及活动卡片：陕西大剧院", "activity_card_not_user_words"),
    ("user", None, "用户发送活动卡片：小無家咖啡馆", "activity_card_not_user_words"),
    ("user", "user_stated", "用户暂时忽略了线下活动推荐：锦江剧场", "superseded_status_event"),
    ("user", "user_stated", "用户确认到达线下活动地点：幸运咖(嘉泰广场店)", "superseded_status_event"),
    ("ai", "daily_summary", "用户分享了大兴善寺的活动卡片，建议去感受历史氛围。", "wrong_direction"),
    # 用户自己说的看法、AI 自己的感受: 不动
    ("user", "user_stated", "用户对新华书店(嫩兴路店)的活动不太感兴趣", None),
    ("ai", "daily_summary", "看到佛山先行书店小聚活动卡片时感到开心。", None),
    ("ai", "ai_authored", "用户分享了一张活动卡片", None),
    ("user", "activity_record", "用户提及活动卡片：X", None),
])
def test_repair_only_targets_card_noise_superseded_events_and_reversed_summaries(side, provenance, content, reason):
    from scripts.repair_activity_memories import classify

    assert classify(side, provenance, content) == reason


async def test_repair_plan_then_apply_archives_noise_and_writes_state_memories(scene, monkeypatch, tmp_path):
    from scripts import repair_activity_memories as repair

    database, user, create, _, memories = scene
    accepted = await create(status="accepted", accepted_at=DAY)
    untouched = await create()  # 没回应过: 不写
    workspace = (await database.query_raw(
        "SELECT workspace_id FROM offline_activity_recommendations WHERE id=$1", accepted))[0]["workspace_id"]
    noise = await database.usermemory.create(data={
        "userId": user, "workspaceId": workspace, "content": "用户提及活动卡片：江边自来水厂旧址",
        "level": 3, "importance": .4, "provenance": "user_stated", "mainCategory": "生活", "subCategory": "其他"})
    opinion = await database.usermemory.create(data={
        "userId": user, "workspaceId": workspace, "content": "用户觉得自来水厂旧址那边太远了",
        "level": 3, "importance": .4, "provenance": "user_stated", "mainCategory": "生活", "subCategory": "其他"})
    changed = await database.usermemory.create(data={
        "userId": user, "workspaceId": workspace, "content": "用户暂时忽略了线下活动推荐：江边自来水厂旧址",
        "level": 3, "importance": .4, "provenance": "user_stated", "mainCategory": "生活", "subCategory": "其他"})
    try:
        plan = await repair.build_plan(database)
        mine = {i["memory_id"]: i["reason"] for i in plan["archive"]}
        assert mine.get(noise.id) == "activity_card_not_user_words" and opinion.id not in mine
        assert mine.get(changed.id) == "superseded_status_event"
        synced = [i["activity_id"] for i in plan["sync"]]
        assert accepted in synced and untouched not in synced
        assert repair.summary({"archive": [], "sync": []}) == {"archive": 0, "archive_by_reason": {}, "sync": 0, "users": 0}
        # 计划生成后被改过的记忆不碰
        await database.usermemory.update(where={"id": changed.id}, data={"content": "用户后来说想去自来水厂旧址"})
        mine_plan = {**plan, "archive": [i for i in plan["archive"] if i["memory_id"] in {noise.id, changed.id}],
                     "sync": [i for i in plan["sync"] if i["activity_id"] == accepted]}
        stats = await repair.apply_plan(database, mine_plan)
        assert stats == {"archived": 1, "archive_skipped_changed": 1, "sync": {"written": 1}}
        rows = {r["id"]: r["is_archived"] for r in await database.query_raw(
            "SELECT id, is_archived FROM memories_user WHERE id = ANY($1::text[])", [noise.id, opinion.id, changed.id])}
        assert rows == {noise.id: True, opinion.id: False, changed.id: False}
        log = await database.query_raw("SELECT operation, new_value FROM memory_changelogs WHERE memory_id=$1", noise.id)
        assert log[0]["operation"] == "repair_archive" and "activity_card_not_user_words" in log[0]["new_value"]
        assert (await memories(accepted))["user"]["content"].endswith("还没去")
        assert await repair.apply_plan(database, mine_plan) == {
            "archived": 0, "archive_skipped_changed": 2, "sync": {"unchanged": 1}}
        with pytest.raises(ValueError):
            await repair.apply_plan(database, {"format": "other"})
        with pytest.raises(ValueError):
            await repair.apply_plan(database, {**mine_plan, "archive": [{**mine_plan["archive"][0], "side": "x"}]})
        # 命令行: 计划文件只给属主读, 输出只有计数
        class Client:  # main() 自己连库断库; 这里转给测试库, 不真的断开
            connect = disconnect = AsyncMock()

            def __getattr__(self, name):
                return getattr(database, name)

        monkeypatch.setattr("app.db.db", Client())
        path = tmp_path / "plan.json"
        out = await repair.main(SimpleNamespace(plan=path, apply=None))
        assert out["mode"] == "plan" and oct(path.stat().st_mode & 0o777) == "0o600"
        path.write_text(json.dumps({"format": "activity-memory-repair-v1", "archive": [], "sync": []}))
        assert await repair.main(SimpleNamespace(plan=None, apply=path)) == {
            "mode": "apply", "archived": 0, "archive_skipped_changed": 0, "sync": {}}
    finally:
        ids = [noise.id, opinion.id, changed.id]
        await database.execute_raw("DELETE FROM memory_changelogs WHERE memory_id = ANY($1::text[])", ids)
        await database.execute_raw("DELETE FROM memories_user WHERE id = ANY($1::text[])", ids)


async def test_expired_events_resync_their_state_memories(scene, monkeypatch):
    from app.services.offline import repository as repo

    database, user, create, _, _ = scene
    monkeypatch.setattr(repo, "db", database)
    synced = []
    monkeypatch.setattr(am, "schedule_sync", lambda activity_id, user_id: synced.append((activity_id, user_id)))
    ident = await create(status="accepted", accepted_at=DAY)
    await database.execute_raw(
        """UPDATE offline_activity_recommendations SET discovery_metadata='{"kind":"event"}'::jsonb,
             expires_at=$2::timestamptz WHERE id=$1""", ident, DAY.isoformat())
    await repo.expire_inactive_events(user)
    assert synced == [(ident, user)]
    await repo.expire_inactive_events(user)  # 已过期的不再触发
    assert synced == [(ident, user)]


def test_state_memories_are_never_merged_or_promoted():
    from app.services.memory.lifecycle.value import l1_promotion_allowed
    from app.services.memory.provenance import COMPRESSION_EXEMPT
    from app.services.memory.storage.reconciliation import _is_write_protected

    row = SimpleNamespace(provenance="activity_record", level=2, mainCategory="生活", subCategory="其他特殊事件")
    assert _is_write_protected(row) and "activity_record" in COMPRESSION_EXEMPT
    assert not l1_promotion_allowed(side="ai", provenance="activity_record", main_category="生活",
                                    sub_category="其他特殊事件", contribution_days=9)


async def test_clearing_skips_sides_that_never_had_a_memory(scene):
    database, user, create, _, _ = scene
    ident = await create(status="accepted", accepted_at=DAY)
    await am.sync(ident, user)
    await database.execute_raw("UPDATE offline_activity_memories SET user_memory_id=NULL WHERE activity_id=$1", ident)
    assert await am.archive_for_user(user) == 1


async def test_auto_archive_resyncs_the_state_memory(monkeypatch):
    from app.services.offline import activity_service

    monkeypatch.setattr(activity_service.repo, "list_due_for_auto_archive",
                        AsyncMock(return_value=[{"id": "a1", "user_id": "u1", "workspace_id": "w1", "title": "X"}]))
    monkeypatch.setattr(activity_service.repo, "mark_archived", AsyncMock(return_value={"id": "a1"}))
    monkeypatch.setattr(activity_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    monkeypatch.setattr(activity_service, "_ensure_journey_note", AsyncMock())
    synced = []
    monkeypatch.setattr(activity_service, "sync_activity_memory", lambda *args: synced.append(args))
    await activity_service.auto_archive_due_activities()
    assert synced == [("a1", "u1")]
