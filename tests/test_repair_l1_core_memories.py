"""存量 L1 清理脚本: 判定口径、只读演练、按计划执行与还原。"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma import Json

from scripts import repair_l1_core_memories as repair
from tests.test_memory_lifecycle_postgres import event, memory, memory_flow, row  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401


def _row(**overrides):
    return {
        "id": "m1", "user_id": "u1", "workspace_id": "w1", "content": "x",
        "main_category": "生活", "sub_category": "其他", "importance": 0.8,
        "current_score": 0.92, "level": 1, "provenance": "user_stated",
        "is_archived": False, **overrides,
    }


class TestClassify:
    def test_user_episodes_and_misc_are_demoted(self):
        assert repair.classify("user", _row()) == ("demote", repair.USER_NON_CORE)
        assert repair.classify("user", _row(main_category="身份", sub_category="其他")) == (
            "demote", repair.USER_NON_CORE)

    def test_user_identity_is_kept(self):
        assert repair.classify("user", _row(main_category="身份", sub_category="现居地")) is None

    def test_promoted_stable_preference_needs_multi_day_evidence(self):
        pref = _row(main_category="偏好", sub_category="禁忌/雷区")
        assert repair.classify("user", pref, contribution_days=1) == (
            "demote", repair.USER_UNSUSTAINED)
        assert repair.classify("user", pref, contribution_days=3) is None
        # 录入时就 ≥0.85 的是用户强调过的, 不是靠晋升进的 L1
        assert repair.classify("user", {**pref, "importance": 0.9}, contribution_days=0) is None

    def test_ai_chat_derived_l1(self):
        assert repair.classify("ai", _row(provenance="ai_authored", main_category="生活",
                                          sub_category="生活")) == ("demote", repair.AI_CHAT_L1)
        assert repair.classify("ai", _row(provenance="daily_summary", main_category="身份")) == (
            "archive", repair.AI_PERSONA_CLAIM)
        assert repair.classify("ai", _row(provenance="ai_authored", main_category="生活",
                                          sub_category="交互")) is None

    def test_ai_persona_seeds_are_never_touched(self):
        for provenance in ("profile_seed", "knowledge_seed", None):
            assert repair.classify("ai", _row(provenance=provenance), offline_overlap=1.0) is None

    def test_offline_copy_flip_is_archived_at_any_level(self):
        flipped = _row(provenance="ai_authored", level=2, main_category="生活", sub_category="生活")
        assert repair.classify("ai", flipped, offline_overlap=0.9) == ("archive", repair.AI_OFFLINE_COPY)
        assert repair.classify("ai", flipped, offline_overlap=0.2) is None

    def test_archived_rows_are_ignored(self):
        assert repair.classify("user", _row(is_archived=True)) is None


def test_copy_overlap_undoes_the_subject_rewrite():
    copy = "大悦城里有不少独立书店和咖啡馆，感觉挺对你胃口的，路过可以逛逛～"
    assert repair.copy_overlap("我觉得大悦城里有不少独立书店和咖啡馆，挺对用户胃口的", copy) >= 0.8
    assert repair.copy_overlap("我头疼好些了，去清凉山走走挺解压的，山里空气好",
                               "头疼好些了，去清凉山走走挺解压的。山里空气好，适合你这种爱逛书店的人") >= 0.8
    assert repair.copy_overlap("我和用户下五子棋时差点被吓出一身冷汗", copy) < 0.2
    assert repair.copy_overlap("", copy) == 0.0


# ── 真实 PostgreSQL ───────────────────────────────────────────────────────


async def _offline_copy_evidence(f, memory, text):
    msg = await f.db.message.create(data={
        "conversationId": f.ids["conversation"], "role": "assistant", "content": text,
        "metadata": Json({"trigger_type": "offline_activity_recommendation"}),
    })
    await f.db.execute_raw(
        """INSERT INTO memory_evidence_links
             (id, memory_id, memory_source, ai_memory_id, user_id, workspace_id, agent_id,
              content_version, source_kind, source_ref, source_message_id, source_role,
              source_version, relation, extractor_version)
           VALUES ($1, $2, 'ai', $2, $3, $4, $5, $6, 'message', $7, $7, 'assistant', $8,
                   'extracted_from', 'test')""",
        str(uuid4()), memory.id, f.ids["owner"], f.ids["workspace"], f.ids["agent"],
        repair.content_hash(memory.content), msg.id, repair.content_hash(text),
    )


async def _receipt(f, memory_id, days_ago):
    msg = await event(f)
    await f.db.execute_raw(
        "INSERT INTO memory_usage_receipts(event_id,memory_side,memory_id,contributed,reward,created_at) "
        "VALUES ($1,'user',$2,TRUE,0.12,NOW() - make_interval(days => $3::int))",
        msg.id, memory_id, days_ago,
    )


@pytest.fixture
async def seeded(memory_flow, monkeypatch):  # noqa: F811
    f = memory_flow
    monkeypatch.setattr(repair, "db", f.db)
    monkeypatch.setattr(repair, "_after_commit", AsyncMock())
    m = {
        "desk": await memory(f, level=1, content="用户已经在工位了"),
        "name": await memory(f, level=1, content="用户叫田野", mainCategory="身份", subCategory="姓名",
                             importance=0.9),
        "taboo_new": await memory(f, level=1, content="用户不吃香菜", mainCategory="偏好",
                                  subCategory="禁忌/雷区", importance=0.8),
        "taboo_old": await memory(f, level=1, content="用户花生过敏", mainCategory="偏好",
                                  subCategory="禁忌/雷区", importance=0.8),
        "ai_episode": await memory(f, "ai", level=1, content="我刚刷到一家湘菜馆",
                                   subCategory="生活", provenance="ai_authored"),
        "ai_claim": await memory(f, "ai", level=1, content="职业是修表匠", mainCategory="身份",
                                 subCategory="其他", provenance="daily_summary"),
        "ai_milestone": await memory(f, "ai", level=1, content="我和用户第一次聊心事",
                                     subCategory="交互", provenance="ai_authored"),
        "ai_seed": await memory(f, "ai", level=1, content="我喜欢春天", mainCategory="情绪",
                                subCategory="其他", provenance="profile_seed"),
        "ai_flip": await memory(f, "ai", level=2, content="我头疼好些了，去清凉山走走挺解压的",
                                subCategory="生活", provenance="ai_authored"),
    }
    for days_ago in (1, 2, 3):
        await _receipt(f, m["taboo_old"].id, days_ago)
    await _offline_copy_evidence(f, m["ai_flip"], "头疼好些了，去清凉山走走挺解压的。山里空气好～")
    yield f, m
    await f.db.execute_raw("DELETE FROM memory_evidence_links WHERE user_id=$1", f.ids["owner"])


def _plan_index(items):
    return {i["id"]: (i["action"], i["reason"]) for i in items}


async def test_dry_run_plans_by_rule_and_writes_nothing(seeded):
    f, m = seeded
    before = await f.db.query_raw(
        "SELECT id, level, is_archived FROM memories_user WHERE user_id=$1 "
        "UNION ALL SELECT id, level, is_archived FROM memories_ai WHERE user_id=$1 ORDER BY id",
        f.ids["owner"])
    items, kept = await repair.dry_run(workspace=f.ids["workspace"])
    plan = _plan_index(items)
    assert plan == {
        m["desk"].id: ("demote", repair.USER_NON_CORE),
        m["taboo_new"].id: ("demote", repair.USER_UNSUSTAINED),
        m["ai_episode"].id: ("demote", repair.AI_CHAT_L1),
        m["ai_claim"].id: ("archive", repair.AI_PERSONA_CLAIM),
        m["ai_flip"].id: ("archive", repair.AI_OFFLINE_COPY),
    }
    assert kept["user 身份/姓名"] == 1 and kept["user 偏好/禁忌/雷区"] == 1
    assert kept["ai 生活/交互"] == 1
    assert all(i["content_sha256"] and len(i["preview"]) <= 30 for i in items)
    after = await f.db.query_raw(
        "SELECT id, level, is_archived FROM memories_user WHERE user_id=$1 "
        "UNION ALL SELECT id, level, is_archived FROM memories_ai WHERE user_id=$1 ORDER BY id",
        f.ids["owner"])
    assert after == before
    assert await f.db.memorychangelog.count(where={"userId": f.ids["owner"],
                                                  "operation": repair.OPERATION}) == 0


async def test_dry_run_transaction_refuses_writes(seeded, monkeypatch):
    f, _ = seeded

    async def sneaky_plan(conn, *, workspace=None):
        await conn.execute_raw("UPDATE memories_user SET level=2 WHERE user_id=$1", f.ids["owner"])
        return [], {}

    monkeypatch.setattr(repair, "build_plan", sneaky_plan)
    with pytest.raises(Exception, match="read-only"):
        await repair.dry_run()


async def test_apply_revalidates_each_row_then_revert_restores(seeded):
    f, m = seeded
    items, _ = await repair.dry_run(workspace=f.ids["workspace"])
    # 审核之后、执行之前: 一条内容被改写, 一条已经被别的流程降级
    await f.db.execute_raw("UPDATE memories_ai SET content='我刚刷到一家粤菜馆' WHERE id=$1",
                           m["ai_episode"].id)
    await f.db.execute_raw("UPDATE memories_user SET level=2 WHERE id=$1", m["taboo_new"].id)

    stats = await repair.apply_plan(items)
    assert stats["applied:" + repair.USER_NON_CORE] == 1
    assert stats["applied:" + repair.AI_PERSONA_CLAIM] == 1
    assert stats["applied:" + repair.AI_OFFLINE_COPY] == 1
    assert stats["skipped:content_changed"] == 1
    assert stats["skipped:rule_no_longer_applies"] == 1
    assert (await row(f, m["desk"].id))["level"] == 2
    assert (await row(f, m["ai_claim"].id, "ai"))["is_archived"] is True
    assert (await row(f, m["ai_flip"].id, "ai"))["is_archived"] is True
    assert (await row(f, m["ai_episode"].id, "ai"))["level"] == 1
    assert (await row(f, m["name"].id))["level"] == 1
    assert (await row(f, m["ai_seed"].id, "ai"))["level"] == 1
    logs = await f.db.memorychangelog.find_many(where={"userId": f.ids["owner"],
                                                      "operation": repair.OPERATION})
    assert {log.memoryId for log in logs} == {m["desk"].id, m["ai_claim"].id, m["ai_flip"].id}
    assert all("reason=" in log.newValue for log in logs)

    # 重复执行同一份计划不会再改任何行
    again = await repair.apply_plan(items)
    assert not any(k.startswith("applied:") for k in again)

    preview = await repair.revert(apply=False, memory_ids=[i["id"] for i in items])
    assert preview["restorable"] == 3 and (await row(f, m["desk"].id))["level"] == 2
    done = await repair.revert(apply=True, memory_ids=[i["id"] for i in items])
    assert done["reverted"] == 3
    assert (await row(f, m["desk"].id))["level"] == 1
    assert (await row(f, m["ai_claim"].id, "ai"))["is_archived"] is False
    assert (await row(f, m["ai_flip"].id, "ai"))["is_archived"] is False
    assert (await repair.revert(apply=True, memory_ids=[i["id"] for i in items]))["reverted"] == 0


async def test_revert_skips_rows_changed_after_repair(seeded):
    f, m = seeded
    items, _ = await repair.dry_run(workspace=f.ids["workspace"])
    desk = [i for i in items if i["id"] == m["desk"].id]
    await repair.apply_plan(desk)
    await f.db.execute_raw("UPDATE memories_user SET is_archived=TRUE WHERE id=$1", m["desk"].id)
    stats = await repair.revert(apply=True, memory_ids=[m["desk"].id])
    assert stats["skipped:changed_since_repair"] == 1 and stats["reverted"] == 0


async def test_reason_filter_limits_execution(seeded):
    f, m = seeded
    items, _ = await repair.dry_run(workspace=f.ids["workspace"])
    stats = await repair.apply_plan(items, reasons={repair.USER_NON_CORE})
    assert stats["applied:" + repair.USER_NON_CORE] == 1
    assert stats["not_selected"] == len(items) - 1
    assert (await row(f, m["ai_claim"].id, "ai"))["is_archived"] is False
