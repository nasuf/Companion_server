"""录入期 L1 钳制 + AI 自我记忆来源排除.

生产 2026-10: 抽取 prompt 的 JSON 示例写死 importance 0.8, 六成以上新记忆恰好
0.80; 一次性状态、对话行为与生活杂项随后被当成核心事实使用。录入时能否直接落
L1 跟生命周期晋升闸门同一口径 (taxonomy.l1_promotion_requirement)。
"""

from __future__ import annotations

import pytest

from tests.test_ai_self_authored_persona_block import _run


def _extraction(main, sub, content, importance=0.9, memory_type="life"):
    return {
        "memories": [{
            "content": content,
            "importance": importance,
            "type": memory_type,
            "main_category": main,
            "sub_category": sub,
            "occur_time": None,
            "recurrence": None,
        }],
        "entities": [], "topics": [], "preferences": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("main,sub,content", [
    ("生活", "其他", "用户已经在工位了"),
    ("情绪", "焦虑", "用户最近很焦虑"),
    ("思维", "其他", "用户觉得有可能"),
])
async def test_user_episodes_and_misc_never_insert_as_l1(main, sub, content):
    calls = await _run("user", _extraction(main, sub, content), f"user: {content}")
    assert len(calls) == 1
    assert calls[0]["level"] == 2
    assert calls[0]["importance"] == pytest.approx(0.84)


@pytest.mark.asyncio
@pytest.mark.parametrize("main,sub,content", [
    ("身份", "现居地", "用户住在西安"),
    ("偏好", "禁忌/雷区", "用户对花生过敏"),
])
async def test_core_facts_still_insert_as_l1(main, sub, content):
    calls = await _run("user", _extraction(main, sub, content), f"user: {content}")
    assert calls[0]["level"] == 1
    assert calls[0]["importance"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_missing_main_category_falls_back_to_legacy_type():
    calls = await _run(
        "user",
        _extraction(None, "现居地", "用户住在西安", memory_type="identity"),
        "user: 我住西安",
    )
    assert calls[0]["level"] == 1


@pytest.mark.asyncio
async def test_ai_chat_memory_is_l1_only_for_relationship_milestones():
    episode = await _run(
        "ai", _extraction("生活", "生活", "我刚刷到一家湘菜馆"), "assistant: 我刚刷到一家湘菜馆",
    )
    assert episode[0]["level"] == 2
    milestone = await _run(
        "ai", _extraction("生活", "交互", "我和用户第一次聊到心事"),
        "assistant: 跟你聊完感觉近了一步",
    )
    assert milestone[0]["level"] == 1


def test_offline_activity_copy_never_feeds_ai_self_memory():
    """推荐/陪伴文案是对用户说的业务文本 ("头疼好些了，去清凉山走走"), 抽成 AI
    自我记忆就成了 AI 第一人称的亲历。"""
    from app.services.memory.recording.chat_sources import excluded_from_chat_extraction

    for trigger in (
        "offline_activity_recommendation", "offline_activity_recommendation_card",
        "offline_activity_accepted", "offline_activity_arrival_guide",
    ):
        assert excluded_from_chat_extraction(
            {"role": "assistant", "metadata": {"trigger_type": trigger}}
        ), trigger
    # 用户自己发出的活动卡片仍是用户的事实来源; 普通主动消息照常进入自我记忆
    assert not excluded_from_chat_extraction(
        {"role": "user", "metadata": {"trigger_type": "offline_activity_accepted_card"}}
    )
    assert not excluded_from_chat_extraction(
        {"role": "assistant", "metadata": {"trigger_type": "scheduled_scene"}}
    )
