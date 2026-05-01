"""Spec §3.2 + §4.2 主动记忆按 topic 过滤 + spec §2.1 stage 4 档单测."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _row(rid: str, text: str, importance: float = 0.5):
    return SimpleNamespace(
        id=rid, summary=text, content=text,
        mainCategory="生活", subCategory="日常",
        importance=importance,
    )


@pytest.mark.asyncio
async def test_load_proactive_memories_uses_rerank_when_topic_present():
    """topic_theme 非空 → utility model rerank 选 ≤3 条按返回顺序排."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "下雨天"), _row("m2", "看了部电影"), _row("m3", "新菜谱")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock, return_value={"ids": ["m2", "m1"]}),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="电影",
        )

    # 按 rerank 顺序: m2 在前
    assert ids == ["m2", "m1"]
    assert "看了部电影" in texts[0]


@pytest.mark.asyncio
async def test_load_proactive_memories_falls_back_to_importance_when_rerank_returns_none():
    """render_prompt 失败返回 None → 回退到 importance 倒排, 不阻塞主动消息发送."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "记忆 A"), _row("m2", "记忆 B")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock, return_value=None),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="任意",
        )

    assert ids == ["m1", "m2"]
    assert len(texts) == 2


@pytest.mark.asyncio
async def test_load_proactive_memories_filters_hallucinated_ids():
    """LLM 返回不在候选集里的 id (幻觉) → 仅保留 valid 子集."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "A"), _row("m2", "B")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock,
              return_value={"ids": ["m1", "ghost-id-not-in-candidates"]}),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="任意",
        )

    assert ids == ["m1"]
    assert len(texts) == 1


@pytest.mark.asyncio
async def test_load_proactive_memories_skips_rerank_when_no_topic():
    """topic_theme 为空 → 不调 rerank, 直接走 importance 倒排."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "A")]
    rerank_call = AsyncMock()
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt", rerank_call),
    ):
        _, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="",
        )

    rerank_call.assert_not_awaited()
    assert ids == ["m1"]


@pytest.mark.asyncio
async def test_determine_proactive_stage_4_tier():
    """Spec §2.1 单维度 4 档: P1/P2/warming/intimate."""
    from app.services.proactive.state import determine_proactive_stage

    cases = [
        (5, "p1_cold"),
        (20, "p1_cold"),
        (21, "p2_cold"),
        (40, "p2_cold"),
        (41, "warming"),
        (80, "warming"),
        (81, "intimate"),
        (100, "intimate"),
    ]
    for intimacy, expected in cases:
        with patch(
            "app.services.proactive.state._load_topic_intimacy",
            new_callable=AsyncMock, return_value=intimacy,
        ):
            stage = await determine_proactive_stage("agent1", "user1")
        assert stage == expected, f"intimacy={intimacy} → expected={expected}, got={stage}"
