from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_time_range_results_bypass_vector_similarity_threshold(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: True)
    monkeypatch.setattr(
        hybrid_mod,
        "parse_time_expressions",
        lambda _: [SimpleNamespace(
            confidence=0.9,
            is_future=False,
            start=now - timedelta(days=1),
            end=now,
        )],
    )
    monkeypatch.setattr(
        hybrid_mod,
        "search_by_time_range",
        AsyncMock(return_value=[{
            "id": "m-time",
            "content": "用户去年生日去了海边",
            "summary": "用户去年生日去了海边",
            "level": 2,
            "importance": 0.7,
            "source": "user",
        }]),
    )

    result = await hybrid_mod.hybrid_retrieve(
        "去年生日那天", "u1", workspace_id="ws1",
    )

    memories = result["memories"]
    assert memories
    assert memories[0].id == "m-time"
    assert memories[0].text == "用户去年生日去了海边"


@pytest.mark.asyncio
async def test_hybrid_rerank_uses_last_accessed_at(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=400)).isoformat()
    recent = (now - timedelta(days=2)).isoformat()
    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[
        {
            "id": "old-but-touched",
            "content": "用户很久前说喜欢爵士乐",
            "summary": "用户很久前说喜欢爵士乐",
            "level": 2,
            "importance": 0.7,
            "similarity": 0.8,
            "created_at": old,
            "last_accessed_at": recent,
            "source": "user",
        },
        {
            "id": "old-only",
            "content": "用户很久前说喜欢摇滚",
            "summary": "用户很久前说喜欢摇滚",
            "level": 2,
            "importance": 0.7,
            "similarity": 0.8,
            "created_at": old,
            "source": "user",
        },
    ]))

    result = await hybrid_mod.hybrid_retrieve(
        "我以前喜欢什么音乐", "u1", workspace_id="ws1",
    )

    memories = result["memories"]
    assert memories
    assert [m.id for m in memories[:2]] == ["old-but-touched", "old-only"]
    assert memories[0].score > memories[1].score


@pytest.mark.asyncio
async def test_hybrid_rerank_keeps_safety_memory_in_top_ten(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    generic = [
        {
            "id": f"generic-{i}",
            "content": f"用户核心身份事实 {i}",
            "summary": f"用户核心身份事实 {i}",
            "level": 1,
            "importance": 0.95,
            "similarity": 0.82,
            "main_category": "身份",
            "sub_category": "其他",
            "source": "user",
        }
        for i in range(12)
    ]
    safety_memory = {
        "id": "safety-memory",
        "content": "用户表达过强烈负面情绪, 有轻生念头",
        "summary": "用户表达过强烈负面情绪, 有轻生念头",
        "level": 1,
        "importance": 0.80,
        "similarity": 0.55,
        "main_category": "情绪",
        "sub_category": "悲伤",
        "source": "user",
    }
    monkeypatch.setattr(
        hybrid_mod,
        "search_similar",
        AsyncMock(return_value=generic + [safety_memory]),
    )

    result = await hybrid_mod.hybrid_retrieve(
        "我快活不下去了", "u1", workspace_id="ws1",
    )

    memories = result["memories"]
    assert memories
    ids = [m.id for m in memories]
    assert "safety-memory" in ids
    assert ids.index("safety-memory") < 3


def test_safety_boost_does_not_treat_positive_emotion_as_crisis_memory():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    positive_emotion = {
        "id": "happy-memory",
        "summary": "用户之前收到礼物时特别开心",
        "content": "用户之前收到礼物时特别开心",
        "importance": 0.8,
        "similarity": 0.8,
        "main_category": "情绪",
        "sub_category": "开心",
        "source": "user",
    }
    negative_emotion = {
        "id": "sad-memory",
        "summary": "用户之前说自己很低落, 有些撑不住",
        "content": "用户之前说自己很低落, 有些撑不住",
        "importance": 0.8,
        "similarity": 0.8,
        "main_category": "情绪",
        "sub_category": "悲伤",
        "source": "user",
    }

    positive_score, positive_reasons = rank_memory_candidate(
        positive_emotion, "我快活不下去了",
    )
    negative_score, negative_reasons = rank_memory_candidate(
        negative_emotion, "我快活不下去了",
    )

    assert "安全/情绪相关" not in positive_reasons
    assert "安全/情绪相关" in negative_reasons
    assert negative_score > positive_score
