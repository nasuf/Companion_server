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
