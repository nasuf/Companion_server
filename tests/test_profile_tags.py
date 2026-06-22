from unittest.mock import AsyncMock

import pytest

from app.services import profile_tags
from app.services.offline import repository as offline_repo


def test_normalize_llm_tags_filters_noise_and_keeps_sources():
    raw = {
        "tags": [
            {
                "label": "音乐爱好者",
                "category": "preference",
                "confidence": 0.91,
                "source_memory_ids": ["m1", "missing", "m2"],
            },
            {
                "label": "职业与经济",
                "category": "work",
                "confidence": 0.99,
                "source_memory_ids": ["m1"],
            },
            {
                "label": "冷战中",
                "category": "relationship",
                "confidence": 0.88,
                "source_memory_ids": ["m2"],
            },
            {
                "label": "随便看看",
                "category": "behavior",
                "confidence": 0.3,
                "source_memory_ids": ["m2"],
            },
        ]
    }

    tags = profile_tags._normalize_llm_tags(raw, valid_memory_ids={"m1", "m2"})

    assert [tag.label for tag in tags] == ["音乐爱好者"]
    assert tags[0].source_memory_ids == ["m1", "m2"]
    assert tags[0].confidence == 0.91


@pytest.mark.asyncio
async def test_offline_tags_prefer_persisted_profile_tags(monkeypatch):
    list_profile_tags = AsyncMock(return_value=["音乐爱好者", "爱逛书店"])
    query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr(offline_repo.profile_tags, "list_profile_tags", list_profile_tags)
    monkeypatch.setattr(offline_repo.db, "query_raw", query_raw)

    assert await offline_repo.list_user_tags(
        "user-1",
        "workspace-1",
        agent_id="agent-1",
    ) == ["音乐爱好者", "爱逛书店"]
    list_profile_tags.assert_awaited_once_with(
        "user-1",
        "workspace-1",
        agent_id="agent-1",
        limit=9,
    )
    query_raw.assert_not_awaited()
