"""Entity/topic/preference linking must be per-memory, not batch-level.

Regression: a batch extracting N memories previously linked every entity /
topic / preference to all N rows, polluting the entity graph and entity recall.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.recording import pipeline as pipeline_mod


def _now():
    return datetime(2026, 7, 5, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_entities_topics_linked_per_memory():
    extraction = {
        "memories": [
            {"content": "用户养了一只猫叫芝麻", "importance": 0.6,
             "main_category": "身份", "sub_category": "宠物",
             "entities": ["芝麻"], "topics": ["宠物"]},
            {"content": "用户在做后端开发工作", "importance": 0.6,
             "main_category": "生活", "sub_category": "工作",
             "entities": ["后端开发"], "topics": ["工作"]},
        ],
        # Batch aggregate carries types + a mix of both memories' entities.
        "entities": [
            {"name": "芝麻", "type": "pet"},
            {"name": "后端开发", "type": "topic"},
        ],
        "preferences": [],
        "topics": ["宠物", "工作"],
    }

    ent_calls: list[dict] = []
    topic_calls: list[dict] = []

    async def _rec_entities(**kwargs):
        ent_calls.append(kwargs)
        return 0

    async def _rec_topics(**kwargs):
        topic_calls.append(kwargs)
        return 0

    ids = iter(["mem-cat", "mem-work"])

    with (
        patch.object(pipeline_mod, "should_extract_memory", return_value=True),
        patch("app.config.settings.enable_memory_prefilter", False),
        patch.object(pipeline_mod, "extract_memories", AsyncMock(return_value=extraction)),
        patch.object(pipeline_mod, "resolve_workspace_id", AsyncMock(return_value="w1")),
        patch.object(pipeline_mod, "store_memory", AsyncMock(side_effect=lambda **kw: next(ids))),
        patch.object(pipeline_mod, "log_memory_evidence", AsyncMock()),
        patch.object(pipeline_mod, "record_entities_for_memory", _rec_entities),
        patch.object(pipeline_mod, "record_topics_for_memory", _rec_topics),
        patch.object(pipeline_mod, "record_preferences_for_memory", AsyncMock(return_value=0)),
    ):
        await pipeline_mod.process_memory_pipeline(
            "u1", "user: 我养了猫叫芝麻\nuser: 我在做后端开发",
            side="user", statement_time=_now(),
        )

    # Each memory gets ONLY its own entity, with the type recovered from the batch.
    by_mem = {c["memory_id"]: c["entities"] for c in ent_calls}
    assert by_mem["mem-cat"] == [{"name": "芝麻", "type": "pet"}]
    assert by_mem["mem-work"] == [{"name": "后端开发", "type": "topic"}]

    topics_by_mem = {c["memory_id"]: c["topics"] for c in topic_calls}
    assert topics_by_mem["mem-cat"] == ["宠物"]
    assert topics_by_mem["mem-work"] == ["工作"]


@pytest.mark.asyncio
async def test_preferences_attributed_only_to_mentioning_memory():
    extraction = {
        "memories": [
            {"content": "用户喜欢吃辣", "importance": 0.6,
             "main_category": "偏好", "sub_category": "饮食喜好",
             "entities": [], "topics": []},
            {"content": "用户养了一只猫", "importance": 0.6,
             "main_category": "身份", "sub_category": "宠物",
             "entities": [], "topics": []},
        ],
        "entities": [],
        "preferences": [{"category": "food", "value": "辣"}],
        "topics": [],
    }

    pref_calls: list[dict] = []

    async def _rec_prefs(**kwargs):
        pref_calls.append(kwargs)
        return 0

    ids = iter(["mem-spicy", "mem-cat"])

    with (
        patch.object(pipeline_mod, "should_extract_memory", return_value=True),
        patch("app.config.settings.enable_memory_prefilter", False),
        patch.object(pipeline_mod, "extract_memories", AsyncMock(return_value=extraction)),
        patch.object(pipeline_mod, "resolve_workspace_id", AsyncMock(return_value="w1")),
        patch.object(pipeline_mod, "store_memory", AsyncMock(side_effect=lambda **kw: next(ids))),
        patch.object(pipeline_mod, "log_memory_evidence", AsyncMock()),
        patch.object(pipeline_mod, "record_entities_for_memory", AsyncMock(return_value=0)),
        patch.object(pipeline_mod, "record_topics_for_memory", AsyncMock(return_value=0)),
        patch.object(pipeline_mod, "record_preferences_for_memory", _rec_prefs),
    ):
        await pipeline_mod.process_memory_pipeline(
            "u1", "user: 我喜欢吃辣\nuser: 我养了猫",
            side="user", statement_time=_now(),
        )

    # "辣" only appears in the spicy memory → only that memory links the pref.
    assert len(pref_calls) == 1
    assert pref_calls[0]["memory_id"] == "mem-spicy"
    assert pref_calls[0]["preferences"] == [{"category": "food", "value": "辣"}]
