"""Tests for fast fact extractor working memory."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.fast_fact import (
    extract_fast_facts,
    facts_for_prompt,
    merge_working_facts,
    update_working_facts,
)


def test_merge_working_facts_replaces_same_slot():
    existing = [
        {
            "category": "occupation",
            "key": "job",
            "value": "designer",
            "confidence": 0.81,
            "updated_at": "2026-03-18T10:00:00+00:00",
            "expires_at": "2026-03-28T10:00:00+00:00",
        }
    ]
    new = [
        {
            "category": "occupation",
            "key": "job",
            "value": "engineer",
            "confidence": 0.91,
            "updated_at": "2026-03-18T11:00:00+00:00",
            "expires_at": "2026-03-28T11:00:00+00:00",
        }
    ]

    merged = merge_working_facts(existing, new)
    assert len(merged) == 1
    assert merged[0]["value"] == "engineer"


def test_facts_for_prompt_formats_category():
    facts = [{"category": "preference", "value": "喜欢寿司"}]
    assert facts_for_prompt(facts) == ["[preference] 喜欢寿司"]


@pytest.mark.asyncio
async def test_extract_fast_facts_uses_model_output():
    payload = {
        "facts": [
            {
                "category": "location",
                "key": "current_city",
                "value": "住在上海",
                "confidence": 0.92,
                "ttl_days": 14,
            }
        ]
    }
    with (
        patch("app.services.chat.fast_fact.get_prompt_text", AsyncMock(return_value="{message}")),
        patch("app.services.chat.fast_fact.invoke_json", AsyncMock(return_value=payload)),
    ):
        facts = await extract_fast_facts("我住在上海")

    assert len(facts) == 1
    assert facts[0]["category"] == "location"
    assert facts[0]["value"] == "住在上海"


@pytest.mark.asyncio
async def test_update_working_facts_merges_and_saves(mock_redis):
    mock_redis.get = AsyncMock(return_value='{"facts":[{"category":"name","key":"name","value":"小明","confidence":0.9,"updated_at":"2026-03-18T10:00:00+00:00","expires_at":"2026-04-28T10:00:00+00:00"}]}')
    with patch("app.services.chat.fast_fact.get_redis", AsyncMock(return_value=mock_redis)):
        with patch(
            "app.services.chat.fast_fact.extract_fast_facts",
            AsyncMock(
                return_value=[
                    {
                        "category": "age",
                        "key": "age",
                        "value": "28岁",
                        "confidence": 0.95,
                        "updated_at": "2026-03-18T11:00:00+00:00",
                        "expires_at": "2026-04-28T11:00:00+00:00",
                    }
                ]
            ),
        ):
            facts = await update_working_facts("conv-1", "我28岁")

    assert len(facts) == 2
    mock_redis.set.assert_awaited()
