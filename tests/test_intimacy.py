"""亲密度系统单元测试。

测试覆盖：
- 亲密度等级映射
- 话题深度映射
- 关系时长计算
- Redis缓存CRUD
- 成长亲密度计算
- 话题亲密度计算
"""

import json
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.relationship.intimacy import (
    _compute_relationship_duration,
    compute_growth_intimacy,
    get_cached_intimacy,
    get_intimacy_data,
    get_intimacy_level,
    get_topic_depth,
    get_topic_intimacy,
    save_intimacy,
    save_topic_intimacy,
)


# --- get_intimacy_level ---

class TestGetIntimacyLevel:
    def test_l1(self):
        result = get_intimacy_level(50)
        assert result["level"] == "L1"
        assert result["label"] == "初识"

    def test_l2(self):
        result = get_intimacy_level(200)
        assert result["level"] == "L2"
        assert result["label"] == "熟悉"

    def test_l3(self):
        result = get_intimacy_level(450)
        assert result["level"] == "L3"
        assert result["label"] == "亲近"

    def test_l4(self):
        result = get_intimacy_level(700)
        assert result["level"] == "L4"
        assert result["label"] == "信任"

    def test_l5(self):
        result = get_intimacy_level(900)
        assert result["level"] == "L5"
        assert result["label"] == "挚友"

    def test_boundary_100(self):
        # 100 is L2 (>=100 and <300)
        result = get_intimacy_level(100)
        assert result["level"] == "L2"

    def test_zero(self):
        result = get_intimacy_level(0)
        assert result["level"] == "L1"

    def test_above_1000(self):
        result = get_intimacy_level(1500)
        assert result["level"] == "L5"


# --- get_topic_depth ---

class TestGetTopicDepth:
    def test_shallow(self):
        result = get_topic_depth(10)
        assert result["depth"] == "浅层"

    def test_medium(self):
        result = get_topic_depth(35)
        assert result["depth"] == "中层"

    def test_deep(self):
        result = get_topic_depth(65)
        assert result["depth"] == "深层"

    def test_core(self):
        result = get_topic_depth(90)
        assert result["depth"] == "核心"

    def test_above_100(self):
        result = get_topic_depth(120)
        assert result["depth"] == "核心"


# --- _compute_relationship_duration ---

class TestComputeRelationshipDuration:
    """G3 sigmoid: min(1000, 1000/(1+e^(-0.1*(days-30)))) / 1000"""

    def test_zero_days(self):
        now = datetime.now(UTC)
        result = _compute_relationship_duration(now)
        # t=0 → 1000/(1+e^3) ≈ 47.4 → 0.047
        assert 0.04 < result < 0.06

    def test_30_days(self):
        """t=30 → sigmoid midpoint → ~0.5"""
        created = datetime.now(UTC) - timedelta(days=30)
        result = _compute_relationship_duration(created)
        assert abs(result - 0.5) < 0.01

    def test_60_days(self):
        """t=60 → ~0.95"""
        created = datetime.now(UTC) - timedelta(days=60)
        result = _compute_relationship_duration(created)
        assert result > 0.9

    def test_365_days_near_1(self):
        created = datetime.now(UTC) - timedelta(days=365)
        result = _compute_relationship_duration(created)
        assert result > 0.99


# --- Redis cache (using patch_intimacy_redis fixture) ---

@pytest.mark.asyncio
class TestIntimacyRedis:
    async def test_get_cached_intimacy_none(self, patch_intimacy_redis):
        patch_intimacy_redis.get.return_value = None
        result = await get_cached_intimacy("agent1", "user1")
        assert result is None

    async def test_get_cached_intimacy_valid(self, patch_intimacy_redis):
        data = {"growth": 500, "level": {"level": "L3"}}
        patch_intimacy_redis.get.return_value = json.dumps(data)
        result = await get_cached_intimacy("agent1", "user1")
        assert result["growth"] == 500

    async def test_save_intimacy(self, patch_intimacy_redis):
        await save_intimacy("agent1", "user1", {"growth": 100})
        patch_intimacy_redis.set.assert_called_once()

    async def test_get_topic_intimacy_default(self, patch_intimacy_redis):
        patch_intimacy_redis.get.return_value = None
        val = await get_topic_intimacy("agent1", "user1")
        assert val == 0.0

    async def test_get_topic_intimacy_existing(self, patch_intimacy_redis):
        patch_intimacy_redis.get.return_value = "42.5"
        val = await get_topic_intimacy("agent1", "user1")
        assert val == 42.5

    async def test_save_topic_intimacy(self, patch_intimacy_redis):
        await save_topic_intimacy("agent1", "user1", 55.0)
        patch_intimacy_redis.set.assert_called_once()


# --- compute_growth_intimacy ---

@pytest.mark.asyncio
class TestComputeGrowthIntimacy:
    async def test_no_activity(self, mock_db, patch_intimacy_redis):
        """Zero conversations → score 0 (only duration contributes)."""
        mock_db.conversation.find_many = AsyncMock(return_value=[])

        created_at = datetime.now(UTC) - timedelta(days=30)

        with patch("app.services.relationship.intimacy.db", mock_db):
            score = await compute_growth_intimacy("agent1", "user1", created_at)
            # Only duration contributes (0.4 weight)
            assert 0 < score < 400


# --- get_intimacy_data ---

@pytest.mark.asyncio
class TestGetIntimacyData:
    async def test_returns_full_structure(self, patch_intimacy_redis):
        cached = {"growth": 300, "level": {"level": "L3", "label": "亲近", "score": 300}}
        patch_intimacy_redis.get.side_effect = [
            json.dumps(cached),  # intimacy cache
            "45.0",              # topic intimacy
        ]
        data = await get_intimacy_data("agent1", "user1")
        assert data["growth_intimacy"] == 300
        assert data["topic_intimacy"] == 45.0
        assert data["level"]["level"] == "L3"
        assert data["topic_depth"]["depth"] == "中层"

    async def test_no_cache(self, patch_intimacy_redis):
        patch_intimacy_redis.get.side_effect = [None, None]
        data = await get_intimacy_data("agent1", "user1")
        assert data["growth_intimacy"] == 0
        assert data["topic_intimacy"] == 0.0
        assert data["level"]["level"] == "L1"
