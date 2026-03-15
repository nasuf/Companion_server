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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.intimacy import (
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
    def test_zero_days(self):
        now = datetime.now(UTC)
        result = _compute_relationship_duration(now)
        assert 0 <= result < 0.01

    def test_30_days(self):
        created = datetime.now(UTC) - timedelta(days=30)
        result = _compute_relationship_duration(created)
        expected = math.log1p(30) / math.log1p(180)
        assert abs(result - expected) < 0.01

    def test_180_days(self):
        created = datetime.now(UTC) - timedelta(days=180)
        result = _compute_relationship_duration(created)
        expected = math.log1p(180) / math.log1p(180)
        assert abs(result - expected) < 0.01  # ~1.0

    def test_365_days_capped(self):
        created = datetime.now(UTC) - timedelta(days=365)
        result = _compute_relationship_duration(created)
        assert result == 1.0  # capped at 1.0


# --- Redis cache ---

@pytest.mark.asyncio
class TestIntimacyRedis:
    async def test_get_cached_intimacy_none(self, mock_redis):
        mock_redis.get.return_value = None
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            result = await get_cached_intimacy("agent1", "user1")
            assert result is None

    async def test_get_cached_intimacy_valid(self, mock_redis):
        data = {"growth": 500, "level": {"level": "L3"}}
        mock_redis.get.return_value = json.dumps(data)
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            result = await get_cached_intimacy("agent1", "user1")
            assert result["growth"] == 500

    async def test_save_intimacy(self, mock_redis):
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            await save_intimacy("agent1", "user1", {"growth": 100})
            mock_redis.set.assert_called_once()

    async def test_get_topic_intimacy_default(self, mock_redis):
        mock_redis.get.return_value = None
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            val = await get_topic_intimacy("agent1", "user1")
            assert val == 0.0

    async def test_get_topic_intimacy_existing(self, mock_redis):
        mock_redis.get.return_value = "42.5"
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            val = await get_topic_intimacy("agent1", "user1")
            assert val == 42.5

    async def test_save_topic_intimacy(self, mock_redis):
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            await save_topic_intimacy("agent1", "user1", 55.0)
            mock_redis.set.assert_called_once()


# --- compute_growth_intimacy ---

@pytest.mark.asyncio
class TestComputeGrowthIntimacy:
    async def test_no_activity(self, mock_db, mock_redis):
        """Zero conversations → score 0 (only duration contributes)."""
        mock_db.conversation.find_many = AsyncMock(return_value=[])
        mock_db.memory.count = AsyncMock(return_value=0)

        created_at = datetime.now(UTC) - timedelta(days=30)

        with (
            patch("app.services.intimacy.db", mock_db),
            patch("app.services.intimacy.get_redis", return_value=mock_redis),
        ):
            score = await compute_growth_intimacy("agent1", "user1", created_at)
            # Only duration contributes (0.4 weight)
            assert 0 < score < 400


# --- get_intimacy_data ---

@pytest.mark.asyncio
class TestGetIntimacyData:
    async def test_returns_full_structure(self, mock_redis):
        cached = {"growth": 300, "level": {"level": "L3", "label": "亲近", "score": 300}}
        mock_redis.get.side_effect = [
            json.dumps(cached),  # intimacy cache
            "45.0",              # topic intimacy
        ]
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            data = await get_intimacy_data("agent1", "user1")
            assert data["growth_intimacy"] == 300
            assert data["topic_intimacy"] == 45.0
            assert data["level"]["level"] == "L3"
            assert data["topic_depth"]["depth"] == "中层"

    async def test_no_cache(self, mock_redis):
        mock_redis.get.side_effect = [None, None]
        with patch("app.services.intimacy.get_redis", return_value=mock_redis):
            data = await get_intimacy_data("agent1", "user1")
            assert data["growth_intimacy"] == 0
            assert data["topic_intimacy"] == 0.0
            assert data["level"]["level"] == "L1"
