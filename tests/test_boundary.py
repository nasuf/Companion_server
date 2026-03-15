"""边界系统单元测试。

测试覆盖：
- 耐心区间映射
- 违禁词检测
- 边界回复生成
- Redis耐心值CRUD
- 耐心恢复逻辑
- 道歉处理
- 热路径边界检查
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.boundary import (
    PATIENCE_MAX,
    PATIENCE_NORMAL_MIN,
    adjust_patience,
    check_banned_keywords,
    check_boundary,
    generate_boundary_response,
    get_patience,
    get_patience_zone,
    handle_apology,
    recover_patience_hourly,
    set_patience,
)


# --- get_patience_zone ---

class TestGetPatienceZone:
    def test_normal_at_100(self):
        assert get_patience_zone(100) == "normal"

    def test_normal_at_70(self):
        assert get_patience_zone(70) == "normal"

    def test_medium_at_69(self):
        assert get_patience_zone(69) == "medium"

    def test_medium_at_30(self):
        assert get_patience_zone(30) == "medium"

    def test_low_at_29(self):
        assert get_patience_zone(29) == "low"

    def test_low_at_1(self):
        assert get_patience_zone(1) == "low"

    def test_blocked_at_0(self):
        assert get_patience_zone(0) == "blocked"

    def test_blocked_negative(self):
        assert get_patience_zone(-10) == "blocked"


# --- check_banned_keywords ---

class TestCheckBannedKeywords:
    def test_no_match(self):
        assert check_banned_keywords("你好，今天天气不错") == []

    def test_single_match(self):
        hits = check_banned_keywords("你这个垃圾AI")
        assert "垃圾AI" in hits

    def test_multiple_matches(self):
        hits = check_banned_keywords("你这个智障白痴")
        assert "智障" in hits
        assert "白痴" in hits

    def test_empty_message(self):
        assert check_banned_keywords("") == []

    def test_partial_no_match(self):
        # "垃圾" alone should not match "垃圾AI"
        assert check_banned_keywords("垃圾分类") == []


# --- generate_boundary_response ---

class TestGenerateBoundaryResponse:
    def test_normal_zone(self):
        resp = generate_boundary_response("normal")
        assert resp in [
            "嗯...你这么说我有点难过。",
            "这样说话不太好吧...",
            "我们好好聊天吧～",
        ]

    def test_blocked_zone(self):
        resp = generate_boundary_response("blocked")
        assert resp in ["...", "我不想和你说话了。"]

    def test_unknown_zone_falls_back_to_normal(self):
        resp = generate_boundary_response("unknown")
        assert resp in [
            "嗯...你这么说我有点难过。",
            "这样说话不太好吧...",
            "我们好好聊天吧～",
        ]


# --- Redis CRUD (mocked) ---

@pytest.mark.asyncio
class TestPatienceRedis:
    async def test_get_patience_default(self, mock_redis):
        mock_redis.get.return_value = None
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await get_patience("agent1", "user1")
            assert val == PATIENCE_MAX

    async def test_get_patience_existing(self, mock_redis):
        mock_redis.get.return_value = "55"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await get_patience("agent1", "user1")
            assert val == 55

    async def test_set_patience_clamp_max(self, mock_redis):
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await set_patience("agent1", "user1", 150)
            assert val == PATIENCE_MAX
            mock_redis.set.assert_called_once()

    async def test_set_patience_clamp_min(self, mock_redis):
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await set_patience("agent1", "user1", -20)
            assert val == 0

    async def test_adjust_patience(self, mock_redis):
        mock_redis.get.return_value = "80"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await adjust_patience("agent1", "user1", -15)
            assert val == 65

    async def test_recover_hourly_normal(self, mock_redis):
        mock_redis.get.return_value = "50"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await recover_patience_hourly("agent1", "user1")
            assert val == 55

    async def test_recover_hourly_skip_max(self, mock_redis):
        mock_redis.get.return_value = "100"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await recover_patience_hourly("agent1", "user1")
            assert val == 100
            mock_redis.set.assert_not_called()

    async def test_recover_hourly_skip_blocked(self, mock_redis):
        mock_redis.get.return_value = "0"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await recover_patience_hourly("agent1", "user1")
            assert val == 0
            mock_redis.set.assert_not_called()

    async def test_recover_hourly_no_record(self, mock_redis):
        mock_redis.get.return_value = None
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await recover_patience_hourly("agent1", "user1")
            assert val == PATIENCE_MAX


# --- handle_apology ---

@pytest.mark.asyncio
class TestHandleApology:
    async def test_apology_from_blocked_restores_to_70(self, mock_redis):
        mock_redis.get.return_value = "0"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await handle_apology("agent1", "user1")
            assert val == PATIENCE_NORMAL_MIN  # 70

    async def test_apology_from_low_restores_to_60(self, mock_redis):
        mock_redis.get.return_value = "20"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await handle_apology("agent1", "user1")
            assert val == 60

    async def test_apology_from_high_keeps_current(self, mock_redis):
        mock_redis.get.return_value = "80"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            val = await handle_apology("agent1", "user1")
            assert val == 80


# --- check_boundary (热路径) ---

@pytest.mark.asyncio
class TestCheckBoundary:
    async def test_no_banned_words_returns_none(self, mock_redis):
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            result = await check_boundary("agent1", "user1", "你好")
            assert result is None

    async def test_banned_word_normal_zone(self, mock_redis):
        mock_redis.get.return_value = "80"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            result = await check_boundary("agent1", "user1", "你这个垃圾AI")
            assert result is not None
            assert result["blocked"] is False
            assert result["zone"] == "normal"
            assert "垃圾AI" in result["hits"]

    async def test_banned_word_blocked_zone(self, mock_redis):
        mock_redis.get.return_value = "0"
        with patch("app.services.boundary.get_redis", return_value=mock_redis):
            result = await check_boundary("agent1", "user1", "你这个垃圾AI")
            assert result is not None
            assert result["blocked"] is True
            assert result["zone"] == "blocked"
