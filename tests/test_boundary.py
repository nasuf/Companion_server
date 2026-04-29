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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.interaction.boundary import (
    PATIENCE_MAX,
    PATIENCE_NORMAL_MIN,
    _BOUNDARY_RESPONSES,
    adjust_patience,
    check_banned_keywords,
    check_boundary,
    compute_repeat_deduction,
    detect_apology,
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

    def test_no_match_innocent(self):
        # Innocent message should not match any banned words
        assert check_banned_keywords("今天天气真好") == []


# --- generate_boundary_response ---

class TestGenerateBoundaryResponse:
    def test_normal_zone(self):
        resp = generate_boundary_response("normal")
        assert resp in _BOUNDARY_RESPONSES["normal"]

    def test_blocked_zone(self):
        resp = generate_boundary_response("blocked")
        assert resp in _BOUNDARY_RESPONSES["blocked"]

    def test_unknown_zone_falls_back_to_normal(self):
        resp = generate_boundary_response("unknown")
        assert resp in _BOUNDARY_RESPONSES["normal"]


# --- Redis CRUD (mocked via patch_boundary_redis fixture) ---

@pytest.mark.asyncio
class TestPatienceRedis:
    async def test_get_patience_default(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = None
        val = await get_patience("agent1", "user1")
        assert val == PATIENCE_MAX

    async def test_get_patience_existing(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "55"
        val = await get_patience("agent1", "user1")
        assert val == 55

    async def test_set_patience_clamp_max(self, patch_boundary_redis):
        val = await set_patience("agent1", "user1", 150)
        assert val == PATIENCE_MAX
        patch_boundary_redis.set.assert_called_once()

    async def test_set_patience_clamp_min(self, patch_boundary_redis):
        val = await set_patience("agent1", "user1", -20)
        assert val == 0

    async def test_adjust_patience(self, patch_boundary_redis):
        # 现在 adjust_patience 走 Redis Lua, eval 返结果即新值
        patch_boundary_redis.eval = AsyncMock(return_value=65)
        val = await adjust_patience("agent1", "user1", -15)
        assert val == 65
        # 验证 Lua script 被调用 + 关键词 KEYS/ARGV 顺序
        patch_boundary_redis.eval.assert_awaited_once()
        call_args = patch_boundary_redis.eval.await_args.args
        assert call_args[1] == 1  # numkeys
        assert "patience" in call_args[2]  # KEYS[1] = patience key
        assert call_args[3] == "-15"  # ARGV[1] = delta
        assert call_args[4] == "100"  # ARGV[2] = PATIENCE_MAX

    async def test_recover_hourly_normal(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "50"
        val = await recover_patience_hourly("agent1", "user1")
        assert val == 60  # spec §2.5: 每小时 +10

    async def test_recover_hourly_skip_max(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "100"
        val = await recover_patience_hourly("agent1", "user1")
        assert val == 100
        patch_boundary_redis.set.assert_not_called()

    async def test_recover_hourly_skip_blocked(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "0"
        val = await recover_patience_hourly("agent1", "user1")
        assert val == 0
        patch_boundary_redis.set.assert_not_called()

    async def test_recover_hourly_no_record(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = None
        val = await recover_patience_hourly("agent1", "user1")
        assert val == PATIENCE_MAX


# --- handle_apology ---

@pytest.mark.asyncio
class TestHandleApology:
    async def test_apology_from_blocked_restores_to_70(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "0"
        val = await handle_apology("agent1", "user1")
        assert val == PATIENCE_NORMAL_MIN  # 70

    async def test_apology_from_low_adds_70(self, patch_boundary_redis):
        # 非拉黑分支走 adjust_patience (Lua eval); mock eval 返新值
        patch_boundary_redis.get.return_value = "20"
        patch_boundary_redis.eval = AsyncMock(return_value=90)
        val = await handle_apology("agent1", "user1")
        assert val == 90  # spec §2.5: 20 + 70

    async def test_apology_from_high_caps_at_100(self, patch_boundary_redis):
        # Lua 内部 clamp 到 PATIENCE_MAX, mock 直接返 100
        patch_boundary_redis.get.return_value = "80"
        patch_boundary_redis.eval = AsyncMock(return_value=100)
        val = await handle_apology("agent1", "user1")
        assert val == 100  # min(80 + 70, 100)


# --- check_boundary (热路径) ---

@pytest.mark.asyncio
class TestCheckBoundary:
    async def test_no_banned_words_returns_none(self, patch_boundary_redis):
        result, patience = await check_boundary("agent1", "user1", "你好")
        assert result is None
        assert patience == 100

    async def test_banned_word_normal_zone(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "80"
        result, patience = await check_boundary("agent1", "user1", "你这个垃圾AI")
        assert result is not None
        assert result["blocked"] is False
        assert result["zone"] == "normal"
        assert "垃圾AI" in result["hits"]
        assert patience == 80

    async def test_banned_word_blocked_zone(self, patch_boundary_redis):
        patch_boundary_redis.get.return_value = "0"
        result, patience = await check_boundary("agent1", "user1", "你这个垃圾AI")
        assert result is not None
        assert result["blocked"] is True
        assert result["zone"] == "blocked"
        assert patience == 0

    async def test_clean_message_blocked_zone(self, patch_boundary_redis):
        """Blocked users are intercepted even with clean messages (PRD §6.5.2.4)."""
        patch_boundary_redis.get.return_value = "0"
        result, patience = await check_boundary("agent1", "user1", "你好")
        assert result is not None
        assert result["blocked"] is True
        assert result["zone"] == "blocked"
        assert patience == 0

    async def test_apology_keyword_blocked_still_returns_blocked_signal(
        self, patch_boundary_redis,
    ):
        """spec §2: blocked 态下道歉关键词不再走热路径捷径放行 ——
        必须返回 blocked signal, 让 _handle_blocked 调 LLM 真诚度判断 +
        handle_apology 写回 patience. 老捷径会让 patience 留在 0, 下条又拒.
        """
        patch_boundary_redis.get.return_value = "0"
        result, patience = await check_boundary("agent1", "user1", "对不起我错了")
        assert result is not None
        assert result["blocked"] is True
        assert result["zone"] == "blocked"
        assert patience == 0


# --- record_attack: 24h ZSET 滚动窗口 (spec §2.4) ---

@pytest.mark.asyncio
class TestRecordAttackZset:
    async def _run_record_attack(self, level, zcard_return):
        """复用模板: mock pipeline + redis.zcard, 调用 record_attack 并返回 (count, pipe, redis)."""
        from app.services.interaction.boundary import record_attack

        pipe = MagicMock()
        pipe.incr = MagicMock()
        pipe.expire = MagicMock()
        pipe.zadd = MagicMock()
        pipe.zremrangebyscore = MagicMock()
        pipe.execute = AsyncMock(return_value=[1, 1, 1, 0, 1])

        redis = AsyncMock()
        redis.pipeline = MagicMock(return_value=pipe)
        redis.zcard = AsyncMock(return_value=zcard_return)

        with patch("app.services.interaction.boundary.get_redis", return_value=redis):
            count = await record_attack("agent1", "user1", level=level)
        return count, pipe, redis

    async def test_record_attack_uses_zset_pipeline(self):
        """K1 攻击应走 ZADD + ZREMRANGEBYSCORE + ZCARD 真滚动窗口流程."""
        count, pipe, redis = await self._run_record_attack("K1", zcard_return=3)
        assert count == 3  # 真实 24h 内累计
        pipe.zadd.assert_called_once()
        pipe.zremrangebyscore.assert_called_once()
        # 验证 cutoff: ZREMRANGEBYSCORE key 0 (now - 86400_000)
        args = pipe.zremrangebyscore.call_args.args
        assert args[1] == 0
        assert isinstance(args[2], int) and args[2] > 0
        redis.zcard.assert_awaited_once()

    async def test_record_attack_cleans_24h_old_entries(self):
        """ZREMRANGEBYSCORE 区间应跨 24h 边界, 老条目被清理后 ZCARD 仅返活跃."""
        # 模拟 ZSET 内原有 5 条, 4 条被清掉, ZCARD 返 1
        count, _pipe, _redis = await self._run_record_attack("K3", zcard_return=1)
        assert count == 1

    async def test_record_attack_no_level_returns_zero(self):
        """无 level (聚合统计 key) 仍走旧 INCR 路径, 返 0, 不走 ZSET."""
        from app.services.interaction.boundary import record_attack

        pipe = MagicMock()
        pipe.incr = MagicMock()
        pipe.expire = MagicMock()
        pipe.zadd = MagicMock()  # 应不被调
        pipe.execute = AsyncMock(return_value=[1, 1])
        redis = AsyncMock()
        redis.pipeline = MagicMock(return_value=pipe)
        redis.zcard = AsyncMock()

        with patch("app.services.interaction.boundary.get_redis", return_value=redis):
            count = await record_attack("agent1", "user1")
        assert count == 0
        pipe.zadd.assert_not_called()
        redis.zcard.assert_not_called()


# --- compute_repeat_deduction: spec §2.4 重复加重公式 ---

class TestComputeRepeatDeduction:
    """⌈base × (1 + 0.5 × (n-1))⌉，受 cap 限制."""

    def test_first_attack_returns_base(self):
        # K1: base=5, n=1 → 5
        assert compute_repeat_deduction("K1", 1) == 5
        assert compute_repeat_deduction("K2", 1) == 15
        assert compute_repeat_deduction("K3", 1) == 40

    def test_second_attack_uses_1_5_multiplier(self):
        # K1: 5 × 1.5 = 7.5 → ceil = 8
        assert compute_repeat_deduction("K1", 2) == 8
        # K2: 15 × 1.5 = 22.5 → ceil = 23 → capped at 25
        assert compute_repeat_deduction("K2", 2) == 23
        # K3: 40 × 1.5 = 60 → capped at 50
        assert compute_repeat_deduction("K3", 2) == 50

    def test_third_attack_uses_2x_multiplier(self):
        # K1: 5 × 2 = 10 → exactly cap
        assert compute_repeat_deduction("K1", 3) == 10

    def test_fourth_attack_caps_at_level_max(self):
        # K1: 5 × 2.5 = 12.5 → ceil = 13 → capped at 10
        assert compute_repeat_deduction("K1", 4) == 10
        # K2: 15 × 2.5 = 37.5 → ceil = 38 → capped at 25
        assert compute_repeat_deduction("K2", 4) == 25
        # K3 早就 cap 在第 2 次 (60 → 50), 后续也是 50
        assert compute_repeat_deduction("K3", 4) == 50

    def test_high_count_stays_at_cap(self):
        for n in (10, 50, 100):
            assert compute_repeat_deduction("K1", n) == 10
            assert compute_repeat_deduction("K2", n) == 25
            assert compute_repeat_deduction("K3", n) == 50

    def test_zero_or_negative_count_treated_as_one(self):
        # n = max(1, count) 内部 clamp
        assert compute_repeat_deduction("K1", 0) == 5
        assert compute_repeat_deduction("K1", -3) == 5

    def test_unknown_level_falls_back_to_k1_base(self):
        assert compute_repeat_deduction("unknown", 1) == 5  # base default
        assert compute_repeat_deduction("unknown", 100) == 10  # cap default


# --- detect_apology LLM 失败兑底 (P2-7 防永久困死) ---

@pytest.mark.asyncio
class TestDetectApologyFallback:
    async def test_llm_failure_with_keyword_grants_threshold_sincerity(self):
        """LLM down + 含道歉关键词 → 返 sincerity=0.5 让 _handle_blocked 阈值通过."""
        with patch(
            "app.services.interaction.boundary.get_prompt_text",
            AsyncMock(return_value="判断{message}"),
        ), patch(
            "app.services.interaction.boundary.invoke_json",
            AsyncMock(side_effect=RuntimeError("LLM timeout")),
        ):
            result = await detect_apology("对不起我错了")
        assert result["is_apology"] is True
        assert result["sincerity"] == 0.5

    async def test_llm_failure_without_keyword_no_unblock(self):
        """LLM down + 无关键词 → 维持原 sincerity=0.0, 不解封."""
        with patch(
            "app.services.interaction.boundary.get_prompt_text",
            AsyncMock(return_value="判断{message}"),
        ), patch(
            "app.services.interaction.boundary.invoke_json",
            AsyncMock(side_effect=RuntimeError("LLM timeout")),
        ):
            result = await detect_apology("我反思了一下")  # 无关键词
        assert result["is_apology"] is False
        assert result["sincerity"] == 0.0

    async def test_llm_success_overrides_keyword(self):
        """LLM 正常返回时 fallback 不触发, 以 LLM 判定为准."""
        with patch(
            "app.services.interaction.boundary.get_prompt_text",
            AsyncMock(return_value="判断{message}"),
        ), patch(
            "app.services.interaction.boundary.invoke_json",
            AsyncMock(return_value={"is_apology": False, "sincerity": 0.1}),
        ):
            result = await detect_apology("对不起")  # 即便有关键词
        assert result["is_apology"] is False
        assert result["sincerity"] == 0.1
