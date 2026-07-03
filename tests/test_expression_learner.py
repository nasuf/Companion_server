"""Phase E3 回归: 表达学习 MVP (Redis 存储版)."""

from __future__ import annotations

import json
import random
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.expression_learner import (
    LEARN_EVERY_N,
    MAX_EXPRESSIONS,
    _validate_items,
    bump_message_counter,
    learn_expressions,
    load_expressions,
    merge_expressions,
    sample_expression_habits,
    weighted_sample,
)

P = "app.services.chat.expression_learner"


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key, ttl):
        pass

    async def delete(self, key):
        self.counters.pop(key, None)
        self.store.pop(key, None)

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class TestValidation:
    def test_accepts_valid_pairs(self):
        items = [{"situation": "表示惊叹", "style": "用 我嘞个xxx"}]
        assert _validate_items(items) == items

    def test_rejects_long_and_empty_fields(self):
        assert _validate_items([{"situation": "x" * 31, "style": "ok"}]) == []
        assert _validate_items([{"situation": "", "style": "ok"}]) == []
        assert _validate_items("not-a-list") == []

    def test_unwraps_dict_with_expressions_key(self):
        wrapped = {"expressions": [{"situation": "s", "style": "y"}]}
        assert len(_validate_items(wrapped)) == 1


class TestMerge:
    def test_duplicate_style_bumps_count(self):
        existing = [{"situation": "旧场景", "style": "对对对", "count": 2}]
        merged = merge_expressions(existing, [{"situation": "新场景", "style": "对对对"}])
        assert len(merged) == 1
        assert merged[0]["count"] == 3
        assert merged[0]["situation"] == "新场景"  # 场景描述取最新

    def test_cap_evicts_lowest_count(self):
        existing = [
            {"situation": f"s{i}", "style": f"style{i}", "count": i}
            for i in range(1, MAX_EXPRESSIONS + 1)
        ]
        merged = merge_expressions(
            existing, [{"situation": "new", "style": "newstyle"}],
        )
        assert len(merged) == MAX_EXPRESSIONS
        styles = {e["style"] for e in merged}
        assert "style1" not in styles  # count=1 的最低频被淘汰


class TestWeightedSample:
    def test_no_replacement_and_k_bound(self):
        exprs = [{"situation": f"s{i}", "style": f"y{i}", "count": 1} for i in range(5)]
        picked = weighted_sample(exprs, 3, rng=random.Random(1))
        assert len(picked) == 3
        assert len({p["style"] for p in picked}) == 3

    def test_high_count_favored(self):
        exprs = [
            {"situation": "a", "style": "hot", "count": 50},
            {"situation": "b", "style": "cold", "count": 1},
        ]
        r = random.Random(7)
        first_picks = [weighted_sample(exprs, 1, rng=r)[0]["style"] for _ in range(200)]
        assert first_picks.count("hot") > 150


@pytest.mark.asyncio
class TestCounterThrottle:
    async def test_learns_only_every_n_messages(self):
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            hits = [await bump_message_counter("a1", "u1") for _ in range(LEARN_EVERY_N)]
        assert hits.count(True) == 1 and hits[-1] is True  # 恰好第 N 条触发

    async def test_redis_failure_returns_false(self):
        with patch(f"{P}.get_redis", AsyncMock(side_effect=RuntimeError("down"))):
            assert await bump_message_counter("a1", "u1") is False


@pytest.mark.asyncio
class TestLearnAndSample:
    async def test_learn_stores_and_sample_renders(self):
        fake = _FakeRedis()
        llm_out = [{"situation": "表示赞同", "style": "用 对对对"}]
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=fake)),
            patch(f"{P}.get_prompt_text", AsyncMock(return_value="{conversation}")),
            patch(f"{P}.invoke_json", AsyncMock(return_value=llm_out)),
            patch(f"{P}.get_utility_model"),
        ):
            n = await learn_expressions("a1", "u1", [
                {"role": "user", "content": f"消息{i}"} for i in range(5)
            ])
            assert n == 1
            stored = await load_expressions("a1", "u1")
            assert stored[0]["style"] == "用 对对对"
            habits = await sample_expression_habits("a1", "u1")
        assert habits == ["当「表示赞同」时，可以「用 对对对」"]

    async def test_too_few_user_messages_skips_llm(self):
        with patch(f"{P}.invoke_json", AsyncMock()) as llm:
            n = await learn_expressions("a1", "u1", [
                {"role": "user", "content": "嗯"},
                {"role": "assistant", "content": "好"},
            ])
        assert n == 0
        llm.assert_not_called()

    async def test_llm_failure_returns_zero(self):
        with (
            patch(f"{P}.get_prompt_text", AsyncMock(return_value="{conversation}")),
            patch(f"{P}.invoke_json", AsyncMock(side_effect=RuntimeError("timeout"))),
            patch(f"{P}.get_utility_model"),
        ):
            n = await learn_expressions("a1", "u1", [
                {"role": "user", "content": f"消息{i}"} for i in range(5)
            ])
        assert n == 0

    async def test_sample_without_agent_returns_empty(self):
        assert await sample_expression_habits(None, "u1") == []


@pytest.mark.asyncio
async def test_prompt_builder_renders_expression_section():
    from app.services.chat.prompt_builder import _build_expression_habits_section
    from app.services.prompting import defaults as d

    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        AsyncMock(return_value=d.CHAT_EXPRESSION_HABITS_SECTION_PROMPT),
    ):
        section = await _build_expression_habits_section(
            ["当「表示赞同」时，可以「用 对对对」"],
        )
    assert section is not None
    assert section.prompt_key == "chat.expression_habits_section"
    assert "不要照抄" in section.body
    assert "对对对" in section.body

    # 空 habits → 不注入
    section_empty = await _build_expression_habits_section([])
    assert section_empty is None
