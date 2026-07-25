"""Guards for the utility-model benchmark's non-LLM parts.

The benchmark decides which model runs on the hot path, so a silent break in
its parsers would let a worse model look identical to a better one.
"""

from __future__ import annotations

import pytest

from evals.utility_model.run_eval import PRICING, cost_cny
from evals.utility_model.tasks import ALL_TASKS


class TestTaskBank:
    def test_every_case_expects_a_declared_label(self):
        """标注了词表以外的答案 = 该用例永远判错, 会静默拉低所有模型."""
        for task in ALL_TASKS:
            for case in task.cases:
                assert case.expected in task.labels, f"{task.name}: {case.expected}"

    def test_每个任务的标签都被覆盖到(self):
        """只测一种答案的任务测不出偏置 (全答"记"也能拿高分)."""
        for task in ALL_TASKS:
            covered = {c.expected for c in task.cases}
            assert len(covered) >= 2, f"{task.name} 只覆盖了 {covered}"

    def test_params_fill_every_placeholder_the_prompt_needs(self):
        keys_by_task = {
            "记忆预筛": {"message"},
            "记忆相关度": {"message", "context"},
            "意图识别": {"user_message", "context"},
            "联网判定": {"message", "context"},
        }
        for task in ALL_TASKS:
            got = set(task.params(task.cases[0]))
            assert got == keys_by_task[task.name], f"{task.name}: {got}"


class TestParsers:
    def test_memorize_negative_checked_before_positive(self):
        """「不记」含「记」—— 先判否定, 否则永远解析成"记"."""
        task = next(t for t in ALL_TASKS if t.name == "记忆预筛")
        assert task.parse("不记") == "不记"
        assert task.parse("记") == "记"
        assert task.parse("这句话应该不记") == "不记"

    def test_relevance_reads_json_level(self):
        task = next(t for t in ALL_TASKS if t.name == "记忆相关度")
        assert task.parse('{"level": "强", "enhanced_query": "x"}') == "强"
        assert task.parse('前缀 {"level":"弱","enhanced_query":""} 后缀') == "弱"
        assert task.parse("完全答非所问") is None

    def test_intent_prefers_longest_label(self):
        """短标签是长标签的子串时不能抢先命中."""
        task = next(t for t in ALL_TASKS if t.name == "意图识别")
        assert task.parse("调用久远记忆") == "调用久远记忆"
        assert task.parse("记录请求、日常交流") == "记录请求"

    def test_web_search_negative_first(self):
        task = next(t for t in ALL_TASKS if t.name == "联网判定")
        assert task.parse("不需要联网") == "不需要联网"
        assert task.parse("需要联网") == "需要联网"

    @pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.name)
    def test_unparseable_output_is_none(self, task):
        assert task.parse("") is None


class TestCost:
    def test_three_tier_pricing_uses_cache_rate(self):
        rows = [{"input_tokens": 1000, "output_tokens": 10, "cached_tokens": 800}]
        # 200 未命中 ×0.2 + 800 命中 ×0.04 + 10 输出 ×2.0, 全部 /1e6
        expected = (200 * 0.2 + 800 * 0.04 + 10 * 2.0) / 1_000_000
        assert cost_cny("doubao-seed-2-0-mini-260428", rows) == pytest.approx(expected)

    def test_cached_never_exceeds_input(self):
        """provider 偶尔报出 cached > input; 不夹取会把成本算成负数."""
        rows = [{"input_tokens": 100, "output_tokens": 0, "cached_tokens": 999}]
        assert cost_cny("doubao-seed-2-0-mini-260428", rows) == pytest.approx(
            100 * 0.04 / 1_000_000
        )

    def test_unknown_model_reports_none_not_zero(self):
        """未配价必须是"不知道", 报 0 会让人以为这个模型免费."""
        assert cost_cny("some-model-we-never-priced", [{"input_tokens": 1}]) is None

    def test_current_production_small_model_is_priced(self):
        assert "qwen3.5-flash" in PRICING
