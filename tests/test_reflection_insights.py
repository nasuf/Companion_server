"""反思归纳层的守卫.

这是记忆系统里唯一会写入**推断**的路径 —— 其他记忆都源自某人真的说过的话。所以
这里守的东西跟别处不同: 不是"有没有算错", 而是"模型有没有超出给它的观察乱推", 以及
"推出来的东西会不会被 AI 当着用户的面说出来"。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.reflection import reflect
from app.services.memory.reflection.reflect import (
    MAX_INSIGHTS_PER_RUN,
    MIN_FACTS_TO_REFLECT,
    REFLECTION_IMPORTANCE,
    _parse_insights,
    generate_insights,
    reflect_for_user,
    reflection_enabled_for,
)
from app.services.memory.reflection.signals import BehaviouralFact


def _facts(n: int = 4) -> list[BehaviouralFact]:
    return [
        BehaviouralFact(key=f"k{i}", statement=f"观察 {i}", sample_size=50)
        for i in range(1, n + 1)
    ]


class TestCitationIsTheMainDefence:
    """模型能编出一句听起来有道理的判断, 但编不出一个指向具体观察的有效引用 ——
    除非那条判断真的是从观察来的。"""

    def test_insight_without_citation_is_dropped(self):
        raw = {"insights": [{"text": "他看起来是个很温和的人", "based_on": []}]}
        assert _parse_insights(raw, fact_count=4) == []

    def test_citation_pointing_at_a_nonexistent_fact_is_dropped(self):
        raw = {"insights": [{"text": "他晚上更愿意聊深一点的话题", "based_on": [9]}]}
        assert _parse_insights(raw, fact_count=4) == []

    def test_valid_citation_survives(self):
        raw = {"insights": [{"text": "他晚上更愿意聊深一点的话题", "based_on": [1, 3]}]}
        parsed = _parse_insights(raw, fact_count=4)
        assert len(parsed) == 1
        assert parsed[0].based_on == [1, 3]

    def test_partial_citations_are_kept_after_filtering(self):
        """引用里混了越界的编号时保留有效的部分, 全越界才丢。"""
        raw = {"insights": [{"text": "他习惯用很多短句而不是长段落", "based_on": [2, 99]}]}
        parsed = _parse_insights(raw, fact_count=4)
        assert parsed[0].based_on == [2]


class TestOutputHygiene:
    def test_output_is_capped(self):
        raw = {"insights": [
            {"text": f"判断 {i} 的内容够长了", "based_on": [1]} for i in range(10)
        ]}
        assert len(_parse_insights(raw, fact_count=4)) == MAX_INSIGHTS_PER_RUN

    @pytest.mark.parametrize("text", ["短", "x" * 200])
    def test_absurd_lengths_are_dropped(self, text):
        raw = {"insights": [{"text": text, "based_on": [1]}]}
        assert _parse_insights(raw, fact_count=4) == []

    def test_malformed_output_yields_nothing(self):
        for raw in (None, "", "not json", {"wrong": []}, {"insights": "x"}, 42):
            assert _parse_insights(raw, fact_count=4) == []

    def test_json_string_is_accepted(self):
        raw = '{"insights": [{"text": "他习惯用很多短句而不是长段落", "based_on": [1]}]}'
        assert len(_parse_insights(raw, fact_count=4)) == 1


class TestThePromptForbidsSayingItOutLoud:
    """产出只用来调整语气和话题选择。「我注意到你总在深夜找我」会让人觉得被监视。"""

    def test_prompt_states_the_insights_are_never_spoken(self):
        from app.services.prompting.defaults import MEMORY_REFLECTION_PROMPT

        assert "绝对不会说给用户听" in MEMORY_REFLECTION_PROMPT

    def test_prompt_carries_a_worked_counter_example(self):
        """光说"不要说出口"不够 —— 反例才让模型分得清两种写法。"""
        from app.services.prompting.defaults import MEMORY_REFLECTION_PROMPT

        assert "我注意到你总在晚上找我" in MEMORY_REFLECTION_PROMPT

    def test_prompt_forbids_health_and_财务_inferences(self):
        """行为规律推不出健康或经济状况, 而这类猜测既不准也冒犯。"""
        from app.services.prompting.defaults import MEMORY_REFLECTION_PROMPT

        for forbidden in ("健康", "心理疾病", "经济状况", "感情状态"):
            assert forbidden in MEMORY_REFLECTION_PROMPT

    def test_prompt_requires_hedged_wording(self):
        from app.services.prompting.defaults import MEMORY_REFLECTION_PROMPT

        assert "看起来/可能/大概" in MEMORY_REFLECTION_PROMPT

    def test_prompt_only_receives_facts_not_raw_messages(self):
        """给原始消息就等于允许它跳过验证过的事实自己推断。"""
        from app.services.prompting.defaults import MEMORY_REFLECTION_PROMPT

        assert "{facts}" in MEMORY_REFLECTION_PROMPT
        for leak in ("{messages}", "{conversation}", "{history}"):
            assert leak not in MEMORY_REFLECTION_PROMPT


class TestWhereInsightsLand:
    @pytest.mark.asyncio
    async def test_insights_land_in_l2_never_l1(self):
        """L1 永不衰减 —— 一条错误推断进去就是永久的人设污染。让它进 L2, 有用的
        会因为被反复检索而自己升上去。"""
        stored: list[dict] = []

        async def _store(**kwargs):
            stored.append(kwargs)
            return "mem-1"

        with patch.object(
            reflect, "collect_behavioural_facts", AsyncMock(return_value=_facts()),
        ), patch.object(
            reflect, "generate_insights",
            AsyncMock(return_value=[reflect.Insight("他习惯用短句表达", [1])]),
        ), patch.object(reflect, "store_memory", _store):
            await reflect_for_user(user_id="u", agent_id="a", workspace_id="w")

        assert stored[0]["level"] == 2
        assert stored[0]["importance"] == REFLECTION_IMPORTANCE
        assert stored[0]["importance"] < 0.85, "落进了 L1 区间"

    @pytest.mark.asyncio
    async def test_insights_are_tagged_as_reflected(self):
        """provenance 是它豁免有损压缩、以及日后可整体撤销的唯一依据。"""
        from app.services.memory.provenance import REFLECTED

        stored: list[dict] = []

        async def _store(**kwargs):
            stored.append(kwargs)
            return "mem-1"

        with patch.object(
            reflect, "collect_behavioural_facts", AsyncMock(return_value=_facts()),
        ), patch.object(
            reflect, "generate_insights",
            AsyncMock(return_value=[reflect.Insight("他习惯用短句表达", [1])]),
        ), patch.object(reflect, "store_memory", _store):
            await reflect_for_user(user_id="u", agent_id="a", workspace_id="w")

        assert stored[0]["provenance"] == REFLECTED

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing_but_still_reports(self):
        """开 flag 前先看它会产出什么。"""
        store = AsyncMock()
        with patch.object(
            reflect, "collect_behavioural_facts", AsyncMock(return_value=_facts()),
        ), patch.object(
            reflect, "generate_insights",
            AsyncMock(return_value=[reflect.Insight("他习惯用短句表达", [1])]),
        ), patch.object(reflect, "store_memory", store):
            stats = await reflect_for_user(
                user_id="u", agent_id="a", workspace_id="w", dry_run=True,
            )

        store.assert_not_awaited()
        assert stats["insights"] == 1
        assert stats["stored"] == 0
        assert stats["preview"][0]["based_on"] == ["k1"], "预览要能看到依据"


class TestInsightsNeverMutateRealMemories:
    """反思写的是推断。它绝不能改到用户真的说过的话 —— 那是把推断伪装成陈述,
    而且原文没有留存。"""

    @pytest.mark.asyncio
    async def test_store_skips_reconciliation(self):
        """reconciliation 只写保护 profile_seed / knowledge_seed。一条 user_stated
        的 L2 记忆会被 update_existing / merge_existing 直接改写内容。"""
        stored: list[dict] = []

        async def _store(**kwargs):
            stored.append(kwargs)
            return "mem-1"

        with patch.object(
            reflect, "collect_behavioural_facts", AsyncMock(return_value=_facts()),
        ), patch.object(
            reflect, "generate_insights",
            AsyncMock(return_value=[reflect.Insight("他习惯用短句表达", [1])]),
        ), patch.object(
            reflect, "_existing_reflection_texts", AsyncMock(return_value=set()),
        ), patch.object(reflect, "store_memory", _store):
            await reflect_for_user(user_id="u", agent_id="a", workspace_id="w")

        assert stored[0]["skip_reconciliation"] is True

    @pytest.mark.asyncio
    async def test_repeated_insight_is_skipped_not_merged(self):
        """同一判断周复一周地生成是预期内的。去重只在自己产出的行之间做, 不去碰
        别人的记忆。"""
        store = AsyncMock(return_value="mem-1")
        with patch.object(
            reflect, "collect_behavioural_facts", AsyncMock(return_value=_facts()),
        ), patch.object(
            reflect, "generate_insights",
            AsyncMock(return_value=[reflect.Insight("他习惯用短句表达", [1])]),
        ), patch.object(
            reflect, "_existing_reflection_texts",
            AsyncMock(return_value={"他习惯用短句表达"}),
        ), patch.object(reflect, "store_memory", store):
            stats = await reflect_for_user(user_id="u", agent_id="a", workspace_id="w")

        store.assert_not_awaited()
        assert stats["stored"] == 0

    @pytest.mark.asyncio
    async def test_dedup_lookup_only_matches_reflected_rows(self):
        from app.services.memory.provenance import REFLECTED

        captured: dict = {}

        async def _query(sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return []

        with patch.object(reflect.db, "query_raw", _query):
            await reflect._existing_reflection_texts(user_id="u", workspace_id="w")

        assert "provenance = $3" in captured["sql"]
        assert REFLECTED in captured["args"]
        assert "workspace_id IS NOT DISTINCT FROM" in captured["sql"]


class TestRefusesToWorkWithoutMaterial:
    @pytest.mark.asyncio
    async def test_too_few_facts_skips_the_llm_entirely(self):
        """三条观察推不出东西, 硬调只会逼它编。"""
        llm = AsyncMock()
        with patch.object(reflect, "invoke_json", llm):
            assert await generate_insights(_facts(MIN_FACTS_TO_REFLECT - 1)) == []
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_prompt_yields_nothing(self):
        from app.services.prompting.store import PromptDisabledError

        with patch.object(
            reflect, "get_prompt_text", AsyncMock(side_effect=PromptDisabledError("x")),
        ):
            assert await generate_insights(_facts()) == []

    @pytest.mark.asyncio
    async def test_llm_failure_is_contained(self):
        with patch.object(
            reflect, "get_prompt_text", AsyncMock(return_value="{facts}{max_insights}"),
        ), patch.object(
            reflect, "invoke_json", AsyncMock(side_effect=RuntimeError("boom")),
        ), patch.object(reflect, "get_utility_model", lambda: object()):
            assert await generate_insights(_facts()) == []


class TestRollout:
    def test_disabled_by_default(self):
        """唯一会写入推断的路径, 默认不能开。"""
        from app.config import Settings

        assert Settings.model_fields["memory_reflection_enabled"].default is False

    def test_allowlist_narrows_the_blast_radius(self):
        from app.config import settings

        with patch.object(settings, "memory_reflection_enabled", True), \
             patch.object(settings, "memory_reflection_workspaces", "ws-a, ws-b"):
            assert reflection_enabled_for("ws-a") is True
            assert reflection_enabled_for("ws-c") is False

    def test_empty_allowlist_means_everyone_once_enabled(self):
        from app.config import settings

        with patch.object(settings, "memory_reflection_enabled", True), \
             patch.object(settings, "memory_reflection_workspaces", ""):
            assert reflection_enabled_for("any-ws") is True

    def test_master_switch_beats_the_allowlist(self):
        from app.config import settings

        with patch.object(settings, "memory_reflection_enabled", False), \
             patch.object(settings, "memory_reflection_workspaces", "ws-a"):
            assert reflection_enabled_for("ws-a") is False
