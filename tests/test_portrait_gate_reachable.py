"""画像的前置条件必须在真实数据分布下可达, 且画像要带上互动行为观察.

这个文件存在的理由: 旧门槛 `L2 ≥ 20 AND L1 ≥ 5` 在生产上**一次都没被满足过**,
user_portraits 表从上线起一直是空的, 而且没有任何报错 —— 前置条件不满足只会记一行
info 日志, 看起来跟"这周没人够格"没有区别。

失效的机制值得记下来: 层级由 importance 推导, 于是这两个数在真实数据里此消彼长。
说身份事实的用户攒 L1, 聊日常的用户攒 L2, 很少有人两样都攒够。两个此消彼长的量
做 AND, 门就关死了。

所以下面用**生产上实际观察到的两种分布**做参数, 而不是随手编的数字。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import portrait as portrait_module
from app.services.portrait import (
    MIN_MEMORIES_FOR_PORTRAIT,
    MIN_USER_MESSAGES_FOR_PORTRAIT,
    check_portrait_preconditions,
)

# 生产实测的两种真实分布 (2026-07-28, 15 个有记忆的用户):
#   身份型  L1=23 L2= 5   —— 用户主动陈述姓名/居住地/生日这类事实
#   日常型  L1= 0 L2=23   —— 用户只聊日常, 没说过身份信息
# 旧门槛下两者都不达标, 各差一个条件。
PRODUCTION_SHAPES = [
    pytest.param(23, 5, id="身份型 L1=23 L2=5"),
    pytest.param(0, 23, id="日常型 L1=0 L2=23"),
    pytest.param(8, 9, id="混合型 L1=8 L2=9"),
]


def _agent(age_hours: float = 48):
    return SimpleNamespace(
        id="agent-1",
        createdAt=datetime.now(UTC) - timedelta(hours=age_hours),
    )


async def _check(l1: int, l2: int, user_messages: int = 100) -> bool:
    """按给定的记忆/消息量跑一次前置检查。"""
    with patch.object(portrait_module, "db") as mock_db, \
         patch.object(
             portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
         ), \
         patch.object(
             portrait_module.memory_repo, "count", AsyncMock(return_value=l1 + l2)
         ):
        mock_db.aiagent = MagicMock(find_unique=AsyncMock(return_value=_agent()))
        mock_db.message = MagicMock(count=AsyncMock(return_value=user_messages))
        return await check_portrait_preconditions("user-1", "agent-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("l1,l2", PRODUCTION_SHAPES)
async def test_real_world_distributions_can_qualify(l1, l2):
    """三种真实形态都该够格 —— 它们的共同点是总量够, 差别只在落在哪一层。"""
    assert await _check(l1, l2) is True, (
        f"L1={l1} L2={l2} (合计 {l1 + l2}) 仍然不够格; "
        "门槛又变成看层级构成了"
    )


@pytest.mark.asyncio
async def test_gate_does_not_depend_on_how_memories_split_across_levels():
    """同样的总量, 无论怎么分布在 L1/L2, 结论必须一致。

    这条直接钉住旧 bug 的成因: 一旦有人再按层级分别设下限, 此消彼长的问题就回来了。
    """
    total = MIN_MEMORIES_FOR_PORTRAIT
    verdicts = {await _check(l1, total - l1) for l1 in range(0, total + 1)}
    assert verdicts == {True}, "同样总量下结论随层级分布变化了"


@pytest.mark.asyncio
async def test_thin_users_are_still_rejected():
    """放宽不等于取消 —— 素材不足时写出来的画像是编的。"""
    assert await _check(2, 3) is False


@pytest.mark.asyncio
async def test_memories_without_conversation_are_rejected():
    """只有记忆没有对话时, 画像会写得像档案摘要而不是对一个人的印象。"""
    assert await _check(50, 50, user_messages=MIN_USER_MESSAGES_FOR_PORTRAIT - 1) is False


@pytest.mark.asyncio
async def test_brand_new_agent_is_rejected():
    with patch.object(portrait_module, "db") as mock_db:
        mock_db.aiagent = MagicMock(
            find_unique=AsyncMock(return_value=_agent(age_hours=2))
        )
        assert await check_portrait_preconditions("user-1", "agent-1") is False


@pytest.mark.asyncio
async def test_counts_recallable_memories_not_a_single_level():
    """确认查询本身问的是 L1+L2, 而不是某一层。"""
    captured: dict = {}

    async def _count(**kwargs):
        captured.update(kwargs)
        return 100

    with patch.object(portrait_module, "db") as mock_db, \
         patch.object(
             portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
         ), \
         patch.object(portrait_module.memory_repo, "count", _count):
        mock_db.aiagent = MagicMock(find_unique=AsyncMock(return_value=_agent()))
        mock_db.message = MagicMock(count=AsyncMock(return_value=100))
        await check_portrait_preconditions("user-1", "agent-1")

    assert captured["where"]["level"] == {"in": [1, 2]}
    assert captured["source"] == "user"


class TestBehaviourFeedsThePortrait:
    """记忆记录用户**说过**的话; 行为观察记录他**怎么跟 AI 相处**。画像是"我对这个
    人的整体了解", 缺了后一半是不完整的。

    这些观察一开始被做成独立记忆条目写进检索池, 实测行不通: 72 条真实消息只有 7%
    能召回, 且多是"时间"这类表层词误配 —— 它们是特质不是事实, 而向量检索按话题
    相似度建索引。画像每轮必然注入, 不需要被检索到, 正是它们该待的地方。
    """

    @pytest.mark.asyncio
    async def test_generation_prompt_receives_behaviour(self):
        from app.services.memory.behaviour_signals import BehaviouralFact

        captured: dict = {}

        async def _invoke(model, prompt):
            captured["prompt"] = prompt
            return "画像内容"

        facts = [BehaviouralFact(key="timing", statement="他多在晚上来", sample_size=40)]
        with patch.object(portrait_module, "db") as mock_db, \
             patch.object(
                 portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
             ), \
             patch.object(
                 portrait_module, "collect_behavioural_facts",
                 AsyncMock(return_value=facts),
             ), \
             patch.object(
                 portrait_module, "get_prompt_text",
                 AsyncMock(return_value="记忆:{memories}\n行为:{behaviour}"),
             ), \
             patch.object(portrait_module, "invoke_text", _invoke), \
             patch.object(portrait_module, "get_utility_model", lambda: object()), \
             patch.object(
                 portrait_module.memory_repo, "find_many",
                 AsyncMock(return_value=[
                     SimpleNamespace(level=2, mainCategory="生活", subCategory="其他",
                                     content="喜欢喝咖啡"),
                 ]),
             ), \
             patch.object(
                 portrait_module, "_refresh_tags_best_effort", AsyncMock()
             ):
            mock_db.userportrait = MagicMock(
                find_first=AsyncMock(return_value=SimpleNamespace(
                    id="p1", version=1, content="旧画像")),
                create=AsyncMock(),
            )
            await portrait_module.generate_portrait("user-1", "agent-1")

        assert "他多在晚上来" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_missing_behaviour_does_not_block_the_portrait(self):
        """统计失败不该让画像生成不出来 —— 记忆那一半仍然值得写。"""
        with patch.object(
            portrait_module, "collect_behavioural_facts",
            AsyncMock(side_effect=RuntimeError("db down")),
        ):
            section = await portrait_module._behaviour_section("u", "a", "w")
        assert section and "暂无" in section

    @pytest.mark.asyncio
    async def test_thin_data_says_so_instead_of_inventing(self):
        with patch.object(
            portrait_module, "collect_behavioural_facts", AsyncMock(return_value=[]),
        ):
            section = await portrait_module._behaviour_section("u", "a", "w")
        assert "不够" in section

    def test_both_prompts_carry_the_behaviour_placeholder(self):
        """只在首次生成时注入的话, 画像会随每周重写慢慢把行为观察冲掉。"""
        from app.services.prompting.defaults import (
            PORTRAIT_GENERATION_PROMPT, PORTRAIT_UPDATE_PROMPT,
        )

        assert "{behaviour}" in PORTRAIT_GENERATION_PROMPT
        assert "{behaviour}" in PORTRAIT_UPDATE_PROMPT

    def test_prompts_forbid_overreaching_inferences(self):
        """行为规律推不出健康或经济状况, 而这类猜测既不准也冒犯。"""
        from app.services.prompting.defaults import PORTRAIT_GENERATION_PROMPT

        for forbidden in ("健康", "心理疾病", "经济状况", "感情状态"):
            assert forbidden in PORTRAIT_GENERATION_PROMPT
        assert "留有余地" in PORTRAIT_GENERATION_PROMPT


class TestWeeklyUpdateKeepsBehaviourAlive:
    """周更这条路容易被忽略 —— 首次生成做对了不代表画像会一直带着行为观察。"""

    def test_update_prompt_carries_the_same_guardrails_as_generation(self):
        """两个模板都会把统计数字交给模型。生成路径禁止的推断, 周更路径同样要禁 ——
        否则每周重写一次就有一次机会把"晚上情绪低落"演绎成健康推断。"""
        from app.services.prompting.defaults import (
            PORTRAIT_GENERATION_PROMPT, PORTRAIT_UPDATE_PROMPT,
        )

        for guard in ("健康", "心理疾病", "经济状况", "感情状态", "留有余地"):
            assert guard in PORTRAIT_UPDATE_PROMPT, f"周更模板缺少约束: {guard}"
            assert guard in PORTRAIT_GENERATION_PROMPT

    def test_update_prompt_tells_the_model_to_keep_prior_behaviour_on_outage(self):
        """周更是覆盖式重写。统计缺失那周若不叮嘱, 模型会顺手把上次写进去的
        相处方式删掉 —— 一次统计故障就抹掉了几周积累。"""
        from app.services.prompting.defaults import PORTRAIT_UPDATE_PROMPT

        assert "保留原有画像里的相处方式" in PORTRAIT_UPDATE_PROMPT

    @pytest.mark.asyncio
    async def test_new_behaviour_alone_triggers_a_rewrite(self):
        """记忆没变不代表没有新信息。只看 changelog 的话, 一个天天来聊但没说出
        新事实的用户, 画像里的相处方式会一直停在几周前。"""
        from app.services.memory.behaviour_signals import BehaviouralFact

        invoked: dict = {}

        async def _invoke(model, prompt):
            invoked["prompt"] = prompt
            return "新画像"

        facts = [BehaviouralFact(key="timing", statement="他多在晚上来", sample_size=40)]
        with patch.object(portrait_module, "db") as mock_db, \
             patch.object(
                 portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
             ), \
             patch.object(
                 portrait_module, "collect_behavioural_facts",
                 AsyncMock(return_value=facts),
             ), \
             patch.object(
                 portrait_module, "get_prompt_text",
                 AsyncMock(return_value="旧:{previous_portrait} 变化:{weekly_changes} 行为:{behaviour}"),
             ), \
             patch.object(portrait_module, "invoke_text", _invoke), \
             patch.object(portrait_module, "get_utility_model", lambda: object()), \
             patch.object(portrait_module, "_refresh_tags_best_effort", AsyncMock()):
            mock_db.userportrait = MagicMock(
                find_first=AsyncMock(return_value=SimpleNamespace(
                    id="p1", version=1, content="旧画像")),
                create=AsyncMock(),
            )
            # 本周没有任何记忆变化
            mock_db.memorychangelog = MagicMock(find_many=AsyncMock(return_value=[]))
            result = await portrait_module.update_portrait_weekly("user-1", "agent-1")

        assert result == "新画像", "有新行为观察却没重写画像"
        assert "他多在晚上来" in invoked["prompt"]
        assert "没有新的记忆变化" in invoked["prompt"], "空变化段会被模型当成记忆被清空"

    @pytest.mark.asyncio
    async def test_nothing_new_at_all_keeps_the_old_portrait(self):
        """既没有记忆变化也没有行为观察时, 不该白调一次 LLM。"""
        invoke = AsyncMock()
        with patch.object(portrait_module, "db") as mock_db, \
             patch.object(
                 portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
             ), \
             patch.object(
                 portrait_module, "collect_behavioural_facts", AsyncMock(return_value=[])
             ), \
             patch.object(portrait_module, "invoke_text", invoke), \
             patch.object(
                 portrait_module, "has_active_profile_tags", AsyncMock(return_value=True)
             ):
            mock_db.userportrait = MagicMock(
                find_first=AsyncMock(return_value=SimpleNamespace(
                    id="p1", version=1, content="旧画像")),
            )
            mock_db.memorychangelog = MagicMock(find_many=AsyncMock(return_value=[]))
            result = await portrait_module.update_portrait_weekly("user-1", "agent-1")

        assert result == "旧画像"
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_statistics_outage_does_not_block_the_weekly_rewrite(self):
        """行为统计挂了不该连累记忆那一半的更新。"""
        with patch.object(
            portrait_module, "collect_behavioural_facts",
            AsyncMock(side_effect=RuntimeError("db down")),
        ):
            facts = await portrait_module._collect_behaviour_facts("u", "a", "w")
        assert facts == []
        assert "不够" in portrait_module._render_behaviour(facts)


def test_missing_placeholder_is_reported_loudly():
    """str.format 对多余 kwarg 是静默忽略的 —— 后台把 {behaviour} 删掉之后, 行为
    观察会无声消失, 画像看起来照常生成。这类"功能还在但不再生效"的失效在这个项目
    里出过好几次, 共同点都是太安静。"""
    calls: list = []
    with patch.object(portrait_module.logger, "error", lambda *a, **kw: calls.append(a)):
        portrait_module._warn_if_behaviour_dropped("portrait.generation", "没有占位符")
        assert calls, "缺占位符时没有告警"
        calls.clear()
        portrait_module._warn_if_behaviour_dropped("portrait.generation", "有 {behaviour}")
        assert not calls, "占位符齐全却告警了"
