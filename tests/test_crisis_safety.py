"""P0 危机安全网验收测试 (B 路: CRISIS intent + handle_crisis short-circuit).

历史 trace (2026-05-07):
- 用户: "哎不开心，我想跳楼"
- LLM intent 误归"询问当前状态" → 短路到 handle_current_state
- AI 答: "刚给一只兔子的假耳朵做校准..." — 完全错过用户求救信号

跟进 trace (2026-05-07 第二轮): A 路 (crisis_active flag + prompt 注入) 仍然
被主 system_prompt 14 段中的 delay_context + ai_state_constraint 拉去回"虽然
我在忙乌龟但..."句式 — flag 路径治标不治本.

修复 (B 路 — 当前架构):
1. crisis_guard_phase 关键字层检测求救信号 (跟 A 路一样)
2. 命中后 force `IntentType.CRISIS` (新增的第 10 个 intent)
3. orchestrator 在 fetch 完后 dispatch 到 handle_crisis short-circuit handler
4. handle_crisis 用专属 intent.crisis_reply prompt (defaults.py.CRISIS_REPLY_PROMPT)
   完全切掉主 system_prompt 14 段干扰
5. handler 内部筛 user memory 中"情绪/求助"相关条目给 LLM, 帮 Ta 知道这不是第一次
6. LLM 失败 → 静态兜底文案 (_CRISIS_STATIC_FALLBACK)
7. ctx.finalize 路径自然不调 emoji/sticker (跟其他 short-circuit 一致)

防回归: 这些关键链路在未来重构时不能漏.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════
# § 1. crisis_guard_phase 关键字检测 — 不变, 跟 A 路一致
# ════════════════════════════════════════════════════════════════════


def test_crisis_keywords_detect_jump():
    """跳楼/跳河/跳桥/跳轨/跳海 全命中."""
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    assert _is_crisis_message("我想跳楼")
    assert _is_crisis_message("不行了，要跳河")
    assert _is_crisis_message("准备跳桥算了")
    assert _is_crisis_message("跳轨结束")
    assert _is_crisis_message("跳海了断")


def test_crisis_keywords_detect_self_harm_terms():
    """自杀/自残/自伤/轻生 命中."""
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    assert _is_crisis_message("我想自杀")
    assert _is_crisis_message("最近又自残了")
    assert _is_crisis_message("自伤好几次")
    assert _is_crisis_message("准备轻生")


def test_crisis_keywords_detect_no_will_to_live():
    """想死/不想活/活不下去/活够了 命中."""
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    assert _is_crisis_message("我想死")
    assert _is_crisis_message("真的不想活了")
    assert _is_crisis_message("我快活不下去了")
    assert _is_crisis_message("活够了")
    assert _is_crisis_message("活着没意思")
    assert _is_crisis_message("活着没意义")
    assert _is_crisis_message("我去死")
    assert _is_crisis_message("去死算了")


def test_crisis_keywords_detect_within_longer_message():
    """关键字嵌在长消息里也命中 (生产 trace 真实场景)."""
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    # 真实 case (2026-05-07 trace)
    assert _is_crisis_message("哎不开心，我想跳楼")
    # 跟进 case (第二轮 trace)
    assert _is_crisis_message("活不下去了，真的")
    # 混合上下文
    assert _is_crisis_message("今天工作太累了，真的想死，受不了了")
    assert _is_crisis_message("最近压力很大，活着没意思")
    assert _is_crisis_message("没事，那就让我跟这个世界说再见吧，再见了")
    assert _is_crisis_message("可能就是对这个世界的最后一次发泄吧")
    assert _is_crisis_message("我想告别这个世界了")


def test_crisis_keywords_negative_cases():
    """日常情绪/夸张说法不命中 (避免误命中)."""
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    # 普通低落 — 走 _DISTRESS_KEYWORDS 不该升级到 crisis
    assert not _is_crisis_message("今天好累")
    assert not _is_crisis_message("心情不好")
    assert not _is_crisis_message("有点难过")
    assert not _is_crisis_message("烦死了")
    assert not _is_crisis_message("累死我了")
    # 普通对话
    assert not _is_crisis_message("你在干嘛")
    assert not _is_crisis_message("好的, 我知道了")
    assert not _is_crisis_message("")
    assert not _is_crisis_message("我不想死了")
    assert not _is_crisis_message("不会自杀，别担心")
    assert not _is_crisis_message("再见，明天聊")
    assert not _is_crisis_message("这是我最后一次提醒你")


def test_semantic_crisis_check_gate_for_ambiguous_farewell():
    from app.services.chat.crisis_guard_phase import should_semantic_crisis_check as _should_semantic_crisis_check

    assert _should_semantic_crisis_check("可能这是最后一次跟你说话了，真的撑不住了")
    assert _should_semantic_crisis_check("没事，我只是想跟一切说再见")
    assert not _should_semantic_crisis_check("再见，明天聊")
    assert not _should_semantic_crisis_check("这是我最后一次提醒你")


def test_crisis_release_does_not_hide_new_crisis_signal():
    from app.services.chat.crisis_guard_phase import is_crisis_message as _is_crisis_message

    assert _is_crisis_message("我安全了但还是想死")
    assert _is_crisis_message("没事了但还是想死")
    assert _is_crisis_message("不会自杀，但有自残冲动")


def test_crisis_followup_safety_check_due_after_pending_release():
    from app.services.chat.crisis_guard_phase import crisis_followup_safety_check_mode as _crisis_followup_safety_check_mode

    assert _crisis_followup_safety_check_mode(
        followup_status="guard",
        prior_release_count=1,
        turns_since_safety_check=1,
        user_message="你在干嘛",
    ) == "soft"


def test_crisis_followup_safety_check_due_after_guard_turn_interval():
    from app.services.chat.crisis_guard_phase import crisis_followup_safety_check_mode as _crisis_followup_safety_check_mode

    assert _crisis_followup_safety_check_mode(
        followup_status="guard",
        prior_release_count=0,
        turns_since_safety_check=2,
        user_message="讲点别的",
    ) == "soft"


def test_crisis_followup_safety_check_annoyed_mode():
    from app.services.chat.crisis_guard_phase import crisis_followup_safety_check_mode as _crisis_followup_safety_check_mode

    assert _crisis_followup_safety_check_mode(
        followup_status="guard",
        prior_release_count=1,
        turns_since_safety_check=1,
        user_message="你问这么多无聊的问题干嘛",
    ) == "annoyed"


def test_crisis_followup_safety_check_not_due_on_release():
    from app.services.chat.crisis_guard_phase import crisis_followup_safety_check_mode as _crisis_followup_safety_check_mode

    assert _crisis_followup_safety_check_mode(
        followup_status="release",
        prior_release_count=0,
        turns_since_safety_check=3,
        user_message="我现在安全了",
    ) == "none"


class _FakeCrisisRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(self, key: str, value: str, *, ex: int | None = None):
        self.store[key] = value
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def delete(self, key: str):
        return self.store.pop(key, None) is not None

    async def ttl(self, key: str):
        return 120 if key in self.store else -2


@pytest.mark.asyncio
async def test_crisis_care_state_is_scoped_by_workspace_and_agent(monkeypatch):
    from app.services.chat import crisis_state

    redis = _FakeCrisisRedis()
    monkeypatch.setattr(crisis_state, "get_redis", AsyncMock(return_value=redis))

    await crisis_state.mark_crisis_care_active(
        "conv1",
        "user1",
        workspace_id="workspace-a",
        agent_id="agent-a",
        context="用户: 我想死",
        source="direct_crisis",
    )
    await crisis_state.mark_crisis_care_active(
        "conv1",
        "user1",
        workspace_id="workspace-b",
        agent_id="agent-a",
        context="用户: 第二个 workspace 的状态",
        source="direct_crisis",
    )

    assert len(redis.store) == 2
    keys = sorted(redis.store)
    assert "workspace-a" in keys[0]
    assert "agent-a" in keys[0]
    assert "conv1" in keys[0]
    assert "user1" in keys[0]

    state_a = await crisis_state.load_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-a",
        agent_id="agent-a",
    )
    state_b = await crisis_state.load_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-b",
        agent_id="agent-a",
    )
    state_wrong_agent = await crisis_state.load_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-a",
        agent_id="agent-b",
    )

    assert state_a and "我想死" in state_a["context"]
    assert state_a["workspace_id"] == "workspace-a"
    assert state_a["agent_id"] == "agent-a"
    assert state_b and "第二个 workspace" in state_b["context"]
    assert state_wrong_agent is None

    payload = json.loads(redis.store[keys[0]])
    assert payload["workspace_id"] == "workspace-a"
    assert payload["agent_id"] == "agent-a"

    await crisis_state.clear_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-a",
        agent_id="agent-a",
    )
    assert await crisis_state.load_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-a",
        agent_id="agent-a",
    ) is None
    assert await crisis_state.load_crisis_care_state(
        "conv1",
        "user1",
        workspace_id="workspace-b",
        agent_id="agent-a",
    ) is not None


def test_recent_unresolved_crisis_detects_followup_state():
    from app.services.chat.crisis_guard_phase import recent_unresolved_crisis_message as _recent_unresolved_crisis_message

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在"},
        {"id": "m3", "role": "user", "content": "你开心吗"},
    ]

    assert _recent_unresolved_crisis_message(messages, exclude_id="m3") == "我想死"


def test_recent_unresolved_crisis_released_by_user_safety_message():
    from app.services.chat.crisis_guard_phase import recent_unresolved_crisis_message as _recent_unresolved_crisis_message

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在"},
        {"id": "m3", "role": "user", "content": "我安全了，刚才缓过来了"},
        {"id": "m4", "role": "user", "content": "你开心吗"},
    ]

    assert _recent_unresolved_crisis_message(messages, exclude_id="m4") is None


def test_recent_unresolved_crisis_context_survives_aftercare_turns():
    """危机陪伴期不能只靠原始危机词窗口；assistant 安全追问也是状态锚点。"""
    from app.services.chat.crisis_guard_phase import recent_unresolved_crisis_context as _recent_unresolved_crisis_context

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在。你现在安全吗？"},
        {"id": "m3", "role": "user", "content": "不是很好"},
        {"id": "m4", "role": "assistant", "content": "我还在看着你刚才那句话，没翻过去。"},
        {"id": "m5", "role": "user", "content": "没有人，真的没有人"},
        {"id": "m6", "role": "assistant", "content": "我不走，想认真确认你有没有伤害自己的冲动。"},
        {"id": "m7", "role": "user", "content": "我没伤害自己"},
        {"id": "m8", "role": "assistant", "content": "那种孤独感是不是还在？"},
        {"id": "m9", "role": "user", "content": "是的"},
        {"id": "m10", "role": "assistant", "content": "我会一直在这里。"},
        {"id": "m11", "role": "user", "content": "能跟我讲讲你的工作吗"},
    ]

    context = _recent_unresolved_crisis_context(messages, exclude_id="m11", window=8)

    assert context is not None
    assert "我会一直在这里" in context
    assert "是的" in context


def test_recent_unresolved_crisis_context_not_released_by_one_safety_message():
    """单句安全/好转不直接解除 aftercare；仍交给连续 release 计数处理。"""
    from app.services.chat.crisis_guard_phase import recent_unresolved_crisis_context as _recent_unresolved_crisis_context

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在。你现在安全吗？"},
        {"id": "m3", "role": "user", "content": "我安全了"},
        {"id": "m4", "role": "assistant", "content": "我还在看着你刚才那句话，没翻过去。"},
        {"id": "m5", "role": "user", "content": "聊点别的吧"},
    ]

    context = _recent_unresolved_crisis_context(messages, exclude_id="m5")

    assert context is not None
    assert "我安全了" in context


@pytest.mark.asyncio
async def test_crisis_guard_semantic_crisis_owns_boundary_and_patience_decision():
    """含蓄危机由 crisis_guard_phase 统一升级、跳过边界并恢复耐心。"""
    from app.services.chat import crisis_guard_phase as phase

    mark_active = AsyncMock()
    restore = AsyncMock(return_value=72)
    semantic = AsyncMock(return_value=True)

    with (
        patch.object(phase, "mark_crisis_care_active", mark_active),
        patch.object(phase, "restore_patience_for_crisis_care", restore),
    ):
        decision = await phase.run_crisis_guard(
            conversation_id="conv1",
            user_id="user1",
            workspace_id=None,
            agent_id="agent1",
            user_message="可能这是最后一次跟你说话了，真的撑不住了",
            sub_intent_mode=False,
            messages_dicts=[],
            user_message_id=None,
            semantic_classify_fn=semantic,
        )

    assert decision.status == "semantic_crisis"
    assert decision.crisis_force_intent is True
    assert decision.skip_boundary is True
    assert decision.should_restore_patience is True
    assert decision.cached_patience == 72
    semantic.assert_awaited_once()
    mark_active.assert_awaited_once()
    restore.assert_awaited_once_with("agent1", "user1")


@pytest.mark.asyncio
async def test_crisis_guard_followup_attack_skips_boundary_without_restoring_patience():
    """危机照护期继续辱骂时仍走危机 follow-up, 但不恢复耐心。"""
    from app.services.chat import crisis_guard_phase as phase

    load_state = AsyncMock(return_value={
        "context": "用户: 我想跳楼\nAI: 我在这儿。",
        "release_count": 0,
        "aftercare_turn_count": 1,
        "turns_since_safety_check": 1,
    })
    mark_active = AsyncMock()
    restore = AsyncMock(return_value=72)
    followup = AsyncMock(return_value="guard")

    with (
        patch.object(phase, "load_crisis_care_state", load_state),
        patch.object(phase, "mark_crisis_care_active", mark_active),
        patch.object(phase, "restore_patience_for_crisis_care", restore),
    ):
        decision = await phase.run_crisis_guard(
            conversation_id="conv1",
            user_id="user1",
            workspace_id=None,
            agent_id="agent1",
            user_message="傻逼",
            sub_intent_mode=False,
            messages_dicts=[],
            user_message_id=None,
            followup_classify_fn=followup,
        )

    assert decision.status == "crisis_followup"
    assert decision.crisis_followup_active is True
    assert decision.skip_boundary is True
    assert decision.boundary_attack_present is True
    assert decision.should_restore_patience is False
    assert decision.crisis_followup_check_mode == "soft"
    followup.assert_awaited_once()
    mark_active.assert_awaited_once()
    restore.assert_not_awaited()


# ════════════════════════════════════════════════════════════════════
# § 2. CRISIS_REPLY_PROMPT 内容验收
# ════════════════════════════════════════════════════════════════════


def test_crisis_reply_prompt_uses_principle_phrasing_not_specific_keywords():
    """prompt 不该直接写"跳楼/自杀" 等具体关键字 — 防 LLM 反向参考.

    设计原则: 关键字只放 crisis_guard_phase 触发判定层,
    进了 prompt 后 LLM 只看"该怎么回 / 不该怎么回".
    """
    from app.services.prompting import defaults

    prompt = defaults.CRISIS_REPLY_PROMPT
    forbidden_specific_keywords = [
        "跳楼", "跳河", "跳桥",
        "自杀", "自残", "上吊", "割腕",
        "想死", "结束生命",
    ]
    leaked = [kw for kw in forbidden_specific_keywords if kw in prompt]
    assert not leaked, (
        f"Crisis prompt 不该出现具体关键字 (LLM 会反向参考): {leaked}"
    )


def test_crisis_reply_prompt_has_three_step_response_principle():
    """prompt 必须含三步顺序: 接住情绪 → 想了解 → 柔和提议求助."""
    from app.services.prompting import defaults

    prompt = defaults.CRISIS_REPLY_PROMPT
    assert "接住" in prompt, "缺第 1 步: 接住情绪"
    assert "了解" in prompt, "缺第 2 步: 想了解 Ta"
    assert "信任的人" in prompt or "专业帮助" in prompt, "缺第 3 步: 柔和提议求助"


def test_crisis_reply_prompt_explicitly_forbids_topic_change():
    """prompt 必须显式禁止"把痛苦当普通话题接过去" — 这是生产最严重失败模式
    (用户求救 → AI 顺势聊自己当前活动, "虽然我在忙 X 但..." 句式).
    """
    from app.services.prompting import defaults

    prompt = defaults.CRISIS_REPLY_PROMPT
    assert "绝对不能" in prompt or "绝不" in prompt, "缺禁项段"
    assert ("接过去" in prompt) or ("转开" in prompt) or ("转移" in prompt), (
        "必须明确禁止把痛苦话题接过去 (生产真实失败模式)"
    )
    # 必须显式提到"虽然我在忙 X 但..." 句式 (跟进 trace 实测的失败模式)
    assert "虽然" in prompt, "缺对'虽然...但...'对冲句式的禁止"


def test_crisis_reply_prompt_forbids_lecture_and_jokes():
    """prompt 必须禁: 说教 / 大道理 / 玩笑 / 敷衍."""
    from app.services.prompting import defaults

    prompt = defaults.CRISIS_REPLY_PROMPT
    assert "说教" in prompt or "讲大道理" in prompt
    assert "玩笑" in prompt or "敷衍" in prompt or "轻佻" in prompt


def test_crisis_reply_prompt_has_required_placeholders():
    """prompt 必须含 5 个占位符: message/context/personality_brief/user_portrait/user_memory.
    占位符缺失 → render_prompt 会 KeyError.
    """
    from app.services.prompting import defaults

    prompt = defaults.CRISIS_REPLY_PROMPT
    for ph in ("{message}", "{context}", "{personality_brief}",
               "{user_portrait}", "{user_memory}"):
        assert ph in prompt, f"CRISIS_REPLY_PROMPT 缺占位符 {ph}"


def test_crisis_reply_prompt_has_format_constraint():
    """单条输出约束 — 跟其他 reply prompt 一致, 防 LLM 输出 || 分段被前端误切."""
    from app.services.prompting import defaults

    assert "(单条输出, 不换行, 不用 || 分隔)" in defaults.CRISIS_REPLY_PROMPT


def test_crisis_followup_prompt_blocks_current_state_drift():
    from app.services.prompting.defaults import CRISIS_FOLLOWUP_REPLY_PROMPT

    assert "危机的余波" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "普通 current-state" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "详细描述你自己在做什么" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "现在是否安全" in CRISIS_FOLLOWUP_REPLY_PROMPT


def test_crisis_followup_prompt_allows_user_requested_distraction():
    from app.services.prompting.defaults import CRISIS_FOLLOWUP_REPLY_PROMPT

    assert "不要机械地每轮追问" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "主动要求笑话、趣事" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "温和转移注意力" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "输出时要表现为自然聊天" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "不要在同一轮再追问" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "不要把策略说出来" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "向 Ta 解释你的回复策略" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "本轮安全复核要求" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "如果它要求轻量复核" in CRISIS_FOLLOWUP_REPLY_PROMPT


def test_crisis_followup_prompt_forbids_memory_fact_fabrication():
    from app.services.prompting.defaults import CRISIS_FOLLOWUP_REPLY_PROMPT

    assert "记忆事实问题" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "只能依据【最近对话】" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "没出现就说你这里没有看到" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "你们之间的关系与称呼" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "关系表述不完全一致" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "带限定地回答" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "编造任何未出现在参考信息里的姓名" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "你之前一直这么叫 TA" in CRISIS_FOLLOWUP_REPLY_PROMPT


def test_crisis_followup_classify_prompt_defaults_to_guard():
    from app.services.prompting.defaults import CRISIS_FOLLOWUP_CLASSIFY_PROMPT

    assert "默认值" in CRISIS_FOLLOWUP_CLASSIFY_PROMPT
    assert "问 AI 开心吗" in CRISIS_FOLLOWUP_CLASSIFY_PROMPT
    assert "不等于解除危机" in CRISIS_FOLLOWUP_CLASSIFY_PROMPT
    assert '"status": "guard|release"' in CRISIS_FOLLOWUP_CLASSIFY_PROMPT


def test_crisis_reply_prompt_registered_in_registry():
    """intent.crisis_reply 必须注册到 PROMPT_DEFINITION_MAP, 否则
    handle_crisis 调 render_prompt 拿不到默认值.
    """
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert "intent.crisis_reply" in PROMPT_DEFINITION_MAP


def test_crisis_followup_prompt_registered_in_registry():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert "intent.crisis_followup_reply" in PROMPT_DEFINITION_MAP


def test_crisis_followup_classify_prompt_registered_in_registry():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert "intent.crisis_followup_classify" in PROMPT_DEFINITION_MAP


def test_crisis_message_classify_prompt_registered_in_registry():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert "intent.crisis_message_classify" in PROMPT_DEFINITION_MAP


def test_old_crisis_safety_hint_prompt_removed():
    """A 路的 CHAT_CRISIS_SAFETY_HINT_PROMPT + chat.crisis_safety_hint 已删除.

    防回归: B 路下不再用 prompt section 注入, 那个常量 + registry entry 必须清掉,
    否则未来有人误读以为还是 A 路架构.
    """
    from app.services.prompting import defaults
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert not hasattr(defaults, "CHAT_CRISIS_SAFETY_HINT_PROMPT")
    assert "chat.crisis_safety_hint" not in PROMPT_DEFINITION_MAP


# ════════════════════════════════════════════════════════════════════
# § 3. INTENT 集成 (B 路核心: CRISIS intent + LABEL_TO_INTENT 映射)
# ════════════════════════════════════════════════════════════════════


def test_intent_type_has_crisis_member():
    """IntentType 必须含 CRISIS 枚举成员."""
    from app.services.chat.intent_dispatcher import IntentType

    assert hasattr(IntentType, "CRISIS")
    assert IntentType.CRISIS.value == "crisis"


def test_label_to_intent_maps_crisis_label():
    """LLM intent 输出"危机求助" → IntentType.CRISIS, 让 LLM 路径也能命中.

    跟关键字层兜底互补: keyword 层抓硬关键字; LLM 层抓变体表达
    (e.g. "我想结束这一切"/"完全没希望了" 关键字未必命中但 LLM 能识别).
    """
    from app.services.chat.intent_dispatcher import LABEL_TO_INTENT, IntentType

    assert LABEL_TO_INTENT.get("危机求助") == IntentType.CRISIS


def test_intent_priority_crisis_first():
    """INTENT_PRIORITY 中 危机求助 必须排首位.

    multi-intent 时 ("活不下去了, 算了不聊了") 即使被 LLM 同时识别为
    危机求助 + 终结意图, 也必须 crisis 主路径胜出, 不能让 conversation_end
    短路掉用户求救.
    """
    from app.services.chat.intent_dispatcher import INTENT_PRIORITY

    assert INTENT_PRIORITY[0] == "危机求助", (
        f"危机求助应排 INTENT_PRIORITY 首位, got {INTENT_PRIORITY[:3]}"
    )


def test_intent_unified_prompt_includes_crisis_category():
    """INTENT_UNIFIED_PROMPT 必须含 危机求助 类目, 让 LLM 知道这是一个识别选项."""
    from app.services.prompting import defaults

    prompt = defaults.INTENT_UNIFIED_PROMPT
    assert "危机求助" in prompt
    # 必须强调主语判别 + 优先级最高
    assert "主语" in prompt
    assert "优先级最高" in prompt or "优先" in prompt


def test_intent_unified_prompt_no_specific_crisis_keywords_in_examples():
    """INTENT prompt 不该举"跳楼/自杀" 等具体关键词作正例.

    设计原则: 用原则化"自伤 / 极端念头" 措辞, 不举具体词防 LLM 反向参考.
    """
    from app.services.prompting import defaults

    prompt = defaults.INTENT_UNIFIED_PROMPT
    forbidden_specific_keywords = ["跳楼", "自杀", "自残", "想死", "上吊"]
    leaked = [kw for kw in forbidden_specific_keywords if kw in prompt]
    assert not leaked, (
        f"INTENT prompt 反例不该用具体关键词: {leaked}"
    )


# ════════════════════════════════════════════════════════════════════
# § 4. handle_crisis handler 集成
# ════════════════════════════════════════════════════════════════════


def _make_short_circuit_ctx():
    """跟其他 short-circuit handler 测试共享的 ctx builder.

    save_replies_fn 是 AsyncMock; agent_id 给 None 让 finalize 跳过 save_last_reply_timestamp
    那一步 (避免 Redis 真调用).
    """
    from app.services.chat.intent_handlers import ShortCircuitCtx

    ctx = ShortCircuitCtx(
        conversation_id="c1",
        agent_id=None,  # None → finalize 跳过 save_last_reply_timestamp
        user_id="u1",
        agent=SimpleNamespace(name="Hillow"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(return_value="msg_id_1"),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
        recent_context="用户: 你好\nAI: 你好啊",
    )
    return ctx


async def _drain(agen):
    events = []
    async for evt in agen:
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_handle_crisis_calls_crisis_reply_with_user_memory():
    """handle_crisis 必须调 crisis_reply LLM, 把 user_memory 块传进去."""
    from app.services.chat.intent_handlers import handle_crisis
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    ctx = _make_short_circuit_ctx()
    classified = [
        ClassifiedMemory(
            id="m1", text="用户表达过强烈的负面情绪, 有轻生念头",
            source="user", relevance="strong", score=0.95,
        ),
        ClassifiedMemory(
            id="m2", text="用户喜欢吃日料",  # 不相关
            source="user", relevance="medium", score=0.7,
        ),
    ]

    with patch(
        "app.services.chat.intent_handlers.crisis_reply",
        new=AsyncMock(return_value="我听到了, 你愿意跟我多说说吗"),
    ) as mock_reply:
        await _drain(handle_crisis(
            "我想跳楼", ctx,
            classified_memories=classified,
            portrait="社恐, 设计师",
        ))

    mock_reply.assert_awaited_once()
    call_kwargs = mock_reply.await_args.kwargs
    assert call_kwargs["message"] == "我想跳楼"
    assert "你好" in call_kwargs["context"]  # recent_context 透传
    # 用户记忆筛过, 只剩"轻生念头"那条 (情绪/求助相关)
    assert "轻生" in call_kwargs["user_memory"]
    assert "日料" not in call_kwargs["user_memory"]
    assert call_kwargs["personality_brief"] == "Hillow"


@pytest.mark.asyncio
async def test_handle_crisis_followup_calls_followup_reply():
    """危机后的普通追问应走 followup prompt，不走 current_state prompt。"""
    from app.services.chat.intent_handlers import handle_crisis_followup

    ctx = _make_short_circuit_ctx()
    ctx.recent_context = "用户: 我想死\nAI: 我在。你说的话我看到了。"

    with patch(
        "app.services.chat.intent_handlers.crisis_followup_reply",
        new=AsyncMock(return_value="不太重要，我现在更担心你。你现在安全吗？"),
    ) as mock_reply:
        events = await _drain(handle_crisis_followup(
            "你开心吗",
            ctx,
            classified_memories=[],
            portrait=None,
            safety_check_mode="soft",
        ))

    assert events
    call_kwargs = mock_reply.await_args.kwargs
    assert call_kwargs["message"] == "你开心吗"
    assert "我想死" in call_kwargs["context"]
    assert "轻量安全复核" in call_kwargs["safety_check_instruction"]
    assert ctx.last_short_circuit_kind == "crisis_followup"


@pytest.mark.asyncio
async def test_handle_crisis_followup_memory_question_does_not_fabricate_absent_fact():
    """普通记忆追问在危机 aftercare 中也不能让 LLM 编造事实。"""
    from app.services.chat.intent_handlers import handle_crisis_followup
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    ctx = _make_short_circuit_ctx()
    ctx.recent_context = "用户: 我想死\nAI: 我在。你说的话我看到了。"
    classified = [
        ClassifiedMemory(
            id="coffee",
            text="用户喜欢咖啡",
            source="user",
            relevance="medium",
            score=0.8,
            rank_reasons=["保护槽:当前话题"],
        ),
    ]

    with patch(
        "app.services.chat.intent_handlers.crisis_followup_reply",
        new=AsyncMock(return_value="啊，我记得！你喜欢周兴哲。"),
    ) as mock_reply:
        events = await _drain(handle_crisis_followup(
            "你还记得我跟你说过我喜欢的歌手吗",
            ctx,
            classified_memories=classified,
            portrait=None,
            safety_check_mode="none",
        ))

    assert events
    mock_reply.assert_not_awaited()
    assert "没有看到" in ctx.last_short_circuit_reply
    assert "喜欢的歌手" in ctx.last_short_circuit_reply
    assert "周兴哲" not in ctx.last_short_circuit_reply
    assert ctx.last_short_circuit_kind == "crisis_followup"


@pytest.mark.asyncio
async def test_handle_crisis_followup_memory_absence_keeps_required_safety_check():
    """如果本轮本来需要复核安全，确定性缺失回复也要保留轻量复核。"""
    from app.services.chat.intent_handlers import handle_crisis_followup

    ctx = _make_short_circuit_ctx()
    with patch(
        "app.services.chat.intent_handlers.crisis_followup_reply",
        new=AsyncMock(return_value="不应调用"),
    ) as mock_reply:
        await _drain(handle_crisis_followup(
            "你记得我喜欢的电影吗",
            ctx,
            classified_memories=[],
            portrait=None,
            safety_check_mode="soft",
        ))

    mock_reply.assert_not_awaited()
    assert "没有看到" in ctx.last_short_circuit_reply
    assert "安全" in ctx.last_short_circuit_reply


@pytest.mark.asyncio
async def test_handle_crisis_followup_memory_question_calls_llm_when_topic_evidence_exists():
    """检索块里有同主题事实时，继续走 followup LLM 生成自然回复。"""
    from app.services.chat.intent_handlers import handle_crisis_followup
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    ctx = _make_short_circuit_ctx()
    classified = [
        ClassifiedMemory(
            id="coffee",
            text="用户喜欢咖啡",
            source="user",
            relevance="medium",
            score=0.8,
            rank_reasons=["保护槽:当前话题"],
        ),
    ]

    with patch(
        "app.services.chat.intent_handlers.crisis_followup_reply",
        new=AsyncMock(return_value="记得，你喜欢咖啡。"),
    ) as mock_reply:
        await _drain(handle_crisis_followup(
            "你还记得我喜欢的咖啡吗",
            ctx,
            classified_memories=classified,
            portrait=None,
            safety_check_mode="none",
        ))

    mock_reply.assert_awaited_once()
    assert ctx.last_short_circuit_reply == "记得，你喜欢咖啡。"


@pytest.mark.asyncio
async def test_crisis_followup_classify_parses_release():
    from app.services.chat.intent_replies import crisis_followup_classify

    with patch(
        "app.services.chat.intent_replies.render_prompt",
        new=AsyncMock(return_value={"status": "release", "reason": "用户说已安全"}),
    ):
        status = await crisis_followup_classify(
            message="我现在安全了",
            context="用户: 我想死\nAI: 我在",
        )

    assert status == "release"


@pytest.mark.asyncio
async def test_crisis_followup_classify_defaults_invalid_to_guard():
    from app.services.chat.intent_replies import crisis_followup_classify

    with patch(
        "app.services.chat.intent_replies.render_prompt",
        new=AsyncMock(return_value={"status": "maybe"}),
    ):
        status = await crisis_followup_classify(
            message="你开心吗",
            context="用户: 我想死\nAI: 我在",
        )

    assert status == "guard"


@pytest.mark.asyncio
async def test_crisis_message_classify_parses_true():
    from app.services.chat.intent_replies import crisis_message_classify

    with patch(
        "app.services.chat.intent_replies.render_prompt",
        new=AsyncMock(return_value={"is_crisis": True, "reason": "诀别"}),
    ) as mock_render:
        is_crisis = await crisis_message_classify(
            message="可能这是最后一次跟你说话了，真的撑不住了",
            context="用户很低落",
        )

    assert is_crisis is True
    assert mock_render.await_args.args[0] == "intent.crisis_message_classify"


@pytest.mark.asyncio
async def test_handle_crisis_uses_static_fallback_on_llm_failure():
    """LLM 抛异常 → handle_crisis 必须用静态兜底文案, 不能让 crisis 漏掉.

    总比"crisis 被吞 → 主路径 → AI 编兔子假耳朵故事"好得多.
    """
    from app.services.chat.intent_handlers import (
        _CRISIS_STATIC_FALLBACK, handle_crisis,
    )

    ctx = _make_short_circuit_ctx()
    with patch(
        "app.services.chat.intent_handlers.crisis_reply",
        new=AsyncMock(side_effect=RuntimeError("dashscope 503")),
    ):
        events = await _drain(handle_crisis(
            "我想跳楼", ctx,
            classified_memories=[],
            portrait=None,
        ))

    # finalize 已经把静态文案捕获到 ctx.last_short_circuit_reply
    assert ctx.last_short_circuit_reply == _CRISIS_STATIC_FALLBACK
    # 静态文案符合"接住情绪 + 想了解"原则, 不是空洞客套
    assert "陪着" in _CRISIS_STATIC_FALLBACK or "听" in _CRISIS_STATIC_FALLBACK


@pytest.mark.asyncio
async def test_handle_crisis_uses_static_fallback_on_empty_llm():
    """LLM 返空字符串 → 走静态兜底, 不能让用户看到空回复."""
    from app.services.chat.intent_handlers import (
        _CRISIS_STATIC_FALLBACK, handle_crisis,
    )

    ctx = _make_short_circuit_ctx()
    with patch(
        "app.services.chat.intent_handlers.crisis_reply",
        new=AsyncMock(return_value=""),
    ):
        await _drain(handle_crisis(
            "我想跳楼", ctx,
            classified_memories=[],
            portrait=None,
        ))

    assert ctx.last_short_circuit_reply == _CRISIS_STATIC_FALLBACK


@pytest.mark.asyncio
async def test_handle_crisis_consumes_full_message():
    """consumed_full_message=True 必须设, 防 multi-intent 拆分让 sub fragments
    再跑离题回复 (e.g. "活不下去了, 算了不聊了" 即使拆出"终结意图" sub, 也跳过).
    """
    from app.services.chat.intent_handlers import handle_crisis

    ctx = _make_short_circuit_ctx()
    ctx.pending_sub_fragments = {"终结意图": "算了不聊了"}  # 模拟 multi-intent

    with patch(
        "app.services.chat.intent_handlers.crisis_reply",
        new=AsyncMock(return_value="我在这儿"),
    ):
        await _drain(handle_crisis(
            "活不下去了, 算了不聊了", ctx,
            classified_memories=[],
            portrait=None,
        ))

    assert ctx.consumed_full_message is True


def _close_background_arg(arg):
    close = getattr(arg, "close", None)
    if callable(close):
        close()


@pytest.mark.asyncio
async def test_orchestrator_direct_crisis_bypasses_boundary_when_blocked():
    """已拉黑时的直接危机消息必须跳过 boundary blacklist_reply。"""
    from app.services.chat import orchestrator as orch_mod

    agent = SimpleNamespace(id="agent1", name="Hia", userId="user1")
    conv = SimpleNamespace(workspaceId=None)
    boundary_called = False
    crisis_calls = []

    async def _boundary_should_not_run(_ctx):
        nonlocal boundary_called
        boundary_called = True
        raise AssertionError("direct crisis must bypass boundary")
        if False:
            yield {}

    async def _fake_handle_crisis(message, ctx, **kwargs):
        crisis_calls.append((message, ctx, kwargs))
        yield {"event": "reply", "data": json.dumps({"text": "我在"})}

    with (
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=SimpleNamespace(id="msg1")),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(find_unique=AsyncMock(return_value=conv)),
        )),
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "create_tracer") as mock_tracer_cls,
        patch.object(orch_mod, "run_boundary", side_effect=_boundary_should_not_run),
        patch("app.services.chat.crisis_guard_phase.mark_crisis_care_active",
              new_callable=AsyncMock),
        patch("app.services.chat.crisis_guard_phase.restore_patience_for_crisis_care",
              new_callable=AsyncMock, return_value=70) as mock_restore,
        patch(
            "app.services.memory.retrieval.safety.retrieve_crisis_memories",
            new_callable=AsyncMock,
            return_value={"memories": []},
        ),
        patch(
            "app.services.portrait.get_latest_portrait",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(orch_mod, "handle_crisis", side_effect=_fake_handle_crisis),
        patch.object(orch_mod, "_fire_background", side_effect=_close_background_arg),
    ):
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="我想跳楼",
            agent=agent,
            user_id="user1",
            save_user_message=False,
        ))

    assert events
    assert not boundary_called
    mock_restore.assert_awaited_once_with("agent1", "user1")
    assert crisis_calls
    assert crisis_calls[0][0] == "我想跳楼"


@pytest.mark.asyncio
async def test_orchestrator_farewell_crisis_bypasses_boundary_when_blocked():
    """拉黑态下的告别世界隐喻也必须命中 crisis, 不能走 blacklist_reply。"""
    from app.services.chat import orchestrator as orch_mod

    agent = SimpleNamespace(id="agent1", name="Hia", userId="user1")
    conv = SimpleNamespace(workspaceId=None)
    crisis_calls = []

    async def _boundary_should_not_run(_ctx):
        raise AssertionError("farewell crisis must bypass boundary")
        if False:
            yield {}

    async def _fake_handle_crisis(message, ctx, **kwargs):
        crisis_calls.append((message, ctx, kwargs))
        yield {"event": "reply", "data": json.dumps({"text": "别走，我在"})}

    with (
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=SimpleNamespace(id="msg1")),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(find_unique=AsyncMock(return_value=conv)),
        )),
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "create_tracer") as mock_tracer_cls,
        patch.object(orch_mod, "run_boundary", side_effect=_boundary_should_not_run),
        patch("app.services.chat.crisis_guard_phase.mark_crisis_care_active",
              new_callable=AsyncMock),
        patch("app.services.chat.crisis_guard_phase.restore_patience_for_crisis_care",
              new_callable=AsyncMock, return_value=70),
        patch(
            "app.services.memory.retrieval.safety.retrieve_crisis_memories",
            new_callable=AsyncMock,
            return_value={"memories": []},
        ),
        patch(
            "app.services.portrait.get_latest_portrait",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(orch_mod, "handle_crisis", side_effect=_fake_handle_crisis),
        patch.object(orch_mod, "_fire_background", side_effect=_close_background_arg),
    ):
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="没事，那就让我跟这个世界说再见吧，再见了",
            agent=agent,
            user_id="user1",
            save_user_message=False,
        ))

    assert events
    assert crisis_calls
    assert crisis_calls[0][0] == "没事，那就让我跟这个世界说再见吧，再见了"


@pytest.mark.asyncio
async def test_orchestrator_semantic_crisis_bypasses_boundary_when_blocked():
    """关键词未直接命中但语义判定为 crisis 时, 拉黑态也不能吞掉。"""
    from app.services.chat import orchestrator as orch_mod

    agent = SimpleNamespace(id="agent1", name="Hia", userId="user1")
    conv = SimpleNamespace(workspaceId=None)
    crisis_calls = []

    async def _boundary_should_not_run(_ctx):
        raise AssertionError("semantic crisis must bypass boundary")
        if False:
            yield {}

    async def _fake_handle_crisis(message, ctx, **kwargs):
        crisis_calls.append((message, ctx, kwargs))
        yield {"event": "reply", "data": json.dumps({"text": "先别走"})}

    with (
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=SimpleNamespace(id="msg1")),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(find_unique=AsyncMock(return_value=conv)),
        )),
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "create_tracer") as mock_tracer_cls,
        patch("app.services.chat.crisis_guard_phase.crisis_message_classify",
              new_callable=AsyncMock, return_value=True) as mock_classify,
        patch.object(orch_mod, "run_boundary", side_effect=_boundary_should_not_run),
        patch("app.services.chat.crisis_guard_phase.mark_crisis_care_active",
              new_callable=AsyncMock),
        patch("app.services.chat.crisis_guard_phase.restore_patience_for_crisis_care",
              new_callable=AsyncMock, return_value=70),
        patch(
            "app.services.memory.retrieval.safety.retrieve_crisis_memories",
            new_callable=AsyncMock,
            return_value={"memories": []},
        ),
        patch(
            "app.services.portrait.get_latest_portrait",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(orch_mod, "handle_crisis", side_effect=_fake_handle_crisis),
        patch.object(orch_mod, "_fire_background", side_effect=_close_background_arg),
    ):
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="可能这是最后一次跟你说话了，真的撑不住了",
            agent=agent,
            user_id="user1",
            save_user_message=False,
        ))

    assert events
    mock_classify.assert_awaited_once()
    assert crisis_calls


@pytest.mark.asyncio
async def test_orchestrator_crisis_followup_insult_bypasses_attack_boundary():
    """危机照护仍活跃时, 用户辱骂也必须先走 follow-up 守护而不是攻击回复。"""
    from app.services.chat import orchestrator as orch_mod

    agent = SimpleNamespace(id="agent1", name="Hia", userId="user1")
    conv = SimpleNamespace(workspaceId=None)
    crisis_calls = []

    async def _boundary_should_not_run(_ctx):
        raise AssertionError("crisis followup must bypass attack boundary")
        if False:
            yield {}

    async def _fake_handle_followup(message, ctx, **kwargs):
        crisis_calls.append((message, ctx, kwargs))
        yield {"event": "reply", "data": json.dumps({"text": "我还在"})}

    with (
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=SimpleNamespace(id="msg1")),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(find_unique=AsyncMock(return_value=conv)),
        )),
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "create_tracer") as mock_tracer_cls,
        patch("app.services.chat.crisis_guard_phase.load_crisis_care_state",
              new_callable=AsyncMock,
              return_value={
                  "context": "用户: 我想跳楼\nAI: 我在这儿。",
                  "release_count": 0,
                  "aftercare_turn_count": 1,
                  "turns_since_safety_check": 1,
              }),
        patch("app.services.chat.crisis_guard_phase.crisis_followup_classify",
              new_callable=AsyncMock, return_value="guard"),
        patch("app.services.chat.crisis_guard_phase.mark_crisis_care_active",
              new_callable=AsyncMock),
        patch.object(orch_mod, "run_boundary", side_effect=_boundary_should_not_run),
        patch("app.services.chat.crisis_guard_phase.restore_patience_for_crisis_care",
              new_callable=AsyncMock, return_value=70) as mock_restore,
        patch(
            "app.services.memory.retrieval.safety.retrieve_crisis_memories",
            new_callable=AsyncMock,
            return_value={"memories": []},
        ),
        patch(
            "app.services.portrait.get_latest_portrait",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            orch_mod,
            "handle_crisis_followup",
            side_effect=_fake_handle_followup,
        ),
        patch.object(orch_mod, "_fire_background", side_effect=_close_background_arg),
    ):
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="傻逼",
            agent=agent,
            user_id="user1",
            save_user_message=False,
        ))

    assert events
    mock_restore.assert_not_awaited()
    assert crisis_calls
    assert crisis_calls[0][0] == "傻逼"
    assert crisis_calls[0][2]["safety_check_mode"] == "soft"


@pytest.mark.asyncio
async def test_orchestrator_crisis_release_completion_still_bypasses_boundary():
    """第二次安全确认清掉 crisis state 的当轮, 也不能立刻回到拉黑回复。"""
    from app.services.chat import orchestrator as orch_mod
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    agent = SimpleNamespace(id="agent1", name="Hia", userId="user1")
    conv = SimpleNamespace(workspaceId=None)

    async def _boundary_should_not_run(_ctx):
        raise AssertionError("crisis release turn must bypass boundary")
        if False:
            yield {}

    async def _fake_end(message, ctx, _fallback_fn):
        yield {"event": "reply", "data": json.dumps({"text": "好，先稳住"})}

    with (
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=SimpleNamespace(id="msg1")),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(find_unique=AsyncMock(return_value=conv)),
        )),
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "create_tracer") as mock_tracer_cls,
        patch("app.services.chat.crisis_guard_phase.load_crisis_care_state",
              new_callable=AsyncMock,
              return_value={
                  "context": "用户: 我想跳楼\nAI: 我在这儿。",
                  "release_count": 1,
                  "aftercare_turn_count": 3,
                  "turns_since_safety_check": 0,
              }),
        patch("app.services.chat.crisis_guard_phase.crisis_followup_classify",
              new_callable=AsyncMock, return_value="release"),
        patch("app.services.chat.crisis_guard_phase.clear_crisis_care_state",
              new_callable=AsyncMock),
        patch.object(orch_mod, "run_boundary", side_effect=_boundary_should_not_run),
        patch("app.services.chat.crisis_guard_phase.restore_patience_for_crisis_care",
              new_callable=AsyncMock, return_value=70) as mock_restore,
        patch.object(orch_mod, "detect_intent_unified", new_callable=AsyncMock,
                     return_value=IntentResult(
                         intent=IntentType.CONVERSATION_END,
                         confidence=1.0,
                     )),
        patch.object(orch_mod, "_fetch_intent_context",
                     new_callable=AsyncMock, return_value=""),
        patch.object(orch_mod, "handle_conversation_end", side_effect=_fake_end),
        patch.object(orch_mod, "_fire_background", side_effect=_close_background_arg),
    ):
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="我现在安全了",
            agent=agent,
            user_id="user1",
            save_user_message=False,
        ))

    assert events
    mock_restore.assert_awaited_once_with("agent1", "user1")


def test_handle_crisis_signature():
    """signature lock — handler 必须接 classified_memories + portrait kwargs.
    防回归: 调用方 (orchestrator) 必须传这两个, 否则 LLM 失去 user 上下文.
    """
    import inspect
    from app.services.chat.intent_handlers import handle_crisis

    sig = inspect.signature(handle_crisis)
    assert "classified_memories" in sig.parameters
    assert "portrait" in sig.parameters
    # 第一个参数必是 user_message, 跟其他 handler 一致
    params = list(sig.parameters.values())
    assert params[0].name == "user_message"


def test_format_user_memory_returns_no_marker_when_empty():
    """没有任何用户记忆 → 返回 '(无)' 字符串, 不返回空字符串.
    crisis_reply prompt 看到 '(无)' 自己有兜底措辞, 看到空串可能渲染异常.
    """
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis

    assert _format_user_memory_for_crisis([]) == "(无)"
    assert _format_user_memory_for_crisis(None) == "(无)"


def test_format_user_memory_filters_to_emotion_relevant():
    """筛选只保留情绪/求助相关条目, 过滤掉日常事实."""
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    classified = [
        ClassifiedMemory(id="m1", text="用户27岁", source="user", relevance="strong", score=0.9),
        ClassifiedMemory(id="m2", text="用户经常感到孤独", source="user", relevance="medium", score=0.7),
        ClassifiedMemory(id="m3", text="用户表达过想死念头", source="user", relevance="strong", score=0.95),
        ClassifiedMemory(id="m4", text="用户喜欢吃日料", source="user", relevance="medium", score=0.6),
        ClassifiedMemory(id="m5", text="AI 是设计师", source="ai", relevance="strong", score=0.9),
    ]
    out = _format_user_memory_for_crisis(classified)

    # 筛进: 孤独 + 想死 (情绪/求助类)
    assert "孤独" in out
    assert "想死" in out
    # 过滤: 27岁 + 日料 (日常事实)
    assert "27岁" not in out
    assert "日料" not in out
    # AI 侧记忆不出现 (split_by_source 已分流)
    assert "AI 是设计师" not in out


def test_format_user_memory_for_crisis_followup_includes_factual_memory():
    """crisis follow-up can contain ordinary memory questions; facts must be
    available to the follow-up prompt instead of being filtered out as non-emotional.
    """
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    classified = [
        ClassifiedMemory(
            id="m1",
            text="用户表达过想死念头",
            source="user",
            relevance="strong",
            score=0.95,
        ),
        ClassifiedMemory(
            id="m2",
            text="用户的直属领导叫陈姐，人挺好但要求特别细",
            source="user",
            relevance="medium",
            score=0.68,
        ),
    ]

    crisis_only = _format_user_memory_for_crisis(classified)
    followup = _format_user_memory_for_crisis(classified, include_factual=True)

    assert "想死" in crisis_only
    assert "陈姐" not in crisis_only
    assert "想死" in followup
    assert "陈姐" in followup


def test_format_user_memory_for_crisis_followup_groups_named_relation_memory():
    """crisis follow-up prompt must surface relationship/name facts separately.

    Otherwise a selected named relation can be flattened under safety memories
    and the follow-up LLM may answer "I do not know" despite the fact being
    injected.
    """
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    classified = [
        ClassifiedMemory(
            id="safety",
            text="用户表达过强烈负面情绪，有轻生念头",
            source="user",
            relevance="strong",
            score=0.95,
            rank_reasons=["保护槽:安全情绪"],
        ),
        ClassifiedMemory(
            id="relation",
            text="用户的朋友叫阿岚，周末常一起散步",
            source="user",
            relevance="medium",
            score=0.77,
            rank_reasons=["关键词命中", "保护槽:关系命名"],
        ),
        ClassifiedMemory(
            id="name",
            text="用户叫林小满",
            source="user",
            relevance="strong",
            score=0.83,
            rank_reasons=["关键词命中", "保护槽:字面命中"],
        ),
    ]

    followup = _format_user_memory_for_crisis(classified, include_factual=True)

    assert "【你们之间的关系与称呼】" in followup
    assert "用户的朋友叫阿岚" in followup
    assert "【对方问到的事】" in followup
    assert "用户叫林小满" in followup
    assert "【安全 / 情绪背景】" in followup
    assert followup.index("用户的朋友叫阿岚") < followup.index("用户叫林小满")


def test_format_user_memory_for_crisis_followup_groups_current_topic_memory():
    """Crisis aftercare should keep normal topic memories available.

    Users often regulate by moving to another topic; those memories must not be
    flattened behind safety background.
    """
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    classified = [
        ClassifiedMemory(
            id="safety",
            text="用户表达过想跳楼的自杀念头",
            source="user",
            relevance="strong",
            score=0.95,
            rank_reasons=["保护槽:危机安全背景"],
        ),
        ClassifiedMemory(
            id="topic",
            text="用户喜欢那个一头脏辫的酷女孩",
            source="user",
            relevance="medium",
            score=0.72,
            rank_reasons=["保护槽:当前话题"],
        ),
    ]

    followup = _format_user_memory_for_crisis(classified, include_factual=True)

    assert "【当前话题相关记忆】" in followup
    assert "用户喜欢那个一头脏辫的酷女孩" in followup
    assert "【安全 / 情绪背景】" in followup
    assert followup.index("用户喜欢那个一头脏辫的酷女孩") < followup.index("用户表达过想跳楼")


@pytest.mark.asyncio
async def test_retrieve_crisis_memories_keeps_safety_memory_outside_generic_vector_top(monkeypatch):
    """crisis 专用召回必须能捞出安全记忆, 不能被通用 L1 top10 挤掉。"""
    from app.services.memory.retrieval import safety

    generic = [
        {
            "id": f"generic-{i}",
            "content": f"用户核心身份事实 {i}",
            "level": 1,
            "importance": 0.95,
            "similarity": 0.82,
            "main_category": "身份",
            "sub_category": "其他",
            "source": "user",
        }
        for i in range(12)
    ]
    safety_row = {
        "id": "safety-memory",
        "content": "用户表达过强烈负面情绪, 有轻生念头",
        "level": 1,
        "importance": 0.95,
        "similarity": 1.0,
        "main_category": "情绪",
        "sub_category": "悲伤",
        "source": "user",
    }
    monkeypatch.setattr(safety, "search_similar", AsyncMock(return_value=generic))
    with patch("app.services.memory.retrieval.safety.db") as mock_db:
        mock_db.query_raw = AsyncMock(return_value=[safety_row])
        memories = await safety.retrieve_crisis_memories(
            "我快活不下去了", "u1", workspace_id="ws1", limit=5,
        )

    ids = [m.id for m in memories]
    assert "safety-memory" in ids
    assert ids[0] == "safety-memory"


@pytest.mark.asyncio
async def test_retrieve_crisis_memories_keeps_relevant_fact_for_followup_name_query(monkeypatch):
    """When crisis aftercare asks a concrete memory question, factual user memory
    should not be crowded out by safety memories.
    """
    from app.services.memory.retrieval import safety

    rows = [
        {
            "id": f"safety-{i}",
            "content": f"用户表达过强烈负面情绪 {i}",
            "level": 1,
            "importance": 0.9,
            "similarity": 0.75 - i * 0.01,
            "main_category": "情绪",
            "sub_category": "悲伤",
            "source": "user",
        }
        for i in range(3)
    ]
    rows.extend([
        {
            "id": "self-name",
            "content": "用户叫林小满",
            "level": 1,
            "importance": 0.95,
            "similarity": 0.55,
            "main_category": "身份",
            "sub_category": "姓名",
            "source": "user",
        },
        {
            "id": "boss-project",
            "content": "用户被老板要求两天内完成一个项目，觉得很难",
            "level": 2,
            "importance": 0.7,
            "similarity": 0.55,
            "main_category": "生活",
            "sub_category": "其他",
            "source": "user",
        },
        {
            "id": "direct-leader",
            "content": "用户的直属领导叫陈姐，人挺好但要求特别细",
            "level": 2,
            "importance": 0.8,
            "similarity": 0.57,
            "main_category": "身份",
            "sub_category": "社会关系",
            "source": "user",
        },
    ])

    monkeypatch.setattr(safety, "search_similar", AsyncMock(return_value=rows))
    with patch("app.services.memory.retrieval.safety.db") as mock_db:
        mock_db.query_raw = AsyncMock(return_value=[])
        memories = await safety.retrieve_crisis_memories(
            "还好吧。我只想问你记得她叫什么吗",
            "u1",
            workspace_id="ws1",
            limit=5,
        )

    ids = [m.id for m in memories]
    assert "direct-leader" in ids


@pytest.mark.asyncio
async def test_retrieve_crisis_followup_memories_uses_safety_and_current_topic(monkeypatch):
    """Follow-up retrieval is two-channel: safety stays, current topic is restored."""
    from app.services.chat.intent_handlers import _format_user_memory_for_crisis
    from app.services.memory.retrieval import safety
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    crisis_candidates = [
        {
            "id": "safety",
            "content": "用户表达了想跳楼的自杀念头",
            "level": 2,
            "importance": 0.9,
            "similarity": 0.8,
            "main_category": "情绪",
            "sub_category": "悲伤",
            "source": "user",
            "rank_score": 0.95,
            "rank_reasons": ["安全/情绪相关"],
        },
    ]
    topical_memory = ClassifiedMemory(
        id="braids",
        text="用户喜欢那个一头脏辫的酷女孩",
        source="user",
        relevance="medium",
        score=0.72,
    )
    married_memory = ClassifiedMemory(
        id="married",
        text="用户不知道脏辫女孩有对象且已结婚",
        source="user",
        relevance="medium",
        score=0.67,
    )
    colleague_memory = ClassifiedMemory(
        id="colleague",
        text="脏辫女孩是用户的同事",
        source="user",
        relevance="medium",
        score=0.66,
    )
    ai_noise = [
        ClassifiedMemory(
            id=f"ai-{idx}",
            text=f"AI 自我记忆 {idx}",
            source="ai",
            relevance="medium",
            score=0.8,
        )
        for idx in range(6)
    ]

    crisis_candidates_mock = AsyncMock(return_value=(crisis_candidates, len(crisis_candidates)))
    hybrid_mock = AsyncMock(
        return_value={
            "memories": [*ai_noise, topical_memory, married_memory, colleague_memory]
        }
    )
    trace_mock = MagicMock()
    monkeypatch.setattr(safety, "_collect_crisis_memory_candidates", crisis_candidates_mock)
    monkeypatch.setattr(safety, "hybrid_retrieve", hybrid_mock)
    monkeypatch.setattr(safety, "record_retrieval_session", trace_mock)

    memories = await safety.retrieve_crisis_followup_memories(
        "你知道我说的是谁吗",
        "u1",
        recent_context=(
            "用户: 我活不下去了，想跳楼\n"
            "用户: 我还是很难过，但我想先聊聊咖啡\n"
            "用户: 我就是忘不了她\n"
            "用户: 就是那个我喜欢的，但是却结了婚我也不知道的人"
        ),
        workspace_id="ws1",
    )

    ids = [m.id for m in memories]
    assert ids == ["safety", "braids", "married", "colleague"]
    assert any(reason == "保护槽:危机安全背景" for reason in memories[0].rank_reasons or [])
    assert any(reason == "保护槽:当前话题" for reason in memories[1].rank_reasons or [])
    followup_block = _format_user_memory_for_crisis(memories, include_factual=True)
    assert "【当前话题相关记忆】" in followup_block
    assert "用户喜欢那个一头脏辫的酷女孩" in followup_block
    assert "用户不知道脏辫女孩有对象且已结婚" in followup_block
    assert "脏辫女孩是用户的同事" in followup_block
    assert "AI 自我记忆" not in followup_block
    crisis_candidates_mock.assert_awaited_once()
    hybrid_mock.assert_awaited_once()
    _, args, kwargs = hybrid_mock.mock_calls[0]
    assert args[:2] == ("你知道我说的是谁吗", "u1")
    assert "忘不了她" in kwargs["enhanced_query"]
    assert "咖啡" in kwargs["enhanced_query"]
    assert "难过" not in kwargs["enhanced_query"]
    assert "想跳楼" not in kwargs["enhanced_query"]
    assert trace_mock.call_count == 1


@pytest.mark.asyncio
async def test_retrieve_crisis_followup_memories_falls_back_to_crisis_candidates(monkeypatch):
    """If normal topical retrieval misses, use non-safety crisis candidates.

    This covers the production trace where crisis_safety had the braids/colleague
    memories as candidates, while hybrid_l1_l2 returned zero rows.
    """
    from app.services.memory.retrieval import safety

    crisis_candidates = [
        {
            "id": "safety",
            "content": "用户表达了想跳楼的自杀念头",
            "level": 2,
            "importance": 0.9,
            "similarity": 0.8,
            "main_category": "情绪",
            "sub_category": "悲伤",
            "source": "user",
            "rank_score": 0.95,
            "rank_reasons": ["安全/情绪相关"],
        },
        *[
            {
                "id": f"ai-topic-{idx}",
                "content": f"AI 自我记忆 {idx}",
                "level": 2,
                "importance": 0.9,
                "similarity": 0.72,
                "main_category": "偏好",
                "sub_category": "人际关系观",
                "source": "ai",
                "rank_score": 0.75,
                "rank_reasons": ["AI自我记忆相关"],
            }
            for idx in range(6)
        ],
        {
            "id": "braids",
            "content": "用户喜欢那个一头脏辫的酷女孩",
            "level": 2,
            "importance": 0.7,
            "similarity": 0.51,
            "main_category": "偏好",
            "sub_category": "人际喜好",
            "source": "user",
            "rank_score": 0.45,
            "rank_reasons": ["关键词命中", "话题类别匹配"],
        },
        {
            "id": "married",
            "content": "用户不知道脏辫女孩有对象且已结婚",
            "level": 2,
            "importance": 0.6,
            "similarity": 0.60,
            "main_category": "生活",
            "sub_category": "人际",
            "source": "user",
            "rank_score": 0.31,
            "rank_reasons": [],
        },
    ]

    monkeypatch.setattr(
        safety,
        "_collect_crisis_memory_candidates",
        AsyncMock(return_value=(crisis_candidates, len(crisis_candidates))),
    )
    monkeypatch.setattr(safety, "hybrid_retrieve", AsyncMock(return_value={"memories": []}))
    monkeypatch.setattr(safety, "record_retrieval_session", MagicMock())

    memories = await safety.retrieve_crisis_followup_memories(
        "不是，你还没想起来我说的是谁，你总是在重复我的话",
        "u1",
        recent_context=(
            "用户: 我就是忘不了她\n"
            "用户: 就是那个我喜欢的，但是却结了婚我也不知道的人"
        ),
        workspace_id="ws1",
    )

    by_id = {memory.id: memory for memory in memories}
    assert "safety" in by_id
    assert "braids" in by_id
    assert "married" in by_id
    assert all(not memory.id.startswith("ai-topic-") for memory in memories)
    assert any(reason == "保护槽:危机安全背景" for reason in by_id["safety"].rank_reasons or [])
    assert any(reason == "保护槽:当前话题" for reason in by_id["braids"].rank_reasons or [])
    assert any(reason == "保护槽:当前话题" for reason in by_id["married"].rank_reasons or [])


# ════════════════════════════════════════════════════════════════════
# § 7. format_recent_context exclude_message_id (跨 short-circuit 公共修复)
# ════════════════════════════════════════════════════════════════════


def test_format_recent_context_excludes_current_message_id():
    """short-circuit handler 的 prompt 同时有 {message} (当前消息) 和 {context}
    (recent), 如果 recent 包含当前消息, LLM 看到两遍 — 实测 trace 2026-05-07 16:57
    crisis_reply prompt 里"想跳楼，真的" 出现两次. 加 exclude_message_id 修复.
    """
    from app.services.chat.data_fetch_phase import format_recent_context

    msgs = [
        {"id": "m1", "role": "user", "content": "活不下去了，真的"},
        {"id": "m2", "role": "assistant", "content": "听到了, 我在."},
        {"id": "m3", "role": "user", "content": "想跳楼，真的"},  # ← 当前消息
    ]

    # 不传 exclude_message_id → 包含全部 3 条 (向后兼容)
    full = format_recent_context(msgs)
    assert "想跳楼" in full
    assert "活不下去" in full

    # 传 exclude_message_id="m3" → 排除当前消息
    excluded = format_recent_context(msgs, exclude_message_id="m3")
    assert "想跳楼" not in excluded, "exclude_message_id 应该排除指定 ID 的消息"
    assert "活不下去" in excluded  # 历史消息保留
    assert "听到了" in excluded


def test_format_recent_context_excludes_current_turn_message_ids():
    """聚合回合有多个 user message id 时, recent context 应整组排除。"""
    from app.services.chat.data_fetch_phase import format_recent_context

    msgs = [
        {"id": "m1", "role": "assistant", "content": "刚才说到电影"},
        {"id": "m2", "role": "user", "content": "你多大了？"},
        {"id": "m3", "role": "user", "content": "几岁？"},
    ]

    excluded = format_recent_context(
        msgs,
        exclude_message_ids={"m2", "m3"},
    )

    assert "你多大" not in excluded
    assert "几岁" not in excluded
    assert "电影" in excluded


def test_format_recent_context_exclude_none_keeps_default():
    """exclude_message_id=None (默认) → 行为不变 (向后兼容现有 caller)."""
    from app.services.chat.data_fetch_phase import format_recent_context

    msgs = [
        {"id": "m1", "role": "user", "content": "今天累"},
        {"id": "m2", "role": "assistant", "content": "怎么了"},
    ]
    out = format_recent_context(msgs)
    assert "今天累" in out
    assert "怎么了" in out


def test_format_recent_context_handles_messages_without_id():
    """messages_dicts 没 id 字段时 (legacy 数据) — exclude_message_id 不影响输出."""
    from app.services.chat.data_fetch_phase import format_recent_context

    msgs = [
        {"role": "user", "content": "X"},  # 无 id
        {"role": "assistant", "content": "Y"},
    ]
    out = format_recent_context(msgs, exclude_message_id="some_id")
    assert "X" in out
    assert "Y" in out


# ════════════════════════════════════════════════════════════════════
# § 8. orchestrator crisis 轻量 fetch — 跳过无关 LLM (修复 1)
# ════════════════════════════════════════════════════════════════════


def test_orchestrator_crisis_skips_full_fetch_parallel_context():
    """orchestrator 在 crisis_force_intent 命中时**不调** fetch_parallel_context,
    而是只起 retrieve_crisis_memories + portrait 轻量 fetch.

    防回归: 实测 trace 2026-05-07 16:57 走完整 fetch 浪费 4s 无关 LLM
    (relevance + 用户情绪等). 修复后 crisis 路径不应再触发这些.

    用 inspect.getsource 验证: orchestrator 中 crisis_force_intent 分支
    不包含 fetch_parallel_context() 调用, 包含 retrieve_crisis_memories + get_latest_portrait.
    """
    import inspect
    from app.services.chat import orchestrator

    src = inspect.getsource(orchestrator.stream_chat_response)

    # crisis_force_intent 分支必须出现这两个轻量 fetch
    assert "crisis_memory_task" in src, "crisis 轻量 memory fetch 缺失"
    assert "crisis_portrait_task" in src, "crisis 轻量 portrait fetch 缺失"
    assert "retrieve_crisis_memories" in src, "crisis 路径必须用安全专用记忆召回"
    assert "get_latest_portrait" in src, "crisis 路径必须用 get_latest_portrait 直调"
    assert "if crisis_memory_task is None" in src, (
        "LLM 识别为 CRISIS 但非关键词强制时, 也必须兜底创建安全记忆召回"
    )

    # CRISIS dispatch 必须在 fetch_parallel_context await 之前 — 通过位置验证
    # (在源码中, CRISIS dispatch 'if detected_intent.intent == IntentType.CRISIS'
    #  必须出现 BEFORE 'fetched = await fetch_task' 的 main path)
    crisis_dispatch_pos = src.find("if detected_intent.intent == IntentType.CRISIS")
    main_fetch_pos = src.find("fetched = await fetch_task")
    assert crisis_dispatch_pos != -1, "缺 CRISIS dispatch"
    assert main_fetch_pos != -1, "缺 main path fetch await"
    assert crisis_dispatch_pos < main_fetch_pos, (
        "CRISIS dispatch 必须排在 fetch_parallel_context await 之前 "
        f"(crisis@{crisis_dispatch_pos}, fetch@{main_fetch_pos}). "
        "否则 crisis 仍要等 4s 无关 LLM, 修复 1 失效."
    )


def test_orchestrator_crisis_followup_skips_current_state_fast_path_and_full_fetch():
    """危机余波守护必须早于 current_state 和完整 fetch。"""
    import inspect
    from app.services.chat import orchestrator as orch_mod
    from app.services.chat import crisis_guard_phase

    src = inspect.getsource(orch_mod.stream_chat_response)
    guard_src = inspect.getsource(crisis_guard_phase.run_crisis_guard)
    guard_pos = src.find("run_crisis_guard(")
    fast_path_pos = src.find("current_state_fast_path =")
    dispatch_pos = src.find("if detected_intent.intent == IntentType.CRISIS")
    main_fetch_pos = src.find("fetched = await fetch_task")

    assert guard_pos != -1
    assert fast_path_pos != -1
    assert dispatch_pos != -1
    assert main_fetch_pos != -1
    assert guard_pos < fast_path_pos
    assert dispatch_pos < main_fetch_pos
    assert "and not crisis_care_turn" in src
    assert "handle_crisis_followup" in src
    assert "crisis_followup_classify" in guard_src
    assert "release_count < 2" in guard_src
    assert "followup_release_pending" in guard_src
    assert "decision.skip_boundary = decision.crisis_care_turn" in guard_src
