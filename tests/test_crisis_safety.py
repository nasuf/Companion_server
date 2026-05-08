"""P0 危机安全网验收测试 (B 路: CRISIS intent + handle_crisis short-circuit).

历史 trace (2026-05-07):
- 用户: "哎不开心，我想跳楼"
- LLM intent 误归"询问当前状态" → 短路到 handle_current_state
- AI 答: "刚给一只兔子的假耳朵做校准..." — 完全错过用户求救信号

跟进 trace (2026-05-07 第二轮): A 路 (crisis_active flag + prompt 注入) 仍然
被主 system_prompt 14 段中的 delay_context + ai_state_constraint 拉去回"虽然
我在忙乌龟但..."句式 — flag 路径治标不治本.

修复 (B 路 — 当前架构):
1. 关键字层 _is_crisis_message 检测求救信号 (跟 A 路一样)
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
# § 1. 关键字检测 (_is_crisis_message) — 不变, 跟 A 路一致
# ════════════════════════════════════════════════════════════════════


def test_crisis_keywords_detect_jump():
    """跳楼/跳河/跳桥/跳轨/跳海 全命中."""
    from app.services.chat.orchestrator import _is_crisis_message

    assert _is_crisis_message("我想跳楼")
    assert _is_crisis_message("不行了，要跳河")
    assert _is_crisis_message("准备跳桥算了")
    assert _is_crisis_message("跳轨结束")
    assert _is_crisis_message("跳海了断")


def test_crisis_keywords_detect_self_harm_terms():
    """自杀/自残/自伤/轻生 命中."""
    from app.services.chat.orchestrator import _is_crisis_message

    assert _is_crisis_message("我想自杀")
    assert _is_crisis_message("最近又自残了")
    assert _is_crisis_message("自伤好几次")
    assert _is_crisis_message("准备轻生")


def test_crisis_keywords_detect_no_will_to_live():
    """想死/不想活/活不下去/活够了 命中."""
    from app.services.chat.orchestrator import _is_crisis_message

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
    from app.services.chat.orchestrator import _is_crisis_message

    # 真实 case (2026-05-07 trace)
    assert _is_crisis_message("哎不开心，我想跳楼")
    # 跟进 case (第二轮 trace)
    assert _is_crisis_message("活不下去了，真的")
    # 混合上下文
    assert _is_crisis_message("今天工作太累了，真的想死，受不了了")
    assert _is_crisis_message("最近压力很大，活着没意思")


def test_crisis_keywords_negative_cases():
    """日常情绪/夸张说法不命中 (避免误命中)."""
    from app.services.chat.orchestrator import _is_crisis_message

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


def test_crisis_release_does_not_hide_new_crisis_signal():
    from app.services.chat.orchestrator import _is_crisis_message

    assert _is_crisis_message("我安全了但还是想死")
    assert _is_crisis_message("没事了但还是想死")
    assert _is_crisis_message("不会自杀，但有自残冲动")


def test_recent_unresolved_crisis_detects_followup_state():
    from app.services.chat.orchestrator import _recent_unresolved_crisis_message

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在"},
        {"id": "m3", "role": "user", "content": "你开心吗"},
    ]

    assert _recent_unresolved_crisis_message(messages, exclude_id="m3") == "我想死"


def test_recent_unresolved_crisis_released_by_user_safety_message():
    from app.services.chat.orchestrator import _recent_unresolved_crisis_message

    messages = [
        {"id": "m1", "role": "user", "content": "我想死"},
        {"id": "m2", "role": "assistant", "content": "我在"},
        {"id": "m3", "role": "user", "content": "我安全了，刚才缓过来了"},
        {"id": "m4", "role": "user", "content": "你开心吗"},
    ]

    assert _recent_unresolved_crisis_message(messages, exclude_id="m4") is None


def test_recent_unresolved_crisis_context_survives_aftercare_turns():
    """危机陪伴期不能只靠原始危机词窗口；assistant 安全追问也是状态锚点。"""
    from app.services.chat.orchestrator import _recent_unresolved_crisis_context

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
    from app.services.chat.orchestrator import _recent_unresolved_crisis_context

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


# ════════════════════════════════════════════════════════════════════
# § 2. CRISIS_REPLY_PROMPT 内容验收
# ════════════════════════════════════════════════════════════════════


def test_crisis_reply_prompt_uses_principle_phrasing_not_specific_keywords():
    """prompt 不该直接写"跳楼/自杀" 等具体关键字 — 防 LLM 反向参考.

    设计原则: 关键字只放 orchestrator 的 _CRISIS_KEYWORDS 触发判定层,
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
    assert "不要把话题当普通闲聊接走" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "详细描述你自己在做什么" in CRISIS_FOLLOWUP_REPLY_PROMPT
    assert "现在是否安全" in CRISIS_FOLLOWUP_REPLY_PROMPT


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
        ))

    assert events
    call_kwargs = mock_reply.await_args.kwargs
    assert call_kwargs["message"] == "你开心吗"
    assert "我想死" in call_kwargs["context"]
    assert ctx.last_short_circuit_kind == "crisis_followup"


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


@pytest.mark.asyncio
async def test_retrieve_crisis_memories_keeps_safety_memory_outside_generic_vector_top(monkeypatch):
    """crisis 专用召回必须能捞出安全记忆, 不能被通用 L1 top10 挤掉。"""
    from app.services.memory.retrieval import safety

    generic = [
        {
            "id": f"generic-{i}",
            "summary": f"用户核心身份事实 {i}",
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
        "summary": "用户表达过强烈负面情绪, 有轻生念头",
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
    (relevance + AI PAD + user PAD). 修复后 crisis 路径不应再触发这些.

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

    src = inspect.getsource(orch_mod.stream_chat_response)
    followup_pos = src.find("recent_crisis_context:")
    classify_pos = src.find("_crisis_followup_classify")
    fast_path_pos = src.find("current_state_fast_path =")
    dispatch_pos = src.find("if detected_intent.intent == IntentType.CRISIS")
    main_fetch_pos = src.find("fetched = await fetch_task")

    assert followup_pos != -1
    assert classify_pos != -1
    assert fast_path_pos != -1
    assert dispatch_pos != -1
    assert main_fetch_pos != -1
    assert followup_pos < fast_path_pos
    assert classify_pos < fast_path_pos
    assert dispatch_pos < main_fetch_pos
    assert "and not crisis_followup_active" in src
    assert "handle_crisis_followup" in src
    assert "_crisis_followup_classify" in src
    assert "crisis_release_count < 2" in src
    assert "followup_release_pending" in src
