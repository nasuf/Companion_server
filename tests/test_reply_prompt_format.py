"""验收: 所有用户可达 reply prompt 在 【输出】 段都显式约束输出格式.

历史问题: 大部分 reply prompt 只说"只输出回复内容", 没规定单/多条 + 断句.
LLM 自由发挥时可能输出"句1\\n句2" 或"句1||句2", 让前端 pre-wrap 渲染奇怪
或被错误的 split 路径切. 现在统一在 【输出】 段拼接 (单条输出, 不换行, 不用
|| 分隔), 主回复路径 chat.response_instruction 例外 (该路径明确允许 || 多条).

防回归: 未来加新 reply prompt 时若忘了带这个约束, 此 test 失败提醒.
"""

from __future__ import annotations

# 应该是"单条输出"约束的 reply prompt (工具型短回复 / 单一目的)
USER_FACING_REPLY_PROMPT_CONSTS = [
    # Memory contradiction (单条 — 矛盾追问/结束语就 1 句)
    "MEMORY_CONTRADICTION_INQUIRY_PROMPT",
    "MEMORY_CONTRADICTION_REPLY_PROMPT",
    # Intent reply (工具型, 单一目的)
    "END_REPLY_PROMPT",
    "SCHEDULE_QUERY_REPLY_PROMPT",
    "CURRENT_STATE_REPLY_PROMPT",
    "DELETION_REPLY_PROMPT",
    "RECORD_CONFIRM_REPLY_PROMPT",
    "RECORD_ASK_TIME_PROMPT",
    # Boundary reply (短约束硬回复)
    "LIGHT_ATTACK_REPLY_PROMPT",
    "MEDIUM_ATTACK_REPLY_PROMPT",
    "SEVERE_ATTACK_REPLY_PROMPT",
    "MEDIUM_PATIENCE_REPLY_PROMPT",
    "LOW_PATIENCE_REPLY_PROMPT",
    "BLACKLIST_REPLY_PROMPT",
    "APOLOGY_REPLY_PROMPT",
    "BOUNDARY_FINAL_WARNING_PROMPT",
    # Proactive (主动一句)
    "PROACTIVE_SILENCE_PLAIN_PROMPT",
    "PROACTIVE_SILENCE_AI_MEMORY_PROMPT",
    "PROACTIVE_SILENCE_USER_MEMORY_PROMPT",
    "PROACTIVE_SILENCE_SCHEDULE_PROMPT",
    "PROACTIVE_MEMORY_AI_PROMPT",
    "PROACTIVE_MEMORY_USER_PROMPT",
    "PROACTIVE_SCHEDULED_SCENE_PROMPT",
    "PROACTIVE_DECAY_FINAL_PROMPT",
    "PROACTIVE_FIRST_GREETING_PROMPT",
    "PROACTIVE_SPECIAL_HOLIDAY_PROMPT",
    "PROACTIVE_SPECIAL_BIRTHDAY_PROMPT",
    "PROACTIVE_SPECIAL_REMINDER_PROMPT",
    "PROACTIVE_SPECIAL_COMBINED_PROMPT",
    "REMINDER_MESSAGE_PROMPT",
]

# 允许多条 || 的 reply prompt (跟主回复一致, 1-3 条随机).
# Phase: tier reply (memory 分级) 也升级为多条, 跟主回复一致, 提升微信多条体感.
MULTI_REPLY_PROMPT_CONSTS = [
    "RESPONSE_INSTRUCTION_PROMPT",  # 主回复
    "WEAK_MEMORY_REPLY_PROMPT",     # tier reply (4 个)
    "MEDIUM_MEMORY_REPLY_PROMPT",
    "STRONG_MEMORY_REPLY_PROMPT",
    "L3_MEMORY_REPLY_PROMPT",
]


def test_all_user_facing_reply_prompts_have_format_constraint():
    """每个用户可达 reply prompt 必须含 (单条输出, 不换行, 不用 || 分隔) 约束.

    防止 LLM 输出 "句1\\n句2" 或 "句1||句2" 触发未预期的拆分行为.
    """
    from app.services.prompting import defaults

    constraint = "(单条输出, 不换行, 不用 || 分隔)"
    missing = []
    for name in USER_FACING_REPLY_PROMPT_CONSTS:
        prompt = getattr(defaults, name, None)
        if prompt is None:
            missing.append(f"{name} (常量不存在)")
            continue
        if constraint not in prompt:
            missing.append(name)

    assert not missing, (
        f"以下 reply prompt 缺少 '{constraint}' 输出约束:\n"
        + "\n".join(f"  - {m}" for m in missing)
        + "\n\n所有用户可达 reply prompt 必须显式声明输出格式, 防 LLM 输出 "
          "'句1\\n句2' 或 '||' 触发拆分."
    )


def test_multi_reply_prompts_have_pipe_format():
    """主回复 + 4 个 tier reply prompt 必须含 "分{n}条 || 分隔" 多条指令."""
    from app.services.prompting import defaults

    for name in MULTI_REPLY_PROMPT_CONSTS:
        prompt = getattr(defaults, name)
        # 必须含 || 分隔 + {n}/{max_per}/{total} 占位符
        assert "||" in prompt, f"{name} 必须含 || 多条指令"
        assert "{n}" in prompt, f"{name} 必须含 {{n}} 占位符"
        assert "{max_per}" in prompt, f"{name} 必须含 {{max_per}} 占位符"
        assert "{total}" in prompt, f"{name} 必须含 {{total}} 占位符"


def test_current_state_prompt_does_not_include_full_schedule():
    """当前状态问句只需要当前活动；完整日程会增加 token 且诱导长回复。"""
    from app.services.prompting import defaults

    prompt = defaults.CURRENT_STATE_REPLY_PROMPT
    assert "AI当前的作息：{current_activity}" in prompt
    assert "AI当前的作息表：{ai_schedule}" not in prompt
    assert "不允许新增具体物件、书名、页面、剧情、地点或巧合细节" in prompt
    assert "云彩分类的书" in prompt


def test_intent_prompt_keeps_current_state_followup_disambiguation():
    """上一轮内容追问不能被 current_state 短路吞掉。"""
    from app.services.prompting import defaults

    prompt = defaults.INTENT_UNIFIED_PROMPT
    assert "用户基于上一轮 AI 的说法表达惊讶、质疑或追问" in prompt
    assert "这么晚还能看到X啊" in prompt


def test_schedule_query_prompt_forbids_using_current_state_for_future_query():
    """未来日程查询不能展开当前作息故事。"""
    from app.services.prompting import defaults

    prompt = defaults.SCHEDULE_QUERY_REPLY_PROMPT
    assert "如果用户问明天/后天/周末/未来，不要提你现在正在做什么" in prompt
    assert "不要编造突发事故" in prompt
    assert "刚才发生的经历" in prompt


def test_tier_reply_prompts_forbid_ungrounded_self_context():
    """tier reply 使用率提升后, 必须防无依据自我活动/经历编造。"""
    from app.services.prompting import defaults

    for name in (
        "WEAK_MEMORY_REPLY_PROMPT",
        "MEDIUM_MEMORY_REPLY_PROMPT",
        "STRONG_MEMORY_REPLY_PROMPT",
        "L3_MEMORY_REPLY_PROMPT",
    ):
        prompt = getattr(defaults, name)
        assert "不要主动编造你正在做什么" in prompt, f"{name} 缺少当前活动编造约束"
        assert "过去经历" in prompt, f"{name} 缺少过往经历编造约束"
        assert "和用户的共同经历" in prompt, f"{name} 缺少共同经历编造约束"


def test_decision_prompts_not_polluted_with_format_constraint():
    """决策/标签/JSON 类 prompt 不该有"单条输出, 不换行" 约束 (它们输出标签
    /分类/JSON, 不是 reply 文本)."""
    from app.services.prompting import defaults

    constraint = "(单条输出, 不换行, 不用 || 分隔)"
    decision_prompts = [
        "MEMORY_JUDGEMENT_USER_PROMPT",
        "MEMORY_JUDGEMENT_AI_PROMPT",
        "MEMORY_EXTRACTION_USER_PROMPT",
        "MEMORY_EXTRACTION_AI_PROMPT",
        "EMOTION_EXTRACTION_PROMPT",
        "ATTACK_TARGET_PROMPT",
        "ATTACK_LEVEL_PROMPT",
        "BANNED_WORD_PROMPT",
        "POSITIVE_INTERACTION_PROMPT",
        "INTENT_UNIFIED_PROMPT",
        "L3_TRIGGER_PROMPT",
        "MEMORY_RELEVANCE_PROMPT",
        "MEMORY_CONTRADICTION_DETECTION_PROMPT",
        "MEMORY_CONTRADICTION_ANALYSIS_PROMPT",
        "DELETION_INTENT_PROMPT",
    ]
    polluted = []
    for name in decision_prompts:
        prompt = getattr(defaults, name, None)
        if prompt is None:
            continue
        if constraint in prompt:
            polluted.append(name)

    assert not polluted, (
        f"以下决策 prompt 不该有 '{constraint}' (它们输出标签/JSON, 不是 reply 文本):\n"
        + "\n".join(f"  - {p}" for p in polluted)
    )
