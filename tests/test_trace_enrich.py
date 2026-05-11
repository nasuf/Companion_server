"""Tests for trace_enrich semantic enrichment.

CI 守 prompt 头部 drift: defaults.py 任何注册过的 prompt 头部 60 字改了,
test_each_registered_prompt_matches 就会失败, 提示更新映射表.
"""

from __future__ import annotations

import pytest

from app.services.chat import trace_enrich
from app.services.prompting import defaults
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.utils import SafeDict


def _fake_llm_step(prompt_text: str, output_text: str = "") -> dict:
    """构造一个 normalized step (mimics public_trace._normalize_step output)."""
    return {
        "id": "step-1",
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": {
            "messages": [[{
                "id": ["langchain", "schema", "messages", "HumanMessage"],
                "kwargs": {"content": prompt_text, "type": "human"},
            }]],
        },
        "outputs": {
            "generations": [[{
                "text": output_text,
                "message": {
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {"content": output_text, "type": "ai"},
                },
            }]],
        },
    }


# ──────────────────────────────────────────────────────────────────────
# 守 drift: 已注册 prompt 必须能识别
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt_const,expected_key", [
    (defaults.MEMORY_RELEVANCE_PROMPT, "memory.relevance"),
    (defaults.INTENT_UNIFIED_PROMPT, "intent.unified"),
    (defaults.INTENT_SPLIT_PROMPT, "intent.split"),
    (defaults.L3_TRIGGER_PROMPT, "memory.l3_trigger"),
    (defaults.MEMORY_CONTRADICTION_DETECTION_PROMPT, "memory.contradiction_detection"),
    (defaults.MEMORY_CONTRADICTION_ANALYSIS_PROMPT, "memory.contradiction_analysis"),
    (defaults.APOLOGY_PROMPT, "boundary.apology"),
    (defaults.POSITIVE_INTERACTION_PROMPT, "boundary.positive_interaction"),
    (defaults.ATTACK_TARGET_PROMPT, "boundary.attack_target"),
    (defaults.ATTACK_LEVEL_PROMPT, "boundary.attack_level"),
    (defaults.BANNED_WORD_PROMPT, "boundary.banned_word"),
    (defaults.MEMORY_JUDGEMENT_USER_PROMPT, "memory.judgement_user"),
    (defaults.MEMORY_JUDGEMENT_AI_PROMPT, "memory.judgement_ai"),
    (defaults.DELETION_INTENT_PROMPT, "memory.deletion_intent"),
    (defaults.REMINDER_PRE_CHECK_PROMPT, "proactive.reminder_pre_check"),
    (defaults.MEMORY_PAIRWISE_CONTRADICTION_PROMPT, "memory.pairwise_contradiction"),
    (defaults.CRISIS_MESSAGE_CLASSIFY_PROMPT, "intent.crisis_message_classify"),
    (defaults.PROACTIVE_MEMORY_TOPIC_RERANK_PROMPT, "proactive.memory_topic_rerank"),
    (defaults.USER_EMOTION_LABEL_PROMPT, "emotion.user_label"),
    (defaults.PERSONALITY_SCORING_PROMPT, "agent.personality_scoring"),
    (defaults.CHARACTER_GENERATION_PROMPT, "character.generation"),
    (defaults.CHARACTER_REPAIR_MISSING_FIELDS_PROMPT, "character.repair_missing_fields"),
    (defaults.LIFE_OVERVIEW_PROMPT, "schedule.life_overview"),
    (defaults.DAILY_SCHEDULE_PROMPT, "schedule.daily_schedule"),
    (defaults.DAILY_SCHEDULE_WITH_USER_MEMORY_PROMPT, "schedule.daily_schedule_with_memory"),
    (defaults.SCHEDULE_DAILY_SUMMARY_PROMPT, "schedule.daily_summary"),
    (defaults.SCHEDULE_DAILY_SUMMARY_MEMORIES_PROMPT, "schedule.daily_summary_memories"),
    (defaults.PORTRAIT_GENERATION_PROMPT, "portrait.generation"),
    (defaults.PORTRAIT_UPDATE_PROMPT, "portrait.update"),
    (defaults.CURRENT_STATE_REPLY_PROMPT, "intent.current_state_reply"),
    (defaults.CRISIS_REPLY_PROMPT, "intent.crisis_reply"),
    (defaults.CRISIS_FOLLOWUP_CLASSIFY_PROMPT, "intent.crisis_followup_classify"),
    (defaults.CRISIS_FOLLOWUP_REPLY_PROMPT, "intent.crisis_followup_reply"),
    (defaults.END_REPLY_PROMPT, "intent.end_reply"),
    (defaults.SCHEDULE_QUERY_REPLY_PROMPT, "intent.schedule_query_reply"),
    (defaults.SCHEDULE_ADJUST_REPLY_PROMPT, "intent.schedule_adjust_reply"),
    (defaults.APOLOGY_REPLY_PROMPT, "boundary.apology_reply"),
    (defaults.DELETION_CONFIRM_PROMPT, "intent.deletion_confirm"),
    (defaults.DELETION_REPLY_PROMPT, "intent.deletion_reply"),
    (defaults.MEMORY_CONTRADICTION_INQUIRY_PROMPT, "memory.contradiction_inquiry"),
    (defaults.MEMORY_CONTRADICTION_REPLY_PROMPT, "memory.contradiction_reply"),
    (defaults.BOUNDARY_FINAL_WARNING_PROMPT, "boundary.final_warning"),
    (defaults.LIGHT_ATTACK_REPLY_PROMPT, "boundary.light_attack_reply"),
    (defaults.MEDIUM_ATTACK_REPLY_PROMPT, "boundary.medium_attack_reply"),
    (defaults.SEVERE_ATTACK_REPLY_PROMPT, "boundary.severe_attack_reply"),
    (defaults.MEDIUM_PATIENCE_REPLY_PROMPT, "boundary.medium_patience_reply"),
    (defaults.LOW_PATIENCE_REPLY_PROMPT, "boundary.low_patience_reply"),
    (defaults.BLACKLIST_REPLY_PROMPT, "boundary.blacklist_reply"),
    (defaults.WEAK_MEMORY_REPLY_PROMPT, "memory.weak_reply"),
    (defaults.MEDIUM_MEMORY_REPLY_PROMPT, "memory.medium_reply"),
    (defaults.STRONG_MEMORY_REPLY_PROMPT, "memory.strong_reply"),
    (defaults.L3_MEMORY_REPLY_PROMPT, "memory.l3_reply"),
    (defaults.DELAY_EXPLANATION_PROMPT, "reply.delay_explanation"),
    (defaults.REMINDER_MESSAGE_PROMPT, "proactive.reminder_message"),
    (defaults.RECORD_CONFIRM_REPLY_PROMPT, "intent.record_confirm_reply"),
    (defaults.RECORD_ASK_TIME_PROMPT, "intent.record_ask_time"),
    (defaults.PROACTIVE_SPECIAL_REMINDER_PROMPT, "proactive.special_reminder"),
    (defaults.PROACTIVE_SILENCE_PLAIN_PROMPT, "proactive.silence_plain"),
    (defaults.PROACTIVE_SILENCE_AI_MEMORY_PROMPT, "proactive.silence_ai_memory"),
    (defaults.PROACTIVE_SILENCE_USER_MEMORY_PROMPT, "proactive.silence_user_memory"),
    (defaults.PROACTIVE_SILENCE_SCHEDULE_PROMPT, "proactive.silence_schedule"),
    (defaults.PROACTIVE_MEMORY_AI_PROMPT, "proactive.memory_ai"),
    (defaults.PROACTIVE_MEMORY_USER_PROMPT, "proactive.memory_user"),
    (defaults.PROACTIVE_SCHEDULED_SCENE_PROMPT, "proactive.scheduled_scene"),
    (defaults.PROACTIVE_DECAY_FINAL_PROMPT, "proactive.decay_final"),
    (defaults.PROACTIVE_FIRST_GREETING_PROMPT, "proactive.first_greeting"),
    (defaults.PROACTIVE_SPECIAL_HOLIDAY_PROMPT, "proactive.special_holiday"),
    (defaults.PROACTIVE_SPECIAL_BIRTHDAY_PROMPT, "proactive.special_birthday"),
    (defaults.PROACTIVE_SPECIAL_COMBINED_PROMPT, "proactive.special_combined"),
    (defaults.SYSTEM_BASE_PROMPT, "chat.system_base"),
    # reply.split_2/split_3 已删除 — 主 LLM 直接按 || 输出, 不再二次拆分
    (defaults.AI_REPLY_EMOTION_PROMPT, "reply.emotion_detection"),
    (defaults.MEMORY_EXTRACTION_USER_PROMPT, "memory.extraction_user"),
    (defaults.MEMORY_EXTRACTION_AI_PROMPT, "memory.extraction_ai"),
])
def test_each_registered_prompt_matches(prompt_const, expected_key):
    """defaults.py 头部 60 字改了 → 此测试立刻挂, 提示更新映射表."""
    step = _fake_llm_step(prompt_const)
    enriched = trace_enrich.enrich_step(step)
    assert enriched["prompt_key"] == expected_key, (
        f"指纹丢失: {expected_key}. 头部很可能改了, 请更新 trace_enrich.py 映射表."
    )
    assert enriched["display_name"] != "ChatOpenAI"
    assert enriched["category"] in ("decision", "data", "reply", "post")


def test_registered_prompt_keys_exist_in_prompt_registry():
    """Trace 里暴露的 prompt_key 必须能跳转到 admin prompt registry."""
    unknown = {
        meta.prompt_key
        for _, meta in trace_enrich._REGISTRY
        if meta.prompt_key not in PROMPT_DEFINITION_MAP
    }
    assert unknown == set()


def test_enriched_prompt_includes_admin_registry_metadata():
    """Trace step 和提示词管理页必须用同一 registry metadata 对齐。"""
    step = _fake_llm_step(defaults.INTENT_UNIFIED_PROMPT)
    enriched = trace_enrich.enrich_step(step)

    definition = PROMPT_DEFINITION_MAP["intent.unified"]
    assert enriched["prompt_key"] == definition.key
    assert enriched["prompt_title"] == definition.title
    assert enriched["prompt_stage"] == definition.stage
    assert enriched["prompt_category"] == definition.category


def test_main_reply_exposes_component_prompt_keys():
    """主回复是拼接 prompt, trace 应列出可编辑组件而不是只给一个泛化标题。"""
    prompt = "\n\n".join([
        f"## 核心规则\n{defaults.SYSTEM_BASE_PROMPT}",
        f"## 反幻觉硬约束\n{defaults.ANTI_HALLUCINATION_HARD_RULE_PROMPT}",
        f"## 你的身份\n你的名字叫Lua。",
        f"## 对话一致性\n{defaults.CONSISTENCY_RULES_PROMPT}",
        f"## 回复要求\n{defaults.RESPONSE_INSTRUCTION_PROMPT.format(n=2, total=220, max_per=80)}",
    ])
    step = _fake_llm_step(prompt)
    enriched = trace_enrich.enrich_step(step)

    keys = [item["prompt_key"] for item in enriched["prompt_components"]]
    assert keys == [
        "chat.system_base",
        "chat.anti_hallucination_hard_rule",
        "chat.consistency_rules",
        "chat.response_instruction",
    ]
    assert "chat.personality_rules" not in keys
    assert "chat.ai_state_constraint" not in keys


def test_structured_prompt_render_trace_overrides_legacy_section_guessing():
    """新 trace 用渲染期 hash+span 元数据定位组件, 不靠 section 标题猜。"""
    prompt = "\n\n".join([
        f"## 核心规则\n{defaults.SYSTEM_BASE_PROMPT}",
        "## 当前情绪\n你们目前的关系是初识。",
        "## 你记得的事情\n(本次没有联想到任何与当前话题相关的记忆)",
    ])
    step = trace_enrich.enrich_step(_fake_llm_step(prompt))
    legacy_keys = [item["prompt_key"] for item in step.get("prompt_components", [])]
    assert legacy_keys == ["chat.system_base"]

    start = prompt.index("你们目前的关系")
    end = start + len("你们目前的关系是初识。")
    trace_enrich.apply_prompt_render_traces([step], [{
        "prompt_hash": trace_enrich.prompt_hash(prompt),
        "prompt_key": "chat.system_base",
        "source": "chat.system_prompt",
        "components": [
            {"prompt_key": "chat.relationship_stage_section", "start": start, "end": end},
        ],
    }])

    components = step["prompt_components"]
    assert [item["prompt_key"] for item in components] == ["chat.relationship_stage_section"]
    assert components[0]["start"] == start
    assert components[0]["end"] == end


def test_main_reply_exposes_ai_state_component_only_when_rendered():
    prompt = "\n\n".join([
        f"## 核心规则\n{defaults.SYSTEM_BASE_PROMPT}",
        f"## 你的隐性状态约束\n{defaults.CHAT_AI_STATE_CONSTRAINT_PROMPT.format(activity='看书', status='busy')}",
        f"## 回复要求\n{defaults.RESPONSE_INSTRUCTION_PROMPT.format(n=1, total=220, max_per=80)}",
    ])
    enriched = trace_enrich.enrich_step(_fake_llm_step(prompt))

    keys = [item["prompt_key"] for item in enriched["prompt_components"]]
    assert keys == ["chat.system_base", "chat.ai_state_constraint", "chat.response_instruction"]


def test_boundary_reply_exposes_persona_lock_component():
    prompt = defaults.BOUNDARY_PERSONA_LOCK_PROMPT + defaults.LIGHT_ATTACK_REPLY_PROMPT
    enriched = trace_enrich.enrich_step(_fake_llm_step(prompt))

    keys = [item["prompt_key"] for item in enriched["prompt_components"]]
    assert keys == ["boundary.persona_lock", "boundary.light_attack_reply"]


def test_trace_prompt_key_coverage_matches_runtime_surfaces():
    """后台 prompt 要么是独立 LLM step, 要么是组合组件, 要么明确标成非运行时。"""
    registered = {meta.prompt_key for _, meta in trace_enrich._REGISTRY}
    covered = (
        registered
        | trace_enrich._COMPONENT_ONLY_PROMPT_KEYS
        | trace_enrich._NON_RUNTIME_PROMPT_KEYS
    )
    assert set(PROMPT_DEFINITION_MAP) - covered == set()


def test_all_defaults_prompt_constants_are_registered():
    """defaults.py 是唯一 prompt 文案源; 其中每个 *_PROMPT 都必须出现在后台 registry。"""
    import ast
    from pathlib import Path

    defaults_path = Path(defaults.__file__)
    tree = ast.parse(defaults_path.read_text(encoding="utf-8"))
    prompt_constant_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.endswith("_PROMPT")
    }

    registered_names = {
        definition.default_text
        for definition in PROMPT_DEFINITION_MAP.values()
    }
    defaults_by_name = {
        name: getattr(defaults, name)
        for name in prompt_constant_names
    }
    missing = {
        name
        for name, value in defaults_by_name.items()
        if value not in registered_names
    }
    assert missing == set()


@pytest.mark.parametrize("prompt_const,expected_key,params", [
    (
        defaults.SCHEDULE_QUERY_REPLY_PROMPT,
        "intent.schedule_query_reply",
        {
            "message": "你明天忙吗",
            "context": "（无）",
            "personality_brief": "真诚朋友",
            "user_portrait": "",
            "current_activity": "",
            "ai_schedule": "09:00-10:00 早餐",
            "ai_portrait": "普通人",
        },
    ),
    (
        defaults.DAILY_SCHEDULE_PROMPT,
        "schedule.daily_schedule",
        {
            "date": "2026-05-09",
            "weekday": "星期六",
            "day_kind": "周末",
            "name": "Lua",
            "age": 28,
            "occupation": "咖啡师",
            "personality_brief": "真诚朋友",
            "overview": "生活稳定",
        },
    ),
    (
        defaults.DELAY_EXPLANATION_PROMPT,
        "reply.delay_explanation",
        {
            "received_time": "13:00",
            "activity": "洗澡",
            "status": "忙碌",
            "current_time": "13:30",
            "delay_minutes": 30,
        },
    ),
])
def test_formatted_runtime_prompts_still_match(prompt_const, expected_key, params):
    """含占位符的 prompt 格式化后也必须能匹配, 避免只测 defaults 原文。"""
    prompt = prompt_const.format_map(SafeDict(params))
    enriched = trace_enrich.enrich_step(_fake_llm_step(prompt))
    assert enriched["prompt_key"] == expected_key


# ──────────────────────────────────────────────────────────────────────
# 兜底分支
# ──────────────────────────────────────────────────────────────────────


def test_unmatched_prompt_falls_back_to_other():
    step = _fake_llm_step("【一段我们没注册的随便什么 prompt 头部】")
    enriched = trace_enrich.enrich_step(step)
    assert enriched["prompt_key"] is None
    assert enriched["category"] == "other"
    assert enriched["display_name"] == "ChatOpenAI"


def test_non_llm_step_skipped():
    step = {
        "id": "x", "name": "chat_request", "run_type": "chain",
        "inputs": {}, "outputs": {},
    }
    enriched = trace_enrich.enrich_step(step)
    assert enriched["category"] == "chain"
    assert enriched["display_name"] == "chat_request"
    assert enriched["prompt_key"] is None


def test_missing_inputs_falls_back():
    step = {
        "id": "x", "name": "ChatOpenAI", "run_type": "llm",
        "inputs": None, "outputs": None,
    }
    enriched = trace_enrich.enrich_step(step)
    assert enriched["category"] == "other"
    assert enriched["prompt_key"] is None


# ──────────────────────────────────────────────────────────────────────
# Decision label 提取
# ──────────────────────────────────────────────────────────────────────


def test_label_memory_relevance_passthrough():
    step = _fake_llm_step(defaults.MEMORY_RELEVANCE_PROMPT, "强")
    enriched = trace_enrich.enrich_step(step)
    assert enriched["decision_label"] == "强"


def test_label_user_emotion_summarized():
    output = '{"emotion": "焦虑", "intensity": 70, "confidence": 0.8}'
    step = _fake_llm_step(defaults.USER_EMOTION_LABEL_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert "焦虑" in enriched["decision_label"]
    assert "70" in enriched["decision_label"]


def test_label_contradiction_no_conflict():
    output = '```json\n{"has_conflict": false}\n```'
    step = _fake_llm_step(defaults.MEMORY_CONTRADICTION_DETECTION_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert enriched["decision_label"] == "无矛盾"


def test_label_contradiction_with_conflict():
    output = '{"has_conflict": true, "conflict_description": "搬家了"}'
    step = _fake_llm_step(defaults.MEMORY_CONTRADICTION_DETECTION_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert "有矛盾" in enriched["decision_label"]
    assert "搬家" in enriched["decision_label"]


# test_label_split_n 已删除 — REPLY_SPLIT_3_PROMPT 已废 (主 LLM 直接按 || 输出)


def test_label_emotion():
    output = '{"emotion": "高兴", "intensity": 75}'
    step = _fake_llm_step(defaults.AI_REPLY_EMOTION_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert "高兴" in enriched["decision_label"]
    assert "75" in enriched["decision_label"]


def test_label_extractor_failure_falls_back_to_truncated_output():
    """label_extractor 抛异常时, 回落到 output 前 30 字."""
    output = "not valid json {{{garbled"
    step = _fake_llm_step(defaults.USER_EMOTION_LABEL_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert enriched["decision_label"] is not None
    assert "not valid json" in enriched["decision_label"]


def test_label_crisis_followup_classify_summarized():
    output = '{"status": "guard", "reason": "用户仅回复谢谢，未明确表示安全"}'
    step = _fake_llm_step(defaults.CRISIS_FOLLOWUP_CLASSIFY_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert enriched["display_name"] == "危机后续状态判定"
    assert enriched["prompt_key"] == "intent.crisis_followup_classify"
    assert "继续保护" in enriched["decision_label"]


def test_enrich_steps_batch_inplace():
    steps = [
        _fake_llm_step(defaults.MEMORY_RELEVANCE_PROMPT, "弱"),
        _fake_llm_step(defaults.USER_EMOTION_LABEL_PROMPT, '{"emotion": "悲伤", "intensity": 55}'),
    ]
    result = trace_enrich.enrich_steps(steps)
    assert result is steps  # in-place
    assert steps[0]["prompt_key"] == "memory.relevance"
    assert steps[1]["prompt_key"] == "emotion.user_label"


# ──────────────────────────────────────────────────────────────────────
# P4b: critical path 标记
# ──────────────────────────────────────────────────────────────────────


def _step(id_, parent_id, started, ended):
    return {
        "id": id_,
        "name": "n",
        "run_type": "chain",
        "parent_id": parent_id,
        "started_at": started,
        "ended_at": ended,
        "inputs": None,
        "outputs": None,
    }


def test_critical_path_simple_serial_chain():
    """root → A → B (串行): 三者都在关键路径上."""
    steps = [
        _step("root", None, "2026-04-28T00:00:00Z", "2026-04-28T00:00:10Z"),
        _step("A", "root", "2026-04-28T00:00:00Z", "2026-04-28T00:00:05Z"),
        _step("B", "A", "2026-04-28T00:00:05Z", "2026-04-28T00:00:10Z"),
    ]
    trace_enrich.enrich_steps(steps)
    by_id = {s["id"]: s for s in steps}
    assert by_id["root"].get("on_critical_path") is True
    assert by_id["A"].get("on_critical_path") is True
    assert by_id["B"].get("on_critical_path") is True


def test_critical_path_parallel_takes_slower():
    """root → A (1s 并行) + B (5s 并行) → only B 在路径上."""
    steps = [
        _step("root", None, "2026-04-28T00:00:00Z", "2026-04-28T00:00:05Z"),
        _step("A", "root", "2026-04-28T00:00:00Z", "2026-04-28T00:00:01Z"),
        _step("B", "root", "2026-04-28T00:00:00Z", "2026-04-28T00:00:05Z"),
    ]
    trace_enrich.enrich_steps(steps)
    by_id = {s["id"]: s for s in steps}
    assert by_id["root"].get("on_critical_path") is True
    assert by_id["B"].get("on_critical_path") is True
    # A 比 B 早结束, 不在关键路径
    assert by_id["A"].get("on_critical_path") is not True


def test_critical_path_empty_safe():
    """空 steps 不报错."""
    trace_enrich.enrich_steps([])  # 不抛


def test_critical_path_no_root_safe():
    """无 root (所有节点都有 parent_id) 优雅处理 — 应该不挂."""
    steps = [
        _step("A", "missing-parent", "2026-04-28T00:00:00Z", "2026-04-28T00:00:01Z"),
    ]
    trace_enrich.enrich_steps(steps)
    # 没 root 就不标 critical_path, 不报错
    assert steps[0].get("on_critical_path") is not True
