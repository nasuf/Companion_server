"""Tests for trace_enrich semantic enrichment.

CI 守 prompt 头部 drift: defaults.py 任何注册过的 prompt 头部 60 字改了,
test_each_registered_prompt_matches 就会失败, 提示更新映射表.
"""

from __future__ import annotations

import pytest

from app.services.chat import trace_enrich
from app.services.prompting import defaults


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
    (defaults.INTENT_SPLIT_PROMPT, "intent.split_multi"),
    (defaults.L3_TRIGGER_PROMPT, "intent.l3_trigger"),
    (defaults.MEMORY_CONTRADICTION_DETECTION_PROMPT, "memory.contradiction_detection"),
    (defaults.MEMORY_CONTRADICTION_ANALYSIS_PROMPT, "memory.contradiction_analysis"),
    (defaults.APOLOGY_PROMPT, "intent.apology_detect"),
    (defaults.ATTACK_TARGET_PROMPT, "boundary.attack_target"),
    (defaults.ATTACK_LEVEL_PROMPT, "boundary.attack_level"),
    (defaults.BANNED_WORD_PROMPT, "boundary.banned_word"),
    (defaults.MEMORY_JUDGEMENT_USER_PROMPT, "memory.pre_filter_user"),
    (defaults.MEMORY_JUDGEMENT_AI_PROMPT, "memory.pre_filter_ai"),
    (defaults.DELETION_INTENT_PROMPT, "memory.deletion_intent"),
    (defaults.MEMORY_PAIRWISE_CONTRADICTION_PROMPT, "memory.pairwise_contradiction"),
    (defaults.AI_PAD_PROMPT, "emotion.ai_pad"),
    (defaults.EMOTION_EXTRACTION_PROMPT, "emotion.user_pad"),
    (defaults.CURRENT_STATE_REPLY_PROMPT, "intent.current_state_reply"),
    (defaults.DELETION_CONFIRM_PROMPT, "memory.deletion_confirm"),
    (defaults.DELETION_REPLY_PROMPT, "memory.deletion_reply"),
    (defaults.MEMORY_CONTRADICTION_INQUIRY_PROMPT, "memory.contradiction_inquiry"),
    (defaults.MEMORY_CONTRADICTION_REPLY_PROMPT, "memory.contradiction_reply"),
    (defaults.BOUNDARY_FINAL_WARNING_PROMPT, "boundary.final_warning"),
    (defaults.LIGHT_ATTACK_REPLY_PROMPT, "boundary.attack_reply_light"),
    (defaults.MEDIUM_ATTACK_REPLY_PROMPT, "boundary.attack_reply_medium"),
    (defaults.SEVERE_ATTACK_REPLY_PROMPT, "boundary.attack_reply_severe"),
    (defaults.MEDIUM_PATIENCE_REPLY_PROMPT, "boundary.patience_medium_reply"),
    (defaults.LOW_PATIENCE_REPLY_PROMPT, "boundary.patience_low_reply"),
    (defaults.BLACKLIST_REPLY_PROMPT, "boundary.blacklist_reply"),
    (defaults.SYSTEM_BASE_PROMPT, "chat.system_base"),
    (defaults.REPLY_SPLIT_2_PROMPT, "reply.split_2"),
    (defaults.REPLY_SPLIT_3_PROMPT, "reply.split_3"),
    (defaults.AI_REPLY_EMOTION_PROMPT, "reply.emotion"),
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


def test_label_pad_summarized():
    output = '{"pleasure": 0.7, "arousal": 0.4, "dominance": 0.6}'
    step = _fake_llm_step(defaults.AI_PAD_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert "偏积极" in enriched["decision_label"]
    assert "中等" in enriched["decision_label"] or "平静" in enriched["decision_label"]


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


def test_label_split_n():
    output = "句一\n句二\n句三"
    step = _fake_llm_step(defaults.REPLY_SPLIT_3_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert enriched["decision_label"] == "拆出 3 句"


def test_label_emotion():
    output = '{"emotion": "高兴", "intensity": 75}'
    step = _fake_llm_step(defaults.AI_REPLY_EMOTION_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    assert "高兴" in enriched["decision_label"]
    assert "75" in enriched["decision_label"]


def test_label_extractor_failure_falls_back_to_truncated_output():
    """label_extractor 抛异常时, 回落到 output 前 30 字."""
    output = "not valid json {{{garbled"
    step = _fake_llm_step(defaults.AI_PAD_PROMPT, output)
    enriched = trace_enrich.enrich_step(step)
    # _label_pad 解析失败 → fallback 截断
    assert enriched["decision_label"] is not None
    assert "not valid json" in enriched["decision_label"]


def test_enrich_steps_batch_inplace():
    steps = [
        _fake_llm_step(defaults.MEMORY_RELEVANCE_PROMPT, "弱"),
        _fake_llm_step(defaults.AI_PAD_PROMPT, '{"pleasure": -0.5, "arousal": 0.2, "dominance": 0.3}'),
    ]
    result = trace_enrich.enrich_steps(steps)
    assert result is steps  # in-place
    assert steps[0]["prompt_key"] == "memory.relevance"
    assert steps[1]["prompt_key"] == "emotion.ai_pad"


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
