"""Trace step semantic enrichment.

把 LangSmith 原始 step (run_type='llm', name='ChatOpenAI' 等) 映射到带语义的字段:
- display_name: 让 PM 一眼看懂的中文功能名 (e.g. "记忆相关度判定")
- category: decision / data / reply / post / other (前端配色用)
- prompt_key: 关联到 prompting registry, 详情面板提供 trace 内编辑
- decision_label: 提取关键决策 (e.g. "弱" / "无矛盾" / "偏积极"), 替代生 output

新 trace 优先使用渲染期记录的 prompt_hash + 组件 span, 不靠文本猜测.
旧 trace 没有渲染期元数据时, 从 prompting registry 的默认模板自动派生稳定指纹做
graceful fallback; 指纹来源仍是 defaults.py, 不在本文件复制 prompt 文案.

供 public_trace.load_public_trace 在返回前调用; 失败时 graceful degrade
(category='other', display_name=run.name).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.trace_components import prompt_hash

logger = logging.getLogger(__name__)


Category = str  # "decision" | "data" | "reply" | "post" | "other"


@dataclass(frozen=True)
class _PromptMeta:
    prompt_key: str
    display_name: str
    category: Category
    label_extractor: Callable[[str], str | None] | None = None


_MAIN_PROMPT_ALWAYS_COMPONENT_KEYS = ["chat.system_base"]

_MAIN_PROMPT_SECTION_COMPONENTS = [
    ("## 反幻觉硬约束", "chat.anti_hallucination_hard_rule"),
    ("## 对话一致性", "chat.consistency_rules"),
    ("## 你的隐性状态约束", "chat.ai_state_constraint"),
    ("## 回复要求", "chat.response_instruction"),
]

_BOUNDARY_BODY_PROMPT_KEYS = {
    "boundary.final_warning",
    "boundary.light_attack_reply",
    "boundary.medium_attack_reply",
    "boundary.severe_attack_reply",
    "boundary.medium_patience_reply",
    "boundary.low_patience_reply",
    "boundary.blacklist_reply",
}

# Registry 中不作为独立 LangSmith LLM step 出现、但会作为组合 prompt 的可编辑片段出现.
_COMPONENT_ONLY_PROMPT_KEYS = {
    "boundary.persona_lock",
    "chat.ai_state_constraint",
    "chat.anti_hallucination_hard_rule",
    "chat.consistency_rules",
    "chat.l3_memory_section",
    "chat.memory_empty_anchor",
    "chat.memory_section_body",
    "chat.relationship_stage_section",
    "chat.response_instruction",
    "chat.special_instruction_appendix",
    "chat.time_memories_section",
    "intent.conversation_end_fallback_instruction",
    "intent.schedule_missing_context",
    "reply.delay_explanation_fallback_instruction",
}

# 当前运行时不参与任何 LLM prompt, 但保留在后台管理里用于历史兼容/未来恢复.
_NON_RUNTIME_PROMPT_KEYS = {"chat.personality_rules"}


def _prompt_admin_meta(prompt_key: str) -> dict[str, Any]:
    definition = PROMPT_DEFINITION_MAP.get(prompt_key)
    payload: dict[str, Any] = {"prompt_key": prompt_key}
    if definition:
        payload.update({
            "title": definition.title,
            "stage": definition.stage,
            "category": definition.category,
            "description": definition.description,
        })
    return payload


def _component_admin_meta(component: dict[str, Any]) -> dict[str, Any] | None:
    prompt_key = component.get("prompt_key")
    if not isinstance(prompt_key, str) or not prompt_key:
        return None
    payload = _prompt_admin_meta(prompt_key)
    for key in ("start", "end", "editable"):
        if key in component:
            payload[key] = component[key]
    return payload


def _main_prompt_components(rendered_prompt: str) -> list[dict[str, Any]]:
    keys = list(_MAIN_PROMPT_ALWAYS_COMPONENT_KEYS)
    for section_header, prompt_key in _MAIN_PROMPT_SECTION_COMPONENTS:
        if section_header in rendered_prompt:
            keys.append(prompt_key)
    return [_prompt_admin_meta(key) for key in keys]


def _boundary_prompt_components(body_prompt_key: str) -> list[dict[str, Any]]:
    return [
        _prompt_admin_meta("boundary.persona_lock"),
        _prompt_admin_meta(body_prompt_key),
    ]


# ─────────────────────────────────────────────────────────────────
# Decision label extractors — 失败时返回 None, 调用方截断 output 兜底
# ─────────────────────────────────────────────────────────────────


def _label_passthrough(output: str) -> str | None:
    """直接用 output (适合输出"强/中/弱"这种单 token 决策)."""
    text = (output or "").strip()
    if not text or len(text) > 30:
        return None
    return text


def _label_strip_codeblock(output: str) -> str:
    """剥掉 markdown 代码块包裹, 取内部. JSON 输出常见这个格式."""
    text = (output or "").strip()
    if text.startswith("```"):
        # ```json\n{...}\n```
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _label_emotion(output: str) -> str | None:
    """Emotion label JSON → '标签 (强度 N)' 摘要."""
    try:
        data = json.loads(_label_strip_codeblock(output))
        label = str(data.get("emotion") or "").strip()
        intensity = int(float(data.get("intensity", 0)))
    except Exception:
        return None
    return f"{label or '中性'} (强度 {max(0, min(100, intensity))})"


def _label_contradiction(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    if data.get("has_conflict"):
        desc = str(data.get("conflict_description") or "").strip()
        return f"有矛盾: {desc[:20]}" if desc else "有矛盾"
    return "无矛盾"


def _label_intent_unified(output: str) -> str | None:
    """输出可能是单一 label 或顿号分隔多 label."""
    text = (output or "").strip()
    if not text:
        return None
    # "日常交流" / "日常交流、终结意图"
    return text[:40]


def _label_apology(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    if data.get("is_apology"):
        sincerity = data.get("sincerity")
        try:
            return f"道歉 (诚意 {float(sincerity):.2f})"
        except Exception:
            return "道歉"
    return "非道歉"


def _label_attack_level(output: str) -> str | None:
    text = (output or "").strip()
    if text in ("K1", "K2", "K3"):
        mapping = {"K1": "K1 轻度", "K2": "K2 中度", "K3": "K3 重度"}
        return mapping[text]
    return _label_passthrough(text)


def _label_split_n(output: str) -> str | None:
    """拆句 prompt 的输出是 N 行, 摘要为'拆出 N 句'."""
    lines = [ln for ln in (output or "").strip().split("\n") if ln.strip()]
    if not lines:
        return None
    return f"拆出 {len(lines)} 句"


def _label_emotion(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
        emo = str(data.get("emotion") or "").strip()
        intensity = data.get("intensity")
        if emo and intensity is not None:
            return f"{emo} (强度 {intensity})"
        return emo or None
    except Exception:
        return None


def _label_judge_remember(output: str) -> str | None:
    """记忆 pre-filter 输出'记/不记'."""
    text = (output or "").strip()
    return text if text in ("记", "不记") else None


def _label_extraction(output: str) -> str | None:
    """记忆抽取输出 JSON memories list, 摘要 '抽到 N 条'."""
    try:
        data = json.loads(_label_strip_codeblock(output))
        items = data.get("memories")
        if isinstance(items, list):
            return f"抽到 {len(items)} 条" if items else "无可抽"
    except Exception:
        pass
    return None


def _label_reply_text(output: str) -> str | None:
    """回复类: 直接用 output 前 40 字."""
    text = (output or "").strip()
    if not text:
        return None
    return text[:40] + ("…" if len(text) > 40 else "")


def _label_crisis_followup_classify(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    status = str(data.get("status") or "").strip()
    if status not in {"guard", "release"}:
        return None
    reason = str(data.get("reason") or "").strip()
    label = "继续保护" if status == "guard" else "解除危机"
    return f"{label}: {reason[:20]}" if reason else label


def _label_json_bool(output: str, field: str, true_label: str, false_label: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    if field not in data:
        return None
    return true_label if bool(data.get(field)) else false_label


def _label_crisis_message(output: str) -> str | None:
    return _label_json_bool(output, "is_crisis", "危机", "非危机")


def _label_schedule_adjust(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    reply = str(data.get("reply") or "").strip()
    adjustment = str(data.get("adjustment") or "").strip()
    if adjustment:
        return f"{adjustment[:30]}"
    if reply:
        return reply[:40] + ("…" if len(reply) > 40 else "")
    return None


def _label_reminder_precheck(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    state = str(data.get("state") or "").strip()
    if not state:
        return None
    reason = str(data.get("reason") or "").strip()
    return f"{state}: {reason[:20]}" if reason else state


def _label_ids(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    ids = data.get("ids")
    if isinstance(ids, list):
        return f"选中 {len(ids)} 条" if ids else "无相关记忆"
    return None


def _label_schedule_items(output: str) -> str | None:
    try:
        data = json.loads(_label_strip_codeblock(output))
    except Exception:
        return None
    if isinstance(data, list):
        return f"生成 {len(data)} 段"
    return None


# ─────────────────────────────────────────────────────────────────
# Legacy prompt 指纹映射表
#
# 新 trace 不依赖这里: 渲染期已经记录 prompt_hash + component span.
# 这里仅服务没有 metadata 的旧 trace, 且 fingerprint 必须从 defaults.py /
# registry 的 default_text 自动派生, 避免 prompt 文案散落在业务代码中.
# ─────────────────────────────────────────────────────────────────


_REGISTRY: list[tuple[str, _PromptMeta]] = []
_FORMAT_FIELD_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*(?:![^}:]+)?(?::[^}]+)?\}")
_PROMPT_LABEL_RE = re.compile(r"^【[^】]+】")
_DISPLAY_SPEC_SUFFIX_RE = re.compile(r"\s*[（(][^）)]*(?:§|spec)[^）)]*[）)]", re.IGNORECASE)
_COMMON_FINGERPRINT_SNIPPETS = (
    "只作为用户的线上好友",
    "和用户不会有任何线下交集",
    "你是一个真人朋友",
    "你不是AI",
    "你是发文字消息的真人朋友",
    "描述自己时**不许把事归因到用户**",
    "不加任何额外说明",
    "只输出回复内容",
    "只输出自然语言描述",
    "严格按以下",
)


def _clean_display_name(display_name: str) -> str:
    """Trace step names are product labels; hide implementation/spec section refs."""
    return _DISPLAY_SPEC_SUFFIX_RE.sub("", display_name).strip()


def _register(fingerprint: str, meta: _PromptMeta) -> None:
    if not fingerprint or len(fingerprint) < 8:
        logger.warning(f"[trace_enrich] fingerprint too short for {meta.prompt_key}, may collide")
    _REGISTRY.append((fingerprint, meta))


def _candidate_fingerprint_chunks(line: str) -> list[str]:
    chunks: list[str] = []
    cleaned = _PROMPT_LABEL_RE.sub("", line.strip()).strip()
    if not cleaned:
        return chunks
    if re.match(r"^\d+[.．]", cleaned) or cleaned.startswith("- "):
        return chunks
    for chunk in _FORMAT_FIELD_RE.split(cleaned):
        value = chunk.strip()
        if not value:
            continue
        if value.endswith(("：\"", ":\"", "：", ":")):
            continue
        parts = [value]
        if any(common in value for common in _COMMON_FINGERPRINT_SNIPPETS):
            parts = [
                part.strip()
                for part in re.split(r"(?<=[。！？.!?])", value)
                if part.strip()
                and not any(common in part for common in _COMMON_FINGERPRINT_SNIPPETS)
            ]
        if "{" in value or "}" in value:
            continue
        for part in parts:
            if part.startswith("（供"):
                continue
            if len(part) >= 8:
                chunks.append(part)
    return chunks


def _default_fingerprint(prompt_key: str) -> str:
    definition = PROMPT_DEFINITION_MAP.get(prompt_key)
    if definition is None:
        raise KeyError(f"Prompt not registered: {prompt_key}")
    candidates: list[str] = []
    for raw_line in definition.default_text.splitlines():
        candidates.extend(_candidate_fingerprint_chunks(raw_line))
    if not candidates:
        compact = _FORMAT_FIELD_RE.sub("", definition.default_text)
        compact = re.sub(r"\s+", " ", compact).strip()
        if compact:
            candidates.append(compact)
    if not candidates:
        raise ValueError(f"Cannot derive prompt fingerprint: {prompt_key}")
    return candidates[0][:90]


def _register_prompt(
    prompt_key: str,
    display_name: str,
    category: Category,
    label_extractor: Callable[[str], str | None] | None,
) -> None:
    _register(_default_fingerprint(prompt_key), _PromptMeta(
        prompt_key, _clean_display_name(display_name), category, label_extractor,
    ))


_PROMPT_FALLBACK_REGISTRATIONS: list[
    tuple[str, str, Category, Callable[[str], str | None] | None]
] = [
    # Decision 类
    ("memory.relevance", "记忆相关度判定", "decision", _label_passthrough),
    ("intent.unified", "统一意图识别", "decision", _label_intent_unified),
    ("intent.split", "多意图拆分", "decision", _label_passthrough),
    ("memory.l3_trigger", "L3 唤醒判定", "decision", _label_passthrough),
    ("memory.contradiction_detection", "L1 矛盾检测", "decision", _label_contradiction),
    ("memory.contradiction_analysis", "矛盾分析", "decision", _label_passthrough),
    ("boundary.apology", "道歉检测", "decision", _label_apology),
    ("boundary.positive_interaction", "正向互动判断", "decision", _label_passthrough),
    ("boundary.attack_target", "攻击目标识别", "decision", _label_passthrough),
    ("boundary.attack_level", "攻击级别识别", "decision", _label_attack_level),
    ("boundary.banned_word", "违禁词判断", "decision", _label_passthrough),
    ("memory.judgement_user", "用户记忆预筛", "decision", _label_judge_remember),
    ("memory.judgement_ai", "AI 自我记忆预筛", "decision", _label_judge_remember),
    ("memory.deletion_intent", "记忆删除/改期意图判定", "decision", _label_passthrough),
    ("proactive.reminder_pre_check", "提醒触发前状态判别", "decision", _label_reminder_precheck),
    ("memory.pairwise_contradiction", "L1 一致性扫描", "decision", _label_passthrough),
    ("memory.reconciliation", "记忆事实演化裁决", "decision", _label_passthrough),
    ("intent.crisis_message_classify", "危机消息语义判定", "decision", _label_crisis_message),
    ("proactive.memory_topic_rerank", "主动话题记忆重排", "decision", _label_ids),
    ("intent.crisis_followup_classify", "危机后续状态判定", "decision", _label_crisis_followup_classify),
    ("music.user_pause_followup_decision", "共听暂停后跟进判定", "decision", _label_passthrough),

    # Data 类
    ("emotion.user_label", "用户情绪标签", "data", _label_emotion),
    ("agent.personality_scoring", "AI 性格打分", "data", _label_passthrough),
    ("character.generation", "AI 背景生成", "data", _label_passthrough),
    ("character.repair_missing_fields", "背景缺字段补齐", "data", _label_passthrough),
    ("schedule.life_overview", "生活画像生成", "data", _label_reply_text),
    ("schedule.daily_schedule_with_memory", "每日作息生成(带记忆)", "data", _label_schedule_items),
    ("schedule.daily_schedule", "每日作息生成", "data", _label_schedule_items),
    ("schedule.daily_summary", "昨日生活总结", "data", _label_reply_text),
    ("schedule.daily_summary_memories", "昨日总结记忆拆分", "data", _label_extraction),
    ("portrait.generation", "用户画像生成", "data", _label_reply_text),
    ("portrait.update", "用户画像更新", "data", _label_reply_text),
    ("portrait.tags", "用户画像标签生成", "data", _label_passthrough),
    ("offline.activity_card", "线下活动推荐卡生成", "data", _label_reply_text),
    ("offline.gift_selection", "线下礼物选择", "data", _label_reply_text),
    ("music.co_listening_context", "共听上下文", "data", _label_reply_text),

    # Reply 类
    ("intent.current_state_reply", "询问当前状态回复", "reply", _label_reply_text),
    ("intent.crisis_reply", "危机求助回复", "reply", _label_reply_text),
    ("intent.crisis_followup_reply", "危机后续跟进回复", "reply", _label_reply_text),
    ("intent.end_reply", "终结意图回复", "reply", _label_reply_text),
    ("intent.schedule_query_reply", "计划查询回复", "reply", _label_reply_text),
    ("intent.schedule_adjust_reply", "作息调整回复", "reply", _label_schedule_adjust),
    ("boundary.apology_reply", "道歉/承诺回复", "reply", _label_reply_text),
    ("intent.deletion_confirm", "删除确认", "reply", _label_reply_text),
    ("intent.deletion_reply", "删除完成回复", "reply", _label_reply_text),
    ("memory.contradiction_inquiry", "矛盾询问", "reply", _label_reply_text),
    ("memory.contradiction_reply", "矛盾化解回复", "reply", _label_reply_text),
    ("boundary.final_warning", "最终警告", "reply", _label_reply_text),
    ("boundary.light_attack_reply", "轻度攻击回复 (K1)", "reply", _label_reply_text),
    ("boundary.medium_attack_reply", "中度攻击回复 (K2)", "reply", _label_reply_text),
    ("boundary.severe_attack_reply", "重度攻击回复 (K3)", "reply", _label_reply_text),
    ("boundary.medium_patience_reply", "中耐心回复", "reply", _label_reply_text),
    ("boundary.low_patience_reply", "低耐心回复", "reply", _label_reply_text),
    ("boundary.blacklist_reply", "拉黑回复", "reply", _label_reply_text),
    ("memory.medium_reply", "中记忆回复", "reply", _label_reply_text),
    ("memory.strong_reply", "强记忆回复", "reply", _label_reply_text),
    ("memory.l3_reply", "久远记忆回复", "reply", _label_reply_text),
    ("memory.weak_reply", "弱记忆回复", "reply", _label_reply_text),
    ("reply.delay_explanation", "延迟解释回复", "reply", _label_reply_text),
    ("proactive.reminder_message", "提醒发送消息", "reply", _label_reply_text),
    ("intent.record_confirm_reply", "记录请求确认回复", "reply", _label_reply_text),
    ("intent.record_ask_time", "记录请求反问时间", "reply", _label_reply_text),
    ("proactive.special_reminder", "特殊日期提醒", "reply", _label_reply_text),
    ("proactive.silence_plain", "沉默唤醒(无记忆)", "reply", _label_reply_text),
    ("proactive.silence_ai_memory", "沉默唤醒(AI记忆)", "reply", _label_reply_text),
    ("proactive.silence_user_memory", "沉默唤醒(用户记忆)", "reply", _label_reply_text),
    ("proactive.silence_schedule", "沉默唤醒(作息)", "reply", _label_reply_text),
    ("proactive.memory_ai", "记忆主动(AI记忆)", "reply", _label_reply_text),
    ("proactive.memory_user", "记忆主动(用户记忆)", "reply", _label_reply_text),
    ("proactive.scheduled_scene", "定时情景(AI作息)", "reply", _label_reply_text),
    ("proactive.decay_final", "衰减最后一次回复", "reply", _label_reply_text),
    ("proactive.first_greeting", "AI首次打招呼", "reply", _label_reply_text),
    ("proactive.special_holiday", "特殊日期(节日)", "reply", _label_reply_text),
    ("proactive.special_birthday", "特殊日期(生日)", "reply", _label_reply_text),
    ("proactive.special_combined", "特殊日期(合并)", "reply", _label_reply_text),
    ("offline.gift_thanks_reply", "礼物感谢回应", "reply", _label_reply_text),
    ("music.accept_invite", "接受共听邀请", "reply", _label_reply_text),
    ("music.agent_join_after_busy", "忙碌后加入共听", "reply", _label_reply_text),
    ("music.agent_late_missed", "错过共听", "reply", _label_reply_text),
    ("music.busy_exit", "忙碌退出共听", "reply", _label_reply_text),
    ("music.busy_reject", "忙碌拒绝共听", "reply", _label_reply_text),
    ("music.proactive_recommend", "主动推荐共听", "reply", _label_reply_text),
    ("music.sleep_reject", "睡眠拒绝共听", "reply", _label_reply_text),
    ("music.switch_track", "切换共听曲目", "reply", _label_reply_text),
    ("music.track_changed_auto", "自动换歌提醒", "reply", _label_reply_text),
    ("music.track_changed_manual", "手动换歌提醒", "reply", _label_reply_text),
    ("music.user_absent_exit", "用户缺席退出共听", "reply", _label_reply_text),
    ("music.user_exit", "用户退出共听", "reply", _label_reply_text),
    ("music.user_pause_exit", "用户暂停退出共听", "reply", _label_reply_text),
    ("chat.system_base", "主回复", "reply", _label_reply_text),

    # Post 类
    ("reply.emotion_detection", "回复情绪识别", "post", _label_emotion),
    ("memory.extraction_user", "用户记忆抽取", "post", _label_extraction),
    ("memory.extraction_ai", "AI 自我记忆抽取", "post", _label_extraction),
]

for _prompt_key, _display_name, _category, _extractor in _PROMPT_FALLBACK_REGISTRATIONS:
    _register_prompt(_prompt_key, _display_name, _category, _extractor)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def _extract_first_user_message(inputs: Any) -> str | None:
    """LangSmith run.inputs 的 messages[0][0] 通常是 HumanMessage,
    但也可能直接是 list of dicts. 兼容多种形式取出 content 字符串."""
    if not isinstance(inputs, dict):
        return None
    messages = inputs.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    # messages 可能是 [[msg, msg, ...]] 或 [msg, msg, ...]
    first_group = messages[0]
    if isinstance(first_group, list):
        candidate = first_group[0] if first_group else None
    else:
        candidate = first_group
    if not isinstance(candidate, dict):
        return None
    kwargs = candidate.get("kwargs") or {}
    content = kwargs.get("content")
    if isinstance(content, str):
        return content
    return None


def _extract_output_text(outputs: Any) -> str:
    """LangSmith run.outputs.generations[0][0].text 通常就是 LLM 输出文本."""
    if not isinstance(outputs, dict):
        return ""
    generations = outputs.get("generations")
    if not isinstance(generations, list) or not generations:
        return ""
    first_group = generations[0]
    if isinstance(first_group, list) and first_group:
        first = first_group[0]
    else:
        first = first_group
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text
    # Fallback: message.kwargs.content
    msg = first.get("message")
    if isinstance(msg, dict):
        kwargs = msg.get("kwargs") or {}
        content = kwargs.get("content")
        if isinstance(content, str):
            return content
    return ""


def enrich_step(step: dict[str, Any]) -> dict[str, Any]:
    """给 normalized step 加 4 个语义字段. 只增不改.

    匹配失败时 (兜底分支): display_name=step.name, category="other",
    prompt_key=None, decision_label=None.

    LLM 流式输出 (qwen3.5-plus 主回复) 的 token_count 通常为 0,
    但 outputs 仍有 text — 仍能识别并用 _label_reply_text 截断.
    """
    if step.get("run_type") != "llm":
        # 非 LLM 节点 (chain / tool 等) 不做语义识别, 只加 category=other
        step["display_name"] = step.get("name") or "Step"
        step["category"] = "chain" if step.get("run_type") == "chain" else "other"
        step["prompt_key"] = None
        step["decision_label"] = None
        return step

    user_msg = _extract_first_user_message(step.get("inputs"))
    if not user_msg:
        step["display_name"] = step.get("name") or "ChatOpenAI"
        step["category"] = "other"
        step["prompt_key"] = None
        step["decision_label"] = None
        return step

    # 找第一个匹配的 fingerprint (substring in user_msg).
    # 多个匹配时取第一个 (注册顺序敏感, 更独特的应放前).
    meta: _PromptMeta | None = None
    for fingerprint, candidate in _REGISTRY:
        if fingerprint in user_msg:
            meta = candidate
            break
    if meta is None:
        step["display_name"] = step.get("name") or "ChatOpenAI"
        step["category"] = "other"
        step["prompt_key"] = None
        step["decision_label"] = None
        return step

    output_text = _extract_output_text(step.get("outputs"))
    label: str | None = None
    if meta.label_extractor:
        try:
            label = meta.label_extractor(output_text)
        except Exception as e:
            logger.debug(
                f"[trace_enrich] label_extractor failed prompt_key={meta.prompt_key}: {e}"
            )
            label = None
    if label is None and output_text:
        # Fallback: 截 output 前 30 字
        label = output_text.strip()[:30]
        if len(output_text.strip()) > 30:
            label += "…"

    step["display_name"] = meta.display_name
    step["category"] = meta.category
    step["prompt_key"] = meta.prompt_key
    admin_meta = _prompt_admin_meta(meta.prompt_key)
    step["prompt_title"] = admin_meta.get("title")
    step["prompt_stage"] = admin_meta.get("stage")
    step["prompt_category"] = admin_meta.get("category")
    step["prompt_description"] = admin_meta.get("description")
    if meta.prompt_key == "chat.system_base":
        step["prompt_components"] = _main_prompt_components(user_msg)
    elif meta.prompt_key in _BOUNDARY_BODY_PROMPT_KEYS:
        step["prompt_components"] = _boundary_prompt_components(meta.prompt_key)
    step["decision_label"] = label
    return step


def enrich_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量包装. 修改原 dict 并返回 (in-place)."""
    for step in steps:
        enrich_step(step)
    _mark_critical_path(steps)
    return steps


def apply_prompt_render_traces(
    steps: list[dict[str, Any]],
    prompt_render_traces: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Attach exact prompt component metadata captured at render time.

    `enrich_step` can still label old traces using fingerprints, but new traces
    should not infer composite prompt parts by section text. The message
    metadata stores a SHA-256 hash of the exact rendered prompt plus span-based
    component provenance; this function joins that metadata back onto the
    LangSmith steps.
    """
    if not prompt_render_traces:
        return steps

    by_hash: dict[str, dict[str, Any]] = {}
    for item in prompt_render_traces:
        if not isinstance(item, dict):
            continue
        h = item.get("prompt_hash")
        if isinstance(h, str) and h:
            by_hash[h] = item
    if not by_hash:
        return steps

    for step in steps:
        if step.get("run_type") != "llm":
            continue
        rendered_prompt = _extract_first_user_message(step.get("inputs"))
        if not rendered_prompt:
            continue
        matched = by_hash.get(prompt_hash(rendered_prompt))
        if not matched:
            continue

        prompt_key = matched.get("prompt_key")
        if isinstance(prompt_key, str) and prompt_key in PROMPT_DEFINITION_MAP:
            step["prompt_key"] = prompt_key
            admin_meta = _prompt_admin_meta(prompt_key)
            step["prompt_title"] = admin_meta.get("title")
            step["prompt_stage"] = admin_meta.get("stage")
            step["prompt_category"] = admin_meta.get("category")
            step["prompt_description"] = admin_meta.get("description")

        components = matched.get("components")
        if isinstance(components, list):
            enriched_components = [
                meta for component in components
                if isinstance(component, dict)
                for meta in [_component_admin_meta(component)]
                if meta is not None
            ]
            if enriched_components:
                step["prompt_components"] = enriched_components
        step["prompt_render_source"] = matched.get("source")
    return steps


# ─────────────────────────────────────────────────────────────────
# P4b: 关键路径标记 - 跑完所有 enrich_step 之后, 算 critical path
# ─────────────────────────────────────────────────────────────────


def _end_ms(step: dict[str, Any]) -> int:
    """从 step.ended_at (ISO8601) 解析为 epoch ms. 缺失返回 0."""
    end_str = step.get("ended_at")
    if not end_str:
        return 0
    try:
        from datetime import datetime
        return int(
            datetime.fromisoformat(str(end_str).replace("Z", "+00:00")).timestamp() * 1000
        )
    except Exception:
        return 0


def _mark_critical_path(steps: list[dict[str, Any]]) -> None:
    """关键路径定义: 从 root 出发, 每层选 ended_at 最晚的 child, 递归到底.

    背后的直觉: parent 完成时间 = max(children 完成时间), 决定 parent
    完成时间的那个 child 是"卡 parent 的瓶颈". 整条链就是导致总耗时的路径,
    优化它能直接缩短整次请求.

    跟"longest path by sum of durations"的严格定义有差别 (并行场景 sum 会
    大于真实 wall-clock latency), 但跟用户对"关键路径"的直觉更对齐 — 我们
    在意的是 wall-clock 慢在哪.

    所有在路径上的 step 加 on_critical_path=True. 路径外的不写字段
    (前端用 step.on_critical_path === true 判断, undefined 视为 false).
    """
    if not steps:
        return
    # 按 parent_id 分组
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for s in steps:
        by_parent.setdefault(s.get("parent_id"), []).append(s)

    # 找 root: parent_id 为 None 的节点中 ended_at 最晚的 (一般只有 1 个)
    roots = by_parent.get(None) or []
    if not roots:
        return
    cur: dict[str, Any] | None = max(roots, key=_end_ms)

    while cur is not None:
        cur["on_critical_path"] = True
        children = by_parent.get(cur.get("id")) or []
        if not children:
            break
        cur = max(children, key=_end_ms)
