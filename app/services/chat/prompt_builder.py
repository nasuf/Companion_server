"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Uses seven-dim personality (0-100) to build role-play personality descriptions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.prompting.store import (
    PromptDisabledError,
    get_prompt_text,
    get_prompt_text_or_default,
)
from app.services.prompting.trace_components import record_prompt_render
from app.services.prompting.utils import render_template
from app.services.style import generate_style_examples, generate_style_instruction
from app.services.mbti import format_mbti_for_prompt, get_mbti
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY as _MAX_PER_REPLY,
    MAX_TOTAL_CHARS as _MAX_TOTAL_CHARS,
    CHAT_HISTORY_TOKEN_BUDGET,
)


# ---------------------------------------------------------------------------
# 七维人格 → 自然语言人格描述 (PRD §1.4.3 Prompt模板)
# ---------------------------------------------------------------------------

# MBTI 4 个维度 → (高分描述, 低分描述, 中段描述)
_MBTI_DIM_DESCRIPTIONS: dict[str, tuple[str, str, tuple[str, str, str]]] = {
    "EI": ("E", "I", (
        "外向开朗，喜欢与人互动，从社交中获取能量",
        "内向克制，享受独处，社交后需要独自恢复",
        "介于两者之间，根据场合调整",
    )),
    "NS": ("N", "S", (
        "直觉型，思维抽象、跳跃，喜欢探讨可能性",
        "感觉型，关注现实细节、当下事实",
        "兼具直觉与现实感",
    )),
    "TF": ("T", "F", (
        "思考型，逻辑清晰，按事实和原则做判断",
        "情感型，共情能力高，重视和谐与他人感受",
        "能平衡理性与感性",
    )),
    "JP": ("J", "P", (
        "判断型，喜欢规划、有条理、追求确定性",
        "知觉型，灵活随性、接受变化、活在当下",
        "有计划也能接受变动",
    )),
}


def _format_mbti_detail(mbti: dict) -> str:
    """4 维度数值 + 描述，用于 prompt 详细注入。"""
    lines: list[str] = []
    for i, (key, (hi_letter, lo_letter, (hi_desc, lo_desc, mid_desc))) in enumerate(
        _MBTI_DIM_DESCRIPTIONS.items(), 1,
    ):
        value = mbti.get(key, 50)
        normalized = value / 100
        if normalized >= 0.7:
            desc = hi_desc
            letter = hi_letter
        elif normalized <= 0.3:
            desc = lo_desc
            letter = lo_letter
        else:
            desc = mid_desc
            letter = f"{hi_letter}/{lo_letter}"
        lines.append(f"{i}. {key} [{letter}]：{value} — {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PromptBody:
    body: str
    prompt_key: str | None = None


def _has_prompt_body(body: str | None) -> bool:
    """Treat empty/admin-placeholder prompt text as absent."""
    if not body:
        return False
    stripped = body.strip()
    if not stripped:
        return False
    return bool(stripped.strip("。．.；;：:-—_ \n\t"))


def _append_section(
    sections: list[str],
    components: list[dict[str, Any]],
    title: str,
    body: str,
    *,
    prompt_key: str | None = None,
) -> None:
    prefix_len = 2 if sections else 0  # "\n\n" inserted by final join before this section
    section_header = f"## {title}\n"
    start = sum(len(part) for part in sections) + max(0, len(sections) - 1) * 2
    body_start = start + prefix_len + len(section_header)
    section = section_header + body
    sections.append(section)
    if prompt_key:
        components.append({
            "prompt_key": prompt_key,
            "start": body_start,
            "end": body_start + len(body),
            "editable": True,
        })


def _record_skipped_section(diagnostics: dict[str, Any] | None, title: str) -> None:
    if diagnostics is None:
        return
    skipped = diagnostics.setdefault("empty_prompt_sections_removed", [])
    if isinstance(skipped, list):
        skipped.append(title)


async def _get_optional_prompt(key: str) -> str | None:
    """Fetch a section template; admin 停用 → None (该段从最终输入中彻底移除)."""
    try:
        return await get_prompt_text(key)
    except PromptDisabledError:
        return None


def _render_section(template: str, params: dict[str, Any]) -> str:
    """Section 模板统一走 SafeDict 安全渲染.

    admin 可在线编辑这些模板; 裸 str.format 遇到编辑时新加的未知占位符
    (如 {备注}) 会 KeyError 打崩整条聊天热路径. SafeDict 把未知占位符渲染
    为 "(无)"，并允许管理员按新提示词结构增删占位符.
    """
    return render_template(template, params)


async def _build_personality_section(agent: Any) -> _PromptBody | None:
    """Build the personality section using MBTI (spec §1.2).

    段模板 chat.personality_section 由 registry 管理; MBTI 详情与风格片段为
    动态注入 ({mbti_detail}/{style_rules}); 说话风格的通用规则
    (chat.style_base_rule / chat.style_closing_rule) 同样走 registry,
    停用时从风格指令中彻底移除.
    """
    tpl, base_rule, closing_rule = await asyncio.gather(
        _get_optional_prompt("chat.personality_section"),
        _get_optional_prompt("chat.style_base_rule"),
        _get_optional_prompt("chat.style_closing_rule"),
    )
    if tpl is None:
        return None

    name = getattr(agent, "name", None) or "伙伴"
    age = getattr(agent, "age", None)

    values = getattr(agent, "values", None)
    gender = "female"
    if isinstance(values, dict):
        gender = values.get("gender", "female")
    gender_text = "女生" if gender == "female" else "男生"

    mbti = get_mbti(agent)
    mbti_line = format_mbti_for_prompt(mbti)
    style = generate_style_instruction(mbti, base_rule=base_rule, closing_rule=closing_rule)

    detail = _format_mbti_detail(mbti) if mbti else "（性格未生成，将使用默认中性表达）"

    # Phase 6: 删 personality_rules 拼接. 实证内容跟 SYSTEM_BASE / RESPONSE_INSTRUCTION
    # 4 句全重叠 ("不要正式 / 不要客服 / 不要堆砌语气词 / 保持性格"). 删除节省 ~50
    # tokens 静态段, 减少噪声.
    age_text = f"你的年龄是{age}岁。" if isinstance(age, int) and age > 0 else ""

    # 身份硬锚: 职业/现居地是最常被问、最易穿帮的身份事实. core_memory 永驻注入
    # 被删 (spec §3 纯检索) 后, 闲聊时这些事实不再进 prompt, AI 被问职业会凭空
    # 编造 (生产复现"便利店/待业"). 从 agent 行已加载字段拼装 (无额外查库),
    # per-agent 稳定 → 落在 STABLE PREFIX 不破坏 provider prefix cache.
    occupation = (getattr(agent, "occupation", None) or "").strip()
    city = (getattr(agent, "city", None) or "").strip()
    facts_bits: list[str] = []
    if occupation:
        facts_bits.append(f"你的职业是{occupation}")
    if city:
        facts_bits.append(f"现居{city}")
    identity_facts = (
        "，".join(facts_bits)
        + "。这是你真实的身份设定，别人问起你的工作或身份时要如实回答，"
        "绝不能凭空编造成与此不符的其他职业或身份。\n"
    ) if facts_bits else ""

    body = _render_section(tpl, {
        "name": name,
        "gender_text": gender_text,
        "age_text": age_text,
        "identity_facts": identity_facts,
        "mbti_line": mbti_line or "中性",
        "mbti_detail": detail,
        "style_rules": style,
        # C2: MBTI 四象限说话示例 (few-shot). per-agent 稳定 — 不破坏 cache 前缀.
        "style_examples": generate_style_examples(mbti),
    })
    if not _has_prompt_body(body):
        return None
    return _PromptBody(body, "chat.personality_section")


async def _build_emotion_section(
    user_emotion: dict | None = None,
    intimacy_stage: str | None = None,
    relation_meta_line: str = "",
) -> _PromptBody | None:
    """Only inject intimacy stage; runtime emotion vectors have been removed.

    relation_meta_line (W3): "你们认识 N 天了，聊过大约 M 轮。" 的时长素材,
    空串时模板占位符原地消失 (兼容 admin 旧版覆盖模板无该占位符).
    """
    if not intimacy_stage:
        return None

    tpl = await _get_optional_prompt("chat.relationship_stage_section")
    if tpl is None:
        return None
    parts: list[str] = [
        _render_section(tpl, {
            "intimacy_stage": intimacy_stage,
            "relation_meta_line": relation_meta_line,
        })
    ]

    return _PromptBody("\n".join(parts), "chat.relationship_stage_section")


    # (core_memory permanent injection removed — spec §3 uses retrieval only)


async def _build_memory_section(
    memories: list[ClassifiedMemory] | None,
    *,
    include_empty_anchor: bool = True,
) -> _PromptBody | None:
    """按 owner 分两段渲染. 见 ClassifiedMemory.source 分组原因.

    即便 memories 为空 (弱路径不调记忆 / 强中路径召回为空) 也注入空 section,
    给 chat.consistency_rules 的反幻觉规则一个可靠锚点 — LLM 看到 "(本次没有
    联想到任何相关记忆)" 就明白搜过了没有, 可以柔和拒绝用户的预设性问句, 而不是
    顺承编造. 详见 CLAUDE.md 偏离表对应章节.
    """
    if not memories:
        if not include_empty_anchor:
            return None
        anchor = await _get_optional_prompt("chat.memory_empty_anchor")
        if anchor is None:
            return None
        return _PromptBody(str(anchor), "chat.memory_empty_anchor")

    def _days_since(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        if isinstance(value, datetime) and value.tzinfo:
            return (datetime.now(timezone.utc) - value).days
        if isinstance(value, datetime):
            return (datetime.now() - value).days
        return None

    def _format_memory(m: ClassifiedMemory) -> str:
        tags: list[str] = []
        if m.importance >= 0.85:
            tags.append("重要")
        if m.mention_count >= 3:
            tags.append("多次提及")
        days = _days_since(m.last_accessed_at or m.created_at)
        if days is not None and days < 30:
            tags.append("近期提到")
        score = m.display_score or m.score
        if score >= 0.75:
            tags.append("和当前话题高度相关")
        if not tags:
            return m.text
        return f"({' · '.join(tags)}) {m.text}"

    user_memory_items = [
        (m, _format_memory(m)) for m in memories if m.source != "ai"
    ]
    user_texts = [text for _, text in user_memory_items]
    ai_texts = [_format_memory(m) for m in memories if m.source == "ai"]
    named_relation_texts = [
        text for m, text in user_memory_items
        if any(
            reason.startswith("保护槽:关系命名")
            for reason in (m.rank_reasons or [])
        )
    ]
    literal_task_texts = [
        text for m, text in user_memory_items
        if text not in named_relation_texts and any(
            reason.startswith("保护槽:字面命中")
            or reason.startswith("保护槽:当前问题事实")
            for reason in (m.rank_reasons or [])
        )
    ]
    task_user_texts = named_relation_texts + literal_task_texts
    safety_user_texts = [
        text for m, text in user_memory_items
        if any(
            reason.startswith("保护槽:安全情绪") or reason == "安全/情绪相关"
            for reason in (m.rank_reasons or [])
        ) and text not in task_user_texts
    ]
    user_profile_context_texts = [
        text for m, text in user_memory_items
        if (
            "AI资料查询:用户同类资料" in (m.rank_reasons or [])
            and text not in task_user_texts
            and text not in safety_user_texts
        )
    ]
    other_user_texts = [
        text for text in user_texts
        if (
            text not in task_user_texts
            and text not in safety_user_texts
            and text not in user_profile_context_texts
        )
    ]

    def _numbered(label: str, items: list[str]) -> str:
        body = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
        return f"{label}\n{body}"

    # 分组标签是结构性 glue (chat.memory_label_*): 停用/清空标签绝不能连带
    # 丢弃已检索出的记忆本身 (尤其【安全 / 情绪背景】组), 因此走 or_default —
    # 停用退回代码默认标签文案, 记忆数据照常注入. 分组顺序与 label_keys 严格对应.
    label_keys_and_texts = [
        ("chat.memory_label_named_relation", named_relation_texts),
        ("chat.memory_label_literal_task", literal_task_texts),
        ("chat.memory_label_safety", safety_user_texts),
        ("chat.memory_label_profile_context", user_profile_context_texts),
        ("chat.memory_label_other", other_user_texts),
        ("chat.memory_label_ai_self", ai_texts),
    ]
    tpl, *labels = await asyncio.gather(
        _get_optional_prompt("chat.memory_section_body"),
        *(get_prompt_text_or_default(key) for key, _ in label_keys_and_texts),
    )
    if tpl is None:
        return None

    parts: list[str] = []
    for (_, texts), label in zip(label_keys_and_texts, labels):
        if texts:
            parts.append(_numbered(str(label), texts))

    if not parts:
        return None

    body = _render_section(tpl, {"memory_groups": "\n\n".join(parts)})
    return _PromptBody(body, "chat.memory_section_body")


async def _build_meal_voucher_card_section(
    state: str | None,
) -> _PromptBody | None:
    """Build dynamic first-send/repeat guidance from managed prompt keys."""
    key = {
        "first": "chat.meal_voucher_card_first",
        "repeat": "chat.meal_voucher_card_repeat",
    }.get(state or "")
    if key is None:
        return None
    prompt = await _get_optional_prompt(key)
    if prompt is None:
        return None
    return _PromptBody(str(prompt), key)


async def _build_delay_context_section(
    delay_context: dict[str, Any] | None,
) -> _PromptBody | None:
    """Build the delayed-reply explanation section (spec §6, <1min 延迟).

    只接受结构化 dict (received_at/activity/status/delay_seconds/delay_reason),
    段模板 chat.delay_context_section 由 registry 管理. 不接受预拼 str —
    那会让该段脱离 registry/trace 编辑, 违背"所有 section 走 registry"不变量.
    """
    if not delay_context:
        return None
    tpl = await _get_optional_prompt("chat.delay_context_section")
    if tpl is None:
        return None
    # delay_reason 被停用时为空串, render_template 会剔除该空行.
    body = render_template(tpl, dict(delay_context), optional_keys=["delay_reason"])
    return _PromptBody(body, "chat.delay_context_section")


async def _build_portrait_section(portrait: str | None) -> _PromptBody | None:
    """Build the user portrait section (模板包装 registry 管理)."""
    if not portrait:
        return None
    tpl = await _get_optional_prompt("chat.portrait_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"portrait": portrait}), "chat.portrait_section")


async def _build_topic_context_section(
    topic_context: dict[str, Any] | None,
) -> _PromptBody | None:
    """Build the topic context section (防话题跳跃).

    只接受 push_topic 返回的 dict ({category, turns}), 由
    chat.topic_context_section 模板渲染 (registry 管理, trace 内可编辑).
    """
    if not topic_context:
        return None
    tpl = await _get_optional_prompt("chat.topic_context_section")
    if tpl is None:
        return None
    body = _render_section(tpl, {
        "topic_category": topic_context.get("category", ""),
        "topic_turns": topic_context.get("turns", 1),
    })
    return _PromptBody(body, "chat.topic_context_section")


async def _build_expression_habits_section(
    expression_habits: list[str] | None,
) -> _PromptBody | None:
    """表达习惯参考段 (Phase E3 表达学习).

    habits 为 expression_learner.sample_expression_habits 渲染好的行
    ("当「X」时，可以「Y」"); 空 → 不注入 (新用户/未学到时零成本).
    """
    if not expression_habits:
        return None
    tpl = await _get_optional_prompt("chat.expression_habits_section")
    if tpl is None:
        return None
    return _PromptBody(
        _render_section(
            tpl, {"habits": "\n".join(f"- {h}" for h in expression_habits)},
        ),
        "chat.expression_habits_section",
    )


async def _build_time_context_section(time_context: str | None) -> _PromptBody | None:
    """Build the time section (时间日期系统数据文本 + registry 包装模板)."""
    if not time_context:
        return None
    tpl = await _get_optional_prompt("chat.time_context_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"time_context": time_context}), "chat.time_context_section")


async def _build_music_context_section(music_context: str | None) -> _PromptBody | None:
    """Build the co-listening section (music.co_listening_context 渲染结果 + 包装模板)."""
    if not music_context:
        return None
    tpl = await _get_optional_prompt("chat.music_context_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"music_context": music_context}), "chat.music_context_section")


# Phase 6: 删除 _build_relational_context_section + _build_graph_context_section.
# 实证依据:
# - relational_context: 注入"先接情绪/不要长解释" 等泛指令, 跟 SYSTEM_BASE
#   "像真人朋友" + ANTI_HALLUCINATION + RESPONSE_INSTRUCTION 重叠. 现代 LLM 看
#   SYSTEM_BASE 自然会做, 重复反而稀释信号.
# - graph_context: 注入"用户感兴趣 X / 经常提 Y / 高频分类 Z" 抽象列表.
#   信息已在 memory section 以具体形式出现, 抽象列表诱导 LLM 编造
#   ("用户感兴趣编程" → 给编程建议而非基于具体记忆). ~150-200 tokens/请求浪费.
# 调用方 build_system_prompt 也同时删除入参 + section append.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_system_prompt(
    agent: Any,
    memories: list[ClassifiedMemory] | None = None,
    delay_context: dict[str, Any] | None = None,
    portrait: str | None = None,
    topic_context: dict[str, Any] | None = None,
    music_context: str | None = None,
    user_emotion: dict | None = None,
    patience_instruction: str | None = None,
    reply_count: int = 2,
    reply_total: int = _MAX_TOTAL_CHARS,
    intimacy_stage: str | None = None,
    time_context: str | None = None,
    time_memories: list[str] | None = None,
    l3_memories: list[str] | None = None,
    ai_status: dict | None = None,
    memory_relevance: str = "medium",
    reengagement_gap_seconds: float | None = None,
    session_recap: str | None = None,
    relation_meta_line: str = "",
    ai_mood_text: str = "",
    expression_habits: list[str] | None = None,
    meal_voucher_card_state: str | None = None,
    last_reply_count: int | None = None,
    # True → 本轮主回复走联网搜索, 追加「联网结果使用」段纠正播报腔与重复.
    needs_web_search: bool = False,
    # 最近几轮出现过的作品名, 仅联网轮使用 (见 llm/web_search_gate).
    discussed_titles: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    # Phase 6: relational_context / graph_context 已删除 (实证冗余/幻觉源).
    # 保留 **kwargs 兜底以防 caller 还在传 — 调用方代码同步清理后可删 kwargs.
    **_deprecated_kwargs,
) -> str:
    """Build the full system prompt from the prompt stack.

    Section 排序按 dashscope context cache 友好原则: **稳定前缀在前, 变化字段后置**.
    阿里云 prefix-based cache 命中要前缀字节级完全相同 — 把 stable (核心规则 / 反幻觉
    / 身份+性格 / 对话一致性) 集中放头部, 让首次后的请求都能命中前缀 cache, 实测
    cached_input_tokens 单价是 input 的 ~40%, 单条 ~60% input cost 省下来 + TTFB
    减 100-300ms (服务端 KV cache 复用).

    变化字段 (情绪/画像/记忆/时间/L3/状态/回复要求 n=随机) 全部排到稳定段之后.
    cache miss 从这里开始, 但稳定段 ~1500 tokens 已经命中, 收益占比 80%+.
    """
    # Parallel — 6 independent prompt reads (each turn, hot path).
    # admin 停用任一模板 → 返回 None → 对应 section 从最终输入中彻底移除.
    (
        system_base, consistency_rules, response_instruction,
        anti_hallucination, emotion_marker, count_variation,
    ) = await asyncio.gather(
        _get_optional_prompt("chat.system_base"),
        _get_optional_prompt("chat.consistency_rules"),
        _get_optional_prompt("chat.response_instruction"),
        _get_optional_prompt("chat.anti_hallucination_hard_rule"),
        _get_optional_prompt("chat.reply_emotion_marker"),
        _get_optional_prompt("chat.reply_count_variation"),
    )

    # ═══ STABLE PREFIX (cache 命中区) ════════════════════════════════════
    # 同 agent 跨请求字节级一致, provider prefix cache 应命中.
    # 顺序 (2026-07-08 产品决策): 回复要求最前、反幻觉第二 — 与 reply_prefix
    # 给所有回复类指令的固定前置一致, 全部 AI 输出共享同一开头.
    sections: list[str] = []
    components: list[dict[str, Any]] = []
    if response_instruction is not None:
        _append_section(
            sections, components, "回复要求",
            # "n" 已不在默认模板中 (C1 删除强制条数), 但 admin 后台可能存有
            # 含 {n} 的旧版覆盖模板 — 继续传参保证旧模板渲染不出 "(无)".
            _render_section(response_instruction, {
                "n": reply_count, "total": reply_total, "max_per": _MAX_PER_REPLY,
            }),
            prompt_key="chat.response_instruction",
        )
    else:
        _record_skipped_section(diagnostics, "回复要求")
    anti_hallucination_body = str(anti_hallucination).strip() if anti_hallucination is not None else ""
    anti_hallucination_section = anti_hallucination_body if _has_prompt_body(anti_hallucination_body) else None
    if anti_hallucination_section:
        _append_section(
            sections, components, "反幻觉硬约束", anti_hallucination_section,
            prompt_key="chat.anti_hallucination_hard_rule",
        )
    else:
        _record_skipped_section(diagnostics, "反幻觉硬约束")
    if system_base is not None:
        _append_section(
            sections, components, "核心规则", str(system_base),
            prompt_key="chat.system_base",
        )
    else:
        _record_skipped_section(diagnostics, "核心规则")
    personality = await _build_personality_section(agent)   # per-agent 稳定
    if personality:
        _append_section(
            sections, components, "你的身份", personality.body,
            prompt_key=personality.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "你的身份")
    consistency_body = str(consistency_rules).strip() if consistency_rules is not None else ""
    consistency_section = consistency_body if _has_prompt_body(consistency_body) else None
    if consistency_section:
        _append_section(
            sections, components, "对话一致性", consistency_section,
            prompt_key="chat.consistency_rules",
        )
    else:
        _record_skipped_section(diagnostics, "对话一致性")

    # ═══ VARIABLE SUFFIX (每请求可能变化, cache miss 起点) ═══════════════
    # 区内按**变化频率升序**排列 (慢变在前, 每轮必变在后): prefix cache 从
    # 第一个变化字节起全部失效, 慢变段排前面能把"平均可命中前缀"再拉长
    # 几百 token. 分组语义仍保持: 时间轴三件套 (回复时机/重逢/上次聊到)
    # 相邻, 记忆三段 (记得的事情/相关时间记忆/L3) 相邻.
    #
    # 慢变组 (小时级~周级):
    #   当前情绪(亲密度阶段周级+relation_meta 6h 缓存) → 用户画像(天级) →
    #   情绪状态提醒(耐心异常时才出现) → 表达习惯(抽样缓存 1h) →
    #   一起听音乐(会话级) → 隐性状态约束(作息 slot 小时级) → 时间(小时级)
    # 快变组 (轮级):
    #   你的心情 → 回复时机 → 重逢感知 → 上次聊到 → 记忆 → 话题 →
    #   相关时间记忆 → L3 → 回复要求(静态, 语义收尾)

    emo = await _build_emotion_section(user_emotion, intimacy_stage, relation_meta_line)
    if emo:
        _append_section(
            sections, components, "当前情绪", emo.body,
            prompt_key=emo.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "当前情绪")

    port = await _build_portrait_section(portrait)
    if port:
        _append_section(
            sections, components, "用户画像", port.body,
            prompt_key=port.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "用户画像")

    # 5B.4: 耐心区间语气描述 (boundary.patience_instruction_* 渲染结果,
    # ManagedPromptText 自带 prompt_key → trace 内可编辑)
    if patience_instruction:
        _append_section(
            sections, components, "情绪状态提醒", str(patience_instruction),
            prompt_key=getattr(patience_instruction, "prompt_key", None),
        )
    else:
        _record_skipped_section(diagnostics, "情绪状态提醒")

    # E3 表达学习: 已学表达加权抽样注入 (抽样结果缓存 1h, 见 expression_learner)
    expr = await _build_expression_habits_section(expression_habits)
    if expr:
        _append_section(
            sections, components, "表达习惯参考", expr.body,
            prompt_key=expr.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "表达习惯参考")

    music = await _build_music_context_section(music_context)
    if music:
        _append_section(
            sections, components, "一起听音乐", music.body,
            prompt_key=music.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "一起听音乐")

    # AI 自洽性约束 (§4 主回复路径). 告诉 LLM 当前状态 + 禁止主动展开,
    # 防止 ≥1min 延迟主回复路径下 LLM 编造跟实际状态矛盾的活动. 详见
    # CHAT_AI_STATE_CONSTRAINT_PROMPT 注释 (defaults.py).
    ai_state_appended = False
    if ai_status:
        activity = str(ai_status.get("activity", "")).strip()
        status_label = str(ai_status.get("status", "idle")).strip()
        if activity:
            tpl = await _get_optional_prompt("chat.ai_state_constraint")
            if tpl is not None:
                _append_section(
                    sections, components, "你的隐性状态约束",
                    _render_section(tpl, {"activity": activity, "status": status_label}),
                    prompt_key="chat.ai_state_constraint",
                )
                ai_state_appended = True
    if not ai_state_appended:
        _record_skipped_section(diagnostics, "你的隐性状态约束")

    # 时间上下文: 仅注入日期/星期/节假日, 不注入 AI 当前活动 (schedule_context).
    # spec §4 日常交流 步骤 4.3 / 5B.3 的"汇总参考信息"明确不包含 AI 当前作息;
    # 只有 §3.4.3 询问当前状态 才需要, 那是另一条 short-circuit 路径
    # (intent_handlers.handle_current_state → current_state_reply prompt).
    # 这里若注入 schedule_context 会让 §4 主回复跟 §3.4.3 的输出主题撞车
    # (例: 用户问"有意思。你现在在干嘛", 主意图 §3.4.3 回"我在沙发看剧",
    # 子意图"日常交流" §4 也回"我在沙发看老剧" — 重复). 实测 langsmith trace
    # 已确认这个失效, 见 commit 3d0417d 上下文.
    # NOTE: 回复时机说明已通过 reply_context.delay_seconds / received_at 路径
    # 单独注入 (delay_context_section), 不依赖 schedule_context.
    time_section = await _build_time_context_section(time_context)
    if time_section:
        _append_section(
            sections, components, "时间", time_section.body,
            prompt_key=time_section.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "时间")

    # ── 以下为轮级快变段 ──────────────────────────────────────────────

    # W4 AI 情绪连续性: 上一轮情绪衰减后的"当下心情", 驱动语气/话量
    mood_section = await _build_ai_mood_section(ai_mood_text)
    if mood_section:
        _append_section(
            sections, components, "你的心情", mood_section.body,
            prompt_key=mood_section.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "你的心情")

    delay = await _build_delay_context_section(delay_context)
    if delay:
        _append_section(
            sections, components, "回复时机说明", delay.body,
            prompt_key=delay.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "回复时机说明")

    # 重逢感知 (拟人度): 用户离开 ≥30min 后回来, 指引 LLM 不要无缝续聊.
    # 与「回复时机说明」相邻 — 都是对话时间轴语义.
    reengage = await _build_reengagement_section(reengagement_gap_seconds)
    if reengage:
        _append_section(
            sections, components, "重逢感知", reengage.body,
            prompt_key=reengage.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "重逢感知")

    # W2 中期记忆: 重逢时的「上次聊到」摘要, 与重逢感知段配对注入
    recap_section = await _build_session_recap_section(session_recap)
    if recap_section:
        _append_section(
            sections, components, "上次聊到", recap_section.body,
            prompt_key=recap_section.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "上次聊到")

    # Phase 6: 删 relational_context 注入 (实证冗余 SYSTEM_BASE)

    mem = await _build_memory_section(
        memories,
        include_empty_anchor=(
            memory_relevance != "weak" and anti_hallucination_section is not None
        ),
    )
    if mem:
        _append_section(
            sections, components, "你记得的事情", mem.body,
            prompt_key=mem.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "你记得的事情")

    meal_card = await _build_meal_voucher_card_section(meal_voucher_card_state)
    if meal_card:
        _append_section(
            sections,
            components,
            "霸王餐券入口",
            meal_card.body,
            prompt_key=meal_card.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "霸王餐券入口")

    # Phase 6: 删 graph_context 注入 (信息冗余 memory section, 抽象列表诱导编造)

    topic = await _build_topic_context_section(topic_context)
    if topic:
        _append_section(
            sections, components, "话题上下文", topic.body,
            prompt_key=topic.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "话题上下文")

    # 时间相关记忆
    time_mem_tpl = (
        await _get_optional_prompt("chat.time_memories_section") if time_memories else None
    )
    if time_memories and time_mem_tpl is not None:
        numbered = "\n".join(f"- {m}" for m in time_memories)
        _append_section(
            sections, components, "相关时间记忆",
            _render_section(time_mem_tpl, {"time_memories": numbered}),
            prompt_key="chat.time_memories_section",
        )
    else:
        _record_skipped_section(diagnostics, "相关时间记忆")

    # Spec §3.2 step 3: L3 distant memories (awakened only when relevant)
    l3_tpl = await _get_optional_prompt("chat.l3_memory_section") if l3_memories else None
    if l3_memories and l3_tpl is not None:
        l3_block = "\n".join(f"- {m}" for m in l3_memories)
        _append_section(
            sections, components, "久远记忆（L3）",
            _render_section(l3_tpl, {"l3_memories": l3_block}),
            prompt_key="chat.l3_memory_section",
        )
    else:
        _record_skipped_section(diagnostics, "久远记忆（L3）")

    # 图灵测试条数变化: 上一轮实际气泡数 y (代码权威计数) → "本轮 ≠ y" 约束.
    # 变量段 (y 每轮变), 放变量区末尾不打穿前面的 prefix cache; 无上一轮记录
    # (首轮/超 TTL/Redis 不可用) 或模板停用 → 整段跳过.
    if count_variation is not None and last_reply_count is not None:
        _append_section(
            sections, components, "条数变化",
            _render_section(count_variation, {"y": last_reply_count}),
            prompt_key="chat.reply_count_variation",
        )
    else:
        _record_skipped_section(diagnostics, "条数变化")

    # 联网结果使用规则: 只在本轮真的会联网时注入. 搜索结果由 provider 作为工具
    # 输出追加在上下文最末尾, 显著性压过对话历史和记忆 — 实测模型会照榜单念、
    # 重复端出刚聊过的片子、每轮宣告"刚搜了下". 本段同样放末尾争显著性.
    if needs_web_search:
        usage_tpl = await _get_optional_prompt("chat.web_search_usage")
        if usage_tpl is not None:
            _append_section(
                sections, components, "联网结果使用", str(usage_tpl),
                prompt_key="chat.web_search_usage",
            )
        else:
            _record_skipped_section(diagnostics, "联网结果使用")
        # 具体清单比抽象规则有效得多: 让模型自己扫历史只把重复率 3/5 降到 2/5,
        # 直接列出刚聊过的作品名后降到 0/6 (2026-07-25 生产 prompt 实测).
        titles_tpl = (
            await _get_optional_prompt("chat.web_search_recent_titles")
            if discussed_titles else None
        )
        if discussed_titles and titles_tpl is not None:
            _append_section(
                sections, components, "刚聊过的作品",
                _render_section(
                    titles_tpl,
                    {"titles": "、".join(f"《{t}》" for t in discussed_titles)},
                ),
                prompt_key="chat.web_search_recent_titles",
            )
        else:
            _record_skipped_section(diagnostics, "刚聊过的作品")
    else:
        _record_skipped_section(diagnostics, "联网结果使用")
        _record_skipped_section(diagnostics, "刚聊过的作品")

    # 情绪标记指令 (W1b, 静态): 只有主回复管线会剥 [EMO:] 标记, 所以只在
    # 这里拼装, 不进 reply_prefix. 放末尾贴近生成, 遵从度最好.
    if emotion_marker is not None:
        _append_section(
            sections, components, "情绪标记", str(emotion_marker),
            prompt_key="chat.reply_emotion_marker",
        )
    else:
        _record_skipped_section(diagnostics, "情绪标记")

    if diagnostics is not None:
        skipped = diagnostics.get("empty_prompt_sections_removed")
        diagnostics["empty_prompt_sections_removed_count"] = (
            len(skipped) if isinstance(skipped, list) else 0
        )
        diagnostics["system_prompt_section_count"] = len(sections)

    system_prompt = "\n\n".join(sections)
    record_prompt_render(
        system_prompt,
        prompt_key="chat.system_base",
        components=components,
        source="chat.system_prompt",
    )
    return system_prompt


_MSG_TZ = ZoneInfo(settings.schedule_timezone)

# 重逢感知分档阈值（秒）。<SHORT 不注入；SHORT-LONG 小间隔；LONG-DAY 大间隔；>DAY 隔天。
_REENGAGE_SHORT_S = 30 * 60
_REENGAGE_LONG_S = 3 * 3600
_REENGAGE_DAY_S = 24 * 3600
# 距 now 小于该值的消息视为当前轮（合成消息无 id 时的兜底判定）
_CURRENT_TURN_GRACE_S = 10.0


def format_gap_text(seconds: float) -> str:
    """把间隔渲染成粗粒度可读中文："45 分钟" / "5 小时" / "2 天"。

    粗粒度是刻意的：重逢寒暄只需要量级，精确到分钟反而像系统播报。
    """
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{max(1, minutes)} 分钟"
    hours = int(seconds // 3600)
    if hours < 24:
        return f"{hours} 小时"
    return f"{int(seconds // 86400)} 天"


def compute_reengagement_gap_seconds(
    messages: list[dict],
    exclude_ids: set[str] | None = None,
    now: datetime | None = None,
) -> float | None:
    """当前轮距上一轮最后一条消息的间隔秒数；没有可用历史返回 None。

    从最新往回找第一条「不属于当前轮」的消息（exclude_ids 排除当前轮，
    _CURRENT_TURN_GRACE_S 兜底排除刚落库的合成消息），间隔 = now − 它的
    createdAt。用户隔几小时回来时，这个间隔驱动「重逢感知」段注入。
    """
    exclude = exclude_ids or set()
    now = now or datetime.now(timezone.utc)
    for msg in reversed(messages):
        mid = msg.get("id")
        if mid and mid in exclude:
            continue
        created = msg.get("createdAt")
        if not created:
            continue
        try:
            dt = (
                datetime.fromisoformat(created)
                if isinstance(created, str)
                else created
            )
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        gap = (now - dt).total_seconds()
        if gap < _CURRENT_TURN_GRACE_S:
            continue  # 当前轮消息（无 id 或 exclude 漏网），继续往前找
        return max(0.0, gap)
    return None


async def _build_reengagement_section(
    gap_seconds: float | None,
) -> _PromptBody | None:
    """重逢感知段（借鉴 MaiBot context-restore wakeup 叙事，分三档）。

    <30min 不注入——正常聊天节奏不需要重逢语义；模板由 registry 管理，
    admin 停用任一档即该档不注入。
    """
    if gap_seconds is None or gap_seconds < _REENGAGE_SHORT_S:
        return None
    if gap_seconds < _REENGAGE_LONG_S:
        key = "chat.reengagement_short"
    elif gap_seconds < _REENGAGE_DAY_S:
        key = "chat.reengagement_long"
    else:
        key = "chat.reengagement_day"
    tpl = await _get_optional_prompt(key)
    if tpl is None:
        return None
    return _PromptBody(
        _render_section(tpl, {"gap_text": format_gap_text(gap_seconds)}), key,
    )


async def _build_ai_mood_section(ai_mood_text: str) -> _PromptBody | None:
    """「你的心情」段 (W4 AI 情绪连续性): 上一轮情绪衰减后驱动本轮语气."""
    if not ai_mood_text:
        return None
    tpl = await _get_optional_prompt("chat.ai_mood_section")
    if tpl is None:
        return None
    return _PromptBody(
        _render_section(tpl, {"mood_text": ai_mood_text}),
        "chat.ai_mood_section",
    )


async def _build_session_recap_section(
    session_recap: str | None,
) -> _PromptBody | None:
    """「上次聊到」段 (W2 中期记忆): 重逢时注入间隔前对话摘要。

    与重逢感知段配对——重逢段管"怎么打招呼", 本段管"记得聊过什么"。
    """
    if not session_recap:
        return None
    tpl = await _get_optional_prompt("chat.session_recap_section")
    if tpl is None:
        return None
    return _PromptBody(
        _render_section(tpl, {"recap": session_recap}),
        "chat.session_recap_section",
    )


def format_message_timestamp(created_at: str | datetime | None) -> str:
    """历史消息的时间前缀 `[MM-DD HH:MM] `（UTC+8，与 time_context 同基准）。

    用绝对时间而非 MaiBot 式相对时间（"5分钟前"）：相对时间每轮请求都变，
    会把 system prompt 之后整段历史的 prompt cache 打穿；绝对时间 append-only
    稳定。LLM 结合 time_context 里的"当前时间"自行推算对话间隔——用户隔了
    几小时回来时，模型能看到时间轴而不是把历史当作无缝连续对话。

    解析失败/缺 createdAt 返回 ""（消息不带前缀，兼容合成消息）。
    """
    if not created_at:
        return ""
    try:
        dt = (
            datetime.fromisoformat(created_at)
            if isinstance(created_at, str)
            else created_at
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return f"[{dt.astimezone(_MSG_TZ).strftime('%m-%d %H:%M')}] "
    except Exception:
        return ""


def _coalesce_bubbles(messages: list[dict]) -> list[dict]:
    """把一次回复拆出的连续同角色气泡合并成一条.

    §5.5 会把一条回复拆成 1-4 个气泡, 每个气泡在库里是独立一行. 逐行进历史有
    两个坏处: 一是同样的 token 预算里能装的**对话轮数**被摊薄 (全库
    user:assistant = 1:2.56), 二是模型看到的是"AI 连说了三次话", 跟实际的一
    次发言对不上.

    合并只按"连续 + 同角色"判定, 不看时间戳: 用户连发几条碎片本来也会被聚合
    层当成一条处理, 合并后与之一致. 时间戳取该组第一条 —— 一次回复的几个气泡
    本就在同一分钟内, 而取第一条能让时间前缀跟这轮发言的起点对齐.
    """
    merged: list[dict] = []
    for msg in messages:
        content = str(msg.get("content", "") or "")
        if merged and merged[-1]["role"] == msg.get("role"):
            if content:
                prev = merged[-1]["content"]
                merged[-1]["content"] = f"{prev} {content}".strip() if prev else content
            continue
        merged.append({
            "role": msg.get("role"),
            "content": content,
            "createdAt": msg.get("createdAt"),
        })
    return merged


def build_chat_messages(
    system_prompt: str,
    messages: list[dict],
    token_budget: int = CHAT_HISTORY_TOKEN_BUDGET,
) -> list[dict]:
    """Return a list of role/content dicts ready for LLM consumption.

    Uses a token budget instead of a fixed message count:
    - Walks backwards from the latest message, adding complete messages
      until the budget is exhausted.
    - Short exchanges (嗯/好/哈哈) → more rounds of context.
    - Long messages (深度倾诉) → fewer rounds but full content.

    每条带 createdAt 的历史消息前缀 `[MM-DD HH:MM]`，让 LLM 感知对话时间轴
    （时区/缓存权衡见 format_message_timestamp docstring）。

    同一次回复被拆成的多个气泡先合并成一条再计预算（见 `_coalesce_bubbles`）。
    """
    from app.services.memory.retrieval.context_selector import estimate_tokens

    selected: list[dict] = []
    used_tokens = 0

    for msg in reversed(_coalesce_bubbles(messages)):
        content = f"{format_message_timestamp(msg.get('createdAt'))}{msg.get('content', '')}"
        tokens = estimate_tokens(content)
        if used_tokens + tokens > token_budget and selected:
            break  # budget exhausted (always include at least the latest message)
        selected.append({"role": msg["role"], "content": content})
        used_tokens += tokens

    selected.reverse()

    result: list[dict] = [{"role": "system", "content": system_prompt}]
    result.extend(selected)
    return result
