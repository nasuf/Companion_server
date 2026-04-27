"""Trace step semantic enrichment.

把 LangSmith 原始 step (run_type='llm', name='ChatOpenAI' 等) 映射到带语义的字段:
- display_name: 让 PM 一眼看懂的中文功能名 (e.g. "记忆相关度判定")
- category: decision / data / reply / post / other (前端配色用)
- prompt_key: 关联到 prompting registry, 详情面板提供"跳转 admin 编辑"
- decision_label: 提取关键决策 (e.g. "弱" / "无矛盾" / "偏积极"), 替代生 output

实现思路: 用每条 prompt 的头部 60 字作为指纹 (defaults.py 里都是固定文本头部),
runtime 比对 LLM 调用 input.messages 的第一条 HumanMessage content. 头部稳定 →
匹配可靠. 头部改了 → 配套测试会立刻发现 (test_trace_enrich.py).

供 public_trace.load_public_trace 在返回前调用; 失败时 graceful degrade
(category='other', display_name=run.name).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


Category = str  # "decision" | "data" | "reply" | "post" | "other"


@dataclass(frozen=True)
class _PromptMeta:
    prompt_key: str
    display_name: str
    category: Category
    label_extractor: Callable[[str], str | None] | None = None


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


def _label_pad(output: str) -> str | None:
    """PAD JSON → '愉悦P · 唤醒A · 支配D' 简短摘要."""
    try:
        data = json.loads(_label_strip_codeblock(output))
        p = float(data.get("pleasure", 0))
        a = float(data.get("arousal", 0))
        d = float(data.get("dominance", 0))
    except Exception:
        return None
    parts = []
    if p > 0.3:
        parts.append("偏积极")
    elif p < -0.3:
        parts.append("偏消极")
    else:
        parts.append("中性")
    parts.append("激动" if a > 0.6 else ("平静" if a < 0.3 else "中等"))
    parts.append("掌控" if d > 0.6 else ("无助" if d < 0.3 else "中性"))
    return " · ".join(parts)


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


# ─────────────────────────────────────────────────────────────────
# Prompt 指纹映射表 — 用每条 prompt 内的"独特子串"做 substring 匹配,
# 不能用头部前缀 — 多个 prompt 共享 "【限定】..." / "用户刚才对你说" 等
# 公共前缀, 会指纹冲突. 也不能用整 prompt 哈希 — 头部含 {var} 占位符,
# format() 后内容变化.
# 每条 fingerprint 是该 prompt 中的稳定独特子串 (≥30 字, 不含 {var}).
# 防 drift: 测试用 defaults.py 的实际 prompt 走一次, 任一指纹不匹配立挂.
# ─────────────────────────────────────────────────────────────────


# 注册项: (fingerprint_substring, meta). 顺序敏感 — 通用 prompt 应放后面,
# 让更长/更独特的指纹先匹配.
_REGISTRY: list[tuple[str, _PromptMeta]] = []


def _register(fingerprint: str, meta: _PromptMeta) -> None:
    """fingerprint 必须是该 prompt 内的稳定独特子串 (不含 {var} 占位符).
    长度建议 30+. 顺序: 注册越早越优先匹配."""
    if not fingerprint or len(fingerprint) < 8:
        logger.warning(f"[trace_enrich] fingerprint too short for {meta.prompt_key}, may collide")
    _REGISTRY.append((fingerprint, meta))


# Decision 类 — fingerprint 选择各 prompt 内的独特短语 (≥30 字)
_register("判断回答用户这句话需不需要查询长期记忆", _PromptMeta(
    "memory.relevance", "记忆相关度判定", "decision", _label_passthrough,
))
_register("分析用户当前消息的主要意图，从以下选项中选出最匹配", _PromptMeta(
    "intent.unified", "统一意图识别", "decision", _label_intent_unified,
))
_register("用户的一句话包含多个意图。请将原话拆分成多个片段", _PromptMeta(
    "intent.split_multi", "多意图拆分", "decision", _label_passthrough,
))
_register("是否对当前提到的记忆表示不满/纠正，或是否在请求回忆更久之前", _PromptMeta(
    "intent.l3_trigger", "L3 唤醒判定", "decision", _label_passthrough,
))
_register("判断用户当前提及的内容是否与你已有的关于该用户的核心记忆", _PromptMeta(
    "memory.contradiction_detection", "L1 矛盾检测", "decision", _label_contradiction,
))
_register("分析用户对矛盾询问的回复，判断矛盾类型、原因及记忆调整方案", _PromptMeta(
    "memory.contradiction_analysis", "矛盾分析", "decision", _label_passthrough,
))
_register("分析以下消息是否包含道歉或承诺改正", _PromptMeta(
    "intent.apology_detect", "道歉检测", "decision", _label_apology,
))
_register("分析用户消息的攻击目标", _PromptMeta(
    "boundary.attack_target", "攻击目标识别", "decision", _label_passthrough,
))
_register("判断用户这句话的冒犯程度。请以朋友的包容心态", _PromptMeta(
    "boundary.attack_level", "攻击级别识别", "decision", _label_attack_level,
))
_register("判断用户消息是否包含违禁内容（包括谐音、缩写、辱骂、涉黄、涉暴等）", _PromptMeta(
    "boundary.banned_word", "违禁词判断", "decision", _label_passthrough,
))
_register("模拟真人记忆，判断这句话是否值得进入记忆", _PromptMeta(
    "memory.pre_filter_user", "用户记忆预筛", "decision", _label_judge_remember,
))
_register("模拟真人自我记忆，判断这句话是否值得进入记忆", _PromptMeta(
    "memory.pre_filter_ai", "AI 自我记忆预筛", "decision", _label_judge_remember,
))
_register("判断用户是否在要求AI忘记/删除某条记忆", _PromptMeta(
    "memory.deletion_intent", "记忆删除意图判定", "decision", _label_passthrough,
))
_register("扫描下面的 L1 记忆列表, 找出语义上互相矛盾的对", _PromptMeta(
    "memory.pairwise_contradiction", "L1 一致性扫描", "decision", _label_passthrough,
))

# Data 类
_register("根据 AI 当前时间、作息状态、正在进行的活动以及对话上下文，推测 AI 此刻的真实情绪", _PromptMeta(
    "emotion.ai_pad", "AI 情绪 (PAD)", "data", _label_pad,
))
_register("分析用户消息的情绪，输出PAD三维值", _PromptMeta(
    "emotion.user_pad", "用户情绪 (PAD)", "data", _label_pad,
))

# Reply 类 — short-circuit handlers
_register("用户正在询问你当前在做什么或最近怎么样。作为朋友，自然地回答对方", _PromptMeta(
    "intent.current_state_reply", "询问当前状态回复 (§3.4.3)", "reply", _label_reply_text,
))
_register("用户希望你忘记某件事。作为朋友，你需要先确认对方具体想忘记什么", _PromptMeta(
    "memory.deletion_confirm", "删除确认", "reply", _label_reply_text,
))
_register("用户确认要你忘记某件事。作为朋友，你表示已经忘记", _PromptMeta(
    "memory.deletion_reply", "删除完成回复", "reply", _label_reply_text,
))
_register("你发现用户刚才说的话，和你记忆中关于对方的一条核心信息有矛盾", _PromptMeta(
    "memory.contradiction_inquiry", "矛盾询问", "reply", _label_reply_text,
))
_register("用户已解释清楚之前记忆矛盾的原因，你表示理解", _PromptMeta(
    "memory.contradiction_reply", "矛盾化解回复", "reply", _label_reply_text,
))
_register("你现在处于低耐心状态，用户仍在攻击你。你非常不高兴", _PromptMeta(
    "boundary.final_warning", "最终警告", "reply", _label_reply_text,
))
_register("这句话让你有点不舒服。请用自然的语气表达你的感受", _PromptMeta(
    "boundary.attack_reply_light", "轻度攻击回复 (K1)", "reply", _label_reply_text,
))
_register("这句话让你明显不开心了。请用认真但不过激的语气", _PromptMeta(
    "boundary.attack_reply_medium", "中度攻击回复 (K2)", "reply", _label_reply_text,
))
_register("这句话让你非常难过/愤怒，严重伤害了感情", _PromptMeta(
    "boundary.attack_reply_severe", "重度攻击回复 (K3)", "reply", _label_reply_text,
))
_register("你现在处于中等耐心状态（有点不高兴，但还没到生气不理人的程度）", _PromptMeta(
    "boundary.patience_medium_reply", "中耐心回复", "reply", _label_reply_text,
))
_register("你现在处于低耐心状态（很不高兴，不太想多说话）", _PromptMeta(
    "boundary.patience_low_reply", "低耐心回复", "reply", _label_reply_text,
))
_register("你已进入拉黑状态。用户给你发消息，但你现在不想回应", _PromptMeta(
    "boundary.blacklist_reply", "拉黑回复", "reply", _label_reply_text,
))
_register("你是一个有血有肉的人，不是AI助手", _PromptMeta(
    "chat.system_base", "主回复 (§4 日常交流)", "reply", _label_reply_text,
))

# Post 类
_register("将以下AI回复内容拆分成2句自然的回复", _PromptMeta(
    "reply.split_2", "回复拆 2 句", "post", _label_split_n,
))
_register("将以下AI回复内容拆分成3句自然的回复", _PromptMeta(
    "reply.split_3", "回复拆 3 句", "post", _label_split_n,
))
_register("判断AI回复文本的情绪及强度", _PromptMeta(
    "reply.emotion", "回复情绪识别", "post", _label_emotion,
))
_register("分析下面的对话，**只抽取【用户】", _PromptMeta(
    "memory.extraction_user", "用户记忆抽取", "post", _label_extraction,
))
_register("分析下面的对话，**只抽取【我】", _PromptMeta(
    "memory.extraction_ai", "AI 自我记忆抽取", "post", _label_extraction,
))


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
    step["decision_label"] = label
    return step


def enrich_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量包装. 修改原 dict 并返回 (in-place)."""
    for step in steps:
        enrich_step(step)
    _mark_critical_path(steps)
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
