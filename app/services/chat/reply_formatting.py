"""回复切分与文本规范化 — 纯函数层（orchestrator 拆分 R2）。

Multi-reply split & validate (PRD §3.2.1/§3.2.2)：按 || / 空行切分主 LLM
输出，校验条数/单条长度/总长度，句末边界截断。除护栏触发的观测日志外
无副作用，从 orchestrator.py 提取；orchestrator re-export 保持导入路径兼容。

护栏失败语义 (2026-07-20 修订): 条数溢出**合并进最后一条**而非丢弃 —
提示词本身教 LLM"遇到句号、空格也拆短句", 超过上限的片段很可能是收尾的
追问/情绪落点, 静默丢弃会让回复说一半. 所有护栏触发打 EVT_REPLY_GUARDRAIL
事件 (conversation_id 由 observability ContextVar 自动附着), 触发率高说明
prompt 失守, 应修 prompt 而不是靠护栏硬扛.
"""

from __future__ import annotations

import logging
import re

from app.observability.events import EVT_REPLY_GUARDRAIL
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY,
    MAX_REPLY_COUNT,
    MAX_TOTAL_CHARS,
)

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'[。！？…～~!?]+')

# 切分分隔符: 优先 || (我们要求 LLM 输出的); 兼容 LLM 自作主张用空行分段的情况
# (\n\n+) — 不切就会被前端 pre-wrap 渲染成同一气泡内带空行.
_REPLY_SPLIT_RE = re.compile(r'\|\||\n{2,}')

# 单条回复内部残留的换行 (单个 \n 或单个 \r) 收敛为空格,
# 避免句子内 "你好\n吗" 这种意外换行被 pre-wrap 渲染成断行.
_INTRA_REPLY_WS_RE = re.compile(r'[\r\n]+')
_LONG_REPLY_SOFT_BREAK_CHARS = "。！？…～~!?，,；;、"
_NON_TERMINAL_REPLY_END_RE = re.compile(r"[，,、；;：:]+$")

# LLM 偶尔会模仿注入到 prompt 里的历史消息时间前缀 (Phase B1 的 [MM-DD HH:MM] /
# 兼容纯 [HH:MM]), 把它写进自己的回复开头 — 主回复与 tier 回复路径都注入了带时间
# 前缀的历史, 都可能被模仿. prompt 层「防模仿」指令不可靠, 这里在文本规范化时统一
# 剥掉行首时间戳前缀 (可能连续多个), 保证不泄漏到用户可见回复 / 落库 content.
#
# 关键: 模型的模仿常常是**畸形**的 (生产复现 "[07-17 45]" — 日期乱编、分钟缺冒号),
# 旧正则只认严格 [\d-\d \d:\d] 会漏掉这类变体, 漏掉一次就落库并被下一轮历史前缀
# 二次污染, 自我强化到每条回复都带戳. 因此改为匹配"方括号内仅由数字/空白/日期时间
# 分隔符组成、且含至少一个 - 或 : 分隔符、长度≥4"的前缀 — 既能兜住畸形时间戳,
# 又不误伤 [3-5] (长度<4) / [捂脸] (非数字) / [1] (无分隔符) 这类正常方括号内容.
_LEADING_TIMESTAMP_RE = re.compile(
    r'^\s*(?:\[(?=[\d\s/:：\-]*[/:：\-])[\d\s/:：\-]{4,}\]\s*)+'
)


def _strip_leading_timestamp(text: str) -> str:
    """Remove one-or-more leading `[MM-DD HH:MM]` / `[HH:MM]` timestamp prefixes."""
    return _LEADING_TIMESTAMP_RE.sub("", text)


# ── 系统标记剥除 (给系统看的行, 任何出口都绝不能漏给用户) ──
#
# 图灵测试版 chat.response_instruction 要求 LLM 输出 [EMO:标签/强度] 情绪标记
# 和 [X]/【1】条数标记. response_instruction 同时是 reply_prefix (前置注入全部
# 回复类 prompt), 生产 2026-07-19 起复现泄漏: "[2]" 单独成消息 / 回复尾部带
# "[EMO:中性/50]\n【1】". 主回复路径的 extract_emotion_marker 只剥 EMO 且只覆盖
# 主路径 — 这里提供全出口通用的剥除函数, 在 split(_clean_reply_part)/短路/
# 主动/音乐/线下消息出口统一收口.
#
# 条数标记只剥"整条就是标记"或"尾部标记"两种形态 (泄漏的实际形态), 不动
# 文中内嵌的方括号数字 — 与 _LEADING_TIMESTAMP_RE 的保守原则一致.
_EMO_MARKER_ANYWHERE_RE = re.compile(r"[\[【]\s*EMO\s*[:：][^\]】]{0,24}[\]】]")
_COUNT_MARKER_TOKEN = r"[\[【]\s*(?:\d{1,2}|[xXyY])\s*[\]】]"
_ONLY_COUNT_MARKER_RE = re.compile(rf"^\s*(?:{_COUNT_MARKER_TOKEN}\s*)+$")
_TRAILING_COUNT_MARKER_RE = re.compile(rf"(?:\s*{_COUNT_MARKER_TOKEN})+\s*$")


def strip_system_markers(text: str) -> str:
    """剥除 LLM 输出中"给系统看"的标记: [EMO:...] 任意位置 + 条数标记
    ([2]/【1】/[X]) 整条或尾部形态. 返回清理后的文本 (可能为空串)."""
    if not text:
        return text
    cleaned = _EMO_MARKER_ANYWHERE_RE.sub("", text)
    if _ONLY_COUNT_MARKER_RE.match(cleaned):
        return ""
    return _TRAILING_COUNT_MARKER_RE.sub("", cleaned).strip()


def _clean_reply_part(text: str) -> str:
    """单条回复内部规范化: 去首尾空白 + 单个换行折叠成空格 + 剥行首时间戳前缀
    + 剥系统标记 (EMO/条数)."""
    cleaned = _INTRA_REPLY_WS_RE.sub(" ", text).strip()
    cleaned = _strip_leading_timestamp(cleaned).strip()
    return strip_system_markers(cleaned)


def _strip_non_terminal_reply_end(text: str) -> str:
    """Remove dangling boundary punctuation from a split message bubble."""
    return _NON_TERMINAL_REPLY_END_RE.sub("", text.rstrip()).rstrip()


def _polish_split_boundaries(parts: list[str]) -> list[str]:
    """Clean punctuation that only exists because a reply was split."""
    if len(parts) <= 1:
        return parts
    polished = [
        _strip_non_terminal_reply_end(part) if idx < len(parts) - 1 else part
        for idx, part in enumerate(parts)
    ]
    return [part for part in polished if part]


def truncate_at_sentence(text: str, max_len: int) -> str:
    """截断至max_len内最后一个句子边界。"""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    match = None
    for m in _SENTENCE_END.finditer(truncated):
        match = m
    if match and match.end() > max_len // 2:
        return truncated[:match.end()]
    return truncated


def _find_soft_reply_cut(text: str, max_len: int) -> int:
    """Find a punctuation cut point for long single-bubble fallbacks."""
    window = text[:max_len]
    min_cut = max(8, max_len // 3)
    best = -1
    for idx, ch in enumerate(window):
        if ch in _LONG_REPLY_SOFT_BREAK_CHARS and idx + 1 >= min_cut:
            best = idx + 1
    return best


def _split_long_reply_part(text: str, max_len: int, max_count: int) -> list[str]:
    """Split an overlong no-delimiter reply at punctuation instead of mid-clause."""
    remaining = text.strip()
    parts: list[str] = []
    consumed = 0
    while remaining and len(parts) < max_count:
        if len(remaining) <= max_len:
            parts.append(remaining)
            consumed += len(remaining)
            remaining = ""
            break
        cut = _find_soft_reply_cut(remaining, max_len)
        if cut <= 0:
            kept = truncate_at_sentence(remaining, max_len)
            parts.append(kept)
            consumed += len(kept)
            remaining = remaining[len(kept):].strip()
            break
        part = remaining[:cut].strip()
        if part:
            parts.append(part)
            consumed += len(part)
        remaining = remaining[cut:].strip()
    if remaining:
        # 无分隔符长独白填满条数预算后仍有剩余 — 这正是"文字墙"护栏该拦的
        # 形态, 不合并 (总量上限也会拦), 只记录损失量供观测.
        logger.info(
            f"[REPLY-GUARDRAIL] long monologue tail dropped: "
            f"kept={consumed} lost={len(remaining)} chars",
            extra={
                "event": EVT_REPLY_GUARDRAIL,
                "action": "drop_long_tail",
                "chars_kept": consumed,
                "chars_lost": len(remaining),
            },
        )
    return parts


def split_and_validate_replies(
    raw: str,
    max_count: int = MAX_REPLY_COUNT,
    max_per_reply: int = MAX_PER_REPLY,
    max_total: int = MAX_TOTAL_CHARS,
) -> list[str]:
    """按 || 或空行分割 LLM 输出, 校验条数/单条长度/总长度.

    LLM 偶尔不按 prompt 用 ||, 改用空行分段 — 不切的话前端 pre-wrap 会把
    \\n\\n 渲染成单气泡里的空行, 视觉跟正常多条回复混淆. 这里把空行也当
    分隔符. 单条内的孤立 \\n 由 _clean_reply_part 折叠成空格.
    """
    has_explicit_split = bool(_REPLY_SPLIT_RE.search(raw))
    parts = [_clean_reply_part(p) for p in _REPLY_SPLIT_RE.split(raw)]
    parts = [p for p in parts if p]
    if not parts:
        return [_clean_reply_part(raw) or "..."]
    if not has_explicit_split and len(parts) == 1 and len(parts[0]) > max_per_reply:
        parts = _split_long_reply_part(parts[0], max_per_reply, max_count)
    else:
        if len(parts) > max_count:
            # 条数溢出 → 合并进最后一条而非丢弃 (空格连接, 与人设"空格代替
            # 逗号"的口语风格一致). 合并后仍受单条/总量上限约束.
            overflow = len(parts) - max_count
            parts = parts[:max_count - 1] + [" ".join(parts[max_count - 1:])]
            logger.info(
                f"[REPLY-GUARDRAIL] merged {overflow} overflow bubbles into last "
                f"(max_count={max_count})",
                extra={
                    "event": EVT_REPLY_GUARDRAIL,
                    "action": "merge_overflow",
                    "overflow_bubbles": overflow,
                    "max_count": max_count,
                },
            )
        truncated_parts = [truncate_at_sentence(p, max_per_reply) for p in parts]
        per_reply_lost = sum(
            len(orig) - len(cut) for orig, cut in zip(parts, truncated_parts)
        )
        if per_reply_lost:
            n_truncated = sum(
                1 for orig, cut in zip(parts, truncated_parts) if len(cut) < len(orig)
            )
            logger.info(
                f"[REPLY-GUARDRAIL] {n_truncated} bubble(s) truncated at "
                f"{max_per_reply} chars, lost={per_reply_lost}",
                extra={
                    "event": EVT_REPLY_GUARDRAIL,
                    "action": "truncate_bubble",
                    "bubbles_truncated": n_truncated,
                    "chars_lost": per_reply_lost,
                },
            )
        parts = truncated_parts
    result: list[str] = []
    total = 0
    for idx, p in enumerate(parts):
        if total + len(p) > max_total:
            remaining = max_total - total
            chars_lost = sum(len(x) for x in parts[idx:])
            if remaining > 5:
                kept = truncate_at_sentence(p, remaining)
                result.append(kept)
                chars_lost -= len(kept)
            logger.info(
                f"[REPLY-GUARDRAIL] total cap {max_total} hit at bubble "
                f"{idx + 1}/{len(parts)}, lost={chars_lost} chars",
                extra={
                    "event": EVT_REPLY_GUARDRAIL,
                    "action": "truncate_total",
                    "max_total": max_total,
                    "bubbles_planned": len(parts),
                    "bubbles_kept": len(result),
                    "chars_lost": chars_lost,
                },
            )
            break
        result.append(p)
        total += len(p)
    result = _polish_split_boundaries(result)
    return result or [parts[0][:max_per_reply]]


def _merge_adjacent_shortest(parts: list[str], max_per: int) -> list[str] | None:
    """Merge the adjacent pair with the smallest combined length → count-1.

    空格连接 (与人设"空格代替逗号"一致); 合并后仍受单条上限约束.
    """
    if len(parts) < 2:
        return None
    best_i = min(
        range(len(parts) - 1),
        key=lambda i: len(parts[i]) + len(parts[i + 1]),
    )
    merged = f"{parts[best_i].rstrip()} {parts[best_i + 1].lstrip()}".strip()
    merged = truncate_at_sentence(merged, max_per)
    out = parts[:best_i] + [merged] + parts[best_i + 2:]
    out = [p for p in out if p]
    return out or None


def _split_longest_at_sentence(parts: list[str], max_count: int) -> list[str] | None:
    """Split the longest bubble at an internal sentence boundary → count+1.

    只在存在句末标点的内部边界处拆 (不制造断句); 找不到可拆条则返回 None.
    """
    if len(parts) >= max_count:
        return None
    for i in sorted(range(len(parts)), key=lambda k: len(parts[k]), reverse=True):
        text = parts[i]
        internal = [m.end() for m in _SENTENCE_END.finditer(text) if 0 < m.end() < len(text)]
        if not internal:
            continue
        mid = len(text) / 2
        cut = min(internal, key=lambda c: abs(c - mid))
        left, right = text[:cut].strip(), text[cut:].strip()
        if left and right:
            return parts[:i] + [left, right] + parts[i + 1:]
    return None


def enforce_count_variation(
    parts: list[str],
    last_count: int | None,
    *,
    max_count: int = MAX_REPLY_COUNT,
    max_per: int = MAX_PER_REPLY,
) -> tuple[list[str], str | None]:
    """图灵测试硬约束: 相邻两轮的可见气泡数不能相同.

    prompt (chat.reply_count_variation) 已请求 LLM "本轮条数 ≠ 上一轮", 但 LLM
    数句子不可靠, 常继续锁在 2-3 条. 这里做**代码级兜底**: 切分后的实际条数若恰
    好等于上一轮, 用 ±1 打破——合并最短相邻对 (少一条) 或在句末边界拆最长条
    (多一条), 全程保持在 [1, max_count] 与单条字数上限内.

    方向: 低条数优先"加"(让 1/4 条也出现, 抵消 2-3 偏好), 高条数优先"减", 边界
    强制单向; 首选方向不可行时退另一方向. 都不可行 (如唯一一条且无内部句末边界)
    原样返回, action='unresolved' 便于观测 prompt 遵从度.
    """
    if last_count is None or not parts:
        return parts, None
    n = len(parts)
    if n != last_count:
        return parts, None

    grow_first = n < max_count and n <= max_count // 2
    order = ("grow", "shrink") if grow_first else ("shrink", "grow")
    for direction in order:
        candidate = (
            _split_longest_at_sentence(parts, max_count) if direction == "grow"
            else _merge_adjacent_shortest(parts, max_per)
        )
        if candidate and 1 <= len(candidate) <= max_count and len(candidate) != last_count:
            logger.info(
                f"[REPLY-COUNT-VARY] {direction} {n}->{len(candidate)} (last={last_count})",
                extra={
                    "event": EVT_REPLY_GUARDRAIL,
                    "action": f"count_vary_{direction}",
                    "from_count": n,
                    "to_count": len(candidate),
                    "last_count": last_count,
                },
            )
            return candidate, direction
    logger.info(
        f"[REPLY-COUNT-VARY] unresolved n={n} last={last_count} (no safe adjustment)",
        extra={
            "event": EVT_REPLY_GUARDRAIL,
            "action": "count_vary_unresolved",
            "from_count": n,
            "last_count": last_count,
        },
    )
    return parts, "unresolved"


