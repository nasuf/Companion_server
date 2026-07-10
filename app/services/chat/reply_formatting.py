"""回复切分与文本规范化 — 纯函数层（orchestrator 拆分 R2）。

Multi-reply split & validate (PRD §3.2.1/§3.2.2)：按 || / 空行切分主 LLM
输出，校验条数/单条长度/总长度，句末边界截断。全部为无副作用的纯文本
函数，从 orchestrator.py 提取；orchestrator re-export 保持导入路径兼容。
"""

from __future__ import annotations

import re

from app.services.prompts.system_prompts import (
    MAX_PER_REPLY,
    MAX_REPLY_COUNT,
    MAX_TOTAL_CHARS,
)

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


def _clean_reply_part(text: str) -> str:
    """单条回复内部规范化: 去首尾空白 + 单个换行折叠成空格 + 剥行首时间戳前缀."""
    cleaned = _INTRA_REPLY_WS_RE.sub(" ", text).strip()
    return _strip_leading_timestamp(cleaned).strip()


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
    while remaining and len(parts) < max_count:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
        cut = _find_soft_reply_cut(remaining, max_len)
        if cut <= 0:
            parts.append(truncate_at_sentence(remaining, max_len))
            break
        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()
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
        parts = parts[:max_count]
        parts = [truncate_at_sentence(p, max_per_reply) for p in parts]
    result: list[str] = []
    total = 0
    for p in parts:
        if total + len(p) > max_total:
            remaining = max_total - total
            if remaining > 5:
                result.append(truncate_at_sentence(p, remaining))
            break
        result.append(p)
        total += len(p)
    result = _polish_split_boundaries(result)
    return result or [parts[0][:max_per_reply]]


