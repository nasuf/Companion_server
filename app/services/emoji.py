"""表情推荐服务。

基于情绪标签匹配表情，纯计算无LLM。
"""

from __future__ import annotations

import random
import re

# 情绪→表情映射
EMOJI_MAP: dict[str, list[str]] = {
    "高兴": ["😄", "😊", "🥰", "😁", "🎉", "✨"],
    "悲伤": ["😢", "😭", "🥺", "💔", "😞"],
    "愤怒": ["😤", "😠", "💢", "🔥"],
    "惊讶": ["😮", "😲", "🤯", "❗"],
    "恐惧": ["😨", "😰", "🫣"],
    "厌恶": ["😒", "🙄", "😑"],
    "中性": ["😊", "🙂", "👌"],
    "焦虑": ["😰", "😟", "🫤", "💭"],
    "失望": ["😞", "😔", "💔"],
    "欣慰": ["😌", "🥰", "💕"],
    "感激": ["🙏", "🥹", "💕", "❤️"],
    "戏谑": ["😏", "😜", "🤪", "😈"],
}

# 正面/负面/中性分类
POSITIVE_EMOTIONS = {"高兴", "欣慰", "感激", "戏谑"}
NEGATIVE_EMOTIONS = {"悲伤", "愤怒", "恐惧", "厌恶", "焦虑", "失望"}
NEUTRAL_EMOTIONS = {"惊讶", "中性"}


def recommend_emoji(
    primary_emotion: str | None = None,
    count: int = 3,
) -> list[str]:
    """推荐表情。优先使用显式情绪标签，未知时回退中性。"""
    pool = EMOJI_MAP.get(primary_emotion or "中性", EMOJI_MAP["中性"])
    return random.sample(pool, min(count, len(pool)))


def should_add_emoji(intensity: int = 0) -> bool:
    """Spec §5.3 步骤 2 (2026-07-08 修订版): P_base = random(0, 0.2),
    P_final = min(0.6, P_base + A × 0.3).

    A 在 spec 里是 AI PAD 唤醒度; PAD 管线已移除 (01ee8d2), 用本条回复
    情绪强度/100 作为 A 的代理信号 — 语义同向 (情绪越强越可能带表情).
    旧版 0.4/0.8/0.5 是滥用来源之一, 按新 spec 整体下调约一半.
    """
    signal = max(0.0, min(1.0, intensity / 100))
    p_base = random.uniform(0, 0.2)
    p_final = min(0.6, max(0, p_base + signal * 0.3))
    return random.random() < p_final


def pick_one_emoji(
    primary_emotion: str | None = None,
    exclude: set[str] | frozenset[str] | None = None,
) -> str:
    """从情绪池随机选一个 emoji。

    exclude: 最近几轮已用过的 emoji（跨轮重复回避, C4 拟人度 — 真人不会
    连着几条都用同一个表情）。整池被排除时回退全池, 保证总能选出。
    """
    pool = EMOJI_MAP.get(primary_emotion or "中性", EMOJI_MAP["中性"])
    candidates = [e for e in pool if e not in (exclude or ())] or list(pool)
    return random.choice(candidates)


def should_add_sticker(intensity: int = 0) -> bool:
    """Emotion intensity based sticker probability."""
    signal = max(0.0, min(1.0, intensity / 100))
    p_base = random.uniform(0, 0.4)
    p_final = min(0.7, max(0, p_base + signal * 0.4))
    return random.random() < p_final


# ── 每条消息 emoji 硬上限 ────────────────────────────────────────────
# 一个"emoji 单元": 旗帜对 | 基础emoji + 可选肤色/VS16 + 任意 (ZWJ+emoji) 续接.
# 覆盖 U+1F000-1FAFF (绝大多数 emoji) / 2600-27BF (杂项符号) / 2B00-2BFF (⭐等).
# 肤色修饰符单独出现时会被算作独立单元并剥除 — 结果仍是"一个可见表情", 可接受.
_EMOJI_UNIT_RE = re.compile(
    "(?:[\U0001F1E6-\U0001F1FF]{2}"
    "|(?:[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]\uFE0F?"
    "(?:\u200D[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]\uFE0F?)*))"
)


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_UNIT_RE.search(text or ""))


def limit_emojis(text: str, max_keep: int = 1) -> str:
    """硬保证: 一条消息最多 max_keep 个 emoji, 超出的按出现顺序剥除.

    针对 LLM 在正文里自行生成多个表情的失效模式 (prompt 只能引导不能保证);
    保留第一个是因为它通常贴着最相关的情绪点. 所有用户可见出口
    (主回复 emit / 短路回复 / 主动消息 / 延迟解释) 统一调用.
    """
    if not text:
        return text
    matches = list(_EMOJI_UNIT_RE.finditer(text))
    if len(matches) <= max_keep:
        return text
    out: list[str] = []
    last = 0
    kept = 0
    for m in matches:
        out.append(text[last:m.start()])
        if kept < max_keep:
            out.append(m.group())
            kept += 1
        last = m.end()
    out.append(text[last:])
    return "".join(out)

