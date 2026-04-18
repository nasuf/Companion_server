"""记忆消息过滤器。

多信号准入，决定是否对消息进行记忆提取。
核心目标是过滤纯寒暄/回声词，同时避免错过简短但高价值的自我披露。
"""

from __future__ import annotations

import re

# 情感词库
_EMOTION_WORDS = {
    "开心", "难过", "伤心", "焦虑", "害怕", "生气", "愤怒", "感动",
    "孤独", "压力", "兴奋", "失望", "后悔", "感恩", "紧张", "无聊",
    "幸福", "痛苦", "委屈", "满足", "期待", "惊讶", "厌恶", "嫉妒",
    "喜欢", "讨厌", "爱", "恨", "想念", "思念", "担心", "不安",
}

_EMOTION_WORDS_EN = {
    "happy", "sad", "upset", "angry", "anxious", "afraid", "excited",
    "lonely", "stressed", "disappointed", "grateful", "worried", "love",
    "hate", "miss", "nervous",
}

# 时间词库
_TIME_WORDS = {
    "昨天", "今天", "明天", "上周", "下周", "去年", "今年", "明年",
    "小时候", "以前", "之前", "最近", "刚才", "刚刚", "上个月", "下个月",
    "周末", "春节", "暑假", "寒假", "毕业", "当时", "那时候",
}

_TIME_WORDS_EN = {
    "yesterday", "today", "tomorrow", "last week", "next week", "last year",
    "this year", "next year", "recently", "before", "just now", "weekend",
    "childhood", "vacation", "christmas", "spring festival",
}

# 事实词库（表示陈述事实信息）
_FACT_WORDS = {
    "是", "叫", "在", "住", "岁", "工作", "学习", "上学", "毕业",
    "喜欢", "讨厌", "不喜欢", "养", "有", "买了", "去了", "来了",
    "家", "公司", "学校", "大学", "城市", "专业", "职业",
}

_FACT_WORDS_EN = {
    "am", "live", "work", "study", "major", "job", "age", "from",
    "born", "birthday", "like", "love", "hate", "prefer", "married",
    "single", "family", "mom", "dad", "wife", "husband", "boyfriend",
    "girlfriend", "pet", "dog", "cat",
}

# 第一人称词
_FIRST_PERSON = {"我", "咱", "俺", "自己"}
_FIRST_PERSON_EN = {"i", "i'm", "im", "my", "me", "mine"}

# "我…"句式模式
_SELF_DISCLOSURE_PATTERNS = [
    re.compile(r"我(是|叫|在|住|有|喜欢|讨厌|想|觉得|认为|打算|准备|希望)"),
    re.compile(r"我(的|们)(家|妈|爸|朋友|同事|老板|同学|女朋友|男朋友|老公|老婆)"),
]

_SELF_DISCLOSURE_PATTERNS_EN = [
    re.compile(r"\bi\s*(?:am|'m)\s+\w+"),
    re.compile(r"\bmy\s+(?:name|job|work|major|family|mom|dad|wife|husband|boyfriend|girlfriend|dog|cat)\b"),
    re.compile(r"\bi\s+(?:live|work|study|like|love|hate|prefer|feel|want|plan|grew up)\b"),
]

_CORE_PROFILE_PATTERNS = [
    re.compile(r"我\d{1,2}岁"),
    re.compile(r"我(在|住在|来自|老家在|工作|上班|读|学|养了|有)(.+)"),
    re.compile(r"(我是|我叫)(.+)"),
    re.compile(r"\b(?:i am|i'm|im)\s+\d{1,2}\b"),
    re.compile(r"\b(?:i am|i'm|im)\s+(?:a|an)\s+\w+"),
    re.compile(r"\bi\s+(?:live|work|study)\s+(?:in|at)\b"),
    re.compile(r"\bmy\s+(?:name|birthday|job|major|family)\s+"),
]


def _word_set(message: str) -> set[str]:
    """Extract lowercase latin words for lightweight English heuristics."""
    return {w.lower() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", message)}


# Spec §2.1.1: 常用应答词/语气词，长度≤2 时直接丢弃
_FILLER_WORDS: set[str] = {
    "嗯", "哦", "好", "对", "啊", "哈", "呢", "吧", "呀", "噢", "唔",
    "ok", "嗯嗯", "哦哦", "好的", "好吧", "行", "是", "对对",
    "哈哈", "嘿", "嘻嘻", "呵呵", "hihi", "yeah", "yep", "nope",
    "mhm", "hmm", "haha", "lol",
}


def should_extract_memory(message: str) -> bool:
    """判断消息是否值得进行记忆提取。

    Spec §2.1.1: 纯语气词或长度≤2的常用应答词直接丢弃。
    """
    if not message or not message.strip():
        return False

    msg = message.strip()

    # Spec §2.1.1: 硬拒短消息 + 语气词
    if len(msg) <= 2 and msg.lower() in _FILLER_WORDS:
        return False
    if msg.lower() in _FILLER_WORDS:
        return False

    msg_lower = msg.lower()
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', message))
    latin_words = _word_set(msg)
    non_space_chars = len(re.sub(r"\s+", "", msg))
    total_weight = 0

    # 强信号：明确的个人资料/自我披露，直接放行。
    if any(p.search(msg_lower if "i" in p.pattern or "my" in p.pattern else msg) for p in _CORE_PROFILE_PATTERNS):
        return True

    # Rule 1: 基础长度 (w=1)
    if chinese_chars >= 5 or non_space_chars >= 12:
        total_weight += 1

    # Rule 2: 第一人称 (w=1)
    if any(p in msg for p in _FIRST_PERSON) or any(p in latin_words for p in _FIRST_PERSON_EN):
        total_weight += 1

    # Rule 3: 情感词 (w=2)
    if any(w in msg for w in _EMOTION_WORDS) or any(w in latin_words for w in _EMOTION_WORDS_EN):
        total_weight += 2

    # Rule 4: 时间词 (w=1)
    if any(w in msg for w in _TIME_WORDS) or any(w in msg_lower for w in _TIME_WORDS_EN):
        total_weight += 1

    # Rule 5: 事实词 (w=1)
    if any(w in msg for w in _FACT_WORDS) or any(w in latin_words for w in _FACT_WORDS_EN):
        total_weight += 1

    # Rule 6: 自我暴露句式 (w=2)
    if any(p.search(msg) for p in _SELF_DISCLOSURE_PATTERNS) or any(p.search(msg_lower) for p in _SELF_DISCLOSURE_PATTERNS_EN):
        total_weight += 2

    # Rule 7: 长消息/含数字的简历式事实 (w=1)
    if chinese_chars >= 30 or non_space_chars >= 30:
        total_weight += 1
    if re.search(r"\d", msg) and (
        any(p in msg for p in _FIRST_PERSON)
        or any(p in latin_words for p in _FIRST_PERSON_EN)
        or any(w in msg for w in {"岁", "年", "月", "天"})
        or any(w in latin_words for w in {"years", "year", "old"})
    ):
        total_weight += 1

    return total_weight >= 2
