"""Shared query pattern helpers for memory retrieval.

These helpers are deterministic gates around LLM relevance. Keep them generic:
they should describe query shape, not any specific agent or user identity.
"""

from __future__ import annotations


_AI_RELATION_VERBS = (
    "知道", "认识", "了解", "熟悉", "听过", "看过", "读过", "玩过",
    "吃过", "喝过", "去过", "用过", "喜欢", "爱", "讨厌", "不喜欢",
    "偏爱", "偏", "常看", "常听", "爱看", "爱听", "觉得", "认为",
)
_AI_RELATION_OBJECT_STRIP = "吗嘛呢么呀吧啊？?！!，,。."
_AI_OPINION_PREFIXES = ("你怎么看", "你如何看", "你对")
_USER_TARGET_PREFIXES = ("你觉得我", "你认为我", "你怎么看我", "你对我")
_USER_INFO_QUERY_PREFIXES = (
    "你知道我", "你了解我", "你还记得我", "你记得我",
    "你记不记得我", "你是否记得我",
)
_LEADING_QUESTION_WORDS = (
    "什么", "哪些", "哪种", "哪类", "哪个", "哪一个", "怎样", "怎么样",
    "如何", "多少", "几", "有没有", "有啥", "有什么",
)
_SUBJECT_FILLERS = (
    "最", "比较", "更", "更喜欢", "会", "还", "也", "都", "平时",
    "通常", "一般", "经常", "常常", "偶尔", "最爱",
)
_OBJECT_PLACEHOLDERS = {"这个", "那个", "这些", "那些", "这", "那", "它", "他", "她"}


def _normalize(text: str) -> str:
    return "".join(text.split())


def _has_relation_object(text: str) -> bool:
    obj = text.strip(_AI_RELATION_OBJECT_STRIP).strip()
    if not obj:
        return False
    for word in _LEADING_QUESTION_WORDS:
        obj = obj.removeprefix(word)
    for word in _SUBJECT_FILLERS:
        obj = obj.removeprefix(word)
    obj = obj.strip(_AI_RELATION_OBJECT_STRIP).strip()
    if not obj:
        return True
    if obj in _OBJECT_PLACEHOLDERS:
        return False
    return len(obj) >= 2


def _has_ai_subject_before(text: str, verb_index: int) -> bool:
    prefix = text[:verb_index]
    subject_index = prefix.rfind("你")
    if subject_index < 0:
        return False
    between = prefix[subject_index + 1:]
    if len(between) > 8:
        return False
    return len(between) <= 8 and "我" not in between


def asks_ai_stable_relation(user_message: str) -> bool:
    """Whether the user asks about the agent's stable relation to a topic.

    Examples: "你喜欢什么电影", "你去过哪些城市", "你怎么看科幻片".
    Counterexamples: "你觉得我怎么样", "你还记得我喜欢什么电影吗".
    """
    text = _normalize(user_message)
    if not text:
        return False
    if text.startswith(_USER_TARGET_PREFIXES):
        return False
    if text.startswith(_USER_INFO_QUERY_PREFIXES):
        return False

    for verb in _AI_RELATION_VERBS:
        idx = text.find(verb)
        if idx >= 0 and _has_ai_subject_before(text, idx):
            before = text[:idx].replace("你", "", 1)
            after = text[idx + len(verb):]
            if _has_relation_object(after) or _has_relation_object(before):
                return True

        prefix = f"你{verb}"
        if text.startswith(prefix) and _has_relation_object(text[len(prefix):]):
            return True

        infix = f"你{verb}"
        if text.endswith(("吗", "嘛", "呢", "么", "？", "?")) and infix in text:
            before, _, after = text.partition(infix)
            if _has_relation_object(before) or _has_relation_object(after):
                return True

    for prefix in _AI_OPINION_PREFIXES:
        if text.startswith(prefix) and _has_relation_object(text[len(prefix):]):
            return True

    return False
