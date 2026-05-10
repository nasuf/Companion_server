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
_USER_TARGET_PREFIXES = (
    "你觉得我", "你认为我", "你怎么看我", "你对我",
    "你猜我", "你看我", "你说我",
)
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
_PROFILE_QUERY_DIMENSIONS: tuple[
    tuple[tuple[str, ...], str, tuple[str, ...]], ...
] = (
    (
        ("多大", "几岁", "年龄", "哪年出生", "什么时候出生", "出生年份", "出生"),
        "年龄 几岁 多大 出生年份",
        ("年龄",),
    ),
    (("生日",), "生日 出生日期", ("生日", "年龄")),
    (("叫什么", "叫啥", "名字", "姓名", "是谁"), "姓名 名字 叫什么", ("姓名",)),
    (
        (
            "做什么", "干什么", "职业", "什么工作", "工作是什么",
            "工作是啥", "在哪工作", "在哪里工作",
        ),
        "职业 工作 做什么",
        ("职业/与经济",),
    ),
    (("哪里人", "哪的人", "家乡"), "家乡 哪里人", ("现居地", "居住", "其他")),
    (("住哪", "住在哪里", "现居"), "现居地 住址 住哪", ("现居地", "居住")),
    (
        (
            "学历", "教育背景", "学校", "上学", "读书", "毕业",
            "初中", "高中", "中学", "普通高中", "大学", "本科",
            "研究生", "专业",
        ),
        "教育背景 学历 学校 上学 读书 初中 高中 中学 普通高中 大学 本科 研究生 专业 毕业",
        ("教育背景",),
    ),
    (("星座",), "星座", ("星座",)),
    (("生肖",), "生肖", ("生肖",)),
    (("血型",), "血型", ("血型",)),
    (("身高",), "身高", ("身高",)),
    (("体型",), "体型", ("体型",)),
    (("性别",), "性别", ("性别",)),
)
_AI_PROFILE_QUERY_TERMS = tuple(
    dict.fromkeys(term for terms, _, _ in _PROFILE_QUERY_DIMENSIONS for term in terms)
)


def _normalize(text: str) -> str:
    return "".join(text.split())


def _join_unique_words(parts: list[str]) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for word in part.split():
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    return " ".join(words)


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


def asks_ai_profile_relation(user_message: str) -> bool:
    """Whether the user asks the agent's stable profile/identity facts.

    This covers shape-level profile questions such as age, name, job, hometown,
    education, birthday, etc. It intentionally excludes "你觉得我/你知道我..."
    cases where the answer should be about the user instead of the agent.
    """
    text = _normalize(user_message)
    if not text:
        return False
    if text.startswith(_USER_TARGET_PREFIXES):
        return False
    if text.startswith(_USER_INFO_QUERY_PREFIXES):
        return False
    lower = text.lower()
    has_ai_subject = (
        "你" in text[:8]
        or lower.startswith("ai")
        or lower.startswith("agent")
    )
    return has_ai_subject and any(term in text for term in _AI_PROFILE_QUERY_TERMS)


def ai_profile_search_query(user_message: str) -> str:
    """Expanded retrieval query for agent profile questions.

    Profile questions need two isolated memory lanes: the agent's own profile
    facts are answer evidence, while matching user facts are only dialogue
    context (for example, avoid asking "你呢" when the user already told us).
    Downstream ranking/selection must keep those roles separate. Return an
    empty string for non profile questions so callers can keep the original
    query.
    """
    text = _normalize(user_message)
    if not asks_ai_profile_relation(text):
        return ""
    parts: list[str] = []
    for terms, expansion, _ in _PROFILE_QUERY_DIMENSIONS:
        if any(term in text for term in terms):
            parts.append(expansion)
    if not parts:
        parts.append("身份 个人资料")
    expanded = _join_unique_words(parts)
    return f"{text} AI {expanded} 用户 {expanded}"


def profile_query_subcategories(query: str) -> set[str]:
    """Return structured profile subcategories implied by a query.

    This helper is intentionally source-agnostic: selector/ranker can use it for
    AI profile questions and user profile questions without duplicating the
    profile taxonomy term map.
    """
    text = _normalize(query)
    matched: set[str] = set()
    for terms, _, sub_categories in _PROFILE_QUERY_DIMENSIONS:
        if any(term in text for term in terms):
            matched.update(sub_categories)
    return matched


def asks_ai_stable_relation(user_message: str) -> bool:
    """Whether the user asks about the agent's stable relation to a topic.

    Examples: "你喜欢什么电影", "你去过哪些城市", "你怎么看科幻片", "你多大了".
    Counterexamples: "你觉得我怎么样", "你还记得我喜欢什么电影吗".
    """
    text = _normalize(user_message)
    if not text:
        return False
    if text.startswith(_USER_TARGET_PREFIXES):
        return False
    if text.startswith(_USER_INFO_QUERY_PREFIXES):
        return False
    if asks_ai_profile_relation(text):
        return True

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
