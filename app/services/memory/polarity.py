"""Phase 3: 文本极性检测 — 修复 bge-m3 反义盲区.

实证 (2026-05-07 measurement):
- "我喜欢咖啡" vs "我不喜欢咖啡": cosine 0.84
- "我住在北京" vs "我不住在北京": cosine 0.89
- "我妈妈很健康" vs "我妈妈生病了": cosine 0.83
- 同义对 ("我爱咖啡" vs "我喜欢咖啡"): cosine 0.93
- 完全无关 ("我喜欢咖啡" vs "今天天气不错"): cosine 0.63

→ 反义对 cosine ≈ 同义对 ± 0.1, embedding 几乎不可区分.

影响:
1. DEDUP (cosine > 0.85 视为重复): 反义对会被误判 → 后写的内容被拒收 → 数据丢失.
   "我住在北京" 已存, 用户搬家后说"我不住北京" → cosine 0.89 > 0.85 → 拒收.
2. RETRIEVAL (cosine > 0.5 视为相关): 反义对都召回 → prompt 同时含正反事实 →
   LLM 拍脑袋选一个, 经常选错.

修复策略 (规则层, 不引入 cross-encoder 模型 / 不加 schema 字段):
- has_negation(text): 检测显式否定标记
- is_polarity_match(a, b): 两文本极性是否一致
- semantic_conflict_reasons(a, b): 检测非否定词也能表达的语义对立
- 用法 1 (DEDUP): 极性不一致 → 不算重复
- 用法 2 (RETRIEVAL): 用户 query 表达明确反向 stance/status, 或事实否定目标能
  与 candidate 对齐时 → 降权

局限 (诚实记录):
- 不识别"几乎不/很少/不一定" 等模糊否定
- 不识别反讽 ("我可'喜欢'咖啡呢")
- 否定常见非否定语境 ("不错"/"差不多"/"不止") 用 _NEUTRALIZE_PHRASES 排除,
  但无穷举. 接受 ~5-10% 假阳率: dedup 假阳 = 多存一条 (无数据丢失);
  retrieval 假阳 = 降权 0.3 (仍召回, LLM 可见).
- 语义对立规则只覆盖高风险高频维度: 偏好立场、伴侣状态/身份、就医状态/阶段。
  它不替代 cross-encoder 或更强 embedding, 但能挡住最常见的"喜欢/讨厌"、
  "前男友/前女友"、"住院/出院"类张冠李戴。
"""

from __future__ import annotations

import re

# 中文否定词. 单字符级 substring 匹配; 大多场景"否定词 + 动词" 模式.
_CN_NEGATIONS: tuple[str, ...] = (
    "不", "没", "无", "未", "非", "否", "勿", "别", "莫",
)

# 英文否定词 / 缩写. 加空格防部分匹配 (e.g. "not" 不该匹配 "note").
_EN_NEGATIONS: tuple[str, ...] = (
    " not ", "n't",  # contractions: don't, won't, isn't, hasn't, etc.
    " no ", " never", " none",
    "without", "lack of", "fail to",
)

# "不/没/差不多" 等非否定语境短语. 检测前先 neutralize 掉这些, 减假阳.
# 不穷举, 只挡最常见的高频假阳: "不错"=好, "差不多"=neutral, "不仅"=并且.
_NEUTRALIZE_PHRASES: tuple[str, ...] = (
    "不错", "不少", "不止", "不仅", "不如", "不愧", "不只", "不光", "不但",
    "不过", "不到", "不久", "差不多", "对不起", "对不住", "可不可以",
    "不用谢", "不客气", "没事", "没错",
)

_SEMANTIC_DIMENSIONS: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "preference": (
        "偏好立场",
        {
            "negative": (
                "不喜欢", "不爱", "不吃", "不喝", "讨厌", "厌恶",
                "反感", "排斥", "雷区", "过敏",
            ),
            "positive": (
                "喜欢", "爱吃", "爱喝", "爱用", "爱看", "最爱",
                "偏爱", "钟意", "很爱",
            ),
        },
    ),
    "partner_status": (
        "伴侣状态",
        {
            "ex": ("前男友", "前女友", "前任男友", "前任女友", "前夫", "前妻", "前任"),
            "current": (
                "男朋友", "女朋友", "男友", "女友", "对象", "伴侣",
                "老公", "老婆", "丈夫", "妻子",
            ),
        },
    ),
    "partner_gender": (
        "伴侣身份",
        {
            "male": ("前男友", "前任男友", "前夫", "男朋友", "男友", "老公", "丈夫"),
            "female": ("前女友", "前任女友", "前妻", "女朋友", "女友", "老婆", "妻子"),
        },
    ),
    "hospital_status": (
        "就医状态",
        {
            "admitted": ("住院", "入院", "住进医院", "进医院", "留院"),
            "discharged": (
                "出院", "不用住院", "不住院", "回家休养", "回家了",
            ),
        },
    ),
    "medical_phase": (
        "就医阶段",
        {
            "surgery": ("手术", "开刀"),
            "discharged": ("出院", "回家休养"),
        },
    ),
}

# Retrieval should be conservative with bare negation. A sentence can contain a
# negation marker without asserting the opposite of a stored fact, e.g. "不是特别复杂"
# or "没那么严重". These patterns identify degree/modality negation so it does
# not become a generic semantic-conflict signal.
_CN_DEGREE_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:不|并不)(?:是)?(?:很|太|大|特别|非常|十分|那么|这么|"
        r"算|够|怎么|咋|至于|一定|完全|见得)"
    ),
    re.compile(r"(?:没|没有)(?:很|太|那么|这么|多|少|大|严重|复杂|困难|容易)"),
)

_CN_FACTUAL_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:不|并不)(?P<predicate>是|叫|姓|住在|住|在|属于|来自|"
        r"去过|做过|学过|写过|会|用|吃|喝|看|听|认识|养|需要|要|想|记得)"
        r"(?P<target>[\u4e00-\u9fffA-Za-z0-9_《》“”\"'·\-]{1,24})?"
    ),
    re.compile(
        r"(?:没|没有|未)(?P<predicate>有|去过|做过|住过|见过|看过|听过|"
        r"吃过|喝过|买过|用过|养过|学过|写过|认识|记得)?"
        r"(?P<target>[\u4e00-\u9fffA-Za-z0-9_《》“”\"'·\-]{1,24})"
    ),
    re.compile(
        r"(?:无|非)(?P<target>[\u4e00-\u9fffA-Za-z0-9_《》“”\"'·\-]{1,24})"
    ),
)

_TARGET_PREFIXES: tuple[str, ...] = (
    "一个", "一名", "一种", "一位", "这个", "那个", "这位", "那位",
    "任何", "什么", "特别", "非常", "很", "太", "比较",
)


def _detect_dimension_stance(
    text: str,
    groups: dict[str, tuple[str, ...]],
) -> str | None:
    """Return a single semantic stance for one dimension, or None when mixed.

    Longer phrases are consumed first so "前女友" does not also become "女友",
    and "不喜欢" does not also become "喜欢".
    """
    remaining = text
    hits: list[str] = []
    keyword_items = [
        (label, keyword)
        for label, keywords in groups.items()
        for keyword in keywords
    ]
    for label, keyword in sorted(keyword_items, key=lambda item: len(item[1]), reverse=True):
        if keyword and keyword in remaining:
            hits.append(label)
            remaining = remaining.replace(keyword, "")

    unique_hits = set(hits)
    if len(unique_hits) == 1:
        return hits[0]
    return None


def detect_semantic_stances(text: str) -> dict[str, str]:
    """Detect coarse semantic stances that bge-style embeddings often confuse."""
    if not text:
        return {}
    stances: dict[str, str] = {}
    for dimension, (_, groups) in _SEMANTIC_DIMENSIONS.items():
        stance = _detect_dimension_stance(text, groups)
        if stance:
            stances[dimension] = stance
    return stances


def _remove_degree_negation(text: str) -> str:
    cleaned = text
    for pattern in _CN_DEGREE_NEGATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _clean_factual_target(target: str | None) -> str:
    if not target:
        return ""
    cleaned = re.split(r"[，。！？?、,.\s]", target, maxsplit=1)[0]
    cleaned = re.sub(r"(吗|嘛|呢|啊|呀|吧|了|过|着|的)$", "", cleaned)
    changed = True
    while changed:
        changed = False
        for prefix in _TARGET_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                changed = True
    return cleaned.strip(" “”、\"'《》")


def _extract_factual_negation_targets(text: str) -> list[str]:
    """Extract concrete targets from factual negation.

    This intentionally ignores generic/degree negation. The target is used only
    when it can be aligned with a candidate memory, so ambiguous negation does
    not suppress unrelated memories.
    """
    if not text:
        return []

    cleaned = _remove_degree_negation(text)
    targets: list[str] = []
    for pattern in _CN_FACTUAL_NEGATION_PATTERNS:
        for match in pattern.finditer(cleaned):
            target = _clean_factual_target(match.groupdict().get("target"))
            if target:
                targets.append(target)
    return targets


def _target_overlaps_candidate(target: str, candidate_text: str) -> bool:
    return len(target) >= 2 and target in candidate_text


def _has_factual_negation_conflict(negated_text: str, affirmative_text: str) -> bool:
    if has_negation(affirmative_text):
        return False
    return any(
        _target_overlaps_candidate(target, affirmative_text)
        for target in _extract_factual_negation_targets(negated_text)
    )


def _semantic_label(dimension: str) -> str:
    label, _ = _SEMANTIC_DIMENSIONS.get(dimension, (dimension, {}))
    return label


def has_negation(text: str) -> bool:
    """检测文本是否含显式否定标记.

    实现: 先去掉常见非否定短语 (e.g. "不错"=good, 不算 negation), 再 substring
    匹配中英否定词. 假阳/假阴接受 ~5-10% (见模块 docstring).
    """
    if not text:
        return False

    # 中文非否定语境短语 neutralize
    cleaned = text
    for phrase in _NEUTRALIZE_PHRASES:
        cleaned = cleaned.replace(phrase, "")

    # 中文否定词 substring (单字符已足够 — neutralize 后剩下的"不/没" 多为真否定)
    if any(neg in cleaned for neg in _CN_NEGATIONS):
        return True

    # 英文否定词 (lowercase + 空格防部分匹配)
    t_lower = " " + cleaned.lower() + " "
    if any(neg in t_lower for neg in _EN_NEGATIONS):
        return True

    return False


def is_polarity_match(text_a: str, text_b: str) -> bool:
    """两文本是否同极性 (都有否定, 或都没有). 用于 dedup 决定真重复 vs 反义.

    True  → 同极性 (典型: 都正向陈述, 都否定陈述)
    False → 反义对, embedding 高 cosine 但语义相反, 不该 dedup
    """
    return has_negation(text_a) == has_negation(text_b)


def semantic_conflict_reasons(text_a: str, text_b: str) -> list[str]:
    """Symmetric semantic conflict check for storage dedup.

    False positives are intentionally safer than false negatives here: a false
    positive stores two near-duplicate rows, while a false negative can discard
    a corrected fact forever.
    """
    reasons: list[str] = []
    stances_a = detect_semantic_stances(text_a)
    stances_b = detect_semantic_stances(text_b)
    same_explicit_stance = any(
        stances_b.get(dimension) == stance
        for dimension, stance in stances_a.items()
    )
    if (
        not same_explicit_stance
        and (
            _has_factual_negation_conflict(text_a, text_b)
            or _has_factual_negation_conflict(text_b, text_a)
        )
    ):
        reasons.append("否定极性")

    for dimension, stance_a in stances_a.items():
        stance_b = stances_b.get(dimension)
        if stance_b and stance_a != stance_b:
            reasons.append(_semantic_label(dimension))
    return reasons


def query_semantic_conflict_reasons(query: str, candidate_text: str) -> list[str]:
    """Directional semantic conflict check for retrieval reranking.

    Only downweight when the query itself expresses a stance/status or an
    aligned factual negation. A broad query like "我对咖啡的看法" should still be
    allowed to retrieve both "喜欢" and "不喜欢/讨厌" memories for context.
    """
    reasons: list[str] = []
    query_stances = detect_semantic_stances(query)
    candidate_stances = detect_semantic_stances(candidate_text)
    same_explicit_stance = any(
        candidate_stances.get(dimension) == stance
        for dimension, stance in query_stances.items()
    )
    if (
        not same_explicit_stance
        and _has_factual_negation_conflict(query, candidate_text)
    ):
        reasons.append("否定极性")

    for dimension, query_stance in query_stances.items():
        candidate_stance = candidate_stances.get(dimension)
        if candidate_stance and query_stance != candidate_stance:
            reasons.append(_semantic_label(dimension))
    return reasons
