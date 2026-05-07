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
- 用法 1 (DEDUP): 极性不一致 → 不算重复
- 用法 2 (RETRIEVAL): 用户 query 有否定 + candidate 无否定 (或反之) → 降权

局限 (诚实记录):
- 不识别"几乎不/很少/不一定" 等模糊否定
- 不识别反讽 ("我可'喜欢'咖啡呢")
- 否定常见非否定语境 ("不错"/"差不多"/"不止") 用 _NEUTRALIZE_PHRASES 排除,
  但无穷举. 接受 ~5-10% 假阳率: dedup 假阳 = 多存一条 (无数据丢失);
  retrieval 假阳 = 降权 0.3 (仍召回, LLM 可见).
"""

from __future__ import annotations

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
