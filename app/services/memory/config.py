"""Memory system shared constants.

Centralized here so thresholds/timeouts are not silently duplicated
across files. Callers should import from this module rather than
redefining literals.
"""

# 本文件及 retrieval/ 下的相似度阈值是针对这个 embedding 模型标定的. 向量的
# 余弦分布随模型变化, 而且不是均匀平移 (低端偏移大, 高端几乎不动), 所以换模型
# 必须整组重标 —— 见 scripts/calibrate_embedding_thresholds.py.
#
# 阈值错配不会抛异常, 只会让 AI 悄悄变笨: 门开太大就往 prompt 里灌噪声, 关太
# 死就整轮失忆. 启动时会拿它跟实际生效的模型比对并告警.
CALIBRATED_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# ── Importance → level (spec §1.4) ────────────────────────────────────────
# 这条换算原本散在三处 (录入管线 / 矛盾解决 / 建号人设), 各写各的字面量。分层规则
# 一旦要调整, 漏掉任何一处都会让同一条记忆在不同路径下落到不同层。
L1_MIN_IMPORTANCE = 0.85
L2_MIN_IMPORTANCE = 0.50
STORE_MIN_IMPORTANCE = 0.10


def level_for_importance(importance: float) -> int:
    """spec §1.4: 85+ → L1, 50-84 → L2, 其余 → L3."""
    if importance >= L1_MIN_IMPORTANCE:
        return 1
    if importance >= L2_MIN_IMPORTANCE:
        return 2
    return 3


# ── Similarity thresholds ──
# 写入去重: cosine > 阈值 判为重复, 跳过写入. spec 没规定写入去重 (part2 §2 录
# 入管线只到 prefilter+extraction 两步), 这是工程兜底.
# 0.9 → 0.85: bge-m3 中文 paraphrase 实测落 0.85-0.92 区间, 0.9 漏过真复述
# (e.g. AI 把 L1 "年轻时太任性错过陪父母旅行" 复述成 L2 时 cosine 0.864).
# 详见 scripts/eval_dedup_ai.py 的实测分布.
#
# 2026-07 换 embedding (bge-m3 → qwen3-embedding:0.6b) 时**刻意保持不变**, 这
# 不是漏改. 检索侧阈值全部下调了, 因为新模型把不相关文本对压得更低; 但相似度
# 尺度的偏移不是均匀平移 —— 高端几乎不动. 拿线上 108 对真实近重复记忆实测:
#
#     真重复对的中位相似度   bge-m3 0.936  →  qwen3 0.940
#
# 近乎相同的文本在任何模型下都拿高分, 所以去重这一端不需要跟着降. 保持 0.85 时
# 新模型抓住 84% 的旧判重对、误判 2% 的"相似但不同"对; 降到 0.80 是 92%/4%.
# 取前者是因为两类错误代价不对称: 误判会把新记忆合并进旧的、**静默丢掉信息**,
# 漏判只是多存一条. 标定数据见 scripts/export_near_duplicate_pairs.py.
DEDUP_THRESHOLD: float = 0.85

# 用户语义删除: 跟 dedup 阈值对齐, 之前 dedup=0.9/删除=0.85 的差值是为了"高于
# 去重阈值避免误删", 现在 dedup 也降到 0.85, 二者保持一致.
DELETION_SIMILARITY_THRESHOLD: float = 0.85

# LLM 意图判定（删除/冲突）最低可信度
LLM_INTENT_MIN_CONFIDENCE: float = 0.8

# ── Lifecycle ──
# importance 衰减后低于此值的记忆自动归档
ARCHIVE_IMPORTANCE_THRESHOLD: float = 0.1

# ── Cache ──
# 检索 / 图 / 摘要结果缓存 TTL（秒）
RETRIEVAL_CACHE_TTL: int = 300
GRAPH_CACHE_TTL: int = 300
EMBEDDING_CACHE_TTL: int = 1800

# ── Reminder recurrence (Part 5 §4.2) ──
# 周期性 recurrence 值. once 是默认 (LLM 缺失/未知 → "once" 兜底), 不需要单独常量.
# `_reminder_matches_date` 按周期匹配; once 走精确日期比对.
RECURRENCE_PERIODIC: frozenset[str] = frozenset({"yearly", "monthly", "weekly", "daily"})
