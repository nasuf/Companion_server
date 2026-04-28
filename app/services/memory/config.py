"""Memory system shared constants.

Centralized here so thresholds/timeouts are not silently duplicated
across files. Callers should import from this module rather than
redefining literals.
"""

# ── Similarity thresholds ──
# 写入去重: cosine > 阈值 判为重复, 跳过写入. spec 没规定写入去重 (part2 §2 录
# 入管线只到 prefilter+extraction 两步), 这是工程兜底.
# 0.9 → 0.85: bge-m3 中文 paraphrase 实测落 0.85-0.92 区间, 0.9 漏过真复述
# (e.g. AI 把 L1 "年轻时太任性错过陪父母旅行" 复述成 L2 时 cosine 0.864).
# 详见 scripts/eval_dedup_ai.py 的实测分布.
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
