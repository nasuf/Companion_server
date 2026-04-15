"""Memory system shared constants.

Centralized here so thresholds/timeouts are not silently duplicated
across files. Callers should import from this module rather than
redefining literals.
"""

# ── Similarity thresholds ──
# 写入去重：cosine > 0.9 判为重复，跳过写入
DEDUP_THRESHOLD: float = 0.9

# 用户语义删除：cosine ≥ 0.85 的记忆才会被"按描述删"。高于去重阈值以避免误删。
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
