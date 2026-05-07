"""Embedding service for memory vectors.

Uses OllamaEmbeddings via LangChain to generate embeddings,
and raw SQL for pgvector storage/retrieval.
Includes Redis caching for embeddings.

Phase 0.4: generate_embedding + store_embedding 加 retry with backoff.
Production observation: Ollama 偶尔 transient 503 / GPU 抢占, 单次失败 = 用户
记忆丢失 (无 retry → store_memory raise → handler 兜底回复但 DB 无 record).
"""

import asyncio
import logging

from app.db import db
from app.observability.events import EVT_EMBEDDING_FAIL, EVT_EMBEDDING_RETRY
from app.services.llm.models import get_embedding_model
from app.services.runtime.cache import cache_embedding, cache_set_embedding

logger = logging.getLogger(__name__)

# Phase 0.4: 重试配置. 总耗时上限: 0.3 + 0.7 = 1s (3 次尝试). 实际生产 Ollama
# transient hiccup 通常 < 500ms 恢复, 1s 窗口足够; 长时间 outage 不该重试更多
# (用户感知延迟 + 资源浪费). 调用方对 raise 已有 fallback (handler try/except).
_EMBEDDING_RETRY_DELAYS = (0.3, 0.7)  # backoff 序列, 长度 = max_retries
_EMBEDDING_MAX_ATTEMPTS = len(_EMBEDDING_RETRY_DELAYS) + 1  # 3 次


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text (with cache + retry).

    Retry: Ollama transient 失败 (503, timeout, GPU contention) 重试最多 3 次.
    每次失败 emit EVT_EMBEDDING_RETRY 事件; 全部失败 emit EVT_EMBEDDING_FAIL.
    """
    cached = await cache_embedding(text)
    if cached:
        return cached

    model = get_embedding_model()

    last_exc: Exception | None = None
    for attempt in range(_EMBEDDING_MAX_ATTEMPTS):
        try:
            embedding = await model.aembed_query(text)
            # 成功路径: 缓存后返回
            try:
                await cache_set_embedding(text, embedding)
            except Exception:
                pass
            return embedding
        except Exception as e:
            last_exc = e
            if attempt < _EMBEDDING_MAX_ATTEMPTS - 1:
                # 还有重试机会
                delay = _EMBEDDING_RETRY_DELAYS[attempt]
                logger.warning(
                    f"[EMBEDDING-RETRY] attempt {attempt + 1}/{_EMBEDDING_MAX_ATTEMPTS} "
                    f"failed: {type(e).__name__}: {e}; retrying in {delay}s",
                    extra={
                        "event": EVT_EMBEDDING_RETRY,
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "delay_sec": delay,
                        "text_len": len(text),
                    },
                )
                await asyncio.sleep(delay)
                continue
            # 最后一次失败 — emit fail 事件后 raise
            logger.error(
                f"[EMBEDDING-FAIL] all {_EMBEDDING_MAX_ATTEMPTS} attempts failed: "
                f"{type(e).__name__}: {e}; user memory will be lost",
                extra={
                    "event": EVT_EMBEDDING_FAIL,
                    "attempts": _EMBEDDING_MAX_ATTEMPTS,
                    "error_type": type(e).__name__,
                    "text_len": len(text),
                    "text_preview": text[:40],
                },
            )
            raise
    # 不可达 (循环必 return 或 raise) — type checker 安抚
    raise last_exc or RuntimeError("generate_embedding exhausted without raise")


async def store_embedding(memory_id: str, embedding: list[float]) -> None:
    """Store an embedding in the memory_embeddings table (with retry).

    INSERT ON CONFLICT 是幂等的, 重试安全. 仅 transient PG 错误 (connection
    drop / pool starvation) 才重试; permanent 错误 (vector dim mismatch) 第一次
    就 raise 不会浪费重试. 每次 retry emit EVT_EMBEDDING_RETRY (stage='store').
    """
    from app.services.memory.retrieval.vector_search import format_vector
    vec_str = format_vector(embedding)

    last_exc: Exception | None = None
    for attempt in range(_EMBEDDING_MAX_ATTEMPTS):
        try:
            await db.execute_raw(
                """
                INSERT INTO memory_embeddings (memory_id, embedding)
                VALUES ($1, $2::extensions.vector)
                ON CONFLICT (memory_id) DO UPDATE SET embedding = $2::extensions.vector
                """,
                memory_id,
                vec_str,
            )
            return
        except Exception as e:
            last_exc = e
            if attempt < _EMBEDDING_MAX_ATTEMPTS - 1:
                delay = _EMBEDDING_RETRY_DELAYS[attempt]
                logger.warning(
                    f"[EMBEDDING-STORE-RETRY] attempt {attempt + 1}/"
                    f"{_EMBEDDING_MAX_ATTEMPTS} failed: {type(e).__name__}; "
                    f"retrying in {delay}s",
                    extra={
                        "event": EVT_EMBEDDING_RETRY,
                        "stage": "store",
                        "attempt": attempt + 1,
                        "memory_id": memory_id[:8],
                        "error_type": type(e).__name__,
                    },
                )
                await asyncio.sleep(delay)
                continue
            raise
    raise last_exc or RuntimeError("store_embedding exhausted without raise")


async def get_embedding(memory_id: str) -> list[float] | None:
    """Retrieve the embedding for a memory."""
    rows = await db.query_raw(
        """
        SELECT embedding::text FROM memory_embeddings WHERE memory_id = $1
        """,
        memory_id,
    )
    if not rows:
        return None
    # Parse the vector string [x,y,z,...] back to list
    vec_str = rows[0]["embedding"]
    if vec_str:
        return [float(v) for v in vec_str.strip("[]").split(",")]
    return None
