"""Phase 0.4: embedding 链路 retry 行为.

生产场景:
- Ollama 偶尔 transient 503 (GPU 抢占, 网络抖动) → 单次失败 = 用户记忆丢失
- 修复: generate_embedding + store_embedding 各有 3 次 retry with backoff (0.3s, 0.7s)
- 全部失败 → emit EVT_EMBEDDING_FAIL + raise (caller 兜底)
- memory row 已写但 embedding 失败 + rollback 也失败 → emit EVT_MEMORY_ORPHAN
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_generate_embedding_succeeds_after_retry():
    """第 1 次失败 + 第 2 次成功 → 返回正确 embedding, 不 raise."""
    from app.services.memory.storage.embedding import generate_embedding

    call_count = {"n": 0}
    expected_vec = [0.1] * 1024

    async def _flaky_embed(text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Ollama 503 transient")
        return expected_vec

    fake_model = MagicMock(aembed_query=AsyncMock(side_effect=_flaky_embed))

    with (
        patch("app.services.memory.storage.embedding.cache_embedding",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.memory.storage.embedding.cache_set_embedding",
              new_callable=AsyncMock),
        patch("app.services.memory.storage.embedding.get_embedding_model",
              return_value=fake_model),
    ):
        result = await generate_embedding("用户喜欢咖啡")

    assert result == expected_vec
    assert call_count["n"] == 2  # 1 失败 + 1 成功


@pytest.mark.asyncio
async def test_generate_embedding_raises_after_max_attempts():
    """3 次都失败 → raise + emit EVT_EMBEDDING_FAIL."""
    from app.services.memory.storage.embedding import (
        generate_embedding, _EMBEDDING_MAX_ATTEMPTS,
    )

    call_count = {"n": 0}

    async def _always_fail(text):
        call_count["n"] += 1
        raise RuntimeError(f"Ollama down attempt {call_count['n']}")

    fake_model = MagicMock(aembed_query=AsyncMock(side_effect=_always_fail))

    with (
        patch("app.services.memory.storage.embedding.cache_embedding",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.memory.storage.embedding.get_embedding_model",
              return_value=fake_model),
    ):
        with pytest.raises(RuntimeError, match="Ollama down"):
            await generate_embedding("用户喜欢咖啡")

    assert call_count["n"] == _EMBEDDING_MAX_ATTEMPTS  # 全部尝试


@pytest.mark.asyncio
async def test_generate_embedding_cache_hit_skips_retry():
    """cache hit → 不调 LLM, 不进入 retry loop."""
    from app.services.memory.storage.embedding import generate_embedding

    cached_vec = [0.5] * 1024
    fake_model = MagicMock(aembed_query=AsyncMock())

    with (
        patch("app.services.memory.storage.embedding.cache_embedding",
              new_callable=AsyncMock, return_value=cached_vec),
        patch("app.services.memory.storage.embedding.get_embedding_model",
              return_value=fake_model),
    ):
        result = await generate_embedding("hi")

    assert result == cached_vec
    fake_model.aembed_query.assert_not_called()


@pytest.mark.asyncio
async def test_store_embedding_retries_on_transient_pg_error():
    """store_embedding INSERT ON CONFLICT 是幂等的, transient PG 错误重试安全."""
    from app.services.memory.storage.embedding import store_embedding

    call_count = {"n": 0}

    async def _flaky_db(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("PG connection drop")
        return None  # 第 3 次成功

    from app.services.memory.storage import embedding as emb_mod
    fake_db = MagicMock(execute_raw=AsyncMock(side_effect=_flaky_db))
    with patch.object(emb_mod, "db", fake_db):
        await store_embedding("mem-123", [0.1] * 1024)

    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_store_embedding_raises_after_all_retries():
    """store_embedding 3 次全失败 → raise."""
    from app.services.memory.storage.embedding import (
        store_embedding, _EMBEDDING_MAX_ATTEMPTS,
    )

    call_count = {"n": 0}

    async def _always_fail(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("PG hard down")

    from app.services.memory.storage import embedding as emb_mod
    fake_db = MagicMock(execute_raw=AsyncMock(side_effect=_always_fail))
    with patch.object(emb_mod, "db", fake_db):
        with pytest.raises(RuntimeError, match="PG hard down"):
            await store_embedding("mem-123", [0.1] * 1024)

    assert call_count["n"] == _EMBEDDING_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_backoff_delays_compound():
    """重试 backoff 实际 sleep (避免极端短时间内 hammer 故障 service)."""
    from app.services.memory.storage import embedding as emb_mod
    import time

    call_count = {"n": 0}

    async def _fail_then_succeed(text):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("transient")
        return [0.1] * 1024

    fake_model = MagicMock(aembed_query=AsyncMock(side_effect=_fail_then_succeed))

    with (
        patch("app.services.memory.storage.embedding.cache_embedding",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.memory.storage.embedding.cache_set_embedding",
              new_callable=AsyncMock),
        patch("app.services.memory.storage.embedding.get_embedding_model",
              return_value=fake_model),
    ):
        start = asyncio.get_event_loop().time()
        await emb_mod.generate_embedding("test")
        elapsed = asyncio.get_event_loop().time() - start

    # 0.3 + 0.7 = 1.0s 最少 (2 次重试 = 2 次 backoff)
    assert elapsed >= 0.9, f"expected ≥ 0.9s elapsed (backoff total), got {elapsed:.2f}s"
    # 但也不能太久 (单次 LLM 调用本身瞬时, 大头是 backoff)
    assert elapsed < 2.0, f"unexpected delay {elapsed:.2f}s, retry overhead too high"
