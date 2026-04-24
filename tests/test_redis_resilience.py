"""Redis 故障降级路径集成测试.

验证 Redis 抛 ConnectionError / TimeoutError 时各热路径 caller 的降级契约:
- push_pending / flush_pending / scan_expired → 返回 None / 空列表, 不崩
- delayed_queue flush / enqueue / scan / append → 相同语义
- 所有失败都记 warning 日志

不测试 socket_timeout 本身 (Redis 客户端库的 feature, 无需覆盖库代码).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis_async


class _BrokenPipeline:
    """Pipeline stub: 所有 staged op no-op, 但 execute() 抛 ConnectionError."""
    def rpush(self, *a, **k): pass
    def expire(self, *a, **k): pass
    def set(self, *a, **k): pass
    def zadd(self, *a, **k): pass
    def delete(self, *a, **k): pass

    async def execute(self):
        raise redis_async.ConnectionError("simulated redis down")


class _BrokenRedis:
    """Minimal Redis stub where every op raises. 用于断言 caller 降级契约."""
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or redis_async.ConnectionError("simulated redis down")

    def pipeline(self):
        return _BrokenPipeline()

    async def eval(self, *a, **k):
        raise self._exc

    async def zrangebyscore(self, *a, **k):
        raise self._exc

    async def zscore(self, *a, **k):
        raise self._exc

    async def get(self, *a, **k):
        raise self._exc

    async def set(self, *a, **k):
        raise self._exc


# ═══════════════════════════════════════════════════════════════════
# aggregation.py
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_aggregation_push_pending_redis_down_swallows(caplog):
    """push_pending 遇 Redis ConnectionError 应 warning + 返回, 不向上抛."""
    from app.services.interaction.aggregation import push_pending

    broken = _BrokenRedis()
    with patch("app.services.interaction.aggregation.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            await push_pending(
                agent_id="a1", user_id="u1",
                conversation_id="c1", text="hi", message_id="m1",
            )

    assert any("[AGG-PUSH] Redis push failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_aggregation_flush_pending_redis_down_returns_empty_tuple(caplog):
    """flush_pending 遇 Redis eval 异常返回 (None, None, None, None), 不崩."""
    from app.services.interaction.aggregation import flush_pending

    broken = _BrokenRedis()
    with patch("app.services.interaction.aggregation.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            result = await flush_pending(agent_id="a1", user_id="u1")

    assert result == (None, None, None, None)
    assert any("[AGG-FLUSH] Redis eval failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_aggregation_scan_expired_redis_down_returns_empty(caplog):
    """scan_expired 遇 Redis zrangebyscore 异常返回 [], scheduler 下 tick 再试."""
    from app.services.interaction.aggregation import scan_expired

    broken = _BrokenRedis()
    with patch("app.services.interaction.aggregation.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            result = await scan_expired()

    assert result == []
    assert any("[AGG-SCAN] Redis zrangebyscore failed" in rec.message for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════
# delayed_queue.py
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delayed_queue_flush_redis_down_returns_empty(caplog):
    """flush_due_delayed_messages Redis eval 失败返回 [], 不崩 scheduler."""
    from app.services.interaction.delayed_queue import flush_due_delayed_messages

    broken = _BrokenRedis()
    with patch("app.services.interaction.delayed_queue.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            result = await flush_due_delayed_messages("conv-1")

    assert result == []
    assert any("[DELAY-FLUSH] Redis eval failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_delayed_queue_enqueue_redis_down_logs_warning(caplog):
    """enqueue_delayed_message Redis zscore 失败 → warning + return, 不抛."""
    from app.services.interaction.delayed_queue import enqueue_delayed_message

    broken = _BrokenRedis()
    with patch("app.services.interaction.delayed_queue.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            await enqueue_delayed_message("conv-1", {"message": "hi"}, delay_seconds=1.0)

    assert any("[DELAY-ENQUEUE] Redis" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_delayed_queue_append_redis_down_returns_false(caplog):
    """enqueue_or_append_delayed Redis eval 失败返回 False."""
    from app.services.interaction.delayed_queue import enqueue_or_append_delayed

    broken = _BrokenRedis()
    with patch("app.services.interaction.delayed_queue.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            result = await enqueue_or_append_delayed("conv-1", {"message": "hi"}, delay_seconds=1.0)

    assert result is False
    assert any("[DELAY-APPEND] Redis eval failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_delayed_queue_scan_redis_down_returns_empty(caplog):
    """scan_due_delayed_messages Redis zrangebyscore 失败返回 []."""
    from app.services.interaction.delayed_queue import scan_due_delayed_messages

    broken = _BrokenRedis()
    with patch("app.services.interaction.delayed_queue.get_redis", AsyncMock(return_value=broken)):
        with caplog.at_level("WARNING"):
            result = await scan_due_delayed_messages()

    assert result == []
    assert any("[DELAY-SCAN] Redis zrangebyscore failed" in rec.message for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════
# readonly mode: require_redis + recheck
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_require_redis_raises_503_when_unhealthy():
    from fastapi import HTTPException

    from app.api.deps import require_redis

    with patch("app.api.deps.is_redis_healthy", return_value=False):
        with pytest.raises(HTTPException) as ei:
            await require_redis()
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_require_redis_passes_when_healthy():
    from app.api.deps import require_redis

    with patch("app.api.deps.is_redis_healthy", return_value=True):
        # 不抛则 pass
        await require_redis()


@pytest.mark.asyncio
async def test_recheck_redis_health_flips_state(caplog):
    """recheck 把 global _redis_healthy 切到最新 ping 结果, transition 打日志."""
    from app.redis_client import mark_redis_healthy, recheck_redis_health

    mark_redis_healthy(True)  # 起点 healthy

    with patch("app.redis_client.redis_health", AsyncMock(return_value=False)):
        with caplog.at_level("WARNING"):
            new = await recheck_redis_health()

    assert new is False
    assert any("[REDIS-HEALTH] state transition: True -> False" in rec.message for rec in caplog.records)

    # 恢复 flag 避免污染后续测试
    mark_redis_healthy(True)
