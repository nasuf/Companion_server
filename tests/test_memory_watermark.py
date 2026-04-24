"""Memory extraction watermark tests.

验证 (conversation_id, side) 水位线切分:
- 无水位线 → 全部当新消息
- 有水位线 → 只抽 createdAt > watermark 的消息
- 该 side 无新消息 → 跳过 LLM 调用
- 抽取成功后水位线推进
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest


UTC = timezone.utc


def _msg(role: str, content: str, ts: datetime) -> dict:
    return {"role": role, "content": content, "createdAt": ts.isoformat()}


@pytest.mark.asyncio
async def test_watermark_absent_extracts_all():
    """首次运行 (无水位线) → 全部当新消息, process_memory_pipeline 收到完整 conversation."""
    from app.services.chat.post_process import _bg_memory_pipeline

    t0 = datetime.now(UTC)
    msgs = [
        _msg("user", "hi", t0),
        _msg("assistant", "hey", t0 + timedelta(seconds=1)),
    ]

    with patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=None)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()) as mock_pipeline:
        await _bg_memory_pipeline("u1", msgs, conversation_id="c1")

    assert mock_pipeline.await_count == 2  # user + ai
    for call in mock_pipeline.await_args_list:
        kwargs = call.kwargs
        assert kwargs["context_conversation"] == ""  # wm 无 → context 空
        assert kwargs["new_conversation"] != ""
    # 推进水位线: user 一次 + ai 一次
    assert mock_set.await_count == 2


@pytest.mark.asyncio
async def test_watermark_splits_context_and_new():
    """有水位线 → 老消息进 context, 新消息进 new; 水位线更新到最新 createdAt."""
    from app.services.chat.post_process import _bg_memory_pipeline

    base = datetime.now(UTC)
    wm_ts = base
    msgs = [
        _msg("user", "old_u", base - timedelta(seconds=2)),
        _msg("assistant", "old_a", base - timedelta(seconds=1)),
        _msg("user", "new_u", base + timedelta(seconds=1)),
        _msg("assistant", "new_a", base + timedelta(seconds=2)),
    ]

    with patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm_ts)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()) as mock_pipeline:
        await _bg_memory_pipeline("u1", msgs, conversation_id="c1")

    # user + ai 两条 pipeline 都应被调用
    assert mock_pipeline.await_count == 2
    # 每条都要有 context (pre-wm 消息) + new (post-wm 消息)
    for call in mock_pipeline.await_args_list:
        kwargs = call.kwargs
        assert "old_u" in kwargs["context_conversation"]
        assert "old_a" in kwargs["context_conversation"]
        assert "new_u" in kwargs["new_conversation"]
        assert "new_a" in kwargs["new_conversation"]

    # 水位线推进到 max(本 side 新消息 createdAt)
    assert mock_set.await_count == 2
    side_ts = {call.args[1]: call.args[2] for call in mock_set.await_args_list}
    assert side_ts["user"] == base + timedelta(seconds=1)
    assert side_ts["ai"] == base + timedelta(seconds=2)


@pytest.mark.asyncio
async def test_watermark_skips_when_no_new_for_side():
    """某 side 无新消息 → 该 side 不调 LLM, 不更新水位线."""
    from app.services.chat.post_process import _bg_memory_pipeline

    base = datetime.now(UTC)
    # user 有新消息, ai 没有 (水位线已覆盖所有 assistant 消息)
    wm_ts_user = base - timedelta(seconds=10)
    wm_ts_ai = base  # 已覆盖下面所有 assistant 消息
    msgs = [
        _msg("assistant", "old_a", base - timedelta(seconds=5)),
        _msg("user", "new_u", base - timedelta(seconds=1)),
    ]

    async def _get_wm(conv_id, side):
        return wm_ts_user if side == "user" else wm_ts_ai

    with patch("app.services.chat.post_process.get_watermark", side_effect=_get_wm), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()) as mock_pipeline:
        await _bg_memory_pipeline("u1", msgs, conversation_id="c1")

    # 只有 user side 被调, ai 跳过
    assert mock_pipeline.await_count == 1
    assert mock_pipeline.await_args.kwargs["side"] == "user"
    # 只推进 user 水位线
    assert mock_set.await_count == 1
    assert mock_set.await_args.args[1] == "user"


@pytest.mark.asyncio
async def test_no_conversation_id_falls_back_to_legacy_behavior():
    """无 conversation_id (proactive sender 等场景) → 不走水位线, 全部当新."""
    from app.services.chat.post_process import _bg_memory_pipeline

    t0 = datetime.now(UTC)
    msgs = [_msg("assistant", "hi", t0)]

    with patch("app.services.chat.post_process.get_watermark", AsyncMock()) as mock_get, \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()) as mock_pipeline:
        await _bg_memory_pipeline("u1", msgs, conversation_id=None)

    mock_get.assert_not_called()
    mock_set.assert_not_called()
    mock_pipeline.assert_awaited_once()
    assert mock_pipeline.await_args.kwargs["context_conversation"] == ""


@pytest.mark.asyncio
async def test_watermark_advances_past_ts_in_mixed_new_msgs():
    """混合 ts 场景: 老 AI 回复有 ts + 新 AI 回复无 ts (主流程 full_response 刚生成).
    水位线必须推进到 >= now(), 否则下轮新回复持久化后 createdAt > wm 会重抽."""
    from app.services.chat.post_process import _bg_memory_pipeline

    wm_old = datetime.now(UTC) - timedelta(hours=1)
    old_ai_ts = datetime.now(UTC) - timedelta(minutes=5)
    # ai 侧有两条: 一条来自 DB (有 ts), 一条是新 reply (无 ts)
    msgs = [
        {"role": "user", "content": "hello", "createdAt": (datetime.now(UTC) - timedelta(minutes=4)).isoformat()},
        {"role": "assistant", "content": "old reply", "createdAt": old_ai_ts.isoformat()},
        {"role": "assistant", "content": "new reply"},  # no createdAt
    ]

    with patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm_old)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()):
        await _bg_memory_pipeline("u1", msgs, conversation_id="c1")

    # ai 水位线必须推进到 >= 现在 (覆盖无 ts 的新 reply), 不能停在 old_ai_ts
    ai_call = next(c for c in mock_set.await_args_list if c.args[1] == "ai")
    new_wm_ai = ai_call.args[2]
    assert new_wm_ai > old_ai_ts, f"ai watermark {new_wm_ai} 应 > old_ai_ts {old_ai_ts}"


@pytest.mark.asyncio
async def test_watermark_advances_with_now_when_new_msgs_lack_timestamp():
    """无 createdAt 的新消息 (新 AI reply / boundary 手工构造) 抽取后水位线仍推进到 now(),
    否则下一轮持久化后 createdAt > 老 wm 会被重抽."""
    from app.services.chat.post_process import _bg_memory_pipeline

    wm_old = datetime.now(UTC) - timedelta(hours=1)
    # 所有消息无 createdAt (模拟 boundary 路径 或 刚生成未持久化)
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]

    with patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm_old)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline", AsyncMock()):
        await _bg_memory_pipeline("u1", msgs, conversation_id="c1")

    # 水位线必须被推进（fallback 到 now()），覆盖 user + ai 两侧
    assert mock_set.await_count == 2
    # 每次推进的 ts 都 > 老 wm（防回退）
    for call in mock_set.await_args_list:
        new_wm = call.args[2]
        assert new_wm > wm_old


@pytest.mark.asyncio
async def test_watermark_redis_helpers():
    """watermark.py 读写 Redis 正确序列化 ISO 时间."""
    from app.services.memory.recording.watermark import get_watermark, set_watermark

    ts = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=ts.isoformat().encode())
    fake_redis.set = AsyncMock()

    with patch("app.services.memory.recording.watermark.get_redis", AsyncMock(return_value=fake_redis)):
        result = await get_watermark("c1", "user")
        assert result == ts

        await set_watermark("c1", "user", ts)
        fake_redis.set.assert_awaited_once()
        call_args = fake_redis.set.await_args.args
        assert "mem:extract:wm:c1:user" in call_args[0]
        assert call_args[1] == ts.isoformat()


@pytest.mark.asyncio
async def test_watermark_redis_down_degrades_gracefully():
    """Redis 异常 → get 返 None (退化成"全部抽"), set 静默失败."""
    from app.services.memory.recording.watermark import get_watermark, set_watermark

    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(side_effect=Exception("redis down"))
    fake_redis.set = AsyncMock(side_effect=Exception("redis down"))

    with patch("app.services.memory.recording.watermark.get_redis", AsyncMock(return_value=fake_redis)):
        assert await get_watermark("c1", "user") is None
        # set 不应抛异常
        await set_watermark("c1", "user", datetime.now(UTC))
