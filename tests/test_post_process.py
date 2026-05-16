"""post_process 单测：save_replies 持久化 + run_post_process 后台任务编排。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_save_replies_persists_dict_metadata():
    """dict reply 的非 text/index 字段全部进 metadata。"""
    from app.services.chat import post_process

    created_calls: list[dict] = []

    async def _fake_create(data):
        created_calls.append(data)
        return MagicMock(id=f"msg-{len(created_calls)}")

    fake_db = MagicMock()
    fake_db.message.create = AsyncMock(side_effect=_fake_create)
    with patch.object(post_process, "db", fake_db):
        first_id = await post_process.save_replies(
            "conv1",
            [
                {"text": "hi", "boundary": True, "zone": "blocked", "sticker_url": None},
                {"text": "okay", "extra": 42},
            ],
        )

    assert first_id == "msg-1"
    md0 = created_calls[0]["metadata"]  # Json wrapper, just trust the dict roundtrip
    md0_dict = md0.data if hasattr(md0, "data") else md0
    assert md0_dict["reply_index"] == 0
    assert md0_dict["boundary"] is True
    assert md0_dict["zone"] == "blocked"
    # None 值被过滤
    assert "sticker_url" not in md0_dict

    md1_dict = created_calls[1]["metadata"]
    md1 = md1_dict.data if hasattr(md1_dict, "data") else md1_dict
    assert md1["reply_index"] == 1
    assert md1["extra"] == 42


@pytest.mark.asyncio
async def test_save_replies_first_carries_trace_id():
    """trace_id 给定时第一条 reply 的 metadata 只带 trace_id (懒触发模式,
    不再写 trace_pending; share + mirror 由用户点 Trace 按钮时通过 retry endpoint 调)."""
    from app.services.chat import post_process

    created_calls: list[dict] = []

    async def _fake_create(data):
        created_calls.append(data)
        return MagicMock(id=f"msg-{len(created_calls)}")

    fake_db = MagicMock()
    fake_db.message.create = AsyncMock(side_effect=_fake_create)
    with patch.object(post_process, "db", fake_db):
        await post_process.save_replies(
            "conv1",
            ["first", "second"],
            trace_id="trace-xyz",
        )

    md0 = created_calls[0]["metadata"]
    md0_dict = md0.data if hasattr(md0, "data") else md0
    assert md0_dict["trace_id"] == "trace-xyz"
    assert "trace_pending" not in md0_dict
    assert "trace_failed" not in md0_dict

    md1 = created_calls[1]["metadata"]
    md1_dict = md1.data if hasattr(md1, "data") else md1
    assert "trace_id" not in md1_dict
    assert "trace_pending" not in md1_dict


@pytest.mark.asyncio
async def test_run_post_process_fires_all_tasks_for_agent():
    """有 agent_id 时，后台任务都被并行 await。"""
    from app.services.chat import post_process

    with patch.object(post_process, "_bg_user_emotion", AsyncMock()) as e, \
         patch.object(post_process, "_bg_memory_pipeline", AsyncMock()) as m, \
         patch.object(post_process, "_bg_trait_adjustment", AsyncMock()) as t, \
         patch.object(post_process, "_bg_positive_recovery", AsyncMock()) as pr:
        await post_process.run_post_process(
            user_id="u1", agent_id="a1", conversation_id="c1",
            user_message="hi", user_message_id="msg-x",
            full_response="hello",
            messages_dicts=[{"role": "user", "content": "hi"}],
        )

    e.assert_awaited_once()
    m.assert_awaited_once()
    t.assert_awaited_once()
    pr.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_post_process_skips_agent_only_tasks_when_no_agent():
    """无 agent_id 时跳过 trait_adjustment + positive_recovery。"""
    from app.services.chat import post_process

    with patch.object(post_process, "_bg_user_emotion", AsyncMock()) as e, \
         patch.object(post_process, "_bg_memory_pipeline", AsyncMock()) as m, \
         patch.object(post_process, "_bg_trait_adjustment", AsyncMock()) as t, \
         patch.object(post_process, "_bg_positive_recovery", AsyncMock()) as pr:
        await post_process.run_post_process(
            user_id="u1", agent_id=None, conversation_id="c1",
            user_message="hi", user_message_id=None,
            full_response="hello",
            messages_dicts=[{"role": "user", "content": "hi"}],
        )

    # 公共任务仍跑
    e.assert_awaited_once()
    m.assert_awaited_once()
    # agent-only 跳过
    t.assert_not_called()
    pr.assert_not_called()


# --- _bg_positive_recovery: spec §2.5 LLM 语义判定门 ---


@pytest.mark.asyncio
async def test_bg_positive_recovery_skipped_for_neutral_message():
    """中性应答 (嗯/哦) → positive_interaction_check 返 False → 不调 +20."""
    from app.services.chat import post_process

    with patch.object(post_process, "get_patience", AsyncMock(return_value=80)), patch.object(
        post_process, "positive_interaction_check", AsyncMock(return_value=False),
    ), patch.object(post_process, "check_positive_recovery", AsyncMock()) as cpr:
        await post_process._bg_positive_recovery("a1", "u1", "嗯")

    cpr.assert_not_called()


@pytest.mark.asyncio
async def test_bg_positive_recovery_fires_for_positive_message():
    """感谢/善意 + patience 在恢复区间 → LLM 返 True → +20."""
    from app.services.chat import post_process

    with patch.object(post_process, "get_patience", AsyncMock(return_value=60)), patch.object(
        post_process, "positive_interaction_check", AsyncMock(return_value=True),
    ), patch.object(post_process, "check_positive_recovery", AsyncMock()) as cpr:
        await post_process._bg_positive_recovery("a1", "u1", "谢谢你")

    cpr.assert_awaited_once_with("a1", "u1")


@pytest.mark.asyncio
async def test_bg_positive_recovery_skipped_at_patience_cap():
    """患者 patience=100 时 +20 必然 no-op, 跳 LLM 调用省 ~200ms qwen-flash."""
    from app.services.chat import post_process

    pic = AsyncMock(return_value=True)
    with patch.object(post_process, "get_patience", AsyncMock(return_value=100)), patch.object(
        post_process, "positive_interaction_check", pic,
    ), patch.object(post_process, "check_positive_recovery", AsyncMock()) as cpr:
        await post_process._bg_positive_recovery("a1", "u1", "谢谢你")

    pic.assert_not_called()  # LLM 不该被调
    cpr.assert_not_called()


@pytest.mark.asyncio
async def test_bg_positive_recovery_skipped_when_blocked():
    """patience ≤ 0 时也跳过 LLM 与 +20: 拉黑只能靠真诚道歉解封, 不走正向恢复."""
    from app.services.chat import post_process

    pic = AsyncMock(return_value=True)
    with patch.object(post_process, "get_patience", AsyncMock(return_value=0)), patch.object(
        post_process, "positive_interaction_check", pic,
    ), patch.object(post_process, "check_positive_recovery", AsyncMock()) as cpr:
        await post_process._bg_positive_recovery("a1", "u1", "谢谢你")

    pic.assert_not_called()
    cpr.assert_not_called()


@pytest.mark.asyncio
async def test_bg_positive_recovery_swallows_exception():
    """positive_interaction_check 异常时不抛, 走 fallback (不发放恢复)."""
    from app.services.chat import post_process

    with patch.object(post_process, "get_patience", AsyncMock(return_value=60)), patch.object(
        post_process, "positive_interaction_check",
        AsyncMock(side_effect=RuntimeError("LLM down")),
    ), patch.object(post_process, "check_positive_recovery", AsyncMock()) as cpr:
        await post_process._bg_positive_recovery("a1", "u1", "谢谢你")

    cpr.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# Per-conversation memory pipeline lock
# ─────────────────────────────────────────────────────────────────
# Regression: 同一 conv 连续 batch 必须 serialize 防双层 race
#   1. 水位线 race (两 batch 都读旧 wm → 重复抽 msg1 的事实)
#   2. SINGLETON storage TOCTOU (两 batch 同时 query+insert (身份/年龄) → L1 重复)
# 生产 case: 2026-05-07 用户 30s 内连发 2 条画像 dump, 28+27 入库且
# L1 生日/年龄各重复. 修法: dict[conv_id, asyncio.Lock] 串行同 conv;
# 不同 conv 完全并行; 无 conv_id 入口 (proactive sender) 不上锁.

@pytest.mark.asyncio
async def test_memory_pipeline_lock_serializes_same_conv():
    """同 conv 两并发调用必须串行执行 (后到的等先到的完成)."""
    import asyncio
    from app.services.chat import post_process

    # 清掉前面 test 残留的锁
    post_process._pipeline_locks.clear()

    started_at: list[float] = []
    finished_at: list[float] = []

    async def fake_do(*args, **kwargs):
        loop = asyncio.get_event_loop()
        started_at.append(loop.time())
        await asyncio.sleep(0.10)  # 模拟 LLM 抽取耗时
        finished_at.append(loop.time())

    with patch.object(post_process, "_do_memory_pipeline", side_effect=fake_do):
        # 同一 conv_id 并发两次
        await asyncio.gather(
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "a"}], "conv-X"),
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "b"}], "conv-X"),
        )

    # 串行: 第二次 start 必须晚于第一次 finish (留 5ms 容差)
    assert len(started_at) == 2 and len(finished_at) == 2
    assert started_at[1] >= finished_at[0] - 0.005, (
        f"second batch started before first finished: "
        f"started={started_at}, finished={finished_at}"
    )


@pytest.mark.asyncio
async def test_memory_pipeline_lock_parallel_across_convs():
    """不同 conv 必须并行 (锁按 conv 隔离, 不影响吞吐)."""
    import asyncio
    from app.services.chat import post_process

    post_process._pipeline_locks.clear()

    started_at: list[float] = []
    finished_at: list[float] = []

    async def fake_do(*args, **kwargs):
        loop = asyncio.get_event_loop()
        started_at.append(loop.time())
        await asyncio.sleep(0.10)
        finished_at.append(loop.time())

    with patch.object(post_process, "_do_memory_pipeline", side_effect=fake_do):
        # 不同 conv 并发
        await asyncio.gather(
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "a"}], "conv-A"),
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "b"}], "conv-B"),
        )

    # 并行: 第二次 start 应在第一次 finish 之前 (重叠)
    assert len(started_at) == 2 and len(finished_at) == 2
    assert started_at[1] < finished_at[0], (
        f"different convs should run in parallel but ran sequentially: "
        f"started={started_at}, finished={finished_at}"
    )


@pytest.mark.asyncio
async def test_memory_pipeline_no_lock_when_conversation_id_none():
    """conversation_id=None (proactive sender) 不上锁, 直接 fall-through."""
    import asyncio
    from app.services.chat import post_process

    post_process._pipeline_locks.clear()
    started_at: list[float] = []
    finished_at: list[float] = []

    async def fake_do(*args, **kwargs):
        loop = asyncio.get_event_loop()
        started_at.append(loop.time())
        await asyncio.sleep(0.05)
        finished_at.append(loop.time())

    with patch.object(post_process, "_do_memory_pipeline", side_effect=fake_do):
        await asyncio.gather(
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "a"}], None),
            post_process._bg_memory_pipeline("u1", [{"role": "user", "content": "b"}], None),
        )

    # 完全并行 (无锁): 第二次 start 应在第一次 finish 之前
    assert started_at[1] < finished_at[0], (
        "None conversation_id should not be locked"
    )
    # 应没在 _pipeline_locks 里写 None 这种坏 key
    assert None not in post_process._pipeline_locks


@pytest.mark.asyncio
async def test_get_pipeline_lock_returns_same_object_per_conv():
    """同 conv 多次拿锁返回同一 Lock 对象 (重要: 否则 serialize 失效)."""
    from app.services.chat import post_process

    post_process._pipeline_locks.clear()
    lock_a1 = post_process._get_pipeline_lock("conv-A")
    lock_a2 = post_process._get_pipeline_lock("conv-A")
    lock_b = post_process._get_pipeline_lock("conv-B")
    assert lock_a1 is lock_a2, "same conv must return same Lock object"
    assert lock_a1 is not lock_b, "different convs must have distinct Lock objects"


@pytest.mark.asyncio
async def test_memory_pipeline_uses_distributed_lock_in_production():
    """Production path adds a Redis distributed lock around the local conv lock."""
    from app.services.chat import post_process

    post_process._pipeline_locks.clear()
    lock_calls: list[dict] = []

    @asynccontextmanager
    async def fake_distributed_lock(*args, **kwargs):
        lock_calls.append({"args": args, "kwargs": kwargs})
        yield True

    with (
        patch.object(post_process.settings, "app_env", "production"),
        patch.object(post_process, "distributed_lock", fake_distributed_lock),
        patch.object(post_process, "_do_memory_pipeline", AsyncMock()) as do_pipeline,
    ):
        await post_process._bg_memory_pipeline(
            "u1",
            [{"role": "user", "content": "a"}],
            conversation_id="conv-prod",
        )

    do_pipeline.assert_awaited_once()
    assert lock_calls
    assert lock_calls[0]["args"] == ("memory_pipeline:conv-prod",)
    assert lock_calls[0]["kwargs"]["ttl_s"] == post_process._PIPELINE_DISTRIBUTED_LOCK_TTL
    assert lock_calls[0]["kwargs"]["fail_open"] is True
