"""post_process 单测：save_replies 持久化 + run_post_process 后台任务编排。"""

from __future__ import annotations

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
    """有 agent_id 时，4 个后台任务都被并行 await（不含可选 save_ai_emotion）。"""
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
