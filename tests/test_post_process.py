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
async def test_save_replies_first_carries_trace_pending():
    """trace_id 给定时第一条 reply 的 metadata 带 trace_id+trace_pending。"""
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
    assert md0_dict["trace_pending"] is True

    md1 = created_calls[1]["metadata"]
    md1_dict = md1.data if hasattr(md1, "data") else md1
    assert "trace_id" not in md1_dict
    assert "trace_pending" not in md1_dict


@pytest.mark.asyncio
async def test_run_post_process_fires_all_tasks_for_agent():
    """有 agent_id 时，5 个后台任务都被并行 await。"""
    from app.services.chat import post_process

    with patch.object(post_process, "_bg_emotion", AsyncMock()) as e, \
         patch.object(post_process, "_bg_summarizer", AsyncMock()) as s, \
         patch.object(post_process, "_bg_memory_pipeline", AsyncMock()) as m, \
         patch.object(post_process, "_bg_trait_adjustment", AsyncMock()) as t, \
         patch.object(post_process, "_bg_positive_recovery", AsyncMock()) as pr:
        await post_process.run_post_process(
            user_id="u1", agent_id="a1", conversation_id="c1",
            user_message="hi", user_message_id="msg-x",
            full_response="hello",
            messages_dicts=[{"role": "user", "content": "hi"}],
            memory_strings=[],
        )

    e.assert_awaited_once()
    s.assert_awaited_once()
    m.assert_awaited_once()
    t.assert_awaited_once()
    pr.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_post_process_skips_agent_only_tasks_when_no_agent():
    """无 agent_id 时跳过 trait_adjustment + positive_recovery。"""
    from app.services.chat import post_process

    with patch.object(post_process, "_bg_emotion", AsyncMock()) as e, \
         patch.object(post_process, "_bg_summarizer", AsyncMock()) as s, \
         patch.object(post_process, "_bg_memory_pipeline", AsyncMock()) as m, \
         patch.object(post_process, "_bg_trait_adjustment", AsyncMock()) as t, \
         patch.object(post_process, "_bg_positive_recovery", AsyncMock()) as pr:
        await post_process.run_post_process(
            user_id="u1", agent_id=None, conversation_id="c1",
            user_message="hi", user_message_id=None,
            full_response="hello",
            messages_dicts=[{"role": "user", "content": "hi"}],
            memory_strings=None,
        )

    # 3 个公共任务仍跑
    e.assert_awaited_once()
    s.assert_awaited_once()
    m.assert_awaited_once()
    # 2 个 agent-only 跳过
    t.assert_not_called()
    pr.assert_not_called()
