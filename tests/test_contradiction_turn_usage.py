"""矛盾追问轮: 检测带上下文, 追问带人设与画像, 检索到的记忆不记作"被用上".

生产 2026-10-21: 追问整条替换主回复, 主 prompt 根本没构建, 这一轮检索选中的 20
条记忆却都拿了 contribution 回报 —— 连带把"用户询问是否和对方在一个地方"推进 L1。
两个执行器必须一致。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

NOW = datetime(2026, 10, 21, 9, 22)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_contradiction_turn_uses_context_and_credits_candidates_only(monkeypatch, executor):
    from app.config import settings
    from app.services.chat import graph_phases, orchestrator as chat
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair

    io = configure_pair(monkeypatch)
    monkeypatch.setattr(settings, "chat_executor", executor)
    io.db.message.find_many.side_effect = lambda **_: [SimpleNamespace(
        id="prev-ai", role="assistant", content="你也在北京吗？",
        createdAt=io.now - timedelta(minutes=1), metadata={},
    )]
    chat.fetch_parallel_context.return_value = FetchedContext(
        memory_relevance="strong", candidate_ids=["cand-1"],
    )
    chat.fetch_parallel_context.return_value.classified_memories = [SimpleNamespace(id="sel-1")]
    chat.detect_l1_contradiction.return_value = {
        "new_info": "我搬到上海了", "old_content": "用户住在北京", "conflict_description": "现居地变化",
    }
    inquiry = AsyncMock(return_value="搬去上海啦？那边住得惯吗")
    monkeypatch.setattr(chat, "generate_contradiction_inquiry", inquiry)
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["搬去上海啦？那边住得惯吗"], "", False, None)))
    submitted: list[dict] = []

    def submit(**kwargs):
        submitted.append(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(chat, "submit_memory_usage", submit)
    monkeypatch.setattr(graph_phases, "submit_memory_usage", submit)

    events = [event async for event in chat.stream_chat_response(
        "c-1", "我搬到上海了", io.agent, "u-1", forced_intent=IntentType.NONE,
        reply_context={"received_at": NOW.isoformat()},
    )]

    assert sum(e["event"] == "done" for e in events) == 1
    detect_kwargs = chat.detect_l1_contradiction.await_args.kwargs
    assert "你也在北京吗？" in detect_kwargs["recent_context"]
    inquiry_kwargs = inquiry.await_args.kwargs
    assert "你也在北京吗？" in inquiry_kwargs["recent_context"]
    assert inquiry_kwargs["personality_brief"] and inquiry_kwargs["personality_brief"] != io.agent.name
    usage = submitted[-1]
    assert usage["selected_ids"] == []
    assert set(usage["candidate_ids"]) == {"sel-1", "cand-1"}
