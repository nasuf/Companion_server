"""Graph/legacy contract and side-effect regression tests with synthetic IO."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.config import settings
from app.services.chat import orchestrator as chat
from app.services.chat.data_fetch_phase import FetchedContext
from app.services.chat.intent_dispatcher import IntentResult, IntentType


def configure_chat(monkeypatch):
    background = []
    now = datetime(2026, 10, 3, tzinfo=timezone.utc)
    history = [
        SimpleNamespace(
            id="u-old", role="user", content="hello", metadata={}, createdAt=now
        ),
        SimpleNamespace(
            id="a-old", role="assistant", content="hello", metadata={}, createdAt=now
        ),
    ]
    database = SimpleNamespace(
        message=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="u-new")),
            find_many=AsyncMock(side_effect=lambda **_: history.copy()),
        ),
        conversation=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(workspaceId="w-1", userId="u-1")
            ),
            update=AsyncMock(),
        ),
    )
    monkeypatch.setattr(chat, "db", database)
    monkeypatch.setattr(chat, "_save_replies", AsyncMock(return_value="a-new"))
    monkeypatch.setattr(chat, "finish_assistant_turn", AsyncMock())
    monkeypatch.setattr(chat, "_background_post_process", AsyncMock())

    def fire(coro):
        background.append(coro.cr_code.co_name if hasattr(coro, "cr_code") else "mock")
        coro.close()

    monkeypatch.setattr(chat, "_fire_background", fire)
    from app.services.chat import multi_intent

    monkeypatch.setattr(multi_intent, "_fire_background", fire)
    monkeypatch.setattr(
        multi_intent, "finish_assistant_turn", chat.finish_assistant_turn
    )
    from app.services import runtime_config

    monkeypatch.setattr(
        runtime_config, "bind_agent_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(runtime_config, "reset_current_agent", MagicMock())
    decision = SimpleNamespace(
        crisis_force_intent=False,
        crisis_followup_active=False,
        crisis_care_turn=False,
        recent_crisis_context="",
        crisis_followup_check_mode=None,
        cached_patience=100,
        skip_boundary=True,
        status="none",
        reason="none",
        semantic_checked=False,
        semantic_detected=False,
        boundary_attack_present=False,
        intent_metadata={},
    )
    monkeypatch.setattr(chat, "run_crisis_guard", AsyncMock(return_value=decision))

    async def noop(*_, **__):
        if False:
            yield

    for name in (
        "resolve_recent_undo",
        "resolve_pending_contradiction",
        "resolve_pending_deletion",
        "resolve_retrieval_feedback_correction",
    ):
        monkeypatch.setattr(chat, name, noop)
    monkeypatch.setattr(chat, "build_filler_emoji_reply", lambda *_, **__: None)
    monkeypatch.setattr(
        chat, "_fetch_intent_context", AsyncMock(return_value="synthetic context")
    )
    monkeypatch.setattr(
        chat,
        "detect_intent_unified",
        AsyncMock(return_value=IntentResult(intent=IntentType.NONE, confidence=0.9)),
    )
    monkeypatch.setattr(
        chat,
        "fetch_parallel_context",
        AsyncMock(return_value=FetchedContext(memory_relevance="weak")),
    )
    monkeypatch.setattr(
        chat, "get_or_build_session_recap", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(chat, "build_topic_continuation", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "maybe_awaken_l3", AsyncMock(return_value=([], "")))
    monkeypatch.setattr(chat, "push_topic", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "get_mbti", lambda _: None)
    monkeypatch.setattr(
        chat, "get_patience_prompt_instruction", AsyncMock(return_value="")
    )
    monkeypatch.setattr(chat, "load_last_reply_count", AsyncMock(return_value=1))
    monkeypatch.setattr(chat, "save_last_reply_count", AsyncMock())
    monkeypatch.setattr(
        chat,
        "_generate_reply",
        AsyncMock(
            return_value=(
                ["ordinary reply"],
                "ordinary reply",
                False,
                {"emotion": "中性"},
            )
        ),
    )
    monkeypatch.setattr(
        chat, "_ai_reply_emotion", AsyncMock(return_value={"emotion": "中性"})
    )

    async def emit(replies, *, emitted_replies, reply_index_offset=0, **_):
        for index, text in enumerate(replies, reply_index_offset):
            row = {"text": text, "index": index}
            emitted_replies.append(row)
            yield {"event": "reply", "data": json.dumps(row)}

    monkeypatch.setattr(chat, "_emit_replies", emit)
    from app.services.speech_output import policy

    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=False))
    from app.services.interaction import chat_management

    monkeypatch.setattr(chat_management, "reply_delay_enabled", lambda: False)
    from app.services.achievements import service

    monkeypatch.setattr(service, "handle_assistant_turn_event", AsyncMock())
    tracer = MagicMock(trace_id="trace-1", safe_trace_id="trace-1", is_active=True)
    tracer.enter.return_value = tracer
    monkeypatch.setattr(chat, "create_tracer", lambda *_, **__: tracer)
    monkeypatch.setattr(settings, "chat_executor", "langgraph")
    monkeypatch.setattr(settings, "chat_graph_all_conversations", False)
    monkeypatch.setattr(settings, "chat_graph_conversation_allowlist", "c-1")
    return SimpleNamespace(
        db=database,
        tracer=tracer,
        decision=decision,
        background=background,
        achievement=service.handle_assistant_turn_event,
        agent=SimpleNamespace(id="a-1", name="Synthetic"),
    )
