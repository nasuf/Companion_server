"""Graph/legacy contract and side-effect regression tests with synthetic IO."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.services.chat import orchestrator as chat
from app.services.chat.intent_dispatcher import IntentResult, IntentType


@pytest.fixture
def chat_io(monkeypatch):
    from tests.graph_harness_support import configure_chat

    return configure_chat(monkeypatch)


async def run(io, **kwargs):
    return [
        event
        async for event in chat.stream_chat_response(
            "c-1", kwargs.pop("text", "今晚想聊天"), io.agent, "u-1", **kwargs
        )
    ]


@pytest.mark.asyncio
async def test_graph_ordinary_turn_saves_and_finishes_once(chat_io):
    events = await run(chat_io)
    assert [e["event"] for e in events] == ["reply", "done"]
    chat_io.db.message.create.assert_awaited_once()
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    assert chat._save_replies.await_args.kwargs["achievement_turn_final"] is False
    chat_io.achievement.assert_called_once()
    chat._background_post_process.assert_called_once()
    assert chat._background_post_process.call_args.kwargs['source_message_ids'] == ['u-new']
    assert chat._background_post_process.call_args.kwargs['reply_turn_id'] == chat._save_replies.await_args.kwargs['achievement_turn_id']
    chat_io.tracer.close.assert_called_once()
    chat.detect_intent_unified.assert_awaited_once_with(
        "今晚想聊天", context="synthetic context"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", [IntentType.CONVERSATION_END, IntentType.NONE])
async def test_explicit_fragments_share_parent_lifecycle(chat_io, monkeypatch, primary):
    labels = {"终结意图": "不聊了", "日常交流": "心情一般"}
    chat.detect_intent_unified.return_value = IntentResult(
        intent=primary, confidence=0.9, metadata={"fragments": labels}
    )

    async def farewell(text, ctx, *_, **__):
        async for event in ctx.finalize("farewell", kind="conversation_end"):
            yield event

    monkeypatch.setattr(chat, "handle_conversation_end", farewell)
    events = await run(chat_io, text="心情一般，不聊了")
    assert [e["event"] for e in events] == ["reply", "reply", "done"]
    assert [json.loads(e["data"])["index"] for e in events[:-1]] == [0, 1]
    chat_io.db.message.create.assert_awaited_once()
    assert chat._save_replies.await_count == 2
    chat.detect_intent_unified.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    chat_io.achievement.assert_called_once()
    assert chat_io.achievement.await_args.kwargs["user_message_ids"] == ["u-new"]
    assert "assistant_texts" not in chat_io.achievement.await_args.kwargs
    chat._background_post_process.assert_called_once()
    assert chat._background_post_process.call_args.kwargs['source_message_ids'] == ['u-new']
    assert chat._background_post_process.call_args.kwargs['reply_turn_id'] == chat._save_replies.await_args.kwargs['achievement_turn_id']
    assert (
        chat._background_post_process.call_args.kwargs["user_message"]
        == "心情一般，不聊了"
    )
    chat_io.tracer.close.assert_called_once()


@pytest.mark.asyncio
async def test_consumed_message_skips_remaining_fragments(chat_io, monkeypatch):
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.CONVERSATION_END,
        confidence=0.9,
        metadata={"fragments": {"终结意图": "不聊了", "日常交流": "好吧"}},
    )

    async def handle(text, ctx, *_):
        ctx.consumed_full_message = True
        async for event in ctx.finalize("farewell", kind="conversation_end"):
            yield event

    monkeypatch.setattr(chat, "handle_conversation_end", handle)
    events = await run(chat_io)
    assert [e["event"] for e in events] == ["reply", "done"]
    chat._generate_reply.assert_not_called()
    chat._save_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_save_does_not_complete_or_submit_background(chat_io):
    chat._save_replies.return_value = None
    with pytest.raises(RuntimeError, match="persistence failed"):
        await run(chat_io)
    chat.finish_assistant_turn.assert_not_called()
    chat._background_post_process.assert_not_called()
    chat_io.achievement.assert_not_called()
    chat_io.tracer.close.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_classification_cleans_prefetch_and_context(
    chat_io, monkeypatch
):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def pending_fetch(**_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    chat.fetch_parallel_context.side_effect = pending_fetch

    async def classify(*_, **__):
        await started.wait()
        raise asyncio.CancelledError()

    chat.detect_intent_unified.side_effect = classify
    with pytest.raises(asyncio.CancelledError):
        await run(chat_io)
    assert closed.is_set()
    chat.finish_assistant_turn.assert_not_called()
    chat._background_post_process.assert_not_called()
    chat_io.tracer.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("old_executor,old_all", [("legacy", "false"), ("langgraph", "false"), ("langgraph", "true")])
async def test_retired_flags_cannot_change_graph_protocol(chat_io, monkeypatch, old_executor, old_all):
    from app.config import Settings
    from app.services.chat import graph_executor
    monkeypatch.setenv("CHAT_EXECUTOR", old_executor)
    monkeypatch.setenv("CHAT_GRAPH_ALL_CONVERSATIONS", old_all)
    monkeypatch.setenv("CHAT_GRAPH_CONVERSATION_ALLOWLIST", "unrelated-conversation")
    fresh = Settings(_env_file=None)
    assert "chat_executor" not in type(fresh).model_fields
    assert not hasattr(graph_executor, "select_executor")
    assert not hasattr(chat, "_stream_legacy_response")
    events = await run(chat_io, save_user_message=False)
    assert [event["event"] for event in events] == ["reply", "done"]
    chat_io.db.message.create.assert_not_awaited()
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_environment_change_during_turn_cannot_replay(chat_io, monkeypatch):
    async def classify(*_, **__):
        monkeypatch.setenv("CHAT_EXECUTOR", "legacy")
        monkeypatch.setenv("CHAT_GRAPH_ALL_CONVERSATIONS", "false")
        return IntentResult(intent=IntentType.NONE, confidence=0.9)
    chat.detect_intent_unified.side_effect = classify
    events = await run(chat_io)
    assert events[-1]["event"] == "done"
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_recursive_child_rejected_before_storage(chat_io):
    with pytest.raises(ValueError, match="recursive child"):
        await run(chat_io, sub_intent_mode=True)
    chat_io.db.conversation.find_unique.assert_not_awaited()
    chat_io.db.message.create.assert_not_awaited()
    chat._save_replies.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_failure_never_replays_persisted_user(chat_io):
    chat.detect_intent_unified.side_effect = RuntimeError("synthetic classifier unavailable")
    with pytest.raises(RuntimeError, match="synthetic classifier unavailable"):
        await run(chat_io)
    chat_io.db.message.create.assert_awaited_once()
    chat.detect_intent_unified.assert_awaited_once()
    chat._save_replies.assert_not_awaited()
    chat.finish_assistant_turn.assert_not_awaited()





@pytest.mark.asyncio
async def test_main_prompt_arguments_ignore_retired_environment(chat_io, monkeypatch):
    from app.services import music
    from app.services.offline import module_settings

    monkeypatch.setattr(music, "get_active_co_listening", AsyncMock(return_value=None))
    monkeypatch.setattr(
        module_settings, "is_activity_enabled", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(chat, "sample_expression_habits", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat, "get_relation_meta", AsyncMock(return_value={}))
    monkeypatch.setattr(chat, "load_ai_mood", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "pick_reply_count_target", lambda *_: 2)
    prompt = AsyncMock(return_value="synthetic system prompt")
    monkeypatch.setattr(chat, "build_system_prompt", prompt)
    monkeypatch.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 0.0)
    generated_messages = []

    async def generate(**kwargs):
        generated_messages.append(await kwargs["chat_messages_factory"]())
        return ["ordinary reply"], "ordinary reply", False, {"emotion": "中性"}

    chat._generate_reply.side_effect = generate
    monkeypatch.setenv("CHAT_EXECUTOR", "legacy")
    await run(chat_io, save_user_message=False)
    legacy = dict(prompt.await_args.kwargs)
    monkeypatch.setenv("CHAT_EXECUTOR", "langgraph")
    await run(chat_io, save_user_message=False)
    graph = dict(prompt.await_args.kwargs)
    legacy.pop("diagnostics")
    graph.pop("diagnostics")
    assert graph == legacy
    assert generated_messages[0] == generated_messages[1]


@pytest.mark.asyncio
async def test_crisis_has_priority_over_intent_pending_and_regular_fetch(
    chat_io, monkeypatch
):
    from app.services import portrait
    from app.services.memory.retrieval import safety

    chat_io.decision.crisis_force_intent = True
    chat_io.decision.crisis_care_turn = True
    discard = AsyncMock()
    monkeypatch.setattr(chat, "discard_pending_states_for_crisis", discard)
    monkeypatch.setattr(safety, "retrieve_crisis_memories", AsyncMock(return_value=[]))
    monkeypatch.setattr(portrait, "get_latest_portrait", AsyncMock(return_value=None))

    async def crisis(text, ctx, **_):
        async for event in ctx.finalize("crisis care", kind="crisis"):
            yield event

    monkeypatch.setattr(chat, "handle_crisis", crisis)
    events = await run(chat_io, text="我不想活了")
    assert json.loads(events[0]["data"])["text"] == "crisis care"
    discard.assert_awaited_once_with("c-1")
    chat.detect_intent_unified.assert_not_called()
    chat.fetch_parallel_context.assert_not_called()
    chat._generate_reply.assert_not_called()
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_confirmation_short_circuits_without_intent_llm(
    chat_io, monkeypatch
):
    async def pending(text, ctx):
        ctx.last_short_circuit_reply = "confirmed"
        events = await ctx.short_circuit_fn(
            "confirmed", ctx.conversation_id, ctx.agent_id, ctx.user_id
        )
        for event in events:
            yield event
        ctx.stopped = True

    monkeypatch.setattr(chat, "resolve_pending_deletion", pending)
    events = await run(chat_io, text="好")
    assert [e["event"] for e in events] == ["reply", "done"]
    chat.detect_intent_unified.assert_not_called()
    chat.fetch_parallel_context.assert_not_called()
    chat._background_post_process.assert_called_once()
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_boundary_block_keeps_memory_background_excluded(chat_io, monkeypatch):
    chat_io.decision.skip_boundary = False

    async def boundary(ctx):
        for event in await ctx.short_circuit_fn(
            "blocked",
            ctx.conversation_id,
            ctx.agent_id,
            ctx.user_id,
            extra_metadata={"boundary": True},
        ):
            yield event
        ctx.stopped = True

    monkeypatch.setattr(chat, "run_boundary", boundary)
    events = await run(chat_io)
    assert [e["event"] for e in events] == ["reply", "done"]
    chat.detect_intent_unified.assert_not_called()
    chat._background_post_process.assert_not_called()
    assert chat.finish_assistant_turn.await_args.kwargs["proactive_reason"] is None


@pytest.mark.asyncio
async def test_real_graph_traces_link_nodes_and_manual_provider(chat_io, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import HumanMessage

    import app.db
    from app.services.chat import local_tracer

    fake = MagicMock()
    fake.tracerun.create = AsyncMock()
    fake.tracerun.upsert = AsyncMock()
    fake.tracerun.update = AsyncMock()
    monkeypatch.setattr(app.db, "db", fake)
    monkeypatch.setattr(settings, "trace_backend", "local")
    from app.services.chat.tracing import create_tracer
    monkeypatch.setattr(chat, "create_tracer", create_tracer)

    async def generate(**_):
        await FakeListChatModel(responses=["fake reply"]).ainvoke(
            [HumanMessage("fake")]
        )
        now = datetime.now(timezone.utc)
        local_tracer.record_manual_llm_run(
            name="SyntheticHTTPProvider",
            model_name="synthetic",
            messages=[{"role": "user", "content": "fake"}],
            output_text="fake reply",
            started_at=now,
            ended_at=now,
        )
        return ["ordinary reply"], "ordinary reply", False, {"emotion": "中性"}

    chat._generate_reply.side_effect = generate
    events = await run(chat_io)
    await asyncio.sleep(0.1)
    assert events[-1]["event"] == "done"
    rows = [c.kwargs["data"] for c in fake.tracerun.create.await_args_list]
    roots = [row for row in rows if row["name"] == "chat_request"]
    assert len(roots) == 1
    metadata = roots[0]["extraJson"].data["metadata"]
    assert {k:v for k,v in metadata.items() if k != 'chat_context_policy'} == {
        "executor": "langgraph", "graph_version": "chat-g01-v1",
        "checkpoint_enabled": False, "usage_scope": "chat",
        "usage_accounting_version": "trace-linked-v2",
    }
    assert metadata['chat_context_policy']['effective_rounds'] == 5
    assert metadata['chat_context_policy']['format_version'] == 'dialogue-turns-v1'
    assert metadata['chat_context_policy']['main_history_mode'] == 'budget'
    assert isinstance(metadata['chat_context_policy']['windows'], dict)
    root = roots[0]["id"]
    nodes = [row for row in rows if row["name"] == "generate_reply"]
    assert len(nodes) == 1
    manual = next(row for row in rows if row["name"] == "SyntheticHTTPProvider")
    assert manual["traceId"] == root
    assert manual["parentId"] == nodes[0]["id"]
    assert local_tracer._local_trace_handler.get() is None
    assert local_tracer._graph_node_parent.get() is None
    graph = next(row for row in rows if row["name"] == "main_chat")
    extra = graph["extraJson"].data
    assert extra["metadata"]["graph_version"] == "chat-g01-v1"
    assert "owner" not in str(graph.get("inputsJson"))


@pytest.mark.asyncio
async def test_failed_graph_root_reports_error(chat_io, monkeypatch):
    import app.db
    from app.services.chat import local_tracer

    fake = MagicMock()
    fake.tracerun.create = AsyncMock()
    fake.tracerun.upsert = AsyncMock()
    fake.tracerun.update = AsyncMock()
    monkeypatch.setattr(app.db, "db", fake)
    monkeypatch.setattr(settings, "trace_backend", "local")
    from app.services.chat.tracing import create_tracer
    monkeypatch.setattr(chat, "create_tracer", create_tracer)
    chat._save_replies.return_value = None
    with pytest.raises(RuntimeError):
        await run(chat_io)
    await asyncio.sleep(0.1)
    root = fake.tracerun.update.await_args.kwargs["data"]
    assert root["status"] == "error"
    assert root["error"] == "RuntimeError"
    assert local_tracer._local_trace_handler.get() is None


def test_checkpoint_and_node_retry_are_disabled():
    from app.services.chat.main_graph import DURABLE_EXECUTION_READY, _build_graph

    graph = _build_graph()
    assert graph.checkpointer is None
    assert DURABLE_EXECUTION_READY is False
    assert all(not node.retry_policy for node in graph.nodes.values())


@pytest.mark.asyncio
async def test_missing_node_output_fails_before_next_phase(chat_io, monkeypatch):
    from app.services.chat import graph_phases
    from app.services.chat.graph_contracts import ChatNodeContractError

    async def incomplete(ctx):
        yield ctx.completed("identify")

    monkeypatch.setattr(graph_phases, "identify", incomplete)
    with pytest.raises(ChatNodeContractError, match="detected_intent"):
        await run(chat_io)
    chat._save_replies.assert_not_called()
    chat.finish_assistant_turn.assert_not_called()
    chat._background_post_process.assert_not_called()


@pytest.mark.asyncio
async def test_voice_reply_binds_after_persistence_and_keeps_wire_fields(
    chat_io, monkeypatch
):
    from app.services.speech_output import delivery, policy

    voice = SimpleNamespace(
        transcript="voice transcript", metadata={"type": "voice", "url": "synthetic"}
    )
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    monkeypatch.setattr(delivery, "prepare_voice_output", AsyncMock(return_value=voice))
    bind = AsyncMock()
    monkeypatch.setattr(delivery, "bind_prepared_voice_output", bind)
    events = await run(chat_io, reply_context={"client_supports_voice": True})
    payload = json.loads(events[0]["data"])
    assert payload["display_mode"] == "voice"
    assert payload["assistant_message_id"] == "a-new"
    assert payload["text"] == "voice transcript"
    bind.assert_awaited_once_with(voice, message_id="a-new")
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_short_voice_persistence_discards_unbound_output(
    chat_io, monkeypatch
):
    from app.services.chat.multi_intent import short_circuit_reply
    from app.services.speech_output import delivery, policy

    voice = SimpleNamespace(transcript="voice transcript", metadata={})
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    monkeypatch.setattr(delivery, "prepare_voice_output", AsyncMock(return_value=voice))
    discard = AsyncMock()
    monkeypatch.setattr(delivery, "discard_prepared_voice_output", discard)
    persist = AsyncMock(side_effect=RuntimeError("write failed"))
    with pytest.raises(RuntimeError, match="write failed"):
        await short_circuit_reply(
            "reply",
            "c-1",
            "a-1",
            "u-1",
            persist,
            agent=chat_io.agent,
            voice_context=policy.VoiceContext.NORMAL_CHAT,
            defer_turn_finalization=True,
        )
    discard.assert_awaited_once_with(voice)
    chat.finish_assistant_turn.assert_not_called()


@pytest.mark.asyncio
async def test_filler_reply_retains_legacy_proactive_reason(chat_io, monkeypatch):
    monkeypatch.setattr(chat, "build_filler_emoji_reply", lambda *_, **__: "😊")
    await run(chat_io, text="嗯")
    assert (
        chat.finish_assistant_turn.await_args.kwargs["proactive_reason"]
        == "short_circuit"
    )
    chat.detect_intent_unified.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_query_and_ordinary_fragment_records_ordinary_ai_memory(
    chat_io, monkeypatch
):
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.SCHEDULE_QUERY,
        confidence=0.9,
        metadata={
            "query_type": "date",
            "fragments": {"计划查询": "你明天有什么安排", "日常交流": "我心情一般"},
        },
    )

    async def handle(text, ctx, **_):
        return (
            True,
            ctx.finalize("query reply", kind="schedule_query"),
            "synthetic schedule",
        )

    monkeypatch.setattr(chat, "handle_schedule_query", handle)
    events = await run(chat_io, text="你明天有什么安排，我心情一般")
    assert [e["event"] for e in events] == ["reply", "reply", "done"]
    assert chat._background_post_process.call_args.kwargs["skip_ai_memory"] is False
    chat._background_post_process.assert_called_once()


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_owned_tasks(chat_io):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def fetch(**_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def classify(*_, **__):
        await started.wait()
        await asyncio.Event().wait()

    chat.fetch_parallel_context.side_effect = fetch
    chat.detect_intent_unified.side_effect = classify
    task = asyncio.create_task(run(chat_io))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    chat._background_post_process.assert_not_called()
    chat.finish_assistant_turn.assert_not_called()
    chat_io.tracer.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,text,handler,kind,shape",
    [
        (
            IntentType.APOLOGY_PROMISE,
            "对不起，我保证不这样了",
            "handle_apology_promise",
            "apology_promise",
            2,
        ),
        (
            IntentType.DELETION,
            "忘记我住在哪里",
            "handle_deletion",
            "deletion_delete",
            2,
        ),
        (
            IntentType.SCHEDULE_ADJUST,
            "你能不能晚点睡，陪我聊会儿",
            "handle_schedule_adjust",
            "schedule_adjust",
            2,
        ),
        (
            IntentType.SCHEDULE_QUERY,
            "你明天有什么安排",
            "handle_schedule_query",
            "schedule_query",
            3,
        ),
        (
            IntentType.CURRENT_STATE,
            "你现在忙吗",
            "handle_current_state",
            "current_state",
            2,
        ),
        (
            IntentType.RECORD_REQUEST,
            "提醒我明天九点喝水",
            "handle_record_request",
            "record_request",
            2,
        ),
    ],
)
async def test_special_routes_preserve_handler_and_single_completion(
    chat_io, monkeypatch, intent, text, handler, kind, shape
):
    async def handle(message, ctx, *_, **__):
        result = (True, ctx.finalize("special reply", kind=kind))
        return (*result, "synthetic schedule") if shape == 3 else result

    stub = AsyncMock(side_effect=handle)
    monkeypatch.setattr(chat, handler, stub)
    events = await run(chat_io, text=text, forced_intent=intent)
    assert [e["event"] for e in events] == ["reply", "done"]
    stub.assert_awaited_once()
    chat._generate_reply.assert_not_called()
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    chat._background_post_process.assert_called_once()
    chat_io.achievement.assert_called_once()
    chat.detect_intent_unified.assert_not_called()


@pytest.mark.asyncio
async def test_current_state_fast_path_still_skips_classifier_and_full_fetch(
    chat_io, monkeypatch
):
    monkeypatch.setattr(chat, "get_cached_schedule", AsyncMock(return_value=[]))

    async def handle(message, ctx, **_):
        return True, ctx.finalize("state reply", kind="current_state")

    monkeypatch.setattr(chat, "handle_current_state", AsyncMock(side_effect=handle))
    events = await run(chat_io, text="你在干嘛呢")
    assert [e["event"] for e in events] == ["reply", "done"]
    chat.detect_intent_unified.assert_not_called()
    chat.fetch_parallel_context.assert_not_called()
    assert chat._background_post_process.call_args.kwargs["skip_ai_memory"] is True


@pytest.mark.parametrize("executor", ["langgraph"])
async def test_background_sources_cover_all_fragments_before_submission(chat_io,monkeypatch,executor):
    chat.detect_intent_unified.return_value=IntentResult(intent=IntentType.NONE,confidence=.9,
        metadata={'fragments':{'日常交流':'今天开心','终结意图':'先不聊了'}})
    async def farewell(text,ctx,*args,**kwargs):
        async for event in ctx.finalize('farewell',kind='conversation_end'):
            yield event
    monkeypatch.setattr(chat,'handle_conversation_end',farewell)
    saved=[]
    async def save(conversation,replies,**kwargs):
        saved.append(kwargs['achievement_turn_id'])
        return 'reply-'+str(len(saved))
    monkeypatch.setattr(chat,'_save_replies',AsyncMock(side_effect=save))
    async def noop():pass
    def background(**kwargs):
        assert len(saved)==2
        assert kwargs['source_message_ids']==['u-new']
        assert all(turn==kwargs['reply_turn_id'] for turn in saved)
        return noop()
    submit=MagicMock(side_effect=background)
    monkeypatch.setattr(chat,'_background_post_process',submit)
    events=await run(chat_io,text='今天开心，先不聊了')
    assert sum(event['event']=='done' for event in events)==1
    submit.assert_called_once()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_interrupted_graph_fragment_does_not_complete_parent(chat_io, monkeypatch, failure):
    chat.detect_intent_unified.return_value = IntentResult(intent=IntentType.NONE, confidence=.9,
        metadata={"fragments": {"日常交流": "今天开心", "终结意图": "先不聊了"}})
    async def interrupted(*args, **kwargs):
        raise failure("fragment interrupted")
        yield
    monkeypatch.setattr(chat, "handle_conversation_end", interrupted)
    with pytest.raises(failure):
        await run(chat_io, text="今天开心，先不聊了")
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_not_awaited()
    chat._background_post_process.assert_not_called()
    chat_io.tracer.close.assert_called_once()
