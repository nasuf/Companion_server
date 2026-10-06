"""G02: fixed-input paired matrix, including real reply generation and prompts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services.chat import orchestrator as chat
from app.services.chat.data_fetch_phase import FetchedContext
from app.services.chat.intent_dispatcher import IntentResult, IntentType
from tests.g02_harness_support import configure_pair
from evals.reply_register.cases import ALL_CASES


async def collect(io, text="今天有点累了", **kwargs):
    # Paired executions share the ingress clock, including a synthetic user row
    # absent from the mocked history. Never compare two wall-clock timestamps.
    kwargs.setdefault("reply_context", {"received_at": io.now.isoformat()})
    return [event async for event in chat.stream_chat_response(
        "c-1", text, io.agent, "u-1", **kwargs
    )]


def comparable_events(events):
    result = []
    for event in events:
        payload = json.loads(event["data"])
        if event["event"] == "done" and "assistant_message_id" in payload:
            # G01 adds the persisted message id to previously id-less text
            # short-circuits. Clients already accept this optional field.
            assert payload.pop("assistant_message_id") == "a-new"
        result.append({"event": event["event"], "data": payload})
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "weak", "medium", "strong", "l3", "tier_empty", "tier_failure",
    "reunion", "relational", "contradiction", "contradiction_failure",
    "music", "music_failure", "media", "aggregation",
])
async def test_paired_reply_paths_and_prompt_inputs(monkeypatch, case):
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            relevance = case if case in ("weak", "medium", "strong") else "weak"
            if case in ("contradiction", "contradiction_failure", "l3"):
                relevance = "strong"
            chat.fetch_parallel_context.return_value = FetchedContext(
                memory_relevance=relevance,
                user_emotion={"emotion": "疲惫", "intensity": 40},
            )
            kwargs = {}
            text = "今天有点累了"
            if case == "l3":
                chat.maybe_awaken_l3.return_value = (["你曾提过小时候养过白猫"], "请求更久")
            elif case in ("tier_empty", "tier_failure"):
                patch.setattr(chat, "_memory_weak_reply", AsyncMock(
                    return_value=None,
                    side_effect=RuntimeError("synthetic tier failure") if case == "tier_failure" else None,
                ))
            elif case == "reunion":
                patch.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 14400.0)
                chat.get_or_build_session_recap.return_value = "上次聊到准备去逛陶艺展"
            elif case == "relational":
                patch.setattr(chat, "detect_relational_context", lambda *_: "陪伴请求")
            elif case in ("contradiction", "contradiction_failure"):
                if case == "contradiction":
                    chat.detect_l1_contradiction.return_value = {"conflict_description": "城市变更"}
                    patch.setattr(chat, "generate_contradiction_inquiry", AsyncMock(return_value="你现在搬到杭州了吗？"))
                else:
                    chat.detect_l1_contradiction.side_effect = RuntimeError("synthetic conflict lookup failure")
            elif case in ("music", "music_failure"):
                from app.services import music
                if case == "music_failure":
                    music.get_active_co_listening.side_effect = RuntimeError("synthetic music lookup failure")
                else:
                    music.get_active_co_listening.return_value = SimpleNamespace(
                        track=SimpleNamespace(title="合成歌曲", artist="合成歌手")
                    )
                # Main prompt is where co-listening context is injected.
                patch.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 14400.0)
            elif case in ("media", "aggregation"):
                if case == "media":
                    # The realtime adapter renders attachments before calling
                    # the orchestrator; model input must use that rendered text.
                    text = "看这个\n\n[图片内容]\n图片1：合成图片里有一只白猫"
                metadata = {"attachments": [{"kind": "image", "vision_summary": "合成图片里有一只白猫"}]}
                history = [
                    SimpleNamespace(id="fragment-1", role="user", content="看", metadata={}, createdAt=io.now),
                    SimpleNamespace(id="fragment-2", role="user", content="这个", metadata=metadata if case == "media" else {}, createdAt=io.now),
                ]
                io.db.message.find_many.side_effect = lambda **_: history.copy()
                kwargs = {"save_user_message": False, "user_message_id": "fragment-2",
                          "reply_context": {"turn_message_ids": ["fragment-1", "fragment-2"],
                                            "received_at": io.now.isoformat()}}
                patch.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 14400.0)
            events = await collect(io, text, **kwargs)
            snapshot = {
                "events": events, "main": io.main_calls, "tier": io.tier_calls,
                "user_saves": io.db.message.create.await_count,
                "assistant_saves": chat._save_replies.await_count,
                "finish": chat.finish_assistant_turn.await_count,
                "background": chat._background_post_process.call_count,
                "conflict_save": chat.save_pending_contradiction.await_count,
            }
            assert snapshot["finish"] == snapshot["background"] == 1
            assert snapshot["assistant_saves"] == 1
            assert [e["event"] for e in events][-1] == "done"
            if case == "contradiction":
                assert not io.main_calls and not io.tier_calls
                assert snapshot["conflict_save"] == 1
            elif case in ("tier_empty", "tier_failure", "reunion", "relational", "music", "music_failure", "media", "aggregation"):
                assert len(io.main_calls) == 1
                prompt = io.main_calls[0][0][0]["content"]
                if case == "music":
                    assert "合成歌曲" in prompt and "合成歌手" in prompt
                if case == "music_failure":
                    assert "合成歌曲" not in prompt
                if case == "media":
                    assert "白猫" in str(io.main_calls[0][0])
                if case == "aggregation":
                    assert str(io.main_calls[0][0]).count("今天有点累了") == 1
            else:
                assert len(io.tier_calls) == 1
                assert io.tier_calls[0][0] == ("l3" if case == "l3" else relevance)
                assert "[10-04 12:00] user: 今天有点累了" in io.tier_calls[0][1]["context"]
            snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mbti", [
    {"type": "ENFJ", "EI": 80, "NS": 80, "TF": 20, "JP": 80},
    {"type": "ISTP", "EI": 20, "NS": 20, "TF": 80, "JP": 20},
])
async def test_paired_personality_context_uses_current_mbti(monkeypatch, mbti):
    from app.services.mbti import get_mbti
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            patch.setattr(chat, "get_mbti", get_mbti)
            io.agent.mbti = {"type": "INTJ", "EI": 20, "NS": 80, "TF": 80, "JP": 80}
            io.agent.currentMbti = mbti
            patch.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 14400.0)
            events = await collect(io)
            assert len(io.main_calls) == 1
            prompt = io.main_calls[0][0][0]["content"]
            assert mbti["type"] in prompt
            assert "INTJ" not in prompt
            snapshots.append((comparable_events(events), io.main_calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_paired_real_current_state_handler_uses_synthetic_domain_io(monkeypatch, fallback):
    from app.services.chat import intent_handlers
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            reply = AsyncMock(return_value=None if fallback else "正在整理工作台呀")
            patch.setattr(intent_handlers, "current_state_reply", reply)
            events = await collect(io, "你在干嘛", forced_intent=IntentType.CURRENT_STATE)
            intent_handlers.resolve_implicit_time.assert_awaited_once()
            assert reply.await_args.kwargs["current_activity"] == "在家整理工作台"
            assert chat._save_replies.await_count == chat.finish_assistant_turn.await_count == 1
            assert bool(io.main_calls) is fallback
            snapshots.append((comparable_events(events), reply.await_args.kwargs, io.main_calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_paired_farewell_and_prompt_fallback(monkeypatch, fallback):
    from app.services.chat import intent_handlers
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            end = AsyncMock(return_value=None if fallback else "好，那晚安啦")
            patch.setattr(intent_handlers, "end_reply", end)
            patch.setattr(chat, "_intent_llm_reply", AsyncMock(return_value="好，那晚安啦"))
            events = await collect(io, "今天先不聊了，晚安", forced_intent=IntentType.CONVERSATION_END)
            assert chat._save_replies.await_count == 1
            assert chat.finish_assistant_turn.await_count == 1
            assert chat._background_post_process.call_count == 1
            assert chat._intent_llm_reply.await_count == int(fallback)
            snapshots.append((comparable_events(events), end.await_args.kwargs))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
async def test_paired_crisis_priority_preserves_context(monkeypatch, followup):
    from app.services import portrait
    from app.services.memory.retrieval import safety
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            io.decision.crisis_force_intent = not followup
            io.decision.crisis_followup_active = followup
            io.decision.crisis_care_turn = True
            io.decision.recent_crisis_context = "前一轮存在危机风险"
            io.decision.intent_metadata = {"followup": True, "safety_check_mode": "soft"} if followup else {}
            clear = AsyncMock()
            patch.setattr(chat, "discard_pending_states_for_crisis", clear)
            patch.setattr(portrait, "get_latest_portrait", AsyncMock(return_value=None))
            patch.setattr(safety, "retrieve_crisis_memories", AsyncMock(return_value=[]))
            patch.setattr(safety, "retrieve_crisis_followup_memories", AsyncMock(return_value=[]))
            calls = []
            async def handle(message, ctx, **kwargs):
                calls.append((message, kwargs))
                async for event in ctx.finalize("我在，先到安全的地方", kind="crisis_followup" if followup else "crisis"):
                    yield event
            patch.setattr(chat, "handle_crisis_followup" if followup else "handle_crisis", handle)
            events = await collect(io, "我现在还是很难受")
            clear.assert_awaited_once_with("c-1")
            chat.detect_intent_unified.assert_not_called()
            chat.fetch_parallel_context.assert_not_called()
            assert chat.finish_assistant_turn.await_count == 1
            assert not io.main_calls and not io.tier_calls
            assert len(calls) == 1
            snapshots.append((comparable_events(events), calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("zone", ["blocked", "medium", "low", "apology_unblock"])
async def test_paired_boundary_stops_before_pending_and_intent(monkeypatch, zone):
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            io.decision.skip_boundary = False
            async def boundary(ctx):
                for event in await ctx.short_circuit_fn(
                    "边界回复", ctx.conversation_id, ctx.agent_id, ctx.user_id,
                    extra_metadata={"boundary": True, "zone": zone},
                ):
                    yield event
                ctx.stopped = True
            patch.setattr(chat, "run_boundary", boundary)
            events = await collect(io)
            chat.detect_intent_unified.assert_not_called()
            chat.fetch_parallel_context.assert_not_called()
            chat._background_post_process.assert_not_called()
            assert chat.finish_assistant_turn.await_args.kwargs["proactive_reason"] is None
            snapshots.append(comparable_events(events))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_paired_actual_emitter_delay_indices_and_received_time(monkeypatch, queued):
    from app.services.chat import reply_post_process as post
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            patch.setattr(chat, "_emit_replies", post.emit_replies)
            patch.setattr(post, "actual_delay_seconds", lambda _: 120.0)
            patch.setattr(post, "_should_explain_delay", lambda _: True)
            patch.setattr(post, "_now_corrected", lambda: io.now)
            patch.setattr(post, "save_ai_mood", AsyncMock())
            patch.setattr(post, "_load_recent_emojis", AsyncMock(return_value=set()))
            patch.setattr(post, "should_add_emoji", lambda *_: False)
            patch.setattr(post, "should_add_sticker", lambda *_: False)
            patch.setattr(post, "maybe_typo", lambda text, *_, **__: text)
            patch.setattr(post.random, "uniform", lambda *_: 0.0)
            delay = AsyncMock(return_value="刚才在收拾工作台")
            patch.setattr(chat, "_delay_explanation_reply", delay)
            context = {"received_at": io.now.isoformat(), "received_status": {"activity": "收拾工作台", "status": "busy"}}
            events = await collect(io, reply_context=context, delivered_from_queue=queued)
            replies = [json.loads(event["data"]) for event in events if event["event"] == "reply"]
            assert [r["index"] for r in replies] == list(range(len(replies)))
            assert replies[0]["delay_explanation"] is True
            assert delay.await_args.kwargs["received_time"] == "12:00"
            assert context["received_at"] == io.now.isoformat()
            assert chat._save_replies.await_count == chat.finish_assistant_turn.await_count == 1
            snapshots.append((comparable_events(events), delay.await_args.kwargs))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["red_packet", "gift"])
@pytest.mark.parametrize("mark_fails", [False, True])
async def test_paired_offering_context_and_receive_failure(monkeypatch, kind, mark_fails):
    from app.services import offerings
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            mark = AsyncMock(side_effect=RuntimeError("synthetic mark failure") if mark_fails else None)
            patch.setattr(offerings, "mark_offering_received", mark)
            context = {kind: {"offering_id": "synthetic-offering", "amount": 8.88,
                              "gift_name": "合成花束", "name": "合成花束", "message": "送给你"}}
            events = await collect(io, reply_context=context)
            assert len(io.main_calls) == 1
            mark.assert_awaited_once_with(offering_id="synthetic-offering", user_id="u-1", conversation_id="c-1")
            assert chat._background_post_process.call_args.kwargs["skip_memory"] is True
            snapshots.append((comparable_events(events), io.main_calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
async def test_three_fragment_parent_collects_usage_and_side_effects_once(monkeypatch):
    from app.services.llm import usage_repo, usage_tracker
    io = configure_pair(monkeypatch)
    usage = AsyncMock()
    monkeypatch.setattr(usage_repo, "write_usage_row", usage)
    original = "忘记我的城市，提醒我明天喝水，今天有点累"
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.DELETION, confidence=0.9,
        metadata={"fragments": {"删除": "忘记我的城市", "记录请求": "提醒我明天喝水", "日常交流": "今天有点累"}},
    )
    received = []
    async def special(message, ctx, **_):
        received.append(ctx.reply_context["received_at"])
        usage_tracker.record("synthetic", 10, 5)
        return True, ctx.finalize("已处理", kind="record_request" if "喝水" in message else "deletion_delete")
    monkeypatch.setattr(chat, "handle_deletion", special)
    monkeypatch.setattr(chat, "handle_record_request", special)
    async def tier(**_):
        usage_tracker.record("synthetic", 10, 5)
        return "先歇一会儿"
    monkeypatch.setattr(chat, "_memory_weak_reply", tier)
    events = await collect(io, original, reply_context={"received_at": io.now.isoformat()})
    replies = [json.loads(e["data"]) for e in events if e["event"] == "reply"]
    assert [r["index"] for r in replies] == [0, 1, 2]
    assert sum(e["event"] == "done" for e in events) == 1
    assert io.db.message.create.await_count == 1
    assert chat._save_replies.await_count == 3
    assert chat.finish_assistant_turn.await_count == 1
    assert io.achievement.call_count == chat._background_post_process.call_count == 1
    assert chat._background_post_process.call_args.kwargs["user_message"] == original
    assert received == [io.now.isoformat(), io.now.isoformat()]
    usage.assert_awaited_once()
    assert usage.await_args.kwargs["summary"]["call_count"] == 3
    assert usage.await_args.kwargs["summary"]["input_tokens"] == 30


SPECIAL_ROUTES = [
    (IntentType.APOLOGY_PROMISE, "对不起，我保证不会了", "handle_apology_promise", "apology_promise", 2),
    (IntentType.DELETION, "忘记我住在哪里", "handle_deletion", "deletion_delete", 2),
    (IntentType.SCHEDULE_ADJUST, "你能不能晚点睡，陪我聊会儿", "handle_schedule_adjust", "schedule_adjust", 2),
    (IntentType.SCHEDULE_QUERY, "你明天有什么安排", "handle_schedule_query", "schedule_query", 3),
    (IntentType.CURRENT_STATE, "你现在忙吗", "handle_current_state", "current_state", 2),
    (IntentType.RECORD_REQUEST, "提醒我明天九点喝水", "handle_record_request", "record_request", 2),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("intent,text,handler,kind,shape", SPECIAL_ROUTES)
@pytest.mark.parametrize("handled", [True, False])
async def test_paired_special_route_success_and_fallthrough(monkeypatch, intent, text, handler, kind, shape, handled):
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            calls = []
            async def handle(message, ctx, *_, **kwargs):
                calls.append((message, kwargs))
                result = (handled, ctx.finalize("专用回复", kind=kind) if handled else None)
                return (*result, "合成作息") if shape == 3 else result
            patch.setattr(chat, handler, handle)
            events = await collect(io, text, forced_intent=intent)
            assert len(calls) == 1
            assert chat.finish_assistant_turn.await_count == 1
            assert chat._background_post_process.call_count == 1
            assert chat._save_replies.await_count == 1
            assert bool(io.main_calls) is not handled
            snapshots.append((comparable_events(events), calls, io.main_calls))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver", ["resolve_recent_undo", "resolve_pending_contradiction", "resolve_pending_deletion", "resolve_retrieval_feedback_correction"])
@pytest.mark.parametrize("expired", [True, False])
async def test_paired_pending_priority_and_expiry(monkeypatch, resolver, expired):
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            calls = []
            async def pending(*args, **kwargs):
                ctx = kwargs.get("ctx") or args[1]
                calls.append(resolver)
                if not expired:
                    ctx.last_short_circuit_reply = "已确认"
                    for event in await ctx.short_circuit_fn("已确认", ctx.conversation_id, ctx.agent_id, ctx.user_id):
                        yield event
                    ctx.stopped = True
            patch.setattr(chat, resolver, pending)
            events = await collect(io)
            assert len(calls) == 1
            assert chat.detect_intent_unified.await_count == int(expired)
            assert chat.finish_assistant_turn.await_count == 1
            assert chat._background_post_process.call_count == 1
            snapshots.append(comparable_events(events))
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("relevance", ["weak", "medium", "strong"])
async def test_full_bank_same_classifier_decision_renders_same_tier_prompt(monkeypatch, case, relevance):
    """Isolate executor semantics from independently sampled classifier outputs.

    Keep actual tier templates/rendering. This is a controlled-input contract
    test, not a real-model quality or production-prompt qualification result.
    """
    from app.services.chat import intent_replies
    snapshots = []
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            patch.setattr(settings, "chat_executor", executor)
            patch.setattr(chat, "detect_relational_context", lambda *_: None)
            chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance=relevance)
            chat._fetch_intent_context.return_value = "\n".join(
                f"{'AI' if role == 'assistant' else '用户'}: {text}" for role, text in case.history
            )
            patch.setattr(intent_replies, "get_chat_model", lambda: "synthetic-model")
            invoke = AsyncMock(return_value="我这会儿想不起来了||你再跟我说说？")
            patch.setattr(intent_replies, "invoke_text", invoke)
            for kind in ("weak", "medium", "strong"):
                patch.setattr(chat, f"_memory_{kind}_reply", getattr(intent_replies, f"memory_{kind}_reply"))
            events = await collect(io, case.message, forced_intent=IntentType.NONE)
            invoke.assert_awaited_once()
            assert not io.main_calls
            assert "用户刚才说：" + case.message in invoke.await_args.args[1]
            assert chat.finish_assistant_turn.await_count == 1
            snapshots.append((comparable_events(events), invoke.await_args.args[1]))
    assert snapshots[0] == snapshots[1]
