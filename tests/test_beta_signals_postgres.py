"""Real migrated PostgreSQL/Redis persistence, with synthetic model/transport IO."""
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from tests.test_proactive_activity_e2e import flow  # isolated resource fixture
from app.services.proactive import emit
from app.services.proactive.event_identity import proactive_event_id
from app.services.interaction import reply_telemetry as telemetry


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['silence_wakeup', 'followup_unfinished', 'first_greeting',
                                  'special_date', 'offline_activity_recommendation', 'reminder'])
async def test_six_proactive_paths_persist_and_deliver_same_identity(flow, kind):
    mid = await emit.emit_proactive_message(conversation_id=flow.conversation, user_id=flow.user,
        agent_id=flow.agent, workspace_id=flow.workspace, message='合成主动消息||第二句话',
        trigger_type=kind, voice_eligible=False)
    row = await flow.db.message.find_unique(where={'id': mid})
    assert row.metadata['proactive_event_id']
    assert row.metadata['proactive_event_id'] == flow.ws.call_args.args[2]['proactive_event_id']
    assert row.metadata['proactive_event_definition_version'] == 1
    other = await emit.emit_proactive_message(conversation_id=flow.conversation, user_id=flow.user,
        agent_id=flow.agent, workspace_id=flow.workspace, message='下一次发送', trigger_type=kind, voice_eligible=False)
    row2 = await flow.db.message.find_unique(where={'id': other})
    assert row.metadata['proactive_event_id'] != row2.metadata['proactive_event_id']


@pytest.mark.asyncio
async def test_card_and_text_share_durable_event_identity(flow):
    from app.services.offline.chat_emit import emit_activity_card, emit_assistant
    ident = proactive_event_id(conversation_id=flow.conversation, kind='recommendation', source_id='synthetic-operation')
    same = proactive_event_id(conversation_id=flow.conversation, kind='recommendation', source_id='synthetic-operation')
    assert same == ident
    assert proactive_event_id(conversation_id='other', kind='recommendation', source_id='synthetic-operation') != ident
    mids = [await emit_assistant(conversation_id=flow.conversation, user_id=flow.user,
        agent_id=flow.agent, workspace_id=flow.workspace, message='合成邀请', real_world_type='activity',
        source_id='synthetic-operation', trigger_type='offline_activity_recommendation',
        extra_metadata={'proactive_event_id': ident})]
    mids.append(await emit_activity_card(conversation_id=flow.conversation, user_id=flow.user,
        agent_id=flow.agent, workspace_id=flow.workspace, activity={'id': 'synthetic-operation', 'title': '合成散步'},
        trigger_type='offline_activity_recommendation_card', status_label='待确定',
        extra_metadata={'proactive_event_id': ident}))
    rows = await flow.db.message.find_many(where={'id': {'in': mids}})
    assert len(rows) == 2 and {r.metadata['proactive_event_id'] for r in rows} == {ident}
    assert {call.args[2]['proactive_event_id'] for call in flow.ws.await_args_list} == {ident}


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['main', 'tier', 'short_circuit', 'web_search'])
async def test_graph_to_database_keeps_timing_and_trace_diagnostics(flow, monkeypatch, path):
    from tests.g02_harness_support import configure_pair
    from app.services.chat import orchestrator as chat, post_process, reply_generate, intent_handlers
    from app.services.chat.intent_dispatcher import IntentType
    from app.services.chat.tracing import _attach_message_trace_metadata
    from app.services import persona_grounding
    real_main = reply_generate._run_main_llm
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(persona_grounding, 'verify_generated_locations', AsyncMock(return_value=[]))
    monkeypatch.setattr(post_process, 'db', flow.db)
    monkeypatch.setattr(post_process, 'fire_background', lambda coro: coro.close())
    monkeypatch.setattr(chat, '_save_replies', post_process.save_replies)
    kwargs = {}
    text = '今天有点累了'
    if path in {'main', 'web_search'}:
        monkeypatch.setattr(chat, 'compute_reengagement_gap_seconds', lambda *_, **__: 14400.)
    if path == 'web_search':
        chat.fetch_parallel_context.return_value.needs_web_search = True
        search = AsyncMock(return_value='合成联网答案')
        monkeypatch.setattr(reply_generate, '_try_web_search_reply', search)
        monkeypatch.setattr(reply_generate, '_run_main_llm', real_main)
    if path == 'short_circuit':
        text = '你在干嘛'
        kwargs['forced_intent'] = IntentType.CURRENT_STATE
        monkeypatch.setattr(intent_handlers, 'current_state_reply', AsyncMock(return_value='在整理工作台'))
    now = datetime.now(UTC)
    context = {'received_at': now.isoformat(), telemetry.CONTEXT_KEY: telemetry.initial_timing(now),
               'delay_seconds': 0., 'delay_reason': 'delay_disabled'}
    events = [e async for e in chat.stream_chat_response(flow.conversation, text, io.agent, 'u-1',
              reply_context=context, save_user_message=False, **kwargs)]
    assert sum(e['event'] == 'done' for e in events) == 1
    rows = await flow.db.message.find_many(where={'conversationId': flow.conversation, 'role': 'assistant'})
    assert rows
    ds = [r.metadata['response_diagnostics'] for r in rows]
    assert all(d['turn_received_at'] == now.isoformat() and d['planned_delay_ms'] == 0 for d in ds)
    assert all(d['aggregation_wait_ms'] == 0 and d['generation_ms'] >= 0 for d in ds)
    assert len({d['first_bubble_at'] for d in ds}) == 1
    detail = {'trace': {}}
    _attach_message_trace_metadata(detail, rows[0].metadata)
    assert detail['response_diagnostics']['first_bubble_at'] == ds[0]['first_bubble_at']
    if path == 'web_search': search.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_redis_aggregation_and_delayed_merge_use_last_receipt(flow, monkeypatch):
    from app.services.interaction import aggregation, delayed_queue
    monkeypatch.setattr(aggregation, 'get_redis', AsyncMock(return_value=flow.redis))
    monkeypatch.setattr(delayed_queue, 'get_redis', AsyncMock(return_value=flow.redis))
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def no_erasure(_): yield
    monkeypatch.setattr(aggregation.user_erasure, 'conversation_write_scope', no_erasure)
    for index in range(2):
        stamp = datetime.now(UTC)
        context = {'received_at': stamp.isoformat(), telemetry.CONTEXT_KEY: telemetry.initial_timing(stamp),
                   'delay_seconds': 0., 'delay_reason': 'delay_disabled'}
        assert await aggregation.push_turn_pending(agent_id=flow.agent, user_id=flow.user,
            conversation_id=flow.conversation, text=f'合成消息{index}', reply_context=context, message_id=f'm{index}')
    text, cid, ctx, mid = await aggregation.flush_turn_pending(agent_id=flow.agent, user_id=flow.user)
    assert ctx[telemetry.CONTEXT_KEY]['received_at'] == stamp.isoformat()
    assert ctx[telemetry.CONTEXT_KEY]['aggregation_wait_ms'] >= 0 and mid == 'm1'
    await delayed_queue.enqueue_or_append_delayed(cid, {'conversation_id': cid, 'message': text,
        'message_id': mid, 'reply_context': ctx}, 0.)
    merged = delayed_queue.merge_delayed_payloads(await delayed_queue.flush_due_delayed_messages(cid))
    timing = telemetry.TurnTiming().first_bubble(merged['reply_context'])
    assert timing['turn_received_at'] == stamp.isoformat()
    assert timing['delayed_queue_wait_ms'] >= 0
