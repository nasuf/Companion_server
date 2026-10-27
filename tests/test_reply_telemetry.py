"""Clock boundaries and real graph paths, with deterministic external IO."""
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.interaction import reply_telemetry as telemetry
from app.services.interaction.reply_context import merge_reply_contexts
from app.services.interaction.delayed_queue import merge_delayed_payloads

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def context(at=NOW):
    return {"received_at": at.isoformat(), telemetry.CONTEXT_KEY: telemetry.initial_timing(at),
            "delay_seconds": 3.0, "delay_reason": "conversation_mode"}


def test_latest_receipt_and_observed_queue_waits_are_separate(monkeypatch):
    monkeypatch.setattr(telemetry, 'utc_now', lambda: NOW + timedelta(seconds=2))
    c = context()
    queued = telemetry.mark_queue(c, 'aggregation')
    assert 'aggregation_entered_at' not in c[telemetry.CONTEXT_KEY]  # fallback not mutated
    monkeypatch.setattr(telemetry, 'utc_now', lambda: NOW + timedelta(seconds=3.2))
    flushed = telemetry.mark_queue(queued, 'aggregation', leaving=True)
    assert flushed[telemetry.CONTEXT_KEY]['aggregation_wait_ms'] == 1200
    delayed = telemetry.mark_queue(flushed, 'delayed')
    turn = telemetry.TurnTiming(started_at=NOW + timedelta(seconds=7), started_clock=100)
    monkeypatch.setattr(telemetry, 'utc_now', lambda: NOW + timedelta(seconds=9))
    monkeypatch.setattr(telemetry.time, 'perf_counter', lambda: 102)
    d = turn.first_bubble(delayed)
    assert d['generation_ms'] == 2000
    assert d['pre_generation_wait_ms'] == 7000
    assert d['delayed_queue_wait_ms'] == 3800
    assert d['planned_delay_ms'] == 3000  # NOT actual 3800
    assert d['turn_received_at'] == NOW.isoformat()
    monkeypatch.setattr(telemetry.time, 'perf_counter', lambda: 120)
    assert turn.first_bubble(delayed) == d  # later bubble/save cannot move the clock


def test_merged_turn_uses_latest_message_measurement_not_first():
    early, late = context(), context(NOW + timedelta(seconds=4))
    early[telemetry.CONTEXT_KEY]['aggregation_wait_ms'] = 5000
    late[telemetry.CONTEXT_KEY]['aggregation_wait_ms'] = 1200
    merged = merge_delayed_payloads([
        {'message': '第一条', 'message_id': 'm1', 'reply_context': early},
        {'message': '第二条', 'message_id': 'm2', 'reply_context': late},
    ])['reply_context']
    assert merged['received_at'] == early['received_at']  # existing product contract
    assert merged[telemetry.CONTEXT_KEY] == late[telemetry.CONTEXT_KEY]
    d = telemetry.TurnTiming().first_bubble(merged)
    assert d['turn_received_at'] == late['received_at']
    assert d['aggregation_wait_ms'] == 1200
    assert merge_reply_contexts(early, {}) == early
    unknown_latest = merge_reply_contexts(early, {'received_at': late['received_at']})
    assert telemetry.TurnTiming().first_bubble(unknown_latest)['turn_received_at'] is None


@pytest.mark.parametrize('value', [None, '', 'bad', '2026-10-27T12:00:00', 3, False])
def test_unknown_and_invalid_receipt_is_never_replaced_with_now(value):
    c = {telemetry.CONTEXT_KEY: {'received_at': value}}
    d = telemetry.TurnTiming().first_bubble(c)
    assert d['turn_received_at'] is None
    assert d['aggregation_wait_ms'] is None
    assert d['planned_delay_ms'] is None
    assert d['delay_reason'] is None
    assert d['delayed_queue_wait_ms'] is None


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, '3'])
def test_invalid_duration_stays_unknown(value):
    d = telemetry.TurnTiming().first_bubble({'delay_seconds': value,
        telemetry.CONTEXT_KEY: {'aggregation_wait_ms': value}})
    assert d['planned_delay_ms'] is None and d['aggregation_wait_ms'] is None


def test_clock_reversal_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr(telemetry, 'utc_now', lambda: NOW - timedelta(seconds=2))
    flushed = telemetry.mark_queue({telemetry.CONTEXT_KEY: {'aggregation_entered_at': NOW.isoformat()}},
                                  'aggregation', leaving=True)
    assert flushed[telemetry.CONTEXT_KEY]['aggregation_wait_ms'] is None
    assert telemetry.mark_queue(None, 'aggregation') is None


@pytest.mark.asyncio
async def test_ingress_clock_does_not_change_existing_delay_policy_clock(monkeypatch):
    from app.services.interaction import reply_context
    from unittest.mock import Mock
    policy_time = NOW + timedelta(seconds=5)
    monkeypatch.setattr(reply_context, 'get_last_reply_timestamp', AsyncMock(return_value=None))
    policy = Mock(return_value={'delay_seconds': 3, 'delay_reason': 'conversation_mode'})
    monkeypatch.setattr(reply_context, 'compute_delay_profile', policy)
    result = await reply_context.build_reply_timing_context(agent_id='a', user_id='u',
        received_status={}, received_at=policy_time, telemetry_received_at=NOW)
    assert policy.call_args.kwargs['received_at'] == policy_time
    assert result['received_at'] == policy_time.isoformat()
    assert result[telemetry.CONTEXT_KEY]['received_at'] == NOW.isoformat()
    unknown = await reply_context.build_reply_timing_context(agent_id='a', user_id='u',
        received_status={}, received_at=policy_time)
    assert unknown[telemetry.CONTEXT_KEY]['received_at'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['main', 'tier', 'short_circuit', 'web_search'])
async def test_all_reply_paths_persist_turn_timing(monkeypatch, path):
    from tests.g02_harness_support import configure_pair
    from app.services.chat import orchestrator as chat, intent_handlers
    from app.services.chat.intent_dispatcher import IntentType
    from app.services import persona_grounding
    from app.services.chat import reply_generate
    real_main = reply_generate._run_main_llm
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(persona_grounding, 'verify_generated_locations', AsyncMock(return_value=[]))
    kwargs = {}
    if path in {'main', 'web_search'}:
        monkeypatch.setattr(chat, 'compute_reengagement_gap_seconds', lambda *_, **__: 14400.)
    if path == 'web_search':
        # Main generator still exercises search routing, external provider is stubbed.
        chat.fetch_parallel_context.return_value.needs_web_search = True
        from app.services.chat import reply_generate
        search = AsyncMock(return_value='合成联网答案')
        monkeypatch.setattr(reply_generate, '_try_web_search_reply', search)
        monkeypatch.setattr(reply_generate, '_run_main_llm', real_main)
    if path == 'short_circuit':
        kwargs['forced_intent'] = IntentType.CURRENT_STATE
        monkeypatch.setattr(intent_handlers, 'current_state_reply', AsyncMock(return_value='正在整理工作台呀'))
    text = '你在干嘛' if path == 'short_circuit' else '今天有点累了'
    events = [e async for e in chat.stream_chat_response('c-1', text, io.agent, 'u-1',
              reply_context=context(), **kwargs)]
    assert sum(e['event'] == 'done' for e in events) == 1
    payloads = chat._save_replies.await_args.args[1]
    ds = [p['response_diagnostics'] for p in payloads]
    assert ds and all(d['turn_received_at'] == NOW.isoformat() for d in ds)
    assert all(d['generation_ms'] >= 0 and d['planned_delay_ms'] == 3000 for d in ds)
    assert len({d['first_bubble_at'] for d in ds}) == 1
    if path == 'tier':
        assert len(io.tier_calls) == 1
    if path == 'web_search':
        search.assert_awaited_once()
    if path == 'short_circuit':
        intent_handlers.current_state_reply.assert_awaited_once()


def test_first_streamed_bubble_not_delayed_by_later_bubbles_or_persistence(monkeypatch):
    import json
    from app.services.chat.graph_runtime import ChatGraphContext
    frame = SimpleNamespace(reply_context=context(), response_diagnostics={})
    owner = ChatGraphContext(frame, 'synthetic', False,
        timing=telemetry.TurnTiming(started_at=NOW, started_clock=100))
    monkeypatch.setattr(telemetry, 'utc_now', lambda: NOW + timedelta(seconds=1))
    monkeypatch.setattr(telemetry.time, 'perf_counter', lambda: 101)
    owner.note_event({'event': 'reply', 'data': json.dumps({'index': 0, 'text': '第一条'})})
    monkeypatch.setattr(telemetry.time, 'perf_counter', lambda: 120)
    owner.note_event({'event': 'reply', 'data': json.dumps({'index': 1, 'text': '第二条'})})
    assert frame.response_diagnostics['generation_ms'] == 1000
