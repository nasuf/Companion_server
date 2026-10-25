"""U8 release gate: real Redis, no models or production traffic."""
import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest
from redis.asyncio import Redis

from app.services.interaction import delayed_queue as q


@pytest.fixture
async def isolated(monkeypatch, unit_media_scope):
    url = os.environ.get("RUNTIME_JOB_TEST_REDIS_URL", "")
    if not url:
        pytest.skip("explicit isolated Redis required")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost"}
    assert os.environ.get("APP_ENV") != "production"
    redis = Redis.from_url(url, decode_responses=True)
    prefix = "test:delay-gate:" + uuid.uuid4().hex + ":"
    for field in ("_DELAYED_LIST_KEY", "_DELAYED_ZSET_KEY", "_LOCK_KEY", "_INFLIGHT_KEY"):
        monkeypatch.setattr(q, field, prefix + getattr(q, field))
    monkeypatch.setattr(q, "get_redis", AsyncMock(return_value=redis))
    try:
        yield redis
    finally:
        keys = [key async for key in redis.scan_iter(match=prefix + "*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


def payload(text="第一句", ident="msg-1"):
    return {"conversation_id": "conv", "user_id": "user", "agent_id": "agent",
            "message": text, "message_id": ident, "reply_context": {"delay_seconds": 0}}


async def test_locked_conversation_must_remain_queued(isolated, monkeypatch):
    from jobs import scheduler
    monkeypatch.setattr(scheduler, "scan_due_user_turns", AsyncMock(return_value=[]))
    await q.enqueue_or_append_delayed("conv", payload(), 0)
    token = await q.try_lock_conversation("conv")
    assert token
    try:
        await scheduler._run_aggregation_scan_body()
        assert await isolated.zcard(q._DELAYED_LIST_KEY.format(cid="conv")) == 1
        assert await q.has_pending_delayed_messages("conv")
    finally:
        await q.unlock_conversation("conv", token)


async def test_enqueue_error_must_reply_without_queued(monkeypatch, unit_media_scope):
    from app.api.realtime import ws
    from app.services.chat import orchestrator
    from app.services.runtime.ws_manager import manager
    broken = AsyncMock()
    broken.eval.side_effect = RuntimeError("isolated write rejected")
    monkeypatch.setattr(q, "get_redis", AsyncMock(return_value=broken))
    captured = []
    async def generate(**kw):
        captured.append(kw)
        yield {"event": "reply", "data": '{"text":"hello back"}'}
        yield {"event": "done", "data": '{"message_id":"reply"}'}
    monkeypatch.setattr(orchestrator, 'stream_chat_response', generate)
    events = []
    async def send(cid, event, data):
        events.append({'type': event, 'data': data})
    monkeypatch.setattr(manager, 'send_event', send)
    socket = AsyncMock()
    await ws._queue_reply_or_error(socket, conversation_id="conv", agent=SimpleNamespace(id="agent"),
        user_id="user", user_message="hello", user_message_id="msg", reply_context={"delay_seconds": 3})
    events += [c.args[0] for c in socket.send_json.await_args_list]
    assert [e['type'] for e in events] == ['reply', 'done']
    assert len(captured) == 1
    assert captured[0]['user_message_id'] == 'msg'
    assert captured[0]['save_user_message'] is False
    assert captured[0]['reply_context']['delay_seconds'] == 0


async def test_rapid_messages_keep_arrival_order_and_merge_once(isolated):
    await q.enqueue_or_append_delayed("conv", payload("甲", "a"), 0)
    await q.enqueue_or_append_delayed("conv", payload("乙", "b"), 0)
    first, second = await asyncio.gather(q.scan_due_delayed_messages(), q.scan_due_delayed_messages())
    all_batches = first + second
    assert len(all_batches) == 1
    merged = q.merge_delayed_payloads(all_batches[0][1])
    assert merged["queued_messages"] == ["甲", "乙"]
    assert merged["reply_context"]["turn_message_ids"] == ["a", "b"]


@pytest.fixture
async def delivery(monkeypatch, isolated):
    from jobs import scheduler as s
    from app.api.realtime import ws
    from app.services.chat import orchestrator
    from app.services.runtime.ws_manager import manager
    monkeypatch.setattr(s, '_delayed_reply_tasks', {})
    monkeypatch.setattr(s, 'scan_due_user_turns', AsyncMock(return_value=[]))
    monkeypatch.setattr(s, '_already_covered', AsyncMock(return_value=False))
    monkeypatch.setattr(s, '_record_job_outcome', AsyncMock())
    monkeypatch.setattr('app.db.db', SimpleNamespace(conversation=SimpleNamespace(
        find_unique=AsyncMock(return_value=SimpleNamespace(agent=SimpleNamespace(id='agent'))))))
    monkeypatch.setattr(manager, 'send_event', AsyncMock())
    generated = []
    release = asyncio.Event()
    async def generate(**kw):
        generated.append(kw)
        await release.wait()
        yield {'event': 'done', 'data': '{"message_id":"reply"}'}
    monkeypatch.setattr(orchestrator, 'stream_chat_response', generate)
    try:
        yield SimpleNamespace(s=s, generated=generated, release=release, manager=manager)
    finally:
        release.set()
        await s.drain_delayed_replies(timeout=1)


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.005)


async def test_due_conversations_generate_concurrently_and_capacity_stays_in_redis(delivery, isolated, monkeypatch):
    s = delivery.s
    monkeypatch.setattr(s, '_DELAYED_REPLY_CONCURRENCY', 2)
    for cid in ['a', 'b', 'c']:
        await q.enqueue_or_append_delayed(cid, {**payload(ident=cid), 'conversation_id': cid}, 0)
    await asyncio.wait_for(s._run_aggregation_scan_body(), .5)
    await wait_for(lambda: len(delivery.generated) == 2)
    assert {x['conversation_id'] for x in delivery.generated} == {'a', 'b'}
    assert await q.has_pending_delayed_messages('c')
    # A blocked model must not block another scan / short-message aggregation.
    await asyncio.wait_for(s._run_aggregation_scan_body(), .5)
    assert s.scan_due_user_turns.await_count == 2
    assert len(delivery.generated) == 2
    for generated in delivery.generated:
        assert generated['save_user_message'] is False
        timing = generated['reply_context']['delayed_delivery_timings'][0]
        assert 0 <= timing['due_to_start_seconds'] < .5
    delivery.release.set()
    await s.drain_delayed_replies()
    await asyncio.sleep(0)
    await s._run_aggregation_scan_body()
    await wait_for(lambda: len(delivery.generated) == 3)
    await s.drain_delayed_replies()
    assert await isolated.exists(q._LOCK_KEY.format(cid='a')) == 0


async def test_competing_claims_same_conversation_only_generate_once(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    first = asyncio.create_task(delivery.s._run_due_delayed_reply('conv'))
    await wait_for(lambda: len(delivery.generated) == 1)
    second = asyncio.create_task(delivery.s._run_due_delayed_reply('conv'))
    await asyncio.wait_for(second, .5)
    assert len(delivery.generated) == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))
    assert not await q.is_reply_inflight('conv')


async def test_cancel_before_dispatch_starts_keeps_queue(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    task = asyncio.create_task(delivery.s._run_due_delayed_reply('conv'))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert await q.has_pending_delayed_messages('conv')
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))


async def test_generation_failure_releases_lease_and_notifies(delivery, isolated, monkeypatch):
    from app.api.realtime import ws
    monkeypatch.setattr(ws, 'stream_to_ws', AsyncMock(side_effect=RuntimeError('model failed')))
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    await delivery.s._run_due_delayed_reply('conv')
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))
    assert not await q.is_reply_inflight('conv')
    assert any(c.args[1] == 'done' and c.args[2].get('message_id') == 'error'
               for c in delivery.manager.send_event.await_args_list)
    delivery.s._record_job_outcome.assert_awaited_once_with('delayed_reply', ok=False, detail='RuntimeError')


async def test_lease_renews_and_lost_owner_does_not_delete_new_owner(isolated):
    token = await q.try_lock_conversation('conv', ttl=1)
    entered = asyncio.Event()
    async def owning():
        try:
            async with q.maintain_conversation_lock('conv', token, ttl=1, interval=.1):
                await q.mark_reply_inflight('conv', ttl=1, token=token)
                entered.set()
                await asyncio.sleep(5)
        finally:
            await q.clear_reply_inflight('conv', token=token)
            await q.unlock_conversation('conv', token)
    task = asyncio.create_task(owning())
    await entered.wait()
    await asyncio.sleep(1.15)
    assert await q.try_lock_conversation('conv') is None
    assert await q.is_reply_inflight('conv')
    await isolated.set(q._LOCK_KEY.format(cid='conv'), 'new-owner', ex=5)
    await isolated.set(q._INFLIGHT_KEY.format(cid='conv'), 'new-owner', ex=5)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert await isolated.get(q._LOCK_KEY.format(cid='conv')) == 'new-owner'
    assert await isolated.get(q._INFLIGHT_KEY.format(cid='conv')) == 'new-owner'


async def test_appended_payload_records_effective_due_not_new_delay(isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    await q.enqueue_or_append_delayed('conv', payload('second', 'msg-2'), 30)
    rows = await q.flush_due_delayed_messages('conv')
    assert len(rows) == 2
    assert rows[0]['due_at'] == rows[1]['due_at']


async def test_shutdown_cancels_owned_generations_and_releases_lease(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    await delivery.s._run_aggregation_scan_body()
    await wait_for(lambda: len(delivery.generated) == 1)
    await delivery.s.drain_delayed_replies(timeout=0)
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))
    assert not delivery.s._delayed_reply_tasks


async def test_locked_conversations_do_not_starve_other_due_work(delivery, isolated, monkeypatch):
    monkeypatch.setattr(delivery.s, '_DELAYED_REPLY_CONCURRENCY', 2)
    for cid in ['a', 'b', 'c']:
        await q.enqueue_or_append_delayed(cid, {**payload(), 'conversation_id': cid}, 0)
    for cid in ['a', 'b']:
        assert await q.try_lock_conversation(cid)
    await delivery.s._run_aggregation_scan_body()
    await wait_for(lambda: len(delivery.generated) == 1)
    assert delivery.generated[0]['conversation_id'] == 'c'
    assert await q.has_pending_delayed_messages('a')
    assert await q.has_pending_delayed_messages('b')


async def test_queue_authorization_rejection_never_generates(monkeypatch):
    from fastapi import HTTPException
    from app.api.realtime import ws
    from app.services.chat import orchestrator
    generate = AsyncMock()
    monkeypatch.setattr(orchestrator, 'stream_chat_response', generate)
    monkeypatch.setattr(ws, 'enqueue_or_append_delayed', AsyncMock(side_effect=HTTPException(401)))
    with pytest.raises(HTTPException):
        await ws._queue_reply(AsyncMock(), conversation_id='conv', agent=SimpleNamespace(id='agent'),
                              user_id='user', user_message='hello', user_message_id='msg',
                              reply_context={'delay_seconds': 3})
    generate.assert_not_called()


async def test_queue_notification_failure_does_not_generate_again(monkeypatch):
    from app.api.realtime import ws
    from app.services.chat import orchestrator
    generate = AsyncMock()
    monkeypatch.setattr(orchestrator, 'stream_chat_response', generate)
    monkeypatch.setattr(ws, 'enqueue_or_append_delayed', AsyncMock(return_value=False))
    socket = AsyncMock()
    socket.send_json.side_effect = ConnectionError('disconnected')
    with pytest.raises(ConnectionError):
        await ws._queue_reply(socket, conversation_id='conv', agent=SimpleNamespace(id='agent'),
                              user_id='user', user_message='hello', user_message_id='msg',
                              reply_context={'delay_seconds': 3})
    generate.assert_not_called()


@pytest.mark.parametrize('case', ['empty', 'covered', 'missing_agent', 'notify_error'])
async def test_delivery_boundaries_release_owned_lock(delivery, isolated, monkeypatch, case):
    if case != 'empty':
        await q.enqueue_or_append_delayed('conv', payload(), 0)
    if case == 'covered':
        monkeypatch.setattr(delivery.s, '_already_covered', AsyncMock(return_value=True))
    elif case == 'missing_agent':
        monkeypatch.setattr('app.db.db.conversation.find_unique', AsyncMock(return_value=None))
    elif case == 'notify_error':
        delivery.manager.send_event.side_effect = RuntimeError('transport failure')
    await delivery.s._run_due_delayed_reply('conv')
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))
    assert not await q.is_reply_inflight('conv')
    assert not delivery.generated


def test_missing_legacy_due_is_unknown_not_zero(caplog):
    from jobs import scheduler
    merged = {'reply_context': {}}
    scheduler._record_delayed_generation_start('conv', [payload()], merged)
    assert merged['reply_context']['delayed_delivery_timings'][0]['due_to_start_seconds'] is None


async def test_later_message_waits_for_own_conversation_but_not_other_conversations(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    await delivery.s._run_aggregation_scan_body()
    await wait_for(lambda: len(delivery.generated) == 1)
    await q.enqueue_or_append_delayed('conv', payload('later', 'msg-2'), 0)
    await delivery.s._run_aggregation_scan_body()
    # A competing worker uses the same distributed lease, not the local registry.
    await delivery.s._run_due_delayed_reply('conv')
    assert len(delivery.generated) == 1
    assert await q.has_pending_delayed_messages('conv')
    delivery.release.set()
    await delivery.s.drain_delayed_replies()
    await delivery.s._run_aggregation_scan_body()
    await wait_for(lambda: len(delivery.generated) == 2)
    assert delivery.generated[-1]['user_message_id'] == 'msg-2'


async def test_lease_redis_failure_cancels_owner_before_expiry(isolated, monkeypatch):
    token = await q.try_lock_conversation('conv', ttl=1)
    broken = AsyncMock(side_effect=ConnectionError('renew unavailable'))
    monkeypatch.setattr(isolated, 'eval', broken)
    loop = asyncio.get_running_loop()
    async def owning():
        async with q.maintain_conversation_lock('conv', token, ttl=1, interval=.05):
            await asyncio.sleep(5)
    started = loop.time()
    task = asyncio.create_task(owning())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    # Retried while the lease was still valid, cancelled before it could expire.
    assert broken.await_count > 1
    assert .5 <= loop.time() - started < 1


async def test_transient_lease_renewal_failure_keeps_owner(isolated, monkeypatch):
    token = await q.try_lock_conversation('conv', ttl=1)
    real_eval = isolated.eval
    calls = []
    async def flaky(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError('one slow renewal')
        return await real_eval(*args, **kwargs)
    monkeypatch.setattr(isolated, 'eval', flaky)
    async def owning():
        async with q.maintain_conversation_lock('conv', token, ttl=1, interval=.1):
            await asyncio.sleep(1.3)
    await asyncio.wait_for(owning(), 2)
    assert len(calls) > 2
    assert await isolated.get(q._LOCK_KEY.format(cid='conv')) == token
    await q.unlock_conversation('conv', token)


def done_errors(manager):
    return [c for c in manager.send_event.await_args_list
            if c.args[1] == 'done' and c.args[2].get('message_id') == 'error']


async def test_cancel_after_dequeue_reports_failure(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    task = asyncio.create_task(delivery.s._run_due_delayed_reply('conv'))
    await wait_for(lambda: len(delivery.generated) == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(done_errors(delivery.manager)) == 1
    assert not await isolated.exists(q._LOCK_KEY.format(cid='conv'))


async def test_cancel_before_dequeue_reports_nothing(delivery, isolated):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    task = asyncio.create_task(delivery.s._run_due_delayed_reply('conv'))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not done_errors(delivery.manager)
    assert await q.has_pending_delayed_messages('conv')


async def test_lock_error_before_dequeue_keeps_queue_without_false_failure(delivery, isolated, monkeypatch):
    await q.enqueue_or_append_delayed('conv', payload(), 0)
    monkeypatch.setattr(delivery.s, 'try_lock_conversation',
                        AsyncMock(side_effect=ConnectionError('lock unavailable')))
    await delivery.s._run_due_delayed_reply('conv')
    assert not done_errors(delivery.manager)
    assert await q.has_pending_delayed_messages('conv')
    delivery.s._record_job_outcome.assert_awaited_once_with('delayed_reply', ok=False, detail='ConnectionError')
