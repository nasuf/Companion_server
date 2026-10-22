"""Real SQL/Redis page traversal and progress races, without provider calls.

These are progress guarantees, not exactly-once memory writes or durable replay.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from app.services.chat import post_process as p
from app.services.memory.recording import watermark as wm, diagnostics as d
from app.services.memory.recording.outcomes import MemoryProgressError, MemorySideOutcome
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def redis():
    # Do not reuse queue unit fixtures: their mocked erasure fence would hide
    # the real owner-deletion race this suite qualifies.
    url = os.environ['RUNTIME_JOB_TEST_REDIS_URL']
    assert urlsplit(url).hostname in {'127.0.0.1', 'localhost'} and urlsplit(url).path == '/15'
    client = Redis.from_url(url, decode_responses=True)
    await client.ping()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def progress(origins, redis, monkeypatch):
    db, uid, wid, _, msg, *_ = origins
    monkeypatch.setattr(p, 'db', db)
    monkeypatch.setattr(wm, 'get_redis', AsyncMock(return_value=redis))
    try:
        yield db, redis, uid, wid, msg
    finally:
        await redis.delete(*(wm._key(msg.conversationId, side) for side in ('user', 'ai')))


@pytest.mark.parametrize('count', [31, 100, 101])
@pytest.mark.parametrize('side', ['user', 'ai'])
async def test_same_millisecond_pages_are_complete_bounded_and_disjoint(progress, monkeypatch, count, side):
    db, redis, uid, wid, msg = progress
    ts = msg.createdAt + timedelta(seconds=1)
    role = 'user' if side == 'user' else 'assistant'
    rows = [{'id': str(uuid4()), 'conversationId': msg.conversationId, 'role': role,
             'content': f'合成时间事实 {i}', 'createdAt': ts} for i in range(count)]
    await db.message.create_many(data=list(reversed(rows)))
    # Actual old-format cursor: a fully consumed timestamp, not a new message ID.
    key = wm._key(msg.conversationId, side)
    await redis.set(key, msg.createdAt.isoformat(), ex=wm._TTL_SEC)
    selected = []
    async def extract(**kw):
        assert kw['statement_time'] == ts
        assert kw['side'] == side
        selected.extend(kw['evidence_message_ids'])
        return []
    monkeypatch.setattr(p, 'process_memory_pipeline', extract)
    iterations = 0
    while True:
        result = await p._pipeline_with_watermark(uid, [], msg.conversationId, side=side, workspace_id=wid)
        assert result.inspected_count <= 30 and result.source_count <= 30
        iterations += 1
        assert iterations <= 5
        if result.status == 'no_new_messages':
            assert result.has_more is False and result.progress == 'unchanged'
            break
        assert result.status == 'completed' and result.progress == 'advanced'
        assert result.has_more is (len(selected) < count)
        position = await wm.get_watermark(msg.conversationId, side)
        if len(selected) < count:
            assert position.message_id == selected[-1]
        else:
            assert position.message_id is None
            assert datetime.fromisoformat(await redis.get(key)) == ts  # completed cohorts are rollback-readable
    assert selected == sorted(r['id'] for r in rows)
    assert len(set(selected)) == count
    cursor = await wm.get_watermark(msg.conversationId, side)
    assert cursor.timestamp == ts and cursor.message_id is None
    assert await wm.get_watermark(msg.conversationId, 'ai' if side == 'user' else 'user') is None


async def test_excluded_page_advances_then_later_fact_is_extracted(progress, monkeypatch):
    from prisma import Json
    db, redis, uid, wid, msg = progress
    ts = msg.createdAt + timedelta(seconds=1)
    ids = sorted(str(uuid4()) for _ in range(31))
    await db.message.create_many(data=[{'id': mid, 'conversationId': msg.conversationId,
        'role': 'assistant', 'content': '排除文本' if i < 30 else '实际时间事实', 'createdAt': ts,
        'metadata': Json({'human_takeover': True}) if i < 30 else Json({})} for i, mid in enumerate(ids)])
    await wm.set_watermark(msg.conversationId, 'ai', msg.createdAt)
    extract = AsyncMock(return_value=[]); monkeypatch.setattr(p, 'process_memory_pipeline', extract)
    first = await p._pipeline_with_watermark(uid, [], msg.conversationId, side='ai', workspace_id=wid)
    assert first.status == 'excluded' and first.source_count == 0 and first.inspected_count == 30
    assert first.has_more is True
    extract.assert_not_awaited()
    second = await p._pipeline_with_watermark(uid, [], msg.conversationId, side='ai', workspace_id=wid)
    assert second.source_count == 1 and second.inspected_count == 1 and second.has_more is False
    assert extract.await_args.kwargs['evidence_message_ids'] == ids[-1:]
    assert '排除文本' not in str(extract.await_args)


async def test_first_window_remains_bounded_and_does_not_backfill(progress, monkeypatch):
    db, redis, uid, wid, msg = progress
    rows = [{'id': str(uuid4()), 'conversationId': msg.conversationId, 'role': 'user',
             'content': 'older not in initial window', 'createdAt': msg.createdAt - timedelta(days=i+1)} for i in range(35)]
    await db.message.create_many(data=rows)
    extract = AsyncMock(return_value=[]); monkeypatch.setattr(p, 'process_memory_pipeline', extract)
    window = [{'id': msg.id, 'role': msg.role, 'content': msg.content, 'createdAt': msg.createdAt}]
    result = await p._pipeline_with_watermark(uid, window, msg.conversationId, side='user', workspace_id=wid)
    assert result.cursor_kind == 'initial_window' and result.has_more is None and result.inspected_count == 1
    assert extract.await_args.kwargs['evidence_message_ids'] == [msg.id]
    second = await p._pipeline_with_watermark(uid, window, msg.conversationId, side='user', workspace_id=wid)
    assert second.status == 'no_new_messages' and extract.await_count == 1


async def test_page_ending_between_timestamp_groups_keeps_legacy_format(progress, monkeypatch):
    db, redis, uid, wid, msg = progress
    ts = msg.createdAt + timedelta(seconds=1)
    await db.message.create_many(data=[{'id': str(uuid4()), 'conversationId': msg.conversationId,
        'role': 'user', 'content': f'合成消息{i}', 'createdAt': ts + timedelta(seconds=i // 30)} for i in range(31)])
    await wm.set_watermark(msg.conversationId, 'user', msg.createdAt)
    extract = AsyncMock(return_value=[]); monkeypatch.setattr(p, 'process_memory_pipeline', extract)
    result = await p._pipeline_with_watermark(uid, [], msg.conversationId, side='user', workspace_id=wid)
    assert result.has_more is True and result.inspected_count == 30
    assert datetime.fromisoformat(await redis.get(wm._key(msg.conversationId, 'user'))) == ts
    result = await p._pipeline_with_watermark(uid, [], msg.conversationId, side='user', workspace_id=wid)
    assert result.inspected_count == 1 and result.has_more is False


async def test_competing_completions_expiry_and_deleted_key_cannot_rewind(progress):
    _, redis, _, _, msg = progress
    conv = msg.conversationId; ts = msg.createdAt
    await wm.set_watermark(conv, 'user', ts, message_id='001')
    before = await wm.get_watermark(conv, 'user')
    results = await asyncio.gather(*[wm.set_watermark(conv, 'user', ts, message_id=f'{i:03}', expected=before)
                                   for i in range(2, 12)], return_exceptions=True)
    assert sum(r is None for r in results) == 1
    assert all(r is None or isinstance(r, MemoryProgressError) and r.reason == 'watermark_conflict' for r in results)
    current = await wm.get_watermark(conv, 'user')
    assert current.message_id > '001'
    assert 0 < await redis.ttl(wm._key(conv, 'user')) <= wm._TTL_SEC
    for timestamp, mid in [(ts - timedelta(seconds=1), '999'), (ts, '000'), (ts, current.message_id)]:
        with pytest.raises(MemoryProgressError, match='watermark_conflict'):
            await wm.set_watermark(conv, 'user', timestamp, message_id=mid, expected=current)
    await redis.delete(wm._key(conv, 'user'))
    with pytest.raises(MemoryProgressError, match='watermark_conflict'):
        await wm.set_watermark(conv, 'user', ts + timedelta(seconds=1), message_id='999', expected=current)
    assert await wm.get_watermark(conv, 'user') is None
    # New initialization cannot be overwritten by a completion that read absence.
    await wm.set_watermark(conv, 'user', ts, message_id='001', expected=None)
    with pytest.raises(MemoryProgressError, match='watermark_conflict'):
        await wm.set_watermark(conv, 'user', ts + timedelta(seconds=1), message_id='999', expected=None)


@pytest.mark.parametrize('raw', ['2026-10-20T18:00:00', '2026-10-21T02:00:00+08:00', '2026-10-20T18:00:00Z'])
async def test_legacy_exact_raw_cas_and_full_cohort_semantics(progress, raw):
    _, redis, _, _, msg = progress
    key = wm._key(msg.conversationId, 'user')
    await redis.set(key, raw)
    before = await wm.get_watermark(msg.conversationId, 'user')
    assert before.timestamp == datetime(2026, 10, 20, 18, tzinfo=UTC)
    assert before.message_id is None and before.raw == raw
    assert wm.includes(before, before.timestamp, 'any-id')
    await wm.set_watermark(msg.conversationId, 'user', before.timestamp + timedelta(seconds=1), message_id='next', expected=before)
    current = await wm.get_watermark(msg.conversationId, 'user')
    assert current.message_id == 'next'
    # Prior binaries reject this shape; they must never silently skip the cohort.
    with pytest.raises(ValueError): datetime.fromisoformat(await redis.get(key))


@pytest.mark.parametrize('change', ['v', 'conversation', 'side', 'revision', 'timestamp', 'message_id', 'invalid_json', 'oversize', 'invalid_bytes'])
async def test_corrupt_or_cross_scope_cursor_is_not_absence(progress, change):
    _, redis, _, _, msg = progress
    key = wm._key(msg.conversationId, 'user')
    data = {'v': 2, 'conversation': msg.conversationId, 'side': 'user', 'revision': uuid4().hex,
            'timestamp': msg.createdAt.isoformat(), 'message_id': 'm'}
    raw = {'invalid_json': '{broken', 'oversize': 'x'*2049, 'invalid_bytes': b'\xff'}.get(change)
    if raw is None:
        data[change] = {'v': True, 'conversation': 'other', 'side': 'ai', 'revision': None,
                        'timestamp': '2026-10-22T01:00:00', 'message_id': 5}[change]
        raw = json.dumps(data)
    await redis.set(key, raw)
    with pytest.raises(MemoryProgressError, match='watermark_invalid'):
        await wm.get_watermark(msg.conversationId, 'user')


async def test_deleted_owner_cannot_create_initial_progress(progress):
    db, redis, uid, _, msg = progress
    await db.user.update(where={'id': uid}, data={'status': 'deleting'})
    with pytest.raises(MemoryProgressError, match='watermark_write_unavailable'):
        await wm.set_watermark(msg.conversationId, 'user', msg.createdAt, message_id=msg.id, expected=None)
    assert await redis.get(wm._key(msg.conversationId, 'user')) is None


async def test_cursor_conflict_keeps_returned_count_but_does_not_claim_progress(progress, monkeypatch):
    _, redis, uid, wid, msg = progress
    async def extract(**kw):
        await wm.set_watermark(msg.conversationId, 'user', msg.createdAt, message_id=msg.id, expected=None)
        return ['confirmed-result']
    monkeypatch.setattr(p, 'process_memory_pipeline', extract)
    result = await p._pipeline_with_watermark(uid, [{'id': msg.id, 'role': msg.role, 'content': msg.content,
        'createdAt': msg.createdAt}], msg.conversationId, side='user', workspace_id=wid)
    assert result.reason == 'watermark_conflict' and result.status == 'failed'
    assert result.memory_count == 1 and result.progress == 'unknown'
    assert d._safe_side(result)['reason'] == 'watermark_conflict'


def test_diagnostics_unknown_fields_are_not_success_or_private_data():
    safe = d._safe_side(MemorySideOutcome('user', 'completed', 'processed', cursor_kind='PRIVATE',
                                       inspected_count=True, has_more=0))
    assert safe['cursor_kind'] == 'unknown' and safe['inspected_count'] is None and safe['has_more'] is None
    assert 'PRIVATE' not in str(safe)
