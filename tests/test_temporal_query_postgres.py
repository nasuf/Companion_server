"""Real PG date filtering and ownership; mock cache, owned synthetic rows only."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from tests.test_memory_evidence_postgres import origins
from app.services.temporal_query import resolve_temporal_query


@pytest.mark.asyncio
async def test_original_message_anchor_filters_real_pg_events_without_rewriting(origins, monkeypatch):
    from app.services.memory.retrieval import hybrid, vector_search
    from app.services.chat.temporal_context import build_turn_clock
    db, uid, wid, mid, message, _, _, users, agents, spaces, convs = origins
    from datetime import timedelta
    now = datetime(2026, 10, 22, 1, tzinfo=timezone.utc)
    original = now - timedelta(days=2)
    message = await db.message.update(where={"id":message.id}, data={"createdAt":original, "content":"昨天去博物馆"})
    await db.usermemory.update(where={"id":mid}, data={"content":"目标日期去博物馆", "occurTime":original-timedelta(days=1)})
    wrong = await db.usermemory.create(data={"userId":uid,"workspaceId":wid,"content":"执行日前一天的其他事件","level":2,"occurTime":now-timedelta(days=1)})
    foreign = await db.usermemory.create(data={"userId":users[1],"workspaceId":spaces[2],"content":"另一个用户的事件","level":2,"occurTime":original-timedelta(days=1)})
    monkeypatch.setattr(vector_search, "db", db)
    monkeypatch.setattr(hybrid, "search_similar_tiers", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid, "search_related_memories_for_query", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid, "cache_set_retrieval", AsyncMock())
    clock = build_turn_clock([{"id":message.id,"role":"user","createdAt":message.createdAt}], exclude_ids={message.id}, now=now)
    query = resolve_temporal_query(message.content, anchor=clock.statement_at, anchor_source=clock.anchor_source)
    result = await hybrid.hybrid_retrieve(message.content, uid, workspace_id=wid, temporal_query=query)
    ids = {m.id for m in result["memories"] or []}
    assert mid in ids and wrong.id not in ids and foreign.id not in ids
    assert (await db.message.find_unique(where={"id":message.id})).content == "昨天去博物馆"
    assert (await db.usermemory.find_unique(where={"id":mid})).occurTime.date() == (original-timedelta(days=1)).date()


@pytest.mark.asyncio
@pytest.mark.parametrize('text,zone,start_text,end_text', [
    ('上个月', 'Asia/Shanghai', '2026-09-01T00:00:00', '2026-10-01T00:00:00'),
    ('昨天上午', 'Asia/Shanghai', '2026-10-20T09:00:00', '2026-10-20T12:00:00'),
    ('昨天9点30分', 'Asia/Shanghai', '2026-10-20T09:30:00', '2026-10-20T09:31:00'),
    ('2024年2月29日', 'Asia/Shanghai', '2024-02-29T00:00:00', '2024-03-01T00:00:00'),
    ('2026年3月8日', 'America/New_York', '2026-03-08T00:00:00', '2026-03-09T00:00:00'),
    ('2026年11月1日', 'America/New_York', '2026-11-01T00:00:00', '2026-11-02T00:00:00'),
])
async def test_pg_half_open_boundaries_include_final_millisecond_and_keep_scope(
        origins, monkeypatch, text, zone, start_text, end_text):
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    from app.services.memory.retrieval import vector_search
    db, uid, wid, _, _, _, _, users, _, spaces, _ = origins
    monkeypatch.setattr(vector_search, 'db', db)
    columns = await db.query_raw("SELECT datetime_precision FROM information_schema.columns WHERE table_name IN ('memories_user','memories_ai') AND column_name='occur_time'")
    assert len(columns) == 2 and all(c['datetime_precision'] == 3 for c in columns)
    q = resolve_temporal_query(text, anchor='2026-12-01T12:00:00Z' if '11月' in text else '2026-10-21T01:00:00Z',
                              anchor_source='persisted_message', timezone_name=zone)
    expected_start, expected_end = [datetime.fromisoformat(t).replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc)
                                    for t in [start_text, end_text]]
    assert q.retrieval_range() == (expected_start, expected_end)
    expected = {'user':set(), 'ai':set()}
    all_ids=[]
    for side, table, model in [('user','memories_user',db.usermemory),('ai','memories_ai',db.aimemory)]:
        for index, stamp in enumerate([expected_start-timedelta(milliseconds=1), expected_start,
                                       expected_end-timedelta(milliseconds=1), expected_end]):
            row=await model.create(data={'userId':uid,'workspaceId':wid,'content':f'合成边界{side}{index}', 'level':2})
            # Use the actual TIMESTAMP(3) storage precision. Sub-millisecond
            # fixtures round to the next boundary and would test different data.
            await db.execute_raw(f'UPDATE {table} SET occur_time=$2::text::timestamp WHERE id=$1',
                                 row.id, stamp.replace(tzinfo=None).isoformat())
            all_ids.append(row.id)
            if index in {1,2}: expected[side].add(row.id)
        for other_uid, other_wid, archived in [(users[1],spaces[2],False),(uid,spaces[1],False),(uid,wid,True)]:
            await model.create(data={'userId':other_uid,'workspaceId':other_wid,'content':'隔离边界',
                                     'level':2,'occurTime':expected_start,'isArchived':archived})
    for side in ['user','ai',None]:
        rows=await vector_search.search_by_time_range(uid,*q.retrieval_range(),source=side,workspace_id=wid,limit=50)
        wanted=expected[side] if side else expected['user']|expected['ai']
        assert {r['id'] for r in rows} == wanted
    # Retrieval has no write side effects; all synthetic boundary rows remain.
    assert await db.usermemory.count(where={'id':{'in':all_ids}}) == 4
    assert await db.aimemory.count(where={'id':{'in':all_ids}}) == 4


@pytest.mark.asyncio
async def test_pg_aggregation_keeps_original_window_and_rejects_partial_clock(origins):
    from app.services.chat.temporal_context import build_turn_clock
    db, _, _, _, message, *_ = origins
    first = datetime(2026, 10, 20, 1, tzinfo=timezone.utc)
    last = datetime(2026, 10, 20, 2, tzinfo=timezone.utc)
    now = datetime(2026, 10, 22, 1, tzinfo=timezone.utc)
    await db.message.update(where={"id": message.id}, data={"createdAt": first})
    second = await db.message.create(data={"conversationId": message.conversationId,
        "role": "user", "content": "一小时前的事情", "createdAt": last})
    records = await db.message.find_many(where={"id": {"in": [message.id, second.id]}}, order={"createdAt": "desc"})
    rows = [{"id": r.id, "role": r.role, "createdAt": r.createdAt} for r in records]
    clock = build_turn_clock(rows, exclude_ids={message.id, second.id}, now=now, received_at=now)
    assert clock.statement_at == first and clock.statement_end_at == last
    q = resolve_temporal_query(second.content, anchor=clock.statement_at,
        anchor_source=clock.anchor_source, anchor_end=clock.statement_end_at)
    assert q.reason == "multiple_statement_instants" and q.retrieval_range() is None
    partial = build_turn_clock(rows[:1], exclude_ids={message.id, second.id}, now=now, received_at=now)
    assert partial.statement_at is None and partial.statement_evidence.missing == 1
    unchanged = await db.message.find_many(where={"id": {"in": [message.id, second.id]}}, order={"createdAt": "desc"})
    assert [(r.id, r.createdAt, r.content) for r in unchanged] == [(r.id, r.createdAt, r.content) for r in records]
