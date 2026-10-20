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
