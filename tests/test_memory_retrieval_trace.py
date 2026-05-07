"""Tests for local memory retrieval trace collection."""

from __future__ import annotations

from datetime import datetime


def test_record_retrieval_session_serializes_and_resets():
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval.trace import (
        record_retrieval_session,
        reset_retrieval_trace,
        snapshot_retrieval_traces,
        start_retrieval_trace,
    )

    token = start_retrieval_trace()
    try:
        session_id = record_retrieval_session(
            strategy="hybrid_l1_l2",
            query="我又睡不着了",
            workspace_id="w1",
            raw_count=2,
            candidate_count=2,
            candidates=[
                {
                    "id": "m1",
                    "source": "user",
                    "level": 1,
                    "summary": "用户长期失眠，夜里容易焦虑",
                    "importance": 0.92,
                    "similarity": 0.83,
                    "rank_score": 0.88,
                    "rank_reasons": ["keyword:失眠"],
                    "updated_at": "2026-05-08T10:00:00",
                },
                {"id": "m2", "summary": "用户喜欢咖啡"},
            ],
            selected=[
                ClassifiedMemory(
                    id="m1",
                    text="用户长期失眠，夜里容易焦虑",
                    relevance="strong",
                    score=0.88,
                    importance=0.92,
                    similarity=0.83,
                    source="user",
                    created_at=datetime(2026, 5, 1, 8, 0, 0),
                    last_accessed_at=datetime(2026, 5, 8, 10, 0, 0),
                    rank_reasons=["keyword:失眠"],
                )
            ],
            notes={"similarity_threshold": 0.45},
        )
        snapshot = snapshot_retrieval_traces()
    finally:
        reset_retrieval_trace(token)

    assert session_id and session_id.startswith("hybrid_l1_l2_")
    assert len(snapshot) == 1
    session = snapshot[0]
    assert session["workspace_id"] == "w1"
    assert session["raw_count"] == 2
    assert session["candidate_count"] == 2
    assert session["selected_count"] == 1
    assert session["notes"] == {"similarity_threshold": 0.45}
    assert session["candidates"][0]["selected"] is True
    assert session["candidates"][0]["score"] == 0.88
    assert session["selected"][0]["id"] == "m1"
    assert session["selected"][0]["last_accessed_at"].startswith("2026-05-08T10:00:00")
    assert snapshot_retrieval_traces() == []


def test_record_retrieval_session_noops_when_not_started():
    from app.services.memory.retrieval.trace import record_retrieval_session

    assert record_retrieval_session(strategy="hybrid_l1_l2", query="hi") is None


def test_memory_trace_item_handles_cached_string_payload():
    from app.services.memory.retrieval.trace import memory_trace_item

    item = memory_trace_item("ClassifiedMemory(text='用户最近压力很大')", selected=True)

    assert item["text"] == "ClassifiedMemory(text='用户最近压力很大')"
    assert item["selected"] is True
    assert item["score"] is None
