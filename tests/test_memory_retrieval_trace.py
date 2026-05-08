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


def test_replace_latest_retrieval_selection_supersedes_old_sessions():
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval.trace import (
        record_retrieval_session,
        replace_latest_retrieval_selection,
        reset_retrieval_trace,
        snapshot_retrieval_traces,
        start_retrieval_trace,
    )

    token = start_retrieval_trace()
    try:
        record_retrieval_session(
            strategy="hybrid_l1_l2",
            query="那他呢",
            candidates=[{"id": "old", "summary": "旧 query 命中的记忆"}],
            selected=[{"id": "old", "summary": "旧 query 命中的记忆"}],
        )
        record_retrieval_session(
            strategy="hybrid_l1_l2",
            query="那他呢",
            enhanced_query="妈妈最近情况",
            candidates=[
                {"id": "m1", "summary": "用户妈妈最近住院", "rank_score": 0.8},
                {"id": "m2", "summary": "用户喜欢咖啡", "rank_score": 0.6},
            ],
            selected=[{"id": "m2", "summary": "用户喜欢咖啡"}],
        )
        replace_latest_retrieval_selection(
            strategy="hybrid_l1_l2",
            selected=[
                ClassifiedMemory(
                    id="m1",
                    text="用户妈妈最近住院",
                    relevance="strong",
                    score=0.8,
                )
            ],
            final_injected=True,
        )
        snapshot = snapshot_retrieval_traces()
    finally:
        reset_retrieval_trace(token)

    assert snapshot[0]["selected"] == []
    assert snapshot[0]["selected_count"] == 0
    assert snapshot[0]["notes"]["superseded_by_later_retrieval"] is True
    assert snapshot[1]["selected_count"] == 1
    assert snapshot[1]["selected"][0]["id"] == "m1"
    assert snapshot[1]["notes"]["final_injected"] is True
    assert snapshot[1]["candidates"][0]["selected"] is True
    assert snapshot[1]["candidates"][1]["selected"] is False


def test_replace_latest_retrieval_selection_can_mark_weak_gate_not_injected():
    from app.services.memory.retrieval.trace import (
        record_retrieval_session,
        replace_latest_retrieval_selection,
        reset_retrieval_trace,
        snapshot_retrieval_traces,
        start_retrieval_trace,
    )

    token = start_retrieval_trace()
    try:
        record_retrieval_session(
            strategy="hybrid_l1_l2",
            query="哈哈",
            candidates=[{"id": "m1", "summary": "候选记忆"}],
            selected=[{"id": "m1", "summary": "候选记忆"}],
        )
        replace_latest_retrieval_selection(
            strategy="hybrid_l1_l2",
            selected=[],
            final_injected=False,
        )
        snapshot = snapshot_retrieval_traces()
    finally:
        reset_retrieval_trace(token)

    assert snapshot[0]["selected"] == []
    assert snapshot[0]["selected_count"] == 0
    assert snapshot[0]["notes"]["final_injected"] is False
    assert snapshot[0]["candidates"][0]["selected"] is False


def test_build_retrieval_quality_analysis_marks_likely_used_and_warnings():
    from app.services.memory.retrieval.trace import build_retrieval_quality_analysis

    retrievals = [{
        "session_id": "hybrid_l1_l2_abc",
        "strategy": "hybrid_l1_l2",
        "query": "妈妈怎么样了",
        "enhanced_query": "用户妈妈最近住院情况",
        "raw_count": 5,
        "candidate_count": 5,
        "selected": [
            {
                "id": "m1",
                "source": "user",
                "text": "用户妈妈最近住院，用户很担心她",
                "score": 0.82,
                "rank_reasons": ["实体命中", "关键词命中"],
            },
            {
                "id": "m2",
                "source": "ai",
                "text": "AI 喜欢在晚上做手作",
                "score": 0.58,
                "rank_reasons": [],
            },
        ],
        "notes": {"final_injected": True},
    }]

    analysis = build_retrieval_quality_analysis(
        retrievals,
        assistant_reply="你妈妈住院这件事我记得，你担心很正常。",
        user_message="妈妈怎么样了",
    )

    assert analysis is not None
    assert analysis["session_count"] == 1
    assert analysis["selected_count"] == 2
    assert analysis["selected_user_count"] == 1
    assert analysis["selected_ai_count"] == 1
    assert analysis["likely_used_count"] == 1
    assert analysis["likely_unused_count"] == 1
    assert analysis["signal_counts"]["entity"] == 1
    assert analysis["signal_counts"]["keyword"] == 1
    assert analysis["signal_counts"]["enhanced_query"] == 1
    assert analysis["quality_metrics"]["visible_use_rate"] == 0.5
    assert analysis["quality_metrics"]["user_memory_share"] == 0.5
    assert analysis["quality_metrics"]["selection_rate"] == 0.4
    assert analysis["items"][0]["likely_used"] is True
    assert {"妈妈", "住院"} & set(analysis["items"][0]["matched_terms"])


def test_build_retrieval_quality_analysis_reports_weak_gate_drop():
    from app.services.memory.retrieval.trace import build_retrieval_quality_analysis

    analysis = build_retrieval_quality_analysis(
        [{
            "session_id": "hybrid_l1_l2_weak",
            "strategy": "hybrid_l1_l2",
            "query": "哈哈",
            "candidate_count": 1,
            "candidates": [{"id": "m1", "text": "候选记忆"}],
            "selected": [],
            "notes": {"final_injected": False},
        }],
        assistant_reply="哈哈",
        user_message="哈哈",
    )

    assert analysis is not None
    warning_codes = {item["code"] for item in analysis["warnings"]}
    assert "candidates_not_injected" in warning_codes
    assert "final_gate_dropped_candidates" in warning_codes
    assert analysis["quality_metrics"]["has_final_gate_drop"] is True


def test_build_memory_retrieval_feedback_detects_next_turn_correction():
    from app.services.memory.retrieval.trace import build_memory_retrieval_feedback

    feedback = build_memory_retrieval_feedback(
        user_message="不是啊，你记错了，我从来没说过我喜欢芒果。",
        previous_assistant_reply="我记得你喜欢芒果，所以给你推荐这个。",
        previous_metadata={
            "memory_retrieval_analysis": {
                "likely_used_count": 1,
                "items": [
                    {
                        "id": "m1",
                        "text": "用户喜欢芒果",
                        "likely_used": True,
                    }
                ],
            },
            "memory_retrievals": [
                {
                    "selected": [
                        {"id": "m1", "text": "用户喜欢芒果"},
                    ]
                }
            ],
        },
    )

    assert feedback is not None
    assert feedback["signal"] == "potential_memory_correction"
    assert feedback["confidence"] >= 0.9
    assert "记错" in feedback["matched_phrases"]
    assert feedback["memory_ids"] == ["m1"]


def test_build_memory_retrieval_feedback_ignores_plain_negation():
    from app.services.memory.retrieval.trace import build_memory_retrieval_feedback

    feedback = build_memory_retrieval_feedback(
        user_message="不是很舒服，今天有点累。",
        previous_assistant_reply="你今天怎么样？",
        previous_metadata={
            "memory_retrievals": [
                {"selected": [{"id": "m1", "text": "用户最近压力大"}]},
            ],
        },
    )

    assert feedback is None


def test_build_memory_retrieval_feedback_requires_retrieval_metadata():
    from app.services.memory.retrieval.trace import build_memory_retrieval_feedback

    feedback = build_memory_retrieval_feedback(
        user_message="你记错了。",
        previous_assistant_reply="我记得你喜欢咖啡。",
        previous_metadata={"trace_id": "t1"},
    )

    assert feedback is None
