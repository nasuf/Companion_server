"""Ranking must prefer the L2 dynamic score (current_score) over importance."""

from __future__ import annotations

from datetime import UTC, datetime

from app.services.memory.retrieval.ranking import rank_memory_candidate


def _mem(**kwargs):
    base = {
        "id": "m1",
        "summary": "用户在做一个副业项目",
        "content": "用户在做一个副业项目",
        "importance": 0.8,
        "similarity": 0.7,
        "source": "user",
        "main_category": "生活",
        "sub_category": "工作",
        "last_accessed_at": datetime.now(UTC).isoformat(),
        "mention_count": 0,
    }
    base.update(kwargs)
    return base


def test_current_score_used_as_ranking_base_when_present():
    decayed, _ = rank_memory_candidate(_mem(current_score=0.4), "副业项目怎么样了")
    fresh, _ = rank_memory_candidate(_mem(current_score=None), "副业项目怎么样了")
    # Decayed dynamic score must rank strictly below the fallback importance.
    assert decayed < fresh


def test_missing_or_bad_current_score_falls_back_to_importance():
    baseline, _ = rank_memory_candidate(_mem(), "副业项目怎么样了")
    absent, _ = rank_memory_candidate(_mem(current_score=None), "副业项目怎么样了")
    garbage, _ = rank_memory_candidate(_mem(current_score="oops"), "副业项目怎么样了")
    assert absent == baseline
    assert garbage == baseline
