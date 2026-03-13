"""Tests for the memory ranker."""

from datetime import datetime, timezone

from app.services.memory.ranker import rank_memories


def test_rank_memories_by_similarity():
    """Higher similarity scores rank higher."""
    candidates = [
        {"id": "1", "summary": "low sim", "similarity": 0.3, "importance": 0.5, "created_at": "2025-01-01T00:00:00"},
        {"id": "2", "summary": "high sim", "similarity": 0.95, "importance": 0.5, "created_at": "2025-01-01T00:00:00"},
    ]
    ranked = rank_memories(candidates, [], top_k=2)
    assert ranked[0]["id"] == "2"


def test_rank_memories_entity_boost():
    """Memories matching context entities get boosted."""
    candidates = [
        {"id": "1", "summary": "I like sushi and ramen", "similarity": 0.5, "importance": 0.5, "created_at": "2025-01-01T00:00:00"},
        {"id": "2", "summary": "went to the gym", "similarity": 0.5, "importance": 0.5, "created_at": "2025-01-01T00:00:00"},
    ]
    ranked = rank_memories(candidates, ["sushi"], top_k=2)
    assert ranked[0]["id"] == "1"


def test_rank_memories_top_k():
    """Only top_k results are returned."""
    candidates = [
        {"id": str(i), "summary": f"memory {i}", "similarity": 0.5, "importance": 0.5, "created_at": "2025-01-01T00:00:00"}
        for i in range(10)
    ]
    ranked = rank_memories(candidates, [], top_k=3)
    assert len(ranked) == 3


def test_rank_memories_empty():
    """Empty input returns empty output."""
    ranked = rank_memories([], [], top_k=10)
    assert ranked == []
