"""Tests for the context selector."""

from app.services.memory.retrieval.context_selector import (
    ClassifiedMemory,
    select_context,
    split_by_source,
)


def test_select_context_within_budget():
    """Memories within token budget are included."""
    candidates = [
        {"id": "1", "summary": "Short memory one", "importance": 0.9, "created_at": "2025-01-01T00:00:00"},
        {"id": "2", "summary": "Short memory two", "importance": 0.8, "created_at": "2025-01-01T00:00:00", "source": "ai"},
    ]
    result = select_context(candidates, token_budget=800)
    assert len(result) == 2
    # select_context 现返回 ClassifiedMemory 数据类
    assert result[0].text == "Short memory one"
    assert result[0].source == "user"  # 默认 user
    assert result[1].text == "Short memory two"
    assert result[1].source == "ai"  # 透传上游


def test_split_by_source_basic():
    mems = [
        ClassifiedMemory(text="u1", relevance="strong", score=0.9, source="user"),
        ClassifiedMemory(text="a1", relevance="medium", score=0.5, source="ai"),
        ClassifiedMemory(text="u2", relevance="medium", score=0.5, source="user"),
    ]
    user_t, ai_t = split_by_source(mems)
    assert user_t == ["u1", "u2"]
    assert ai_t == ["a1"]


def test_split_by_source_handles_none_and_empty():
    assert split_by_source(None) == ([], [])
    assert split_by_source([]) == ([], [])


def test_select_context_empty():
    """Empty candidates returns empty list."""
    result = select_context([], token_budget=800)
    assert result == []


def test_select_context_budget_limit():
    """Stops adding when token budget exceeded."""
    # Each "word" ~1.3 tokens; create memories that will exceed budget
    long_text = "word " * 200  # ~260 tokens each
    candidates = [
        {"id": str(i), "summary": long_text, "importance": 0.5, "created_at": "2025-01-01T00:00:00"}
        for i in range(10)
    ]
    result = select_context(candidates, token_budget=400)
    # Should not include all 10
    assert len(result) < 10
