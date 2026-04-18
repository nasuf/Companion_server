"""Tests for the context selector."""

from app.services.memory.retrieval.context_selector import select_context


def test_select_context_within_budget():
    """Memories within token budget are included."""
    candidates = [
        {"id": "1", "summary": "Short memory one", "importance": 0.9, "created_at": "2025-01-01T00:00:00"},
        {"id": "2", "summary": "Short memory two", "importance": 0.8, "created_at": "2025-01-01T00:00:00"},
    ]
    result = select_context(candidates, token_budget=800)
    assert len(result) == 2
    assert "Short memory one" in result[0]


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
