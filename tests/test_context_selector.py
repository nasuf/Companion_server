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


def test_select_context_protects_safety_memory_from_top10_truncation():
    """Safety/emotional user memories must not be silently pushed to #11."""
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 人设记忆 {i}",
            "source": "ai",
            "importance": 1.0,
            "similarity": 0.9,
            "rank_score": 0.95 - i * 0.01,
        }
        for i in range(10)
    ]
    candidates.append({
        "id": "safety",
        "summary": "用户表达过强烈负面情绪，有轻生念头",
        "source": "user",
        "main_category": "情绪",
        "sub_category": "悲伤",
        "importance": 0.95,
        "similarity": 0.54,
        "rank_score": 0.42,
        "rank_reasons": ["安全/情绪相关"],
    })

    result = select_context(candidates, token_budget=800, max_items=10)

    assert len(result) == 10
    safety = next((m for m in result if m.id == "safety"), None)
    assert safety is not None
    assert "保护槽:安全情绪" in (safety.rank_reasons or [])
    assert sum(1 for m in result if m.source == "ai") == 9


def test_select_context_keeps_user_memory_floor_when_ai_scores_dominate():
    """User-facing companion chat should not inject only AI persona memories."""
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 高分记忆 {i}",
            "source": "ai",
            "rank_score": 0.98 - i * 0.01,
        }
        for i in range(6)
    ] + [
        {
            "id": f"user-{i}",
            "summary": f"用户相关事实 {i}",
            "source": "user",
            "rank_score": 0.52 - i * 0.01,
        }
        for i in range(3)
    ]

    result = select_context(candidates, token_budget=800, max_items=6)

    assert len(result) == 6
    user_memories = [m for m in result if m.source == "user"]
    assert len(user_memories) == 3
    assert all("保护槽:用户记忆" in (m.rank_reasons or []) for m in user_memories)


def test_select_context_protects_literal_user_keyword_match():
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 高分记忆 {i}",
            "source": "ai",
            "rank_score": 0.9 - i * 0.01,
        }
        for i in range(5)
    ]
    candidates.append({
        "id": "wife-surgery",
        "summary": "用户的妻子之前做过手术",
        "source": "user",
        "rank_score": 0.4,
        "rank_reasons": ["关键词命中"],
    })

    result = select_context(candidates, token_budget=800, max_items=5)

    assert any(m.id == "wife-surgery" for m in result)
    matched = next(m for m in result if m.id == "wife-surgery")
    assert "保护槽:字面命中" in (matched.rank_reasons or [])


def test_select_context_protects_named_relation_memory_before_self_name():
    candidates = [
        {
            "id": f"safety-{i}",
            "summary": f"用户表达过强烈负面情绪 {i}",
            "source": "user",
            "main_category": "情绪",
            "sub_category": "悲伤",
            "importance": 0.9,
            "rank_score": 1.2 - i * 0.01,
            "rank_reasons": ["安全/情绪相关"],
        }
        for i in range(3)
    ]
    candidates.extend([
        {
            "id": "self-name",
            "summary": "用户叫林小满",
            "source": "user",
            "main_category": "身份",
            "sub_category": "姓名",
            "rank_score": 0.85,
            "rank_reasons": ["关键词命中", "话题类别匹配"],
        },
        {
            "id": "direct-leader",
            "summary": "用户的直属领导叫陈姐，人挺好但要求特别细",
            "source": "user",
            "main_category": "身份",
            "sub_category": "社会关系",
            "rank_score": 0.78,
            "rank_reasons": ["关键词命中", "话题类别匹配"],
        },
        {
            "id": "empty-feeling",
            "summary": "用户感到心里空落落的",
            "source": "user",
            "main_category": "情绪",
            "sub_category": "孤独",
            "importance": 0.6,
            "rank_score": 0.98,
            "rank_reasons": ["安全/情绪相关"],
        },
    ])

    result = select_context(
        candidates,
        token_budget=800,
        max_items=5,
        query="还好吧。我只想问你记得她叫什么吗",
    )

    ids = [m.id for m in result]
    assert "direct-leader" in ids
    matched = next(m for m in result if m.id == "direct-leader")
    assert "保护槽:关系命名" in (matched.rank_reasons or [])
