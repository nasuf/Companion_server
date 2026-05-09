"""Tests for the context selector."""

from app.services.memory.retrieval.context_selector import (
    ClassifiedMemory,
    select_context,
    split_by_source,
)


def test_select_context_includes_complete_short_memories():
    """Short memories are included as complete items."""
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


def test_select_context_skips_abnormally_long_single_memory():
    """Token budget is only a per-item guard; long memories are skipped intact."""
    # Each long item is skipped as a whole instead of being truncated.
    long_text = "word " * 200  # ~260 tokens each
    candidates = [
        {"id": str(i), "summary": long_text, "importance": 0.5, "created_at": "2025-01-01T00:00:00"}
        for i in range(10)
    ]
    result = select_context(candidates, token_budget=400)
    assert result == []


def test_select_context_uses_independent_source_quotas_not_global_top10():
    """User and AI memories each get their own quota, so one side cannot crowd out the other."""
    candidates = [
        {
            "id": f"user-{i}",
            "summary": f"用户记忆 {i}",
            "source": "user",
            "rank_score": 0.8 - i * 0.01,
        }
        for i in range(12)
    ] + [
        {
            "id": f"ai-{i}",
            "summary": f"AI 记忆 {i}",
            "source": "ai",
            "rank_score": 0.78 - i * 0.01,
        }
        for i in range(12)
    ]

    result = select_context(candidates, token_budget=80)

    assert len(result) == 20
    assert sum(1 for m in result if m.source == "user") == 10
    assert sum(1 for m in result if m.source == "ai") == 10


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

    assert len(result) == 11
    safety = next((m for m in result if m.id == "safety"), None)
    assert safety is not None
    assert "保护槽:安全情绪" in (safety.rank_reasons or [])
    assert sum(1 for m in result if m.source == "ai") == 10


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

    assert len(result) == 9
    user_memories = [m for m in result if m.source == "user"]
    assert len(user_memories) == 3
    assert all("保护槽:用户记忆" in (m.rank_reasons or []) for m in user_memories)


def test_select_context_protects_ai_self_memory_for_agent_preference_query():
    """Agent self-preference questions should not be filled by unrelated user memories."""
    candidates = [
        {
            "id": f"user-{i}",
            "summary": f"用户普通记忆 {i}",
            "source": "user",
            "rank_score": 0.82 - i * 0.01,
        }
        for i in range(8)
    ]
    candidates.append({
        "id": "ai-movie",
        "summary": "我喜欢烧脑科幻电影，也喜欢轻松喜剧",
        "source": "ai",
        "rank_score": 0.41,
    })

    result = select_context(
        candidates,
        token_budget=800,
        max_items=5,
        query="你喜欢什么电影啊",
    )

    matched = next((m for m in result if m.id == "ai-movie"), None)
    assert matched is not None
    assert "保护槽:AI自我记忆" in (matched.rank_reasons or [])
    assert sum(1 for m in result if m.source == "ai") == 1


def test_select_context_keeps_user_floor_for_user_preference_recall_query():
    """Recall questions about the user are not mistaken for AI self queries."""
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 记忆 {i}",
            "source": "ai",
            "rank_score": 0.9 - i * 0.01,
        }
        for i in range(5)
    ] + [
        {
            "id": f"user-{i}",
            "summary": f"用户偏好记忆 {i}",
            "source": "user",
            "rank_score": 0.45 - i * 0.01,
        }
        for i in range(3)
    ]

    result = select_context(
        candidates,
        token_budget=800,
        max_items=5,
        query="你还记得我喜欢什么电影吗",
    )

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


def test_select_context_protects_high_similarity_vector_hit():
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 高分记忆 {i}",
            "source": "ai",
            "similarity": 0.62,
            "rank_score": 0.95 - i * 0.01,
        }
        for i in range(8)
    ]
    candidates.append({
        "id": "exact-vector-hit",
        "summary": "自由时间无特别安排",
        "source": "user",
        "similarity": 0.99,
        "rank_score": 0.32,
        "rank_reasons": ["高相似向量命中"],
    })

    result = select_context(candidates, token_budget=800, max_items=6)

    matched = next((m for m in result if m.id == "exact-vector-hit"), None)
    assert matched is not None
    assert "保护槽:高相似向量" in (matched.rank_reasons or [])


def test_select_context_does_not_treat_positive_emotion_as_safety_slot():
    candidates = [
        {
            "id": f"ai-{i}",
            "summary": f"AI 高分记忆 {i}",
            "source": "ai",
            "rank_score": 0.9 - i * 0.01,
        }
        for i in range(5)
    ]
    candidates.extend([
        {
            "id": "love-ai",
            "summary": "用户表达了对 AI 的喜爱之情",
            "source": "user",
            "main_category": "情绪",
            "sub_category": "感激",
            "importance": 0.95,
            "rank_score": 0.3,
        },
        {
            "id": "apology",
            "summary": "用户向AI道歉并承诺以后不再犯类似错误",
            "source": "user",
            "main_category": "情绪",
            "sub_category": "遗憾",
            "importance": 0.8,
            "rank_score": 0.29,
        },
        {
            "id": "sadness",
            "summary": "用户最近感到很难过",
            "source": "user",
            "main_category": "情绪",
            "sub_category": "悲伤",
            "importance": 0.8,
            "rank_score": 0.28,
        },
    ])

    result = select_context(
        candidates,
        token_budget=800,
        max_items=5,
        query="我刚才状态不好，你记得吗",
    )

    ids = [m.id for m in result]
    assert "sadness" in ids
    assert "love-ai" not in ids
    assert "apology" not in ids
    matched = next(m for m in result if m.id == "sadness")
    assert "保护槽:安全情绪" in (matched.rank_reasons or [])


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


def test_ranker_prioritizes_relation_name_over_self_and_ai_names():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    query = "她叫什么"
    leader = {
        "id": "leader",
        "summary": "用户的直属领导叫陈姐，人挺好但要求特别细",
        "source": "user",
        "main_category": "身份",
        "sub_category": "社会关系",
        "importance": 0.8,
        "similarity": 0.57,
    }
    user_name = {
        "id": "user-name",
        "summary": "用户叫林小满",
        "source": "user",
        "main_category": "身份",
        "sub_category": "姓名",
        "importance": 0.95,
        "similarity": 0.51,
    }
    ai_name = {
        "id": "ai-name",
        "summary": "我叫Hillow",
        "source": "ai",
        "main_category": "身份",
        "sub_category": "姓名",
        "importance": 0.95,
        "similarity": 0.53,
    }

    leader_score, leader_reasons = rank_memory_candidate(leader, query)
    user_score, user_reasons = rank_memory_candidate(user_name, query)
    ai_score, ai_reasons = rank_memory_candidate(ai_name, query)

    assert "关系命名相关" in leader_reasons
    assert "关系名查询降权:用户本人姓名" in user_reasons
    assert "关系名查询降权:AI记忆" in ai_reasons
    assert leader_score > user_score
    assert leader_score > ai_score


def test_ranker_protects_exact_text_even_with_low_importance():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    exact = {
        "id": "exact",
        "summary": "早上被流浪猫吵醒",
        "source": "user",
        "importance": 0.2,
        "similarity": 1.0,
    }
    generic = {
        "id": "generic",
        "summary": "去公园观察流浪猫狗的行为模式，顺便投喂食物",
        "source": "ai",
        "importance": 0.95,
        "similarity": 0.67,
    }

    exact_score, exact_reasons = rank_memory_candidate(exact, "早上被流浪猫吵醒")
    generic_score, _ = rank_memory_candidate(generic, "早上被流浪猫吵醒")

    assert "精确文本命中" in exact_reasons
    assert exact_score > generic_score


def test_ranker_prioritizes_user_safety_over_ai_emotion_story():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    user_safety = {
        "id": "user-safety",
        "summary": "用户表达了强烈的负面情绪，有轻生念头",
        "source": "user",
        "main_category": "情绪",
        "sub_category": "悲伤",
        "importance": 0.85,
        "similarity": 0.55,
    }
    ai_story = {
        "id": "ai-story",
        "summary": "生病发烧躺在出租屋床上，那一刻觉得世界好空旷。",
        "source": "ai",
        "main_category": "情绪",
        "sub_category": "孤独",
        "importance": 0.95,
        "similarity": 0.72,
    }

    user_score, user_reasons = rank_memory_candidate(
        user_safety,
        "我现在好多了，但还是有点空",
    )
    ai_score, ai_reasons = rank_memory_candidate(
        ai_story,
        "我现在好多了，但还是有点空",
    )

    assert "安全/情绪相关" in user_reasons
    assert "安全查询降权:AI情绪记忆" in ai_reasons
    assert user_score > ai_score


def test_ranker_prioritizes_user_preference_over_ai_preference():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    user_pref = {
        "id": "user-pref",
        "summary": "用户对芒果过敏",
        "source": "user",
        "main_category": "偏好",
        "sub_category": "饮食禁忌",
        "importance": 0.8,
        "similarity": 0.56,
    }
    ai_pref = {
        "id": "ai-pref",
        "summary": "我讨厌吃动物内脏",
        "source": "ai",
        "main_category": "偏好",
        "sub_category": "饮食禁忌",
        "importance": 0.95,
        "similarity": 0.70,
    }

    user_score, user_reasons = rank_memory_candidate(user_pref, "我不喜欢什么")
    ai_score, ai_reasons = rank_memory_candidate(ai_pref, "我不喜欢什么")

    assert "用户偏好相关" in user_reasons
    assert "用户偏好查询降权:AI记忆" in ai_reasons
    assert user_score > ai_score


def test_ranker_prioritizes_ai_preference_for_agent_self_query():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    ai_pref = {
        "id": "ai-pref",
        "summary": "我喜欢烧脑科幻电影",
        "source": "ai",
        "main_category": "偏好",
        "sub_category": "审美爱好",
        "importance": 0.7,
        "similarity": 0.56,
    }
    user_pref = {
        "id": "user-pref",
        "summary": "用户喜欢爱情片",
        "source": "user",
        "main_category": "偏好",
        "sub_category": "审美爱好",
        "importance": 0.95,
        "similarity": 0.72,
    }

    ai_score, ai_reasons = rank_memory_candidate(ai_pref, "你喜欢什么电影啊")
    user_score, user_reasons = rank_memory_candidate(user_pref, "你喜欢什么电影啊")

    assert "AI自我记忆相关" in ai_reasons
    assert "AI自我查询降权:用户记忆" in user_reasons
    assert ai_score > user_score


def test_ranker_keeps_user_identity_as_context_for_ai_profile_query():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    ai_age = {
        "id": "ai-age",
        "summary": "我今年26岁",
        "source": "ai",
        "main_category": "身份",
        "sub_category": "年龄",
        "importance": 0.95,
        "similarity": 0.62,
    }
    user_age = {
        "id": "user-age",
        "summary": "用户28岁",
        "source": "user",
        "main_category": "身份",
        "sub_category": "年龄",
        "importance": 0.95,
        "similarity": 0.58,
    }

    ai_score, ai_reasons = rank_memory_candidate(ai_age, "AI 年龄 用户 年龄")
    user_score, user_reasons = rank_memory_candidate(user_age, "AI 年龄 用户 年龄")

    assert "AI自我记忆相关" in ai_reasons
    assert "AI资料查询:用户同类资料" in user_reasons
    assert "AI自我查询降权:用户记忆" not in user_reasons
    assert ai_score > 0
    assert user_score > 0


def test_select_context_keeps_user_identity_for_ai_profile_query():
    candidates = [
        {
            "id": "ai-age",
            "summary": "我今年26岁",
            "source": "ai",
            "main_category": "身份",
            "sub_category": "年龄",
            "rank_score": 0.7,
            "similarity": 0.62,
        },
        {
            "id": "user-age",
            "summary": "用户28岁",
            "source": "user",
            "main_category": "身份",
            "sub_category": "年龄",
            "rank_score": 0.36,
            "similarity": 0.49,
        },
    ] + [
        {
            "id": f"ai-other-{i}",
            "summary": f"AI 其他资料 {i}",
            "source": "ai",
            "rank_score": 0.69 - i * 0.01,
            "similarity": 0.4,
        }
        for i in range(4)
    ]

    result = select_context(candidates, token_budget=800, query="AI 年龄 用户 年龄")
    texts = [m.text for m in result]

    assert "我今年26岁" in texts
    assert "用户28岁" in texts


def test_ranker_prioritizes_user_reminders_over_generic_identity():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    reminder = {
        "id": "reminder",
        "summary": "用户周三晚上 8 点要跟陈姐开 review",
        "source": "user",
        "main_category": "生活",
        "sub_category": "提醒",
        "importance": 0.7,
        "similarity": 0.46,
    }
    identity = {
        "id": "identity",
        "summary": "用户今年27岁",
        "source": "user",
        "main_category": "身份",
        "sub_category": "年龄",
        "importance": 1.0,
        "similarity": 0.59,
    }

    reminder_score, reminder_reasons = rank_memory_candidate(
        reminder,
        "我最近有什么提醒事项",
    )
    identity_score, _ = rank_memory_candidate(identity, "我最近有什么提醒事项")

    assert "用户提醒相关" in reminder_reasons
    assert reminder_score > identity_score


def test_ranker_prioritizes_user_identity_over_ai_identity():
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    user_identity = {
        "id": "user-age",
        "summary": "用户今年27岁",
        "source": "user",
        "main_category": "身份",
        "sub_category": "年龄",
        "importance": 0.8,
        "similarity": 0.54,
    }
    ai_identity = {
        "id": "ai-age",
        "summary": "我今年21岁",
        "source": "ai",
        "main_category": "身份",
        "sub_category": "年龄",
        "importance": 0.95,
        "similarity": 0.62,
    }
    relation_memory = {
        "id": "leader",
        "summary": "用户的直属领导叫陈姐",
        "source": "user",
        "main_category": "身份",
        "sub_category": "社会关系",
        "importance": 0.8,
        "similarity": 0.57,
    }

    user_score, user_reasons = rank_memory_candidate(
        user_identity,
        "你记得我的基本信息吗",
    )
    ai_score, ai_reasons = rank_memory_candidate(
        ai_identity,
        "你记得我的基本信息吗",
    )
    relation_score, relation_reasons = rank_memory_candidate(
        relation_memory,
        "我的老板叫什么名字",
    )

    assert "用户身份相关" in user_reasons
    assert "用户身份查询降权:AI身份记忆" in ai_reasons
    assert user_score > ai_score
    assert "关系命名相关" in relation_reasons
    assert "用户身份相关" not in relation_reasons
    assert relation_score > user_score
