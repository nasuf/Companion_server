"""Phase 2.1 + 2.3 step 1 测试.

2.1: L1 (importance ≥ 0.85) 不被 time_freshness 衰减压低.
2.3: prompt 模板不再注入 raw PAD 数值 ((0.50, 0.30, 0.50) 这种 LLM 看不懂).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


# ═══════════════════════════════════════════════════════════════════
# Phase 2.1: L1 freshness floor
# ═══════════════════════════════════════════════════════════════════


def test_l1_importance_skips_freshness_decay():
    """L1 (importance >= 0.85) 即便 1 年没访问, freshness 至少 1.0."""
    from app.services.memory.retrieval.relevance import compute_display_score

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=400)

    # L1 身份记忆: importance=0.95, similarity=0.9
    l1_score = compute_display_score(
        importance=0.95, last_accessed_at=one_year_ago, similarity=0.9,
    )
    # freshness 被钳到 1.0, 而不是原 0.4 (>365 天)
    # 实际算: 0.95 × 1.0 × 0.9 = 0.855
    assert l1_score == 0.95 * 1.0 * 0.9, (
        f"L1 expected 0.855 (freshness floor 1.0), got {l1_score:.3f}"
    )


def test_l2_importance_still_subject_to_decay():
    """L2 (importance < 0.85) 仍正常衰减 — Phase 2.1 只豁免 L1."""
    from app.services.memory.retrieval.relevance import compute_display_score

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=400)

    # L2 偏好记忆: importance=0.6
    l2_score = compute_display_score(
        importance=0.6, last_accessed_at=one_year_ago, similarity=0.8,
    )
    # freshness 0.4 (>365 天), 不豁免: 0.6 × 0.4 × 0.8 = 0.192
    assert abs(l2_score - 0.6 * 0.4 * 0.8) < 1e-6, (
        f"L2 expected 0.192 (no floor), got {l2_score:.3f}"
    )


def test_l1_recent_unchanged():
    """L1 在新鲜窗口内 (<30 天) freshness=1.2, floor 不影响 (1.2 > 1.0)."""
    from app.services.memory.retrieval.relevance import compute_display_score

    yesterday = datetime.now(timezone.utc) - timedelta(days=1)

    score = compute_display_score(
        importance=0.95, last_accessed_at=yesterday, similarity=0.9,
    )
    # 0.95 × 1.2 × 0.9 = 1.026 (max(1.2, 1.0) = 1.2)
    assert abs(score - 0.95 * 1.2 * 0.9) < 1e-6


def test_l1_beats_recent_unrelated_l2():
    """场景: 用户问"你叫什么", L1 身份 (1 年没访问) 应排在新闲聊 L2 前.

    Phase 2.1 修复前: L1 freshness=0.4 → display_score 被压
    Phase 2.1 修复后: L1 floor=1.0 → 高 importance 主导排序
    """
    from app.services.memory.retrieval.relevance import compute_display_score

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=400)
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    # L1 "我叫 Nina" — 高 importance 直接命中
    l1_identity = compute_display_score(
        importance=0.95, last_accessed_at=one_year_ago, similarity=0.9,
    )
    # 新 L2 "用户喜欢咖啡" — 低 importance 弱命中
    l2_recent = compute_display_score(
        importance=0.5, last_accessed_at=one_week_ago, similarity=0.5,
    )
    assert l1_identity > l2_recent, (
        f"L1 身份 ({l1_identity:.3f}) 必须排在新 L2 闲聊 ({l2_recent:.3f}) 前"
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 2.3 step 1: 删除 prompt 中的 raw PAD numbers
# ═══════════════════════════════════════════════════════════════════


def test_no_raw_pad_placeholder_in_prompts():
    """所有 prompt 模板不应再含 {pleasure}/{arousal}/{dominance} 占位符
    (除 emotion 抽取本身的输出 schema)."""
    from app.services.prompting import defaults as d

    src = open(d.__file__).read()
    # 抽取 prompt 自身的 output schema 含 pleasure/arousal/dominance 是合法的
    # 用占位符语法 {pleasure} 检测 — schema 只用文本 "pleasure" 不带 {}
    placeholder_count = src.count("{pleasure}") + src.count("{arousal}") + src.count("{dominance}")
    assert placeholder_count == 0, (
        f"defaults.py 仍有 {placeholder_count} 处 PAD 占位符, 应已被 Phase 2.3 全删"
    )


def test_emotion_section_no_raw_pad_vector():
    """prompt_builder._build_emotion_section 不再注入 raw PAD vector."""
    import asyncio
    from app.services.chat.prompt_builder import _build_emotion_section

    user_emotion = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.6}
    section = asyncio.run(_build_emotion_section(
        user_emotion=user_emotion, intimacy_stage="挚友",
    ))
    # 不应含 raw vector 字符串
    assert section is not None
    assert "PAD" not in section
    assert "0.50" not in section and "0.30" not in section
    # intimacy_stage 仍应注入
    assert "挚友" in section


def test_emotion_section_only_intimacy_no_pad():
    """仅有 intimacy_stage, 无 user_emotion → 仍正常输出 (不报错)."""
    import asyncio
    from app.services.chat.prompt_builder import _build_emotion_section

    section = asyncio.run(_build_emotion_section(
        user_emotion=None, intimacy_stage="初识",
    ))
    assert section is not None
    assert "初识" in section


def test_emotion_section_no_intimacy_returns_none():
    """无 intimacy_stage 也无其他可注入信息 → 返 None (不污染 prompt)."""
    import asyncio
    from app.services.chat.prompt_builder import _build_emotion_section

    section = asyncio.run(_build_emotion_section(
        user_emotion={"pleasure": 0.5}, intimacy_stage=None,
    ))
    # 没 intimacy → 整段 None (不再有 PAD section 可输出)
    assert section is None


def test_pad_params_no_longer_imported_in_chat_paths():
    """生产链路 (intent_replies, contradiction, boundary) 不应 import pad_params."""
    paths = [
        "app/services/chat/intent_replies.py",
        "app/services/memory/interaction/contradiction.py",
        "app/services/interaction/boundary.py",
    ]
    for p in paths:
        src = open(f"/Users/songtao/Projects/companion/Companion_server/{p}").read()
        assert "pad_params" not in src, (
            f"{p} 仍引用 pad_params, 应已被 Phase 2.3 step 1 清理"
        )
