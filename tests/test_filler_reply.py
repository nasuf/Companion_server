"""Phase E2 回归: 纯语气词概率性仅表情回应."""

from __future__ import annotations

import random

from app.services.chat.filler_reply import (
    FILLER_EMOJI_PROBABILITY,
    build_filler_emoji_reply,
    is_question_like,
)
from app.services.emoji import EMOJI_MAP


class _AlwaysHit(random.Random):
    """random() 永远命中概率闸门, choice 走真实逻辑."""

    def random(self):
        return 0.0


class _NeverHit(random.Random):
    def random(self):
        return 0.999


class TestQuestionGuard:
    def test_question_endings_detected(self):
        assert is_question_like("要我再陪你一会儿吗")
        assert is_question_like("你觉得呢？")
        assert is_question_like("去不去嘛")
        assert not is_question_like("晚安，做个好梦")
        assert not is_question_like(None)

    def test_filler_after_ai_question_never_shortcuts(self):
        """护栏: "好"/"嗯" 是对 AI 提问的答复, 必须交给意图管线."""
        out = build_filler_emoji_reply(
            "好", previous_assistant_text="要我再陪你一会儿吗?", rng=_AlwaysHit(),
        )
        assert out is None


class TestFillerMatching:
    def test_positive_filler_gets_happy_emoji(self):
        out = build_filler_emoji_reply("哈哈哈", rng=_AlwaysHit())
        assert out in EMOJI_MAP["高兴"]

    def test_neutral_filler_gets_neutral_emoji(self):
        out = build_filler_emoji_reply("嗯嗯", rng=_AlwaysHit())
        assert out in EMOJI_MAP["中性"]

    def test_trailing_punctuation_normalized(self):
        assert build_filler_emoji_reply("好的~", rng=_AlwaysHit()) is not None
        assert build_filler_emoji_reply("哈哈！", rng=_AlwaysHit()) is not None

    def test_non_filler_returns_none(self):
        assert build_filler_emoji_reply("今天加班到十点", rng=_AlwaysHit()) is None
        assert build_filler_emoji_reply("嗯，但是我有点难过", rng=_AlwaysHit()) is None

    def test_probability_gate(self):
        assert build_filler_emoji_reply("嗯", rng=_NeverHit()) is None
        # 默认概率应是少数派行为 (大多数时候仍走完整管线)
        assert 0 < FILLER_EMOJI_PROBABILITY <= 0.5
