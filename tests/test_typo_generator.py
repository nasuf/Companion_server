"""Phase E1 回归: 错别字生成器 (2026-07-03 起默认开启, .env 可关)."""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.typo import _CONFUSION_PAIRS, maybe_typo


class TestMaybeTypo:
    def test_rate_zero_never_mutates(self):
        for _ in range(50):
            text, corr = maybe_typo("我在做的事", rate=0.0)
            assert text == "我在做的事" and corr is None

    def test_rate_one_always_mutates_when_confusable(self):
        rng = random.Random(42)
        text, _ = maybe_typo("我在做的事", rate=1.0, rng=rng)
        assert text != "我在做的事"
        assert len(text) == len("我在做的事")  # 只替换不增删

    def test_no_confusable_chars_returns_unchanged(self):
        text, corr = maybe_typo("哈哈哈哈", rate=1.0)
        assert text == "哈哈哈哈" and corr is None

    def test_correction_is_the_original_char(self):
        rng = random.Random(1)
        found_correction = False
        for _ in range(100):
            text, corr = maybe_typo("我的书", rate=1.0, rng=rng)
            if corr is not None:
                found_correction = True
                assert corr == "的"  # 纠正字是原正确字
                assert "得" in text  # 错字进了正文
        assert found_correction  # ~50% 概率, 100 次必然出现

    def test_at_most_one_typo_per_message(self):
        rng = random.Random(7)
        original = "我的的的的的"
        text, _ = maybe_typo(original, rate=1.0, rng=rng)
        diff = sum(a != b for a, b in zip(original, text))
        assert diff == 1

    def test_no_meaning_flipping_pairs(self):
        """守卫: 意义翻转类混淆对 (买/卖, 带/戴) 不允许进词表."""
        for a, b in (("买", "卖"), ("带", "戴"), ("大", "打")):
            assert _CONFUSION_PAIRS.get(a) != b
            assert _CONFUSION_PAIRS.get(b) != a


def test_typo_enabled_by_default():
    """2026-07-03 产品决策: 错别字生成器默认开启 (可 .env 关)."""
    from app.config import Settings

    assert Settings.model_fields["typo_enabled"].default is True
    assert 0 < Settings.model_fields["typo_rate"].default <= 0.1


@pytest.mark.asyncio
async def test_emit_replies_typo_flag_off_never_mutates():
    """typo_enabled=False 时正文绝不被修改 (关闭开关的行为契约)."""
    from app.services.chat.reply_post_process import emit_replies

    P = "app.services.chat.reply_post_process"
    emitted: list[dict] = []
    with (
        patch(f"{P}.should_add_emoji", return_value=False),
        patch(f"{P}.should_add_sticker", return_value=False),
        patch(f"{P}._load_recent_emojis", AsyncMock(return_value=set())),
        patch(f"{P}.actual_delay_seconds", return_value=None),
        patch(f"{P}.asyncio.sleep", AsyncMock()),
        patch(f"{P}.settings") as mock_settings,
    ):
        mock_settings.typo_enabled = False
        async for _ in emit_replies(
            ["我在做的事"],
            reply_context=None, reply_index_offset=0, sub_intent_mode=False,
            agent=None, user_message="hi",
            delay_reply_fn=AsyncMock(), fallback_fn=AsyncMock(),
            emitted_replies=emitted, reply_emotion=None, conversation_id="c1",
        ):
            pass
    assert emitted[0]["text"] == "我在做的事"


@pytest.mark.asyncio
async def test_emit_replies_typo_enabled_keeps_typo_without_correction_bubble():
    from app.services.chat.reply_post_process import emit_replies

    P = "app.services.chat.reply_post_process"
    emitted: list[dict] = []
    with (
        patch(f"{P}.should_add_emoji", return_value=False),
        patch(f"{P}.should_add_sticker", return_value=False),
        patch(f"{P}._load_recent_emojis", AsyncMock(return_value=set())),
        patch(f"{P}.actual_delay_seconds", return_value=None),
        patch(f"{P}.asyncio.sleep", AsyncMock()),
        patch(f"{P}.settings") as mock_settings,
        patch(f"{P}.maybe_typo", return_value=("我再做的事", "在")),
    ):
        mock_settings.typo_enabled = True
        mock_settings.typo_rate = 1.0
        async for _ in emit_replies(
            ["我在做的事"],
            reply_context=None, reply_index_offset=0, sub_intent_mode=False,
            agent=None, user_message="hi",
            delay_reply_fn=AsyncMock(), fallback_fn=AsyncMock(),
            emitted_replies=emitted, reply_emotion=None, conversation_id="c1",
        ):
            pass

    assert emitted[0]["text"] == "我再做的事"
    assert len(emitted) == 1
    assert "typo_correction" not in emitted[0]
