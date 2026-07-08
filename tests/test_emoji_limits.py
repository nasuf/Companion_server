"""Emoji 节制 (spec §5.3 修订版 + 每条消息 ≤1 硬保证) 测试.

- 概率公式: P_base = random(0, 0.2), P_final = min(0.6, P_base + A × 0.3)
- limit_emojis: 一条消息最多 1 个 emoji, 超出按出现顺序剥除
- 三个用户可见出口 (emit_replies / 短路回复 / 主动消息) 都收口
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.emoji import contains_emoji, limit_emojis


# ═══════════════════════════════════════════════════════════════════
# 概率公式 (spec §5.3 修订版)
# ═══════════════════════════════════════════════════════════════════


class TestShouldAddEmojiFormula:
    def test_base_range_per_spec(self):
        """P_base 必须采样自 random(0, 0.2) — 旧版 0.4 是滥用来源."""
        from app.services import emoji as emoji_mod

        with patch.object(emoji_mod.random, "uniform", return_value=0.1) as uniform, \
             patch.object(emoji_mod.random, "random", return_value=0.99):
            emoji_mod.should_add_emoji(50)
        uniform.assert_called_once_with(0, 0.2)

    def test_coefficient_and_threshold(self):
        """P_final = P_base + A×0.3: A=1.0, base=0.2 → 0.5."""
        from app.services import emoji as emoji_mod

        with patch.object(emoji_mod.random, "uniform", return_value=0.2):
            with patch.object(emoji_mod.random, "random", return_value=0.499):
                assert emoji_mod.should_add_emoji(100) is True
            with patch.object(emoji_mod.random, "random", return_value=0.501):
                assert emoji_mod.should_add_emoji(100) is False

    def test_zero_intensity_only_base(self):
        from app.services import emoji as emoji_mod

        with patch.object(emoji_mod.random, "uniform", return_value=0.15):
            with patch.object(emoji_mod.random, "random", return_value=0.149):
                assert emoji_mod.should_add_emoji(0) is True
            with patch.object(emoji_mod.random, "random", return_value=0.151):
                assert emoji_mod.should_add_emoji(0) is False

    def test_statistical_upper_bound(self):
        """新公式理论上限 0.2+0.3=0.5 (< cap 0.6): 高强度下命中率不该超过 ~55%."""
        from app.services.emoji import should_add_emoji

        hits = sum(should_add_emoji(100) for _ in range(2000))
        assert hits < 2000 * 0.55


# ═══════════════════════════════════════════════════════════════════
# limit_emojis 硬上限
# ═══════════════════════════════════════════════════════════════════


class TestLimitEmojis:
    def test_adjacent_pair_keeps_first(self):
        assert limit_emojis("困得不行😣🤔") == "困得不行😣"

    def test_separated_keeps_first_only(self):
        assert limit_emojis("好呀😅走不走🤪今晚见😈") == "好呀😅走不走今晚见"

    def test_single_emoji_untouched(self):
        assert limit_emojis("差点在梦里接你电话了😴") == "差点在梦里接你电话了😴"

    def test_no_emoji_untouched(self):
        assert limit_emojis("普通文本，没有表情。") == "普通文本，没有表情。"

    def test_empty_and_none_safe(self):
        assert limit_emojis("") == ""

    def test_zwj_family_counts_as_one(self):
        """ZWJ 组合序列 (👨‍👩‍👧) 是一个 emoji 单元, 不能被拆断."""
        text = "全家福👨\u200d👩\u200d👧再来一个😄"
        out = limit_emojis(text)
        assert "👨\u200d👩\u200d👧" in out
        assert "😄" not in out

    def test_flag_pair_counts_as_one(self):
        out = limit_emojis("国旗🇨🇳加油💪")
        assert "🇨🇳" in out and "💪" not in out

    def test_variation_selector_kept_with_first(self):
        assert limit_emojis("爱你❤\ufe0f啦😘") == "爱你❤\ufe0f啦"

    def test_contains_emoji(self):
        assert contains_emoji("哈哈😄")
        assert not contains_emoji("哈哈")
        assert not contains_emoji("")


# ═══════════════════════════════════════════════════════════════════
# 出口收口: emit_replies
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_emit_replies_strips_extra_llm_emojis_and_skips_append():
    """LLM 正文自带 2 个 emoji → 剥到 1; 已有 emoji 的消息不再追加装饰;
    正文表情计入回合上限 (后续 reply 不再贴)."""
    from app.services.chat.reply_post_process import emit_replies
    from app.services.emoji import _EMOJI_UNIT_RE

    P = "app.services.chat.reply_post_process"
    emitted: list[dict] = []
    with (
        patch(f"{P}.should_add_emoji", return_value=True),   # 想加也不许加
        patch(f"{P}.should_add_sticker", return_value=True),
        patch(f"{P}._load_recent_emojis", AsyncMock(return_value=set())),
        patch(f"{P}._remember_emoji", AsyncMock()),
        patch(f"{P}.recommend_sticker", AsyncMock()) as sticker,
        patch(f"{P}.actual_delay_seconds", return_value=None),
        patch(f"{P}.asyncio.sleep", AsyncMock()),
        patch(f"{P}.save_ai_mood", AsyncMock()),
    ):
        async for _ in emit_replies(
            ["困得不行😣🤔", "那哪算正经聊天"],
            reply_context=None,
            reply_index_offset=0,
            sub_intent_mode=False,
            agent=None,
            user_message="hi",
            delay_reply_fn=AsyncMock(),
            fallback_fn=AsyncMock(),
            emitted_replies=emitted,
            reply_emotion={"emotion": "戏谑", "intensity": 90},
            conversation_id="c1",
        ):
            pass

    assert emitted[0]["text"] == "困得不行😣"          # 剥到 1 个
    assert len(_EMOJI_UNIT_RE.findall(emitted[1]["text"])) == 0  # 回合上限: 不再追加
    sticker.assert_not_awaited()  # 正文已有 emoji → sticker 也不贴


@pytest.mark.asyncio
async def test_emit_replies_append_path_still_works_without_llm_emoji():
    """正文无 emoji 时, 追加路径行为不变 (每回合最多 1 个)."""
    from app.services.chat.reply_post_process import emit_replies
    from app.services.emoji import EMOJI_MAP

    P = "app.services.chat.reply_post_process"
    emitted: list[dict] = []
    with (
        patch(f"{P}.should_add_emoji", return_value=True),
        patch(f"{P}.should_add_sticker", return_value=False),
        patch(f"{P}._load_recent_emojis", AsyncMock(return_value=set())),
        patch(f"{P}._remember_emoji", AsyncMock()),
        patch(f"{P}.actual_delay_seconds", return_value=None),
        patch(f"{P}.asyncio.sleep", AsyncMock()),
        patch(f"{P}.save_ai_mood", AsyncMock()),
    ):
        async for _ in emit_replies(
            ["第一条", "第二条"],
            reply_context=None,
            reply_index_offset=0,
            sub_intent_mode=False,
            agent=None,
            user_message="hi",
            delay_reply_fn=AsyncMock(),
            fallback_fn=AsyncMock(),
            emitted_replies=emitted,
            reply_emotion={"emotion": "高兴", "intensity": 90},
            conversation_id="c1",
        ):
            pass

    decorated = [r for r in emitted if any(e in r["text"] for e in EMOJI_MAP["高兴"])]
    assert len(decorated) == 1


# ═══════════════════════════════════════════════════════════════════
# 出口收口: 短路回复 + 主动消息
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_short_circuit_reply_limits_emojis(monkeypatch):
    from app.services.chat import multi_intent

    monkeypatch.setattr(multi_intent, "save_last_reply_timestamp", AsyncMock())
    monkeypatch.setattr(multi_intent, "_fire_background", lambda coro: coro.close())

    saved: list = []

    async def fake_save(conversation_id, replies, **kwargs):
        saved.extend(replies)

    events = await multi_intent.short_circuit_reply(
        "好嘞记上了😜😈明天见",
        "c1", "a1", "u1", fake_save,
    )

    import json as _json
    text = _json.loads(events[0]["data"])["text"]
    assert text == "好嘞记上了😜明天见"


@pytest.mark.asyncio
async def test_emit_proactive_limits_emojis(monkeypatch):
    from app.services.proactive import emit as emit_mod

    created = SimpleNamespace(id="msg-1", createdAt=None)
    fake_db = MagicMock()
    fake_db.message.create = AsyncMock(return_value=created)
    fake_db.proactivechatlog.create = AsyncMock()
    monkeypatch.setattr(emit_mod, "db", fake_db)
    monkeypatch.setattr(
        emit_mod, "manager", SimpleNamespace(send_to_workspace=AsyncMock()),
    )

    await emit_mod.emit_proactive_message(
        conversation_id="c1", user_id="u1", agent_id="a1", workspace_id="w1",
        message="今晚月色真好😍✨🌙",
        trigger_type="silence_wakeup",
    )

    stored = fake_db.message.create.call_args.kwargs["data"]["content"]
    assert stored == "今晚月色真好😍"


def test_no_roleplay_rule_has_emoji_restraint():
    """守卫: 主动消息系列的反旁白规则必须带 emoji 节制 (≤1, 多数不用),
    且不再出现鼓励性的「emoji 自然体现」表述与带表情的正例."""
    from app.services.prompting.defaults import _NO_ROLEPLAY_RULE

    assert "最多一个 emoji" in _NO_ROLEPLAY_RULE
    assert "😴" not in _NO_ROLEPLAY_RULE
