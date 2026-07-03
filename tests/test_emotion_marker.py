"""W1b 回归: 回复情绪并入主 LLM ([EMO:标签/强度] 标记尾).

解析成功 → 跳过串行的 ai_reply_emotion 小模型调用 (关键路径 -300~600ms,
每条消息 -1 次 LLM); 解析失败 → 回退原小模型路径; 标记任何情况下都从
正文剥除, 绝不泄漏给用户.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.chat.reply_generate import extract_emotion_marker


class TestExtractEmotionMarker:
    def test_valid_marker_parsed_and_stripped(self):
        text, emo = extract_emotion_marker("好呀，走！\n[EMO:高兴/85]")
        assert text == "好呀，走！"
        assert emo == {"emotion": "高兴", "intensity": 85}

    def test_fullwidth_variants_accepted(self):
        text, emo = extract_emotion_marker("嗯……【EMO：失望／30】")
        assert text == "嗯……"
        assert emo == {"emotion": "失望", "intensity": 30}

    def test_no_marker_returns_none(self):
        text, emo = extract_emotion_marker("普通回复没有标记")
        assert text == "普通回复没有标记" and emo is None

    def test_invalid_label_stripped_but_none(self):
        """标签不在 12 类 → 返 None 回退小模型, 但标记仍被剥除防泄漏."""
        text, emo = extract_emotion_marker("文本 [EMO:开心到飞起/85]")
        assert emo is None
        assert "EMO" not in text

    def test_multiple_markers_take_last_strip_all(self):
        text, emo = extract_emotion_marker(
            "句1 [EMO:中性/20]||句2 [EMO:高兴/70]",
        )
        assert emo == {"emotion": "高兴", "intensity": 70}
        assert "EMO" not in text
        assert "句1" in text and "句2" in text

    def test_intensity_clamped_to_100(self):
        _, emo = extract_emotion_marker("x [EMO:高兴/999]")
        assert emo["intensity"] == 100

    def test_labels_align_with_emoji_map(self):
        """守卫: 标记标签集必须与 EMOJI_MAP 12 类完全一致 (spec §5 step 1)."""
        from app.services.chat.reply_generate import _VALID_REPLY_EMOTIONS
        from app.services.emoji import EMOJI_MAP

        assert _VALID_REPLY_EMOTIONS == frozenset(EMOJI_MAP.keys())


@pytest.mark.asyncio
class TestGenerateReplyEmotionSource:
    """集成: 主 LLM 路径下标记命中 → 不调 reply_emotion_fn; 未命中 → 回退."""

    async def _run(self, llm_output: str, emotion_fn):
        from unittest.mock import patch

        from app.services.chat.intent_dispatcher import IntentResult, IntentType
        from app.services.chat import reply_generate as rg

        diagnostics: dict = {}
        with patch.object(
            rg, "_run_main_llm", AsyncMock(return_value=(llm_output, False)),
        ):
            replies, raw, is_fallback, emo = await rg.generate_reply(
                contradiction_inquiry=None,
                detected_intent=IntentResult(intent=IntentType.SCHEDULE_QUERY, confidence=1.0),
                memory_relevance="medium",
                relational_context="用户在表达情绪",  # 阻断 tier, 强制主 LLM 路径
                schedule_context=None,
                delay_context=None,
                l3_memories=[],
                classified_memories=[],
                messages_dicts=[],
                portrait=None,
                prompt_user_emotion=None,
                user_message="hi",
                agent=None,
                reply_count=1,
                max_reply_count=3,
                max_total=150,
                tier_fns={},
                truncate_fn=lambda t, n: t[:n],
                pipe_fallback_fn=lambda raw, c, p, t: [raw],
                chat_messages=[{"role": "system", "content": "S"}],
                reply_emotion_fn=emotion_fn,
                diagnostics=diagnostics,
            )
        return replies, emo, diagnostics

    async def test_marker_present_skips_emotion_llm(self):
        emotion_fn = AsyncMock(return_value={"emotion": "中性", "intensity": 10})
        replies, emo, diag = await self._run("好呀！\n[EMO:高兴/80]", emotion_fn)
        emotion_fn.assert_not_called()  # 省掉 1 次小模型调用
        assert emo == {"emotion": "高兴", "intensity": 80}
        assert diag["reply_emotion_source"] == "main_marker"
        assert all("EMO" not in r for r in replies)  # 正文无泄漏

    async def test_marker_missing_falls_back_to_emotion_llm(self):
        emotion_fn = AsyncMock(return_value={"emotion": "中性", "intensity": 10})
        replies, emo, diag = await self._run("好呀，没有标记", emotion_fn)
        emotion_fn.assert_awaited_once()
        assert emo == {"emotion": "中性", "intensity": 10}
        assert diag["reply_emotion_source"] == "fallback_llm"
