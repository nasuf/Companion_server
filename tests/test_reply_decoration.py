"""Phase C4/C5 回归: emoji 跨轮去重 + 每轮上限 + 延迟解释概率化."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.reply_post_process import (
    _load_recent_emojis,
    _should_explain_delay,
    emit_replies,
)
from app.services.emoji import EMOJI_MAP, pick_one_emoji


class TestPickOneEmojiExclude:
    def test_excludes_recent(self):
        pool = set(EMOJI_MAP["高兴"])
        recent = set(list(pool)[:-1])  # 只留一个可选
        remaining = pool - recent
        for _ in range(20):
            assert pick_one_emoji("高兴", exclude=recent) in remaining

    def test_full_exclusion_falls_back_to_pool(self):
        pool = set(EMOJI_MAP["高兴"])
        assert pick_one_emoji("高兴", exclude=pool) in pool

    def test_no_exclude_backward_compatible(self):
        assert pick_one_emoji("高兴") in set(EMOJI_MAP["高兴"])


class TestShouldExplainDelay:
    def test_under_one_minute_never(self):
        assert not any(_should_explain_delay(59) for _ in range(50))

    def test_probabilistic_between_thresholds(self):
        """1-5min 档 p=0.35: 既不是永远解释, 也不是从不解释."""
        results = [_should_explain_delay(120) for _ in range(300)]
        assert any(results) and not all(results)

    def test_long_delay_more_likely(self):
        short = sum(_should_explain_delay(120) for _ in range(500))
        long = sum(_should_explain_delay(3600) for _ in range(500))
        assert long > short


@pytest.mark.asyncio
async def test_emit_replies_caps_one_emoji_per_turn():
    """C4: 一个回合最多 1 个 emoji (原来每条 reply 独立掷骰)."""
    P = "app.services.chat.reply_post_process"
    emitted: list[dict] = []
    with (
        patch(f"{P}.should_add_emoji", return_value=True),   # 每条都想加
        patch(f"{P}.should_add_sticker", return_value=False),
        patch(f"{P}._load_recent_emojis", AsyncMock(return_value=set())),
        patch(f"{P}._remember_emoji", AsyncMock()) as remember,
        patch(f"{P}.actual_delay_seconds", return_value=None),
        patch(f"{P}.asyncio.sleep", AsyncMock()),
    ):
        async for _ in emit_replies(
            ["第一条", "第二条", "第三条"],
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
    assert len(decorated) == 1, f"每轮应恰好 1 条带 emoji, got {len(decorated)}"
    remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_recent_emojis_redis_failure_returns_empty():
    with patch(
        "app.redis_client.get_redis",
        AsyncMock(side_effect=RuntimeError("down")),
    ):
        assert await _load_recent_emojis("c1") == set()


class TestSafeTemplateRendering:
    """D3 回归: registry 模板被 admin 编辑出两类人为失误时不炸调用链路."""

    def test_unknown_placeholder_renders_fallback(self):
        from app.services.prompting.utils import safe_format

        out = safe_format("任务: {instruction} 未知: {nonexistent}", {"instruction": "x"})
        assert "任务: x" in out
        assert "(无)" in out  # 未知占位符兜底而不是 KeyError

    def test_literal_json_braces_fall_back_to_unrendered(self):
        """模板含字面 JSON 大括号 (没写成 {{...}}) → 返回原文并告警, 不抛 ValueError."""
        from app.services.prompting.utils import safe_format

        tpl = '任务: {instruction}\n输出 JSON: {"result": "..."}'
        out = safe_format(tpl, {"instruction": "解释延迟"})
        assert out == tpl  # 渲染失败保底返回未渲染原文

    def test_strict_format_would_raise_in_both_cases(self):
        """对照: 旧的严格 .format 在同样输入下必炸 — 证明 safe_format 迁移有意义."""
        import pytest as _pytest

        with _pytest.raises(KeyError):
            "任务: {instruction} 未知: {nonexistent}".format(instruction="x")
        with _pytest.raises((ValueError, KeyError)):
            '输出 JSON: {"result": "..."}'.format()  # 裸 format: KeyError
        with _pytest.raises(ValueError):
            # format_map + SafeDict 仍炸 (invalid format spec) — safe_format 的兜底价值
            from app.services.prompting.utils import SafeDict
            '输出 JSON: {"result": "..."}'.format_map(SafeDict({}))
