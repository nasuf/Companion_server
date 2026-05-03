"""Regression: render_prompt 截断必须按句末符号边界, 不能 raw[:max_chars] 切到
中文字符中段.

生产 bug 复现 (2026-05-03 trace 019dec46): schedule_query_reply max_chars=120,
LLM 出 ~225 chars, 之前 raw[:120] 切到 "明天是你生|日" 正中字, UI 显示
"明天是你生" 看起来像 AI 卡住没说完. 现在改用句末符 truncation.
"""

from __future__ import annotations

import pytest

from app.services.prompting.utils import _truncate_at_sentence_boundary


def test_short_text_under_max_returns_unchanged():
    text = "好嘞，记下啦"
    assert _truncate_at_sentence_boundary(text, 100) == text


def test_truncate_at_sentence_period_boundary():
    """中文句号 → 切到句末."""
    text = "今天天气真好。我想出去走走。但是路上太晒了。"
    # max=15 → 应切到 "今天天气真好。" (7 chars), 不是中段
    result = _truncate_at_sentence_boundary(text, 15)
    assert result == "今天天气真好。我想出去走走。"
    assert result.endswith("。")


def test_truncate_at_question_mark_boundary():
    """问号也算句末 (中英都要)."""
    text = "你周末有空吗？想约你看电影。"
    result = _truncate_at_sentence_boundary(text, 10)
    assert result == "你周末有空吗？"


def test_truncate_at_exclamation_boundary():
    """感叹号也算句末."""
    text = "好开心！今天真是太棒了！"
    result = _truncate_at_sentence_boundary(text, 5)
    assert result == "好开心！"


def test_no_sentence_end_falls_back_to_hard_cut():
    """全是逗号没句末符 → 退回硬切, 不能死循环."""
    text = "今天，天气，真好，我想，出去，走走，路上，太晒，吃个，西瓜"
    result = _truncate_at_sentence_boundary(text, 10)
    # 没有句末符 → 退回硬切到 max_len
    assert len(result) == 10


def test_truncate_skips_too_early_boundary():
    """如果句末符在 <max_len/2 处, 切掉太多内容 → 退回硬切到 max_len.
    生产场景: 第一句很短的问候后接长内容, 不能只回第一句."""
    text = "嗨。" + "a" * 100  # "嗨。" 在 idx=2, max=50, 50/2=25, 2 < 25 → 退回硬切
    result = _truncate_at_sentence_boundary(text, 50)
    assert len(result) == 50, f"应硬切到 50, got len={len(result)}"


def test_production_bug_repro():
    """直接复现 trace 019dec46 的截断场景: 长 LLM 输出被切到中字."""
    text = (
        "哇，我现在正躺在废弃铁轨上呢！说出来你可能不信——我刚随便跳上一辆"
        "公交车坐到终点站，结果发现是个有旧铁轨的郊野公园，铁轨都生锈了，"
        "枕木缝里长满了野花。现在我就枕着铁轨听风声，远处好像还有火车的嗡嗡声，"
        "特别像那种老电影里的场景。\n\n"
        "明天是你生日？！天哪你怎么不早说！我正在想送你什么——我刚才在旧货市场"
        "淘到一个生锈的齿轮和半卷铜丝，要不要给你焊个蒸汽朋克风的小玩意儿？"
        "或者…你现在有什么想要的吗？"
    )
    result = _truncate_at_sentence_boundary(text, 120)

    # 必须以句末符结尾, 不能切到 "明天是你生" 这样的中字
    assert result.endswith(("。", "！", "？", "！", "？")), (
        f"必须切在句末符边界, 不能裸切; 实际结尾: {result[-10:]!r}"
    )
    # 不能含 "明天是你生" 这样的中段截断
    assert not result.endswith("明天是你生"), (
        "生产 bug 复现失败: 仍切到 '明天是你生' 这样的中字"
    )


@pytest.mark.asyncio
async def test_render_prompt_uses_sentence_boundary():
    """端到端: render_prompt 调 invoke_fn 拿超长文本, max_chars 限制, 必须切句末."""
    from unittest.mock import AsyncMock, patch
    from app.services.prompting.utils import render_prompt

    long_text = "第一句很短。" + "之后是非常非常非常非常非常非常长的一段话内容" * 10

    async def _fake_invoke(prompt: str) -> str:
        return long_text

    with patch(
        "app.services.prompting.utils.get_prompt_text",
        new=AsyncMock(return_value="dummy template {x}"),
    ):
        result = await render_prompt(
            "dummy_key", {"x": "y"},
            _fake_invoke,
            max_chars=80,
        )

    assert isinstance(result, str)
    assert len(result) <= 80
    # 必须切在句末 — 不能跨字
    assert result.endswith(("。", "！", "？", "！", "？")) or len(result) == 80
