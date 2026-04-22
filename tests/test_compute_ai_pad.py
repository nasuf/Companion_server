"""Spec §3.2 AIPAD值判断 单测：compute_ai_pad 行为覆盖。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_compute_ai_pad_happy_path():
    """LLM 返回合法 JSON → 直接返回 clamp 后的 PAD。"""
    from app.services.relationship.emotion import compute_ai_pad

    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="[prompt]{current_status}{current_activity}{recent_context}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(return_value={"pleasure": 0.4, "arousal": 0.6, "dominance": 0.55}),
    ):
        pad = await compute_ai_pad(
            current_time="2026-04-22 15:30 周三",
            schedule_status="空闲",
            current_activity="散步",
            recent_context="用户: 今天天气真好\nAI: 是啊很适合散步",
        )

    assert pad == {"pleasure": 0.4, "arousal": 0.6, "dominance": 0.55}


@pytest.mark.asyncio
async def test_compute_ai_pad_clamps_out_of_range_values():
    """LLM 返回越界值 → 按 PAD 范围裁剪。"""
    from app.services.relationship.emotion import compute_ai_pad

    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="[p]{current_status}{current_activity}{recent_context}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(return_value={"pleasure": 1.8, "arousal": -0.3, "dominance": 2.0}),
    ):
        pad = await compute_ai_pad(
            current_time="2026-04-22 03:00 周三",
            schedule_status="空闲", current_activity="发呆", recent_context="（无）",
        )

    assert pad["pleasure"] == 1.0       # 上限 1.0
    assert pad["arousal"] == 0.0        # 下限 0.0
    assert pad["dominance"] == 1.0      # 上限 1.0


@pytest.mark.asyncio
async def test_compute_ai_pad_llm_failure_falls_back_to_defaults():
    """LLM 调用抛异常 → 返回中性默认。"""
    from app.services.relationship.emotion import compute_ai_pad

    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="[p]{current_status}{current_activity}{recent_context}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(side_effect=RuntimeError("llm down")),
    ):
        pad = await compute_ai_pad(
            current_time="2026-04-22 02:00 周三",
            schedule_status="睡眠", current_activity="睡觉", recent_context="（无）",
        )

    assert pad == {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}


@pytest.mark.asyncio
async def test_compute_ai_pad_feeds_spec_inputs_into_prompt():
    """prompt format 接收的参数必须严格对齐 spec §3.2 的 3 项参考信息。"""
    from app.services.relationship.emotion import compute_ai_pad

    # 捕获 format 实际传入的模板文本
    captured_prompts: list[str] = []

    async def _fake_invoke(model, prompt):
        captured_prompts.append(prompt)
        return {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}

    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="T={current_time} S={current_status} A={current_activity} C={recent_context}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(side_effect=_fake_invoke),
    ):
        await compute_ai_pad(
            current_time="2026-04-22 10:15 周三",
            schedule_status="忙碌",
            current_activity="开会",
            recent_context="用户: 你在干嘛",
        )

    assert captured_prompts == ["T=2026-04-22 10:15 周三 S=忙碌 A=开会 C=用户: 你在干嘛"]
