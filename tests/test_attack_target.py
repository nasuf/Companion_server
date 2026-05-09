from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.intent_replies import attack_target_classify
from app.services.prompting import defaults


@pytest.mark.asyncio
async def test_attack_target_family_name_question_is_not_attack():
    with patch(
        "app.services.chat.intent_replies._classify_label",
        AsyncMock(return_value="攻击AI"),
    ) as classify_mock:
        result = await attack_target_classify(
            "没，你妈妈叫什么还记得吗",
            "用户: 你有兄弟姐妹吗\nAI: 没有呢，我是独生女",
        )

    assert result == "无负面意图"
    classify_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_attack_target_family_insult_still_uses_classifier():
    with patch(
        "app.services.chat.intent_replies._classify_label",
        AsyncMock(return_value="攻击AI"),
    ) as classify_mock:
        result = await attack_target_classify("去你妈的")

    assert result == "攻击AI"
    classify_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_attack_target_mixed_family_question_with_insult_uses_classifier():
    with patch(
        "app.services.chat.intent_replies._classify_label",
        AsyncMock(return_value="攻击AI"),
    ) as classify_mock:
        result = await attack_target_classify("你妈妈叫什么，傻逼")

    assert result == "攻击AI"
    classify_mock.assert_awaited_once()


def test_attack_target_prompt_guards_family_fact_questions():
    prompt = defaults.ATTACK_TARGET_PROMPT

    assert "你妈妈叫什么还记得吗" in prompt
    assert "问 AI 的家庭/身份事实" in prompt
    assert '"你妈" /' not in prompt
