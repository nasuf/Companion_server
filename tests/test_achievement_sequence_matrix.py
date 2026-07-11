from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.achievements.rules import user_message_rules


def _rows(texts: list[str]) -> list[dict]:
    return [{"content": text} for text in texts]


async def _unlocked_ids(texts: list[str]) -> set[int]:
    with patch.object(
        user_message_rules,
        "unlock_achievement",
        AsyncMock(),
    ) as unlock:
        await user_message_rules._check_sequences(
            "u1",
            "a1",
            "w1",
            "c1",
            _rows(texts),
        )
    return {call.kwargs["achievement_id"] for call in unlock.await_args_list}


POSITIVE_SEQUENCE_CASES = [
    (8, ["一", "二二", "三三三", "四四四四四"]),
    (19, ["。同甲", "！同乙", "同丙"]),
    (37, [f"{char}内容" for char in "甲乙丙丁戊己庚辛壬癸"]),
    (38, ["一", "二二", "三三三", "四四四四", "五五五五五"]),
    (41, ["五五五五五", "四四四四", "三三三", "二二", "一"]),
    (42, ["长" * 12, "短" * 4] * 3),
    (47, [f"第{index}颗心" for index in range(6)]),
    (57, ["重复内容", "重复内容", "重复内容"]),
    (70, [f"问题{index}?补充" for index in range(10)]),
    (71, ["完全相同！"] * 10),
]


NEGATIVE_SEQUENCE_CASES = [
    (8, ["一", "二", "三", "超过五个字符"]),
    (19, ["同甲", "异乙", "同丙"]),
    (37, [f"{char}内容" for char in "甲乙丙丁戊己庚辛壬甲"]),
    (38, ["一", "二二", "二二", "四四四四", "五五五五五"]),
    (41, ["五五五五五", "四四四四", "四四四四", "二二", "一"]),
    (42, ["长" * 12, "短" * 4] * 2 + ["普通消息"]),
    (47, ["共同甲", "共同乙", "共同丙", "共同丁", "共同戊", "毫无关联"]),
    (57, ["重复内容", "重复内容", "不同内容"]),
    (70, [f"问题{index}?" for index in range(9)] + ["没有问号"]),
    (71, ["完全相同！"] * 9 + ["完全相同"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "texts"), POSITIVE_SEQUENCE_CASES)
async def test_sequence_achievement_positive_cases(
    achievement_id: int,
    texts: list[str],
):
    assert achievement_id in await _unlocked_ids(texts)


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "texts"), NEGATIVE_SEQUENCE_CASES)
async def test_sequence_achievement_negative_cases(
    achievement_id: int,
    texts: list[str],
):
    assert achievement_id not in await _unlocked_ids(texts)
