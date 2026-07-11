from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.achievements.rules import daily_rollup_rules


LOCAL_TZ = timezone(timedelta(hours=8))
LOCAL_DAY = datetime(2026, 6, 1, tzinfo=LOCAL_TZ)
PAIR = {
    "user_id": "u1",
    "agent_id": "a1",
    "workspace_id": "w1",
    "conversation_id": "c1",
}


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 1, hour, minute, tzinfo=LOCAL_TZ).astimezone(UTC)


def _messages(texts: list[str], *, start_hour: int = 10) -> list[dict]:
    return [
        {
            "content": text,
            "created_at": _at(start_hour, index % 60),
        }
        for index, text in enumerate(texts)
    ]


@dataclass(frozen=True)
class DailyScenario:
    achievement_id: int
    rows: list[dict]
    user_chars: int = 1
    ai_chars: int = 0
    role_counts: tuple[int, int] = (1, 0)
    unique_48h: bool = False
    quick_replies: bool = False
    previous_opener_match: bool = False
    day_flags: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    schedule_streaks: frozenset[int] = field(default_factory=frozenset)


POSITIVE_DAILY_CASES = [
    DailyScenario(2, _messages(["唯一消息"])),
    DailyScenario(5, _messages(["第一天", "第二天"]), unique_48h=True),
    DailyScenario(
        6,
        _messages(["晚上甲", "晚上乙", "晚上丙"], start_hour=18),
        day_flags=frozenset({("evening_3_day", 3)}),
    ),
    DailyScenario(7, _messages(["午间甲", "午间乙"], start_hour=12)),
    DailyScenario(
        18,
        [
            {"content": "早", "created_at": _at(7)},
            {"content": "晚", "created_at": _at(19)},
        ],
    ),
    DailyScenario(
        26,
        _messages(["同一句开场", "其他"]),
        previous_opener_match=True,
    ),
    DailyScenario(28, _messages(["一", "二", "三"])),
    DailyScenario(33, _messages(["短" * 10] * 20), user_chars=200),
    DailyScenario(36, _messages([f"纯文字{index}" for index in range(20)])),
    DailyScenario(43, _messages(["两字", "中间内容", "三字"])),
    DailyScenario(
        44,
        _messages(["用户"] * 10),
        user_chars=10,
        ai_chars=31,
        role_counts=(10, 10),
    ),
    DailyScenario(45, _messages(["偶数"] * 10)),
    DailyScenario(46, _messages(["奇数一"] * 10)),
    DailyScenario(
        50,
        [
            {"content": "早", "created_at": _at(7)},
            {"content": "晚", "created_at": _at(19)},
        ],
        day_flags=frozenset({("span_12h_day", 3)}),
    ),
    DailyScenario(
        51,
        _messages([f"纯文字{index}" for index in range(20)]),
        day_flags=frozenset({("clean_chat_day", 2)}),
    ),
    DailyScenario(
        56,
        _messages(["白天甲", "白天乙", "白天丙"], start_hour=10),
        day_flags=frozenset({("sleep_respect_day", 7)}),
    ),
    DailyScenario(60, _messages(["万字"]), user_chars=10000),
    DailyScenario(62, _messages(["回复"]), quick_replies=True),
    DailyScenario(74, _messages(["状态"]), schedule_streaks=frozenset({7})),
    DailyScenario(80, _messages(["总字数"]), user_chars=40, ai_chars=60),
    DailyScenario(
        90,
        [
            {"content": "首", "created_at": _at(1, 20)},
            {"content": "尾", "created_at": _at(2, 10)},
        ],
    ),
    DailyScenario(93, _messages(["状态"]), schedule_streaks=frozenset({30})),
]


NEGATIVE_DAILY_CASES = [
    DailyScenario(2, _messages(["一", "二"])),
    DailyScenario(5, _messages(["第一天", "第二天"]), unique_48h=False),
    DailyScenario(6, _messages(["晚上甲", "晚上乙", "晚上丙"], start_hour=18)),
    DailyScenario(
        7,
        [
            {"content": "午间", "created_at": _at(12)},
            {"content": "越界", "created_at": _at(14)},
        ],
    ),
    DailyScenario(
        18,
        [
            {"content": "早", "created_at": _at(7)},
            {"content": "还不够晚", "created_at": _at(18, 59)},
        ],
    ),
    DailyScenario(26, _messages(["同一句开场", "其他"])),
    DailyScenario(28, _messages(["一", "二"])),
    DailyScenario(33, _messages(["短" * 10] * 19 + ["超过十个文字啊啊啊啊啊"])),
    DailyScenario(36, _messages(["纯文字"] * 19 + ["带标点！"])),
    DailyScenario(43, _messages(["两字", "三个字"])),
    DailyScenario(
        44,
        _messages(["用户"] * 10),
        user_chars=10,
        ai_chars=31,
        role_counts=(10, 9),
    ),
    DailyScenario(45, _messages(["偶数"] * 9 + ["奇数一"])),
    DailyScenario(46, _messages(["奇数一"] * 9 + ["偶数"])),
    DailyScenario(
        50,
        [
            {"content": "早", "created_at": _at(7)},
            {"content": "晚", "created_at": _at(19)},
        ],
    ),
    DailyScenario(51, _messages([f"纯文字{index}" for index in range(20)])),
    DailyScenario(56, _messages(["白天甲", "白天乙", "白天丙"], start_hour=10)),
    DailyScenario(60, _messages(["不足"]), user_chars=9999),
    DailyScenario(62, _messages(["回复"]), quick_replies=False),
    DailyScenario(74, _messages(["状态"])),
    DailyScenario(80, _messages(["总字数"]), user_chars=40, ai_chars=59),
    DailyScenario(
        90,
        [
            {"content": "首", "created_at": _at(1, 20)},
            {"content": "尾", "created_at": _at(2, 11)},
        ],
    ),
    DailyScenario(93, _messages(["状态"]), schedule_streaks=frozenset({7})),
]


async def _run_scenario(scenario: DailyScenario) -> set[int]:
    async def _flag(
        _user_id: str,
        _agent_id: str,
        event_type: str,
        _local_day: datetime,
        days: int,
    ) -> bool:
        return (event_type, days) in scenario.day_flags

    async def _schedule_streak(**kwargs) -> bool:
        return int(kwargs["days"]) in scenario.schedule_streaks

    with (
        patch.object(
            daily_rollup_rules,
            "_day_user_messages",
            AsyncMock(return_value=scenario.rows),
        ),
        patch.object(
            daily_rollup_rules,
            "_day_role_char_counts",
            AsyncMock(return_value=(scenario.user_chars, scenario.ai_chars)),
        ),
        patch.object(
            daily_rollup_rules,
            "_day_role_message_counts",
            AsyncMock(return_value=scenario.role_counts),
        ),
        patch.object(
            daily_rollup_rules,
            "_day_has_all_quick_replies",
            AsyncMock(return_value=scenario.quick_replies),
        ),
        patch.object(
            daily_rollup_rules,
            "_has_complete_unique_48h_window",
            AsyncMock(return_value=scenario.unique_48h),
        ),
        patch.object(
            daily_rollup_rules,
            "_previous_day_first_message_matches",
            AsyncMock(return_value=scenario.previous_opener_match),
        ),
        patch.object(
            daily_rollup_rules,
            "_has_consecutive_day_flags",
            AsyncMock(side_effect=_flag),
        ),
        patch.object(
            daily_rollup_rules,
            "has_schedule_status_streak",
            AsyncMock(side_effect=_schedule_streak),
        ),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(
            daily_rollup_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await daily_rollup_rules._run_daily_rollup_for_pair(
            PAIR,
            LOCAL_DAY,
            _at(0),
        )

    return {call.kwargs["achievement_id"] for call in unlock.await_args_list}


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", POSITIVE_DAILY_CASES, ids=lambda item: str(item.achievement_id))
async def test_daily_achievement_positive_matrix(scenario: DailyScenario):
    assert scenario.achievement_id in await _run_scenario(scenario)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", NEGATIVE_DAILY_CASES, ids=lambda item: str(item.achievement_id))
async def test_daily_achievement_negative_matrix(scenario: DailyScenario):
    assert scenario.achievement_id not in await _run_scenario(scenario)


@pytest.mark.asyncio
async def test_daily_rollup_isolates_one_pair_failure_from_following_pairs():
    pairs = [
        {**PAIR, "user_id": "broken-user"},
        {**PAIR, "user_id": "healthy-user"},
    ]
    with (
        patch.object(
            daily_rollup_rules.db,
            "query_raw",
            AsyncMock(return_value=pairs),
        ),
        patch.object(
            daily_rollup_rules,
            "_run_daily_rollup_for_pair",
            AsyncMock(side_effect=[RuntimeError("broken"), None]),
        ) as run_pair,
    ):
        with pytest.raises(RuntimeError, match="1 pair"):
            await daily_rollup_rules.run_daily_rollup(LOCAL_DAY)

    assert run_pair.await_count == 2
