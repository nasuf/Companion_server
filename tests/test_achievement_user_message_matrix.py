from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.achievements.events import UserMessageAchievementEvent
from app.services.achievements.rules import user_message_rules


LOCAL_TZ = timezone(timedelta(hours=8))


def _at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(
        2026, 6, 1, hour, minute, second, tzinfo=LOCAL_TZ
    ).astimezone(UTC)


def _row(text: str, at: datetime) -> dict:
    return {"content": text, "created_at": at}


@dataclass(frozen=True)
class UserScenario:
    achievement_id: int
    text: str = "普通消息"
    at: datetime = field(default_factory=lambda: _at(12))
    today: list[dict] | None = None
    user_chars: int = 1
    birthday: tuple[int, int] | None = None
    event_counts: dict[str, int] = field(default_factory=dict)
    schedule_bucket: str | None = None


def _question_rows(count: int) -> list[dict]:
    return [_row(f"问题{index}?", _at(10, index)) for index in range(count)]


def _scene_rows(complete: bool) -> list[dict]:
    rows = [
        _row("上午", _at(9, 30)),
        _row("晚上", _at(19, 30)),
    ]
    if complete:
        rows.append(_row("深夜", _at(23, 30)))
    return rows


POSITIVE_USER_CASES = [
    UserScenario(1),
    UserScenario(9, text="哈哈"),
    UserScenario(21, text="你之后有什么安排"),
    UserScenario(27, text="好呀～"),
    UserScenario(31, today=_question_rows(5)),
    UserScenario(48, schedule_bucket="sleep", event_counts={"sleep_wakeup_period": 10}),
    UserScenario(52, at=_at(2, 30), event_counts={"late_night_day": 10}),
    UserScenario(53, at=_at(5, 30), event_counts={"early_morning_day": 10}),
    UserScenario(58, text="晚安啦", at=_at(22), event_counts={"goodnight_late": 3}),
    UserScenario(60, user_chars=10000),
    UserScenario(63, today=_scene_rows(True)),
    UserScenario(64, today=[_row("消息", _at(10))] * 100),
    UserScenario(69, text="一二三四五六七八九十甲", at=_at(14, 15)),
    UserScenario(72, text="嗯！", event_counts={"um_message": 50}),
    UserScenario(73, today=[_row("消息", _at(10))] * 200),
    UserScenario(76, at=_at(13, 14)),
    UserScenario(82, at=_at(5, 20)),
    UserScenario(83, text="生日快乐", birthday=(6, 1)),
    UserScenario(94, at=_at(23, 59, 58), event_counts={"midnight_edge_message": 10}),
    UserScenario(95, at=_at(6, 11), birthday=(6, 11)),
]


NEGATIVE_USER_CASES = [
    UserScenario(9, text="哈哈！"),
    UserScenario(21, text="我已经有计划了"),
    UserScenario(27, text="好呀~"),
    UserScenario(31, today=_question_rows(4)),
    UserScenario(48, schedule_bucket="sleep", event_counts={"sleep_wakeup_period": 9}),
    UserScenario(52, at=_at(1, 59), event_counts={"late_night_day": 10}),
    UserScenario(53, at=_at(7), event_counts={"early_morning_day": 10}),
    UserScenario(58, text="晚安啦", at=_at(21, 59), event_counts={"goodnight_late": 3}),
    UserScenario(60, user_chars=9999),
    UserScenario(63, today=_scene_rows(False)),
    UserScenario(64, today=[_row("消息", _at(10))] * 99),
    UserScenario(69, text="一二三四五六七八九十", at=_at(14, 15)),
    UserScenario(72, text="嗯！", event_counts={"um_message": 49}),
    UserScenario(73, today=[_row("消息", _at(10))] * 199),
    UserScenario(76, at=_at(13, 15)),
    UserScenario(82, at=_at(5, 21)),
    UserScenario(83, text="生日快乐", birthday=(6, 2)),
    UserScenario(94, at=_at(0, 0, 3), event_counts={"midnight_edge_message": 10}),
    UserScenario(95, at=_at(6, 12), birthday=(6, 11)),
]


async def _evaluate(scenario: UserScenario) -> set[int]:
    today = scenario.today or [_row(scenario.text, scenario.at)]

    async def _event_count(_user_id: str, _agent_id: str, event_type: str) -> int:
        return scenario.event_counts.get(event_type, 0)

    observation = None
    if scenario.schedule_bucket:
        observation = SimpleNamespace(
            bucket=scenario.schedule_bucket,
            period_key="2026-06-01:23:00-07:00:sleep",
        )

    with (
        patch.object(
            user_message_rules,
            "_message_created_at",
            AsyncMock(return_value=scenario.at),
        ),
        patch.object(
            user_message_rules,
            "_day_user_messages",
            AsyncMock(return_value=today),
        ),
        patch.object(
            user_message_rules,
            "_day_role_char_counts",
            AsyncMock(return_value=(scenario.user_chars, 0)),
        ),
        patch.object(
            user_message_rules,
            "_birthday_mmdd",
            AsyncMock(return_value=scenario.birthday),
        ),
        patch.object(
            user_message_rules,
            "_event_count",
            AsyncMock(side_effect=_event_count),
        ),
        patch.object(
            user_message_rules,
            "record_schedule_status_chat",
            AsyncMock(return_value=observation),
        ),
        patch.object(
            user_message_rules,
            "has_schedule_status_streak",
            AsyncMock(return_value=False),
        ),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await user_message_rules.evaluate_user_message(
            UserMessageAchievementEvent(
                user_id="u1",
                agent_id="a1",
                workspace_id="w1",
                conversation_id="c1",
                message_id="m1",
                text=scenario.text,
            )
        )

    return {call.kwargs["achievement_id"] for call in unlock.await_args_list}


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", POSITIVE_USER_CASES, ids=lambda item: str(item.achievement_id))
async def test_user_message_achievement_positive_matrix(scenario: UserScenario):
    assert scenario.achievement_id in await _evaluate(scenario)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", NEGATIVE_USER_CASES, ids=lambda item: str(item.achievement_id))
async def test_user_message_achievement_negative_matrix(scenario: UserScenario):
    assert scenario.achievement_id not in await _evaluate(scenario)
