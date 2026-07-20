"""Cloned agents must get a day-one schedule.

The daily-schedule cron runs pre-dawn only; without this dispatch an afternoon
signup keeps ai_status=None (no delay profile / 隐性状态约束) until tomorrow.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_template.clone import _dispatch_day_one_schedule


def _agent():
    return SimpleNamespace(id="agent-1", name="小伴", mbti={"type": "ISFP"})


@pytest.mark.asyncio
async def test_dispatch_generates_schedule_in_background():
    fb = MagicMock()
    gen = AsyncMock()
    overview = AsyncMock(return_value="一个普洱客服员的生活")
    with patch("app.services.runtime.tasks.fire_background", fb), \
         patch("app.services.schedule_domain.schedule.generate_daily_schedule", gen), \
         patch("app.services.schedule_domain.schedule.get_life_overview", overview), \
         patch("app.services.mbti.get_mbti", return_value={"type": "ISFP"}):
        _dispatch_day_one_schedule(_agent(), "user-1")

        fb.assert_called_once()
        # Drive the captured background coroutine to completion.
        await fb.call_args.args[0]

    gen.assert_awaited_once()
    args = gen.await_args
    assert args.args[0] == "agent-1"
    assert args.args[1] == "小伴"
    assert args.kwargs["user_id"] == "user-1"
    assert args.kwargs["life_overview"] == "一个普洱客服员的生活"


@pytest.mark.asyncio
async def test_generation_failure_is_swallowed():
    """Best-effort: an LLM/DB failure inside the task must not propagate."""
    fb = MagicMock()
    gen = AsyncMock(side_effect=RuntimeError("llm down"))
    with patch("app.services.runtime.tasks.fire_background", fb), \
         patch("app.services.schedule_domain.schedule.generate_daily_schedule", gen), \
         patch("app.services.schedule_domain.schedule.get_life_overview", AsyncMock(return_value=None)), \
         patch("app.services.mbti.get_mbti", return_value=None):
        _dispatch_day_one_schedule(_agent(), "user-1")
        await fb.call_args.args[0]  # must not raise


def test_dispatch_failure_is_swallowed():
    """fire_background itself failing must not break the clone flow."""
    with patch(
        "app.services.runtime.tasks.fire_background",
        MagicMock(side_effect=RuntimeError("loop closed")),
    ):
        _dispatch_day_one_schedule(_agent(), "user-1")  # no raise
