"""l2_dynamics 兼容入口必须按 user 作用域委托给纯衰减维护."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_adjust_side_delegates_to_scoped_decay_backstop():
    from app.services.memory.lifecycle import l2_dynamics
    expected={"total":2,"promoted":0,"demoted":1,"adjusted":1}
    with patch("app.services.memory.lifecycle.lazy_update.sweep_stale_values", new=AsyncMock(return_value={"user":expected})) as sweep:
        assert await l2_dynamics._adjust_side("user","user-1") == expected
    sweep.assert_awaited_once_with(sources=("user",),user_id="user-1")
