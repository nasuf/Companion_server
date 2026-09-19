"""Career picker helpers used by template create."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.career import get_active_career_by_id, pick_random_active_careers


def _career(career_id: str, title: str, status: str = "active"):
    return SimpleNamespace(
        id=career_id,
        title=title,
        duties="d",
        socialValue="s",
        clients="c",
        status=status,
    )


@pytest.mark.asyncio
async def test_get_active_career_by_id_skips_archived():
    with patch("app.services.career.db") as mock_db:
        mock_db.careertemplate.find_unique = AsyncMock(
            return_value=_career("c1", "咖啡师", status="archived"),
        )
        assert await get_active_career_by_id("c1") is None


@pytest.mark.asyncio
async def test_pick_random_active_careers_unique_then_cycles():
    rows = [_career("a", "A"), _career("b", "B")]
    with patch("app.services.career.db") as mock_db:
        mock_db.careertemplate.find_many = AsyncMock(return_value=rows)
        two = await pick_random_active_careers(2)
        four = await pick_random_active_careers(4)

    assert {row["title"] for row in two} == {"A", "B"}
    assert len(four) == 4
    assert {row["title"] for row in four} == {"A", "B"}


@pytest.mark.asyncio
async def test_pick_random_active_careers_empty_pool():
    with patch("app.services.career.db") as mock_db:
        mock_db.careertemplate.find_many = AsyncMock(return_value=[])
        assert await pick_random_active_careers(3) == [None, None, None]
