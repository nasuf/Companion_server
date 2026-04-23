"""Tests for character profile initialization — age clamp per spec §1.3."""

from datetime import date

import pytest

from app.api.admin.character import AGE_MAX, AGE_MIN, clamp_agent_age


def _age_from_birthday(birthday_iso: str, today_iso: str = "2026-04-23") -> int:
    bd = date.fromisoformat(birthday_iso)
    today = date.fromisoformat(today_iso)
    return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))


class TestAgeClampToSpecRange:
    """Spec §1.3: AI 年龄在 20-29 之间随机. clamp_agent_age 是最后防线."""

    def test_birthday_in_valid_range(self):
        """正常 24 岁 → 24."""
        assert clamp_agent_age(_age_from_birthday("2002-01-01")) == 24

    def test_birthday_too_old(self):
        """LLM 生成 1990 年生 → 36 岁 → clamp 到 29."""
        assert clamp_agent_age(_age_from_birthday("1990-05-15")) == AGE_MAX

    def test_birthday_too_young(self):
        """LLM 生成 2010 年生 → 16 岁 → clamp 到 20."""
        assert clamp_agent_age(_age_from_birthday("2010-03-01")) == AGE_MIN

    def test_birthday_edge_20(self):
        """恰好 20 岁不被进一步 clamp."""
        assert clamp_agent_age(_age_from_birthday("2006-01-01")) == AGE_MIN

    def test_birthday_edge_29(self):
        """恰好 29 岁不被进一步 clamp."""
        assert clamp_agent_age(_age_from_birthday("1997-01-01")) == AGE_MAX

    def test_birthday_today_age_0(self):
        """荒诞输入 (今天生) → age=0 → clamp 到 20."""
        assert clamp_agent_age(_age_from_birthday("2026-04-23")) == AGE_MIN

    def test_birthday_future(self):
        """未来日期 → 负数 age → clamp 到 20."""
        assert clamp_agent_age(_age_from_birthday("2030-01-01")) == AGE_MIN


@pytest.mark.parametrize(
    "bd,today,expected",
    [
        ("1995-06-01", "2026-04-23", 29),  # 30.8 → clamp 29
        ("2004-12-31", "2026-04-23", 21),
        ("2008-07-01", "2026-04-23", 20),  # 17.8 → clamp 20
    ],
)
def test_parametric_clamp(bd: str, today: str, expected: int):
    assert clamp_agent_age(_age_from_birthday(bd, today)) == expected


def test_clamp_bounds_exposed_as_constants():
    """Spec §1.3 区间常量必须被 character.py 导出, 避免 magic number."""
    assert AGE_MIN == 20
    assert AGE_MAX == 29
    assert clamp_agent_age(0) == AGE_MIN
    assert clamp_agent_age(99) == AGE_MAX
