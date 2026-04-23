"""Tests for character profile initialization — age clamp per spec §1.3."""

from datetime import date

import pytest


class TestAgeClampToSpecRange:
    """Spec §1.3: AI 年龄在 20-29 之间随机.

    实现: LLM 生成 birthday → 反推 age → 用 max(20, min(29, age)) 硬钳.
    LLM prompt 虽然有区间 hint, 不保证总命中, runtime clamp 是最后防线.
    """

    def _clamp(self, birthday_iso: str, today_iso: str = "2026-04-23") -> int:
        """复现 character.py:372-384 的年龄反推 + clamp 逻辑."""
        bd = date.fromisoformat(birthday_iso)
        today = date.fromisoformat(today_iso)
        age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        return max(20, min(29, age))

    def test_birthday_in_valid_range(self):
        """正常 24 岁 → 24."""
        assert self._clamp("2002-01-01") == 24

    def test_birthday_too_old(self):
        """LLM 生成 1990 年生 → 36 岁 → clamp 到 29."""
        assert self._clamp("1990-05-15") == 29

    def test_birthday_too_young(self):
        """LLM 生成 2010 年生 → 16 岁 → clamp 到 20."""
        assert self._clamp("2010-03-01") == 20

    def test_birthday_edge_20(self):
        """恰好 20 岁不被进一步 clamp."""
        # 2006 - 2026 = 20 (生日已过)
        assert self._clamp("2006-01-01") == 20

    def test_birthday_edge_29(self):
        """恰好 29 岁不被进一步 clamp."""
        assert self._clamp("1997-01-01") == 29

    def test_birthday_today_age_0(self):
        """荒诞输入 (今天生) → age=0 → clamp 到 20."""
        assert self._clamp("2026-04-23") == 20

    def test_birthday_future(self):
        """未来日期 → 负数 age → clamp 到 20."""
        assert self._clamp("2030-01-01") == 20


@pytest.mark.parametrize(
    "bd,today,expected",
    [
        ("1995-06-01", "2026-04-23", 29),  # 30.8 → clamp 29
        ("2004-12-31", "2026-04-23", 21),
        ("2008-07-01", "2026-04-23", 20),  # 17.8 → clamp 20
    ],
)
def test_parametric_clamp(bd: str, today: str, expected: int):
    bd_d = date.fromisoformat(bd)
    today_d = date.fromisoformat(today)
    age = today_d.year - bd_d.year - ((today_d.month, today_d.day) < (bd_d.month, bd_d.day))
    assert max(20, min(29, age)) == expected
