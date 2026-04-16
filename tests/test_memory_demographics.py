"""Unit tests for demographics helpers (constellation / zodiac / distributions)."""

from collections import Counter

import pytest

from app.services.memory.demographics import (
    derive_constellation,
    derive_zodiac,
    sample_blood_type,
    sample_ethnicity,
)


class TestConstellation:
    @pytest.mark.parametrize("birthday, expected", [
        ("1998-01-19", "摩羯座"),
        ("1998-01-20", "摩羯座"),
        ("1998-01-21", "水瓶座"),
        ("1998-02-18", "水瓶座"),
        ("1998-02-19", "水瓶座"),
        ("1998-02-20", "双鱼座"),
        ("1998-03-20", "双鱼座"),
        ("1998-03-21", "双鱼座"),
        ("1998-03-22", "白羊座"),
        ("1998-06-15", "双子座"),
        ("1998-08-23", "狮子座"),
        ("1998-08-24", "处女座"),
        ("1998-12-21", "射手座"),
        ("1998-12-22", "射手座"),
        ("1998-12-23", "摩羯座"),
        ("1998-12-31", "摩羯座"),
    ])
    def test_edge_cases(self, birthday, expected):
        assert derive_constellation(birthday) == expected

    @pytest.mark.parametrize("bad", [None, "", "not-a-date", "1998/06/15", "abc"])
    def test_invalid_input_returns_none(self, bad):
        assert derive_constellation(bad) is None


class TestZodiac:
    @pytest.mark.parametrize("birthday, expected", [
        ("2000-06-15", "龙"),   # anchor
        ("2001-06-15", "蛇"),
        ("2012-06-15", "龙"),   # one cycle later
        ("1988-06-15", "龙"),   # one cycle earlier
        ("1999-06-15", "兔"),
        ("1998-06-15", "虎"),
        ("1997-06-15", "牛"),
        ("1996-06-15", "鼠"),
    ])
    def test_cycles(self, birthday, expected):
        assert derive_zodiac(birthday) == expected

    @pytest.mark.parametrize("bad", [None, "", "bad"])
    def test_invalid(self, bad):
        assert derive_zodiac(bad) is None


class TestBloodType:
    def test_returns_valid_blood_type(self):
        for seed in ("a", "b", "c", "d", "e"):
            assert sample_blood_type(seed) in {"O型", "A型", "B型", "AB型"}

    def test_seeded_deterministic(self):
        assert sample_blood_type("same-seed") == sample_blood_type("same-seed")

    def test_distribution_roughly_matches(self):
        """Over many samples, distribution should lean toward O型 (34%) > A型 (31%)."""
        counts: Counter[str] = Counter()
        for i in range(2000):
            counts[sample_blood_type(f"seed-{i}")] += 1
        # All four types should appear.
        assert set(counts.keys()) == {"O型", "A型", "B型", "AB型"}
        # O型 is most common, AB型 rarest.
        assert counts["O型"] > counts["AB型"]
        # AB型 should be clearly rarer than others (true weight 0.08).
        assert counts["AB型"] < counts["A型"]


class TestEthnicity:
    def test_returns_known_value(self):
        """Sample should be one of the listed ethnicities (at minimum 汉族 dominant)."""
        results = {sample_ethnicity(f"s-{i}") for i in range(100)}
        assert "汉族" in results

    def test_seeded_deterministic(self):
        assert sample_ethnicity("foo") == sample_ethnicity("foo")

    def test_distribution_majority_han(self):
        counts: Counter[str] = Counter()
        for i in range(2000):
            counts[sample_ethnicity(f"seed-{i}")] += 1
        # 汉族 is 91.5% — should dominate.
        assert counts["汉族"] / sum(counts.values()) > 0.80
