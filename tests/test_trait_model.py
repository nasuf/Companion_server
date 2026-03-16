"""七维人格模型转换服务测试。"""

from unittest.mock import MagicMock

from app.services.trait_model import (
    big_five_to_seven_dim,
    get_dim,
    get_seven_dim,
    seven_dim_to_big_five,
)


class TestSevenDimToBigFive:
    def test_neutral(self):
        seven = {"活泼度": 50, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50}
        b5 = seven_dim_to_big_five(seven)
        assert abs(b5["extraversion"] - 0.5) < 0.01
        assert abs(b5["openness"] - 0.5) < 0.01
        assert abs(b5["conscientiousness"] - 0.5) < 0.01

    def test_high_lively(self):
        seven = {"活泼度": 100, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50}
        b5 = seven_dim_to_big_five(seven)
        assert b5["extraversion"] == 1.0

    def test_clamped(self):
        seven = {"活泼度": 200, "理性度": -50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50}
        b5 = seven_dim_to_big_five(seven)
        assert b5["extraversion"] == 1.0
        for v in b5.values():
            assert 0 <= v <= 1


class TestBigFiveToSevenDim:
    def test_neutral(self):
        b5 = {"extraversion": 0.5, "openness": 0.5, "conscientiousness": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}
        seven = big_five_to_seven_dim(b5)
        assert seven["活泼度"] == 50
        assert seven["脑洞度"] == 50
        assert seven["计划度"] == 50

    def test_all_values_in_range(self):
        b5 = {"extraversion": 1.0, "openness": 0.0, "conscientiousness": 1.0, "agreeableness": 0.0, "neuroticism": 1.0}
        seven = big_five_to_seven_dim(b5)
        for v in seven.values():
            assert 0 <= v <= 100


class TestRoundTrip:
    def test_roundtrip_preserves_key_dims(self):
        original = {"活泼度": 80, "理性度": 30, "感性度": 70, "计划度": 60, "随性度": 40, "脑洞度": 90, "幽默度": 75}
        b5 = seven_dim_to_big_five(original)
        back = big_five_to_seven_dim(b5)
        # 活泼度 and 脑洞度 should be preserved exactly (direct mapping)
        assert back["活泼度"] == original["活泼度"]
        assert back["脑洞度"] == original["脑洞度"]
        assert back["计划度"] == original["计划度"]


class TestGetSevenDim:
    def test_from_current_traits(self):
        agent = MagicMock()
        agent.currentTraits = {"活泼度": 80, "理性度": 60}
        agent.sevenDimTraits = {"活泼度": 70}
        agent.personality = {"extraversion": 0.3}
        result = get_seven_dim(agent)
        assert result["活泼度"] == 80

    def test_from_seven_dim_traits(self):
        agent = MagicMock()
        agent.currentTraits = None
        agent.sevenDimTraits = {"活泼度": 70, "理性度": 50}
        agent.personality = {"extraversion": 0.3}
        result = get_seven_dim(agent)
        assert result["活泼度"] == 70

    def test_fallback_from_big_five(self):
        agent = MagicMock()
        agent.currentTraits = None
        agent.sevenDimTraits = None
        agent.personality = {"extraversion": 0.8, "openness": 0.5, "conscientiousness": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}
        result = get_seven_dim(agent)
        assert result["活泼度"] == 80

    def test_null_personality(self):
        agent = MagicMock()
        agent.currentTraits = None
        agent.sevenDimTraits = None
        agent.personality = None
        result = get_seven_dim(agent)
        assert result["活泼度"] == 50  # default


class TestGetDim:
    def test_normal(self):
        assert abs(get_dim({"活泼度": 80}, "活泼度") - 0.8) < 0.01

    def test_missing_dim(self):
        assert abs(get_dim({}, "活泼度") - 0.5) < 0.01

    def test_boundary(self):
        assert get_dim({"活泼度": 0}, "活泼度") == 0.0
        assert get_dim({"活泼度": 100}, "活泼度") == 1.0
