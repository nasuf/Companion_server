"""情绪系统单元测试。

测试覆盖：
- PAD状态更新（亲密度加权融合+共情向量）
- 情绪→语气映射
- 七维人格基线情绪计算
- 情绪稳定性系数
- 12标签PAD映射
- 记忆情绪影响（权重0.2）
- Clamp/Lerp辅助函数
"""

import pytest

from app.services.emotion import (
    PAD_LABEL_TABLE,
    _PAD_DEFAULTS,
    _clamp,
    _clamp_pad,
    _lerp_pad,
    apply_memory_emotion_influence,
    compute_baseline_emotion,
    compute_emotional_stability,
    emotion_to_tone,
    label_to_pad,
    quick_emotion_estimate,
    update_emotion_state,
    pad_to_label,
)


# --- _clamp ---

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_below_min(self):
        assert _clamp(-2.0) == -1.0

    def test_custom_range(self):
        assert _clamp(1.5, 0.0, 1.0) == 1.0
        assert _clamp(-0.5, 0.0, 1.0) == 0.0


# --- _clamp_pad ---

class TestClampPad:
    def test_pleasure_range(self):
        assert _clamp_pad("pleasure", -1.5) == -1.0
        assert _clamp_pad("pleasure", 1.5) == 1.0
        assert _clamp_pad("pleasure", 0.5) == 0.5

    def test_arousal_range(self):
        assert _clamp_pad("arousal", -0.5) == 0.0
        assert _clamp_pad("arousal", 1.5) == 1.0
        assert _clamp_pad("arousal", 0.5) == 0.5

    def test_dominance_range(self):
        assert _clamp_pad("dominance", -0.5) == 0.0
        assert _clamp_pad("dominance", 1.5) == 1.0
        assert _clamp_pad("dominance", 0.5) == 0.5


# --- _lerp_pad ---

class TestLerpPad:
    def test_no_change_at_rate_0(self):
        current = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        target = {"pleasure": -0.5, "arousal": 0.9, "dominance": 0.8}
        result = _lerp_pad(current, target, 0.0)
        assert result == current

    def test_full_change_at_rate_1(self):
        current = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        target = {"pleasure": -0.5, "arousal": 0.9, "dominance": 0.8}
        result = _lerp_pad(current, target, 1.0)
        for dim in ("pleasure", "arousal", "dominance"):
            assert abs(result[dim] - target[dim]) < 1e-6

    def test_midpoint(self):
        current = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        target = {"pleasure": 1.0, "arousal": 1.0, "dominance": 1.0}
        result = _lerp_pad(current, target, 0.5)
        for dim in ("pleasure", "arousal", "dominance"):
            assert abs(result[dim] - 0.5) < 1e-6


# --- update_emotion_state (with empathy vector) ---

class TestUpdateEmotionState:
    def test_with_empathy_vector(self):
        """With seven_dim, empathy vector is applied."""
        current = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"pleasure": -0.8, "arousal": 0.9, "dominance": 0.2}
        seven_dim = {"感性度": 80}

        result = update_emotion_state(current, input_emotion, topic_intimacy=50.0, seven_dim=seven_dim)
        assert isinstance(result["pleasure"], float)
        assert -1.0 <= result["pleasure"] <= 1.0

    def test_without_seven_dim(self):
        """Without seven_dim, default 0.5 sensitivity used."""
        current = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"pleasure": -0.8, "arousal": 0.9, "dominance": 0.2}

        result = update_emotion_state(current, input_emotion, topic_intimacy=50.0)
        assert isinstance(result["pleasure"], float)

    def test_high_intimacy_more_user_influence(self):
        """High intimacy → larger β (more user influence)."""
        current = {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}
        input_emotion = {"pleasure": 1.0, "arousal": 1.0, "dominance": 1.0}

        low = update_emotion_state(current, input_emotion, topic_intimacy=10.0)
        high = update_emotion_state(current, input_emotion, topic_intimacy=90.0)
        assert high["pleasure"] > low["pleasure"]

    def test_missing_keys(self):
        """Missing PAD dims default to _PAD_DEFAULTS (pleasure=0, arousal=0.5, dominance=0.5)."""
        current = {}
        input_emotion = {"pleasure": 0.5}
        result = update_emotion_state(current, input_emotion)
        assert result["arousal"] is not None

    def test_arousal_dominance_clamped_to_0_1(self):
        """Arousal and dominance stay in [0,1] after fusion."""
        current = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        input_emotion = {"pleasure": -1.0, "arousal": 0.0, "dominance": 0.0}
        result = update_emotion_state(current, input_emotion)
        assert result["arousal"] >= 0.0
        assert result["dominance"] >= 0.0


# --- emotion_to_tone ---

class TestEmotionToTone:
    def test_positive(self):
        assert emotion_to_tone({"pleasure": 0.8, "arousal": 0.6, "dominance": 0.6}) == "热情而笃定"

    def test_negative(self):
        assert emotion_to_tone({"pleasure": -0.5, "arousal": 0.3, "dominance": 0.2}) == "难过而退缩"

    def test_anxious(self):
        assert emotion_to_tone({"pleasure": -0.5, "arousal": 0.8, "dominance": 0.3}) == "焦虑而紧绷"


# --- compute_baseline_emotion ---

class TestComputeBaselineEmotion:
    def test_neutral_big_five(self):
        personality = {
            "openness": 0.5, "conscientiousness": 0.5,
            "extraversion": 0.5, "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
        result = compute_baseline_emotion(personality)
        assert abs(result["pleasure"]) < 1e-6
        assert abs(result["arousal"] - 0.5) < 1e-6
        assert abs(result["dominance"] - 0.5) < 1e-6

    def test_with_seven_dim_neutral(self):
        seven = {"活泼度": 50, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50}
        result = compute_baseline_emotion({}, seven_dim=seven)
        assert abs(result["pleasure"] - 0.2) < 0.01
        assert abs(result["arousal"] - 0.5) < 0.01

    def test_high_lively_high_humor(self):
        seven = {"活泼度": 90, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 90}
        result = compute_baseline_emotion({}, seven_dim=seven)
        assert result["pleasure"] > 0.3

    def test_values_clamped(self):
        seven = {"活泼度": 100, "理性度": 0, "感性度": 100, "计划度": 0, "随性度": 100, "脑洞度": 100, "幽默度": 100}
        result = compute_baseline_emotion({}, seven_dim=seven)
        assert -1.0 <= result["pleasure"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        assert 0.0 <= result["dominance"] <= 1.0

    def test_big_five_arousal_dominance_in_0_1(self):
        """Big Five fallback should produce arousal/dominance in [0,1]."""
        personality = {
            "openness": 0.0, "conscientiousness": 0.0,
            "extraversion": 0.0, "agreeableness": 0.0,
            "neuroticism": 1.0,
        }
        result = compute_baseline_emotion(personality)
        assert 0.0 <= result["arousal"] <= 1.0
        assert 0.0 <= result["dominance"] <= 1.0


# --- compute_emotional_stability ---

class TestEmotionalStability:
    def test_neutral(self):
        seven = {"理性度": 50, "计划度": 50, "感性度": 50, "随性度": 50}
        assert abs(compute_emotional_stability(seven) - 0.5) < 0.01

    def test_high_rational(self):
        seven = {"理性度": 100, "计划度": 100, "感性度": 0, "随性度": 0}
        assert compute_emotional_stability(seven) > 0.7

    def test_high_emotional(self):
        seven = {"理性度": 0, "计划度": 0, "感性度": 100, "随性度": 100}
        assert compute_emotional_stability(seven) < 0.3


# --- PAD label table ---

class TestPADLabelTable:
    def test_all_12_labels(self):
        assert len(PAD_LABEL_TABLE) == 12

    def test_label_to_pad(self):
        result = label_to_pad("joy")
        assert result is not None
        assert result["pleasure"] > 0

    def test_pad_to_label_happy(self):
        assert pad_to_label({"pleasure": 0.8, "arousal": 0.7, "dominance": 0.6}) == "高兴"

    def test_pad_to_label_sad(self):
        assert pad_to_label({"pleasure": -0.6, "arousal": 0.3, "dominance": 0.2}) == "悲伤"

    def test_all_labels_arousal_dominance_in_range(self):
        """All 12 labels must have arousal/dominance in [0, 1]."""
        for label, pad in PAD_LABEL_TABLE.items():
            assert 0.0 <= pad["arousal"] <= 1.0, f"{label} arousal out of range: {pad['arousal']}"
            assert 0.0 <= pad["dominance"] <= 1.0, f"{label} dominance out of range: {pad['dominance']}"


# --- apply_memory_emotion_influence ---

class TestMemoryEmotionInfluence:
    def test_no_memories(self):
        current = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        result = apply_memory_emotion_influence(current, [])
        assert result == current

    def test_default_weight_is_0_2(self):
        current = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        memories = [{"pleasure": 1.0, "arousal": 1.0, "dominance": 1.0}]
        result = apply_memory_emotion_influence(current, memories)
        for dim in ("pleasure", "arousal", "dominance"):
            assert abs(result[dim] - 0.2) < 1e-6


# --- quick_emotion_estimate ---

class TestQuickEmotionEstimate:
    def test_happy_keyword(self):
        result = quick_emotion_estimate("哈哈太好了")
        assert result is not None
        assert result["pleasure"] > 0

    def test_sad_keyword(self):
        result = quick_emotion_estimate("好难过啊")
        assert result is not None
        assert result["pleasure"] < 0

    def test_grateful_keyword(self):
        result = quick_emotion_estimate("谢谢你")
        assert result is not None
        assert result["pleasure"] > 0

    def test_no_match(self):
        result = quick_emotion_estimate("今天天气不错")
        assert result is None

    def test_returns_copy_not_shared_reference(self):
        """Returned dict should be a copy, not a reference into PAD_LABEL_TABLE."""
        result = quick_emotion_estimate("哈哈太好了")
        assert result is not None
        assert result is not PAD_LABEL_TABLE["高兴"]
        assert result == PAD_LABEL_TABLE["高兴"]


# --- _PAD_DEFAULTS ---

class TestPADDefaults:
    def test_pleasure_default_is_zero(self):
        assert _PAD_DEFAULTS["pleasure"] == 0.0

    def test_arousal_default_is_half(self):
        assert _PAD_DEFAULTS["arousal"] == 0.5

    def test_dominance_default_is_half(self):
        assert _PAD_DEFAULTS["dominance"] == 0.5
