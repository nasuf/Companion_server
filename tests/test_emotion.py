"""情绪系统单元测试。

测试覆盖：
- VAD状态更新（亲密度加权融合+共情向量）
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
    _clamp,
    _lerp_vad,
    apply_memory_emotion_influence,
    compute_baseline_emotion,
    compute_emotional_stability,
    emotion_to_tone,
    label_to_vad,
    update_emotion_state,
    vad_to_label,
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


# --- _lerp_vad ---

class TestLerpVad:
    def test_no_change_at_rate_0(self):
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        target = {"valence": -0.5, "arousal": 0.9, "dominance": -0.1}
        result = _lerp_vad(current, target, 0.0)
        assert result == current

    def test_full_change_at_rate_1(self):
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        target = {"valence": -0.5, "arousal": 0.9, "dominance": -0.1}
        result = _lerp_vad(current, target, 1.0)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim] - target[dim]) < 1e-6

    def test_midpoint(self):
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        target = {"valence": 1.0, "arousal": 1.0, "dominance": 1.0}
        result = _lerp_vad(current, target, 0.5)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim] - 0.5) < 1e-6


# --- update_emotion_state (with empathy vector) ---

class TestUpdateEmotionState:
    def test_with_empathy_vector(self):
        """With seven_dim, empathy vector is applied."""
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}
        seven_dim = {"感性度": 80}

        result = update_emotion_state(current, input_emotion, topic_intimacy=50.0, seven_dim=seven_dim)
        assert isinstance(result["valence"], float)
        assert -1.0 <= result["valence"] <= 1.0

    def test_without_seven_dim(self):
        """Without seven_dim, default 0.5 sensitivity used."""
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}

        result = update_emotion_state(current, input_emotion, topic_intimacy=50.0)
        assert isinstance(result["valence"], float)

    def test_high_intimacy_more_user_influence(self):
        """High intimacy → larger β (more user influence)."""
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        input_emotion = {"valence": 1.0, "arousal": 1.0, "dominance": 1.0}

        low = update_emotion_state(current, input_emotion, topic_intimacy=10.0)
        high = update_emotion_state(current, input_emotion, topic_intimacy=90.0)
        assert high["valence"] > low["valence"]

    def test_missing_keys(self):
        """Missing VAD dims default to 0."""
        current = {}
        input_emotion = {"valence": 0.5}
        result = update_emotion_state(current, input_emotion)
        assert result["arousal"] is not None


# --- emotion_to_tone ---

class TestEmotionToTone:
    def test_positive(self):
        assert emotion_to_tone({"valence": 0.8, "arousal": 0.6, "dominance": 0.5}) == "enthusiastic and confident"

    def test_negative(self):
        assert emotion_to_tone({"valence": -0.5, "arousal": -0.3, "dominance": -0.7}) == "sad and withdrawn"

    def test_anxious(self):
        assert emotion_to_tone({"valence": -0.5, "arousal": 0.8, "dominance": -0.3}) == "anxious and stressed"


# --- compute_baseline_emotion ---

class TestComputeBaselineEmotion:
    def test_neutral_big_five(self):
        personality = {
            "openness": 0.5, "conscientiousness": 0.5,
            "extraversion": 0.5, "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
        result = compute_baseline_emotion(personality)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim]) < 1e-6

    def test_with_seven_dim_neutral(self):
        seven = {"活泼度": 50, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50}
        result = compute_baseline_emotion({}, seven_dim=seven)
        assert abs(result["valence"] - 0.2) < 0.01
        assert abs(result["arousal"] - 0.5) < 0.01

    def test_high_lively_high_humor(self):
        seven = {"活泼度": 90, "理性度": 50, "感性度": 50, "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 90}
        result = compute_baseline_emotion({}, seven_dim=seven)
        assert result["valence"] > 0.3

    def test_values_clamped(self):
        seven = {"活泼度": 100, "理性度": 0, "感性度": 100, "计划度": 0, "随性度": 100, "脑洞度": 100, "幽默度": 100}
        result = compute_baseline_emotion({}, seven_dim=seven)
        for dim in ("valence", "arousal", "dominance"):
            assert -1.0 <= result[dim] <= 1.0


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

    def test_label_to_vad(self):
        result = label_to_vad("joy")
        assert result is not None
        assert result["valence"] > 0

    def test_vad_to_label_happy(self):
        assert vad_to_label({"valence": 0.7, "arousal": 0.4, "dominance": 0.3}) == "快乐"

    def test_vad_to_label_sad(self):
        assert vad_to_label({"valence": -0.7, "arousal": -0.3, "dominance": -0.5}) == "悲伤"


# --- apply_memory_emotion_influence ---

class TestMemoryEmotionInfluence:
    def test_no_memories(self):
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        result = apply_memory_emotion_influence(current, [])
        assert result == current

    def test_default_weight_is_0_2(self):
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        memories = [{"valence": 1.0, "arousal": 1.0, "dominance": 1.0}]
        result = apply_memory_emotion_influence(current, memories)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim] - 0.2) < 1e-6
