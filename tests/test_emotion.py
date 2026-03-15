"""情绪系统单元测试。

测试覆盖：
- VAD状态更新（亲密度加权融合）
- 情绪→语气映射
- 人格基线情绪计算
- 记忆情绪影响
- Clamp/Lerp辅助函数
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.emotion import (
    _clamp,
    _lerp_vad,
    apply_memory_emotion_influence,
    compute_baseline_emotion,
    emotion_to_tone,
    update_emotion_state,
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


# --- update_emotion_state ---

class TestUpdateEmotionState:
    def test_default_intimacy(self):
        """Default intimacy=50 → balanced weights."""
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}

        result = update_emotion_state(current, input_emotion)
        # At intimacy=50: alpha=0.6, beta=0.4
        for dim in ("valence", "arousal", "dominance"):
            expected = 0.6 * current[dim] + 0.4 * input_emotion[dim]
            assert abs(result[dim] - expected) < 1e-6

    def test_zero_intimacy(self):
        """intimacy=0 → AI retains more (alpha=0.7, beta=0.3)."""
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}

        result = update_emotion_state(current, input_emotion, topic_intimacy=0.0)
        for dim in ("valence", "arousal", "dominance"):
            expected = 0.7 * current[dim] + 0.3 * input_emotion[dim]
            assert abs(result[dim] - expected) < 1e-6

    def test_max_intimacy(self):
        """intimacy=100 → equal weights (alpha=0.5, beta=0.5)."""
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}

        result = update_emotion_state(current, input_emotion, topic_intimacy=100.0)
        for dim in ("valence", "arousal", "dominance"):
            expected = 0.5 * current[dim] + 0.5 * input_emotion[dim]
            assert abs(result[dim] - expected) < 1e-6

    def test_missing_keys(self):
        """Missing VAD dims default to 0."""
        current = {}
        input_emotion = {"valence": 0.5}
        result = update_emotion_state(current, input_emotion)
        assert result["arousal"] == 0.0
        assert result["dominance"] == 0.0


# --- emotion_to_tone ---

class TestEmotionToTone:
    def test_positive(self):
        assert emotion_to_tone({"valence": 0.8, "arousal": 0.6, "dominance": 0.5}) == "enthusiastic and confident"

    def test_negative(self):
        assert emotion_to_tone({"valence": -0.5, "arousal": -0.3, "dominance": -0.7}) == "sad and withdrawn"

    def test_zero_maps_to_positive_octant(self):
        tone = emotion_to_tone({"valence": 0.0, "arousal": 0.0, "dominance": 0.0})
        assert isinstance(tone, str) and len(tone) > 0

    def test_anxious(self):
        assert emotion_to_tone({"valence": -0.5, "arousal": 0.8, "dominance": -0.3}) == "anxious and stressed"

    def test_calm(self):
        assert emotion_to_tone({"valence": 0.6, "arousal": -0.4, "dominance": 0.3}) == "calm and content"


# --- compute_baseline_emotion ---

class TestComputeBaselineEmotion:
    def test_neutral_personality(self):
        """All 0.5 (neutral) → baseline ~0."""
        personality = {
            "openness": 0.5, "conscientiousness": 0.5,
            "extraversion": 0.5, "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
        result = compute_baseline_emotion(personality)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim]) < 1e-6

    def test_high_extraversion(self):
        """High extraversion → positive valence."""
        personality = {"extraversion": 1.0, "agreeableness": 0.5, "neuroticism": 0.0, "openness": 0.5, "conscientiousness": 0.5}
        result = compute_baseline_emotion(personality)
        assert result["valence"] > 0
        # arousal: (e-0.5)*0.3 + (n-0.5)*0.4 = 0.15 - 0.2 = -0.05 (low neuroticism lowers arousal)
        assert result["arousal"] < 0

    def test_high_neuroticism(self):
        """High neuroticism → negative valence, higher arousal."""
        personality = {"extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 1.0, "openness": 0.5, "conscientiousness": 0.5}
        result = compute_baseline_emotion(personality)
        assert result["valence"] < 0
        assert result["arousal"] > 0

    def test_empty_personality(self):
        """Empty personality defaults to 0.5 → zero baseline."""
        result = compute_baseline_emotion({})
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim]) < 1e-6

    def test_values_clamped(self):
        """Extreme personality values still produce clamped output."""
        personality = {"extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 0.0, "openness": 1.0, "conscientiousness": 1.0}
        result = compute_baseline_emotion(personality)
        for dim in ("valence", "arousal", "dominance"):
            assert -1.0 <= result[dim] <= 1.0


# --- apply_memory_emotion_influence ---

class TestMemoryEmotionInfluence:
    def test_no_memories(self):
        current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
        result = apply_memory_emotion_influence(current, [])
        assert result == current

    def test_single_memory(self):
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        memories = [{"valence": 1.0, "arousal": 1.0, "dominance": 1.0}]
        result = apply_memory_emotion_influence(current, memories, influence_weight=0.1)
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim] - 0.1) < 1e-6

    def test_multiple_memories_averaged(self):
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        memories = [
            {"valence": 1.0, "arousal": 0.0, "dominance": 0.0},
            {"valence": -1.0, "arousal": 0.0, "dominance": 0.0},
        ]
        result = apply_memory_emotion_influence(current, memories, influence_weight=0.1)
        # Average of 1.0 and -1.0 is 0.0, so no change
        assert abs(result["valence"]) < 1e-6

    def test_default_weight(self):
        current = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        memories = [{"valence": 1.0, "arousal": 1.0, "dominance": 1.0}]
        result = apply_memory_emotion_influence(current, memories)
        # default weight is 0.05
        for dim in ("valence", "arousal", "dominance"):
            assert abs(result[dim] - 0.05) < 1e-6
