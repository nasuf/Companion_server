"""Tests for the emotion system."""

from app.services.emotion import update_emotion_state, emotion_to_tone


def test_update_emotion_state():
    """Test VAD emotion state update formula."""
    current = {"valence": 0.5, "arousal": 0.3, "dominance": 0.4}
    input_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}

    result = update_emotion_state(current, input_emotion)

    # new = old * 0.9 + input * 0.1
    assert abs(result["valence"] - (0.5 * 0.9 + (-0.8) * 0.1)) < 1e-6
    assert abs(result["arousal"] - (0.3 * 0.9 + 0.9 * 0.1)) < 1e-6
    assert abs(result["dominance"] - (0.4 * 0.9 + (-0.5) * 0.1)) < 1e-6


def test_update_emotion_state_defaults():
    """Test emotion update with missing keys."""
    current = {}
    input_emotion = {"valence": 0.5}

    result = update_emotion_state(current, input_emotion)

    assert result["valence"] == 0.05  # 0*0.9 + 0.5*0.1
    assert result["arousal"] == 0.0
    assert result["dominance"] == 0.0


def test_emotion_to_tone_positive():
    """Test positive emotion maps to enthusiastic tone."""
    emotion = {"valence": 0.8, "arousal": 0.6, "dominance": 0.5}
    tone = emotion_to_tone(emotion)
    assert tone == "enthusiastic and confident"


def test_emotion_to_tone_negative():
    """Test negative emotion maps to sad tone."""
    emotion = {"valence": -0.5, "arousal": -0.3, "dominance": -0.7}
    tone = emotion_to_tone(emotion)
    assert tone == "sad and withdrawn"


def test_emotion_to_tone_neutral():
    """Test zero emotion returns a valid tone."""
    emotion = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    tone = emotion_to_tone(emotion)
    assert isinstance(tone, str)
    assert len(tone) > 0
