"""Phase 7 prompt hardening guards.

Locks in: tier prompts share one persona preamble and carry the anti-
hallucination + anti-roleplay rules; intent-label whitelist is single-sourced;
emotion enums stay consistent; delay/recap contradictions are resolved.
"""

from __future__ import annotations

from app.services.prompting import defaults as d
from app.services.prompting.registry import PROMPT_DEFINITION_MAP


TIER_KEYS = ["memory.weak_reply", "memory.medium_reply",
             "memory.strong_reply", "memory.l3_reply"]


def test_tier_prompts_share_preamble():
    for key in TIER_KEYS:
        text = PROMPT_DEFINITION_MAP[key].default_text
        assert d._TIER_PERSONA_PREAMBLE in text, f"{key} missing shared preamble"


def test_tier_prompts_have_anti_hallucination_and_anti_roleplay():
    for key in TIER_KEYS:
        text = PROMPT_DEFINITION_MAP[key].default_text
        assert "事实底线" in text, f"{key} missing anti-hallucination block"
        assert "旁白" in text, f"{key} missing anti-roleplay rule"
        assert "不是记忆" in text or "不是 X 真发生过的证据" in text, (
            f"{key} missing 'user question is not evidence' rule"
        )


def test_intent_labels_single_sourced_and_exclude_crisis():
    from app.services.chat.intent_replies import _INTENT_LABELS
    from app.services.chat.intent_dispatcher import LABEL_TO_INTENT

    assert _INTENT_LABELS == {l for l in LABEL_TO_INTENT if l != "危机求助"}
    assert "危机求助" not in _INTENT_LABELS  # handled upstream in crisis_guard


def test_emotion_enum_consistent_across_prompts_and_code():
    from app.services.relationship.emotion import EMOTION_LABELS
    from app.services.chat.reply_generate import _VALID_REPLY_EMOTIONS

    assert set(EMOTION_LABELS) == set(_VALID_REPLY_EMOTIONS)
    assert len(EMOTION_LABELS) == 12
    # both emotion prompts enumerate the same 12 labels
    for label in EMOTION_LABELS:
        assert label in d.USER_EMOTION_LABEL_PROMPT
        assert label in d.AI_REPLY_EMOTION_PROMPT


def test_delay_explanation_no_longer_self_contradicts():
    text = PROMPT_DEFINITION_MAP["reply.delay_explanation"].default_text
    # old contradiction: provided time fields yet said "不提及具体时间"
    assert "不提及具体时间" not in text
    assert "模糊说法" in text


def test_recap_defers_to_reengagement():
    text = PROMPT_DEFINITION_MAP["chat.session_recap_section"].default_text
    assert "重逢感知为准" in text


def test_extraction_prompts_have_grounding_rule():
    for key in ["memory.extraction_user", "memory.extraction_ai"]:
        assert "事实依据" in PROMPT_DEFINITION_MAP[key].default_text
