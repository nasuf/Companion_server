from __future__ import annotations

from app.services.speech_output.client import count_billable_characters


DEFAULT_STYLE_INSTRUCTION = "像熟人聊天，口语自然，避免播音腔。"
MAX_INSTRUCTION_BILLABLE_CHARACTERS = 100

_EMOTION_TAGS = {
    "高兴": "[excited]",
    "悲伤": "[sad]",
    "愤怒": "[angry]",
    "惊讶": "[amazed]",
    "恐惧": "[trembling]",
    "厌恶": "[scornful]",
    "焦虑": "[trembling]",
    "失望": "[sad]",
    "欣慰": "[empathetic]",
    "感激": "[empathetic]",
    "戏谑": "[mischievously]",
}


def instruction_billable_characters(value: str | None) -> int:
    return count_billable_characters((value or "").strip())


def resolve_style_instruction(value: str | None) -> str:
    instruction = (value or "").strip() or DEFAULT_STYLE_INSTRUCTION
    if instruction_billable_characters(instruction) > (
        MAX_INSTRUCTION_BILLABLE_CHARACTERS
    ):
        raise ValueError("TTS instruction exceeds provider limit")
    return instruction


def decorate_text_with_emotion(
    text: str,
    emotion: str | None,
    intensity: int | float | None,
    *,
    enabled: bool,
    scale: float,
) -> str:
    """Prefix one provider-supported control tag without changing transcript."""
    clean_text = (text or "").strip()
    if not clean_text or not enabled:
        return clean_text
    label = (emotion or "中性").strip() or "中性"
    try:
        score = max(
            0,
            min(
                100,
                round(float(intensity or 0) * max(0.0, min(2.0, float(scale)))),
            ),
        )
    except (TypeError, ValueError):
        score = 0
    if score < 25:
        return clean_text
    tag = _EMOTION_TAGS.get(label)
    return f"{tag}{clean_text}" if tag else clean_text
