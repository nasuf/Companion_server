from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass

from app.services.speech_output.client import count_billable_characters


DEFAULT_STYLE_INSTRUCTION = "像熟人聊天，口语自然，按语意停顿，避免播音腔。"
MAX_INSTRUCTION_BILLABLE_CHARACTERS = 100
logger = logging.getLogger(__name__)

_EMOTION_TAGS = {
    "高兴": "[excited]",
    "悲伤": "[sad]",
    "愤怒": "[angry]",
    "惊讶": "[amazed]",
    "恐惧": "[trembling]",
    "厌恶": "[scornful]",
    "失望": "[sad]",
    "欣慰": "[empathetic]",
    "感激": "[empathetic]",
    "戏谑": "[mischievously]",
}

_EMOTION_STYLES = {
    "高兴": "开心轻快",
    "悲伤": "低落难过",
    "愤怒": "生气严肃",
    "惊讶": "意外惊讶",
    "恐惧": "害怕不安",
    "厌恶": "反感不悦",
    "焦虑": "担忧犹豫",
    "失望": "失落遗憾",
    "欣慰": "温暖放松",
    "感激": "真诚感谢",
    "戏谑": "轻松俏皮",
}
# Only the planner may add provider controls; quoted/user text must not inject
# them into speech. Preserve ordinary square-bracket text (e.g. [项目名]).
_CONTROL_RE = re.compile(
    r"\[(?:sad|amazed|deep and loud shouting|trembling|angry|excited|sarcastic|"
    r"curious|like dracula|bored|tired|scornful|shouting|asmr|panicked|"
    r"mischievously|empathetic|whispers|reluctantly|crying|serious|very slowly|"
    r"very fast|gasp|sighing|clears throat|giggles|laughing|cough|snorts)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpeechPlan:
    text: str
    instruction: str


def emotion_score(intensity: int | float | None, scale: float) -> int:
    try:
        value, factor = float(intensity or 0), float(scale)
        if not math.isfinite(value) or not math.isfinite(factor):
            return 0
        return round(min(100, max(0, min(100, value)) * max(0, min(2, factor))))
    except (TypeError, ValueError, OverflowError):
        return 0


async def resolve_voice_emotion(
    text: str,
    emotion: str | None,
    intensity: int | float | None,
    *,
    enabled: bool,
    detect_missing: bool,
) -> tuple[str | None, int | float | None]:
    """Reuse chat signals; classify only missing signals on other voice paths."""
    if (
        not enabled or not detect_missing
        or (emotion is not None and intensity is not None)
    ):
        return emotion, intensity
    try:
        from app.services.chat.intent_replies import ai_reply_emotion

        async with asyncio.timeout(3):
            result = await ai_reply_emotion(text)
        return result.get("emotion"), result.get("intensity")
    except Exception as exc:
        logger.warning("TTS emotion detection failed: %s", type(exc).__name__)
        return emotion, intensity


def build_speech_plan(
    text: str,
    emotion: str | None,
    intensity: int | float | None,
    *,
    instruction: str | None,
    enabled: bool,
    scale: float,
) -> SpeechPlan:
    """Shared by delivery and audition; never rewrite the displayed transcript."""
    from app.services.emoji import limit_emojis

    clean_text = " ".join(
        limit_emojis(_CONTROL_RE.sub("", text or ""), max_keep=0).split()
    )
    if not clean_text:
        raise ValueError("TTS text contains no speakable content")
    base = resolve_style_instruction(instruction)
    label = (emotion or "中性").strip()
    score = emotion_score(intensity, scale) if enabled else 0
    mood = _EMOTION_STYLES.get(label)
    if mood and score >= 25:
        degree = "略带" if score < 50 else "带有" if score < 70 else "明显"
        combined = f"{base}本句{degree}{mood}。"
        # Preserve the whole admin instruction, never silently truncate it.
        if instruction_billable_characters(combined) <= MAX_INSTRUCTION_BILLABLE_CHARACTERS:
            base = combined
        # A single nonverbal event requires an explicit textual cue. No random
        # laughter/breaths, and anxiety alone does not imply trembling.
        cue = None
        event = None
        if label in {"高兴", "戏谑"}:
            cue = re.match(r"^哈{2,}[，,。.!！\s]*", clean_text)
            event = "[giggles]"
        elif label in {"悲伤", "失望", "焦虑"}:
            cue = re.match(r"^唉[，,。…！!\s]+", clean_text)
            event = "[sighing]"
        if cue and clean_text[cue.end():].strip():
            clean_text = f"{event}{clean_text[cue.end():]}"
    return SpeechPlan(
        text=decorate_text_with_emotion(clean_text, label, score, enabled=enabled, scale=1),
        instruction=base,
    )


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
    score = emotion_score(intensity, scale)
    if score < 70:
        return clean_text
    tag = _EMOTION_TAGS.get(label)
    return f"{tag}{clean_text}" if tag else clean_text
