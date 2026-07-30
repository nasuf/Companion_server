from __future__ import annotations

_STYLE_BY_EMOTION = {
    "高兴": "带一点自然的笑意和轻快感",
    "悲伤": "低落但克制，不要戏剧化哭腔",
    "愤怒": "明显不悦但控制音量，不要吼叫",
    "惊讶": "略带惊讶，起伏自然，不要夸张",
    "恐惧": "有一点紧张和不安，但保持清晰",
    "厌恶": "语气疏离、略带反感，但不要夸张",
    "中性": "自然平静",
    "焦虑": "稍显担心和急切，但语速不要过快",
    "失望": "失落、语气稍沉，但保持克制",
    "欣慰": "温和放松，带一点安心的笑意",
    "感激": "真诚温暖，语气柔和",
    "戏谑": "轻松俏皮，像熟人间自然打趣",
}


def build_style_instruction(
    emotion: str | None,
    intensity: int | float | None,
) -> str:
    """Map the existing reply emotion signal to a stable TTS instruction."""
    label = (emotion or "中性").strip() or "中性"
    try:
        score = max(0, min(100, int(float(intensity or 0))))
    except (TypeError, ValueError):
        score = 0

    base = _STYLE_BY_EMOTION.get(label, _STYLE_BY_EMOTION["中性"])
    if score < 30:
        strength = "情绪只轻微流露"
    elif score < 70:
        strength = "情绪程度适中"
    else:
        strength = "情绪清楚可感，但仍然克制"
    return (
        "像给熟悉的人发送手机语音一样自然口语化，"
        f"{base}，{strength}。保持原本音色，吐字清楚，停顿自然，"
        "不要播音腔、旁白腔、舞台表演感或额外添加内容。"
    )
