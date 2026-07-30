from __future__ import annotations

_STYLE_BY_EMOTION = {
    "高兴": "笑意自然地落在关键词上，节奏略轻快，句尾可以轻微上扬",
    "悲伤": "声音稍低、节奏稍慢，在难过的位置留一点真实停顿，不使用表演式哭腔",
    "愤怒": "短句更干脆、重音明确，带真实的不悦，保持日常说话音量而不是喊叫",
    "惊讶": "开头反应稍快，音高和节奏有自然变化，像当下真的刚听到这件事",
    "恐惧": "带一点紧张和不安，呼吸与停顿可以稍不均匀，但不要戏剧化",
    "厌恶": "语气略微疏离，重音落在反感的内容上，不使用固定冷脸腔",
    "中性": "松弛自然，像随口说话，语调有细小起伏而不是平直朗读",
    "焦虑": "带真实的担心，局部可以稍快，在关键处自然停一下，不机械加速",
    "失望": "声音稍沉，句尾自然回落，留出一点没说满的失落感",
    "欣慰": "像轻轻松了一口气，节奏放松，笑意温和而不刻意",
    "感激": "语气真诚温暖，关键词略微放软，避免礼貌客服式表达",
    "戏谑": "像熟人间临时起意的打趣，轻重音灵活，带一点自然笑意",
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
        strength = "情绪藏在语气里，整体松弛自然"
    elif score < 70:
        strength = "情绪清楚但不刻意，每句话保留自然起伏"
    else:
        strength = "情绪明显，允许更大的节奏和音高变化，但不要套用统一的夸张腔"
    return (
        "把这句话当成一条私下发给熟人的手机语音，不是在朗读文案。"
        "先理解整句话再开口，节奏跟随语义自然变化；允许自然连读、轻重音、"
        f"短停顿和轻微气息，避免每个字等长等重。{base}；{strength}。"
        "保持说话人的固有音色并完整保留原文，不增删台词。"
        "像真实的人当下开口，不要使用播音、客服、旁白、广告或舞台表演腔。"
    )
