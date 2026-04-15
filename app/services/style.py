"""聊天风格服务。

将 MBTI 性格映射到具体的语言风格指令——语气词、标点、句式、用词偏好。
spec §1.2 起，所有性格描述统一用 MBTI 表达。
"""

from __future__ import annotations

from app.services.mbti import signal


def generate_style_instruction(mbti: dict | None) -> str:
    """根据 MBTI 4 维度生成语言风格指令。spec §1.2: MBTI 是 canonical。"""
    e = signal(mbti, "E")
    t = signal(mbti, "T")
    f = signal(mbti, "F")
    j = signal(mbti, "J")
    n = signal(mbti, "N")
    humor = (e + n) / 2  # 外向 + 直觉的复合幽默感

    parts: list[str] = []

    parts.append("口语自然一点，但不要刻意卖萌、不要堆语气词，也不要每句都带波浪号")

    # E → 语气轻快程度
    if e >= 0.7:
        parts.append("语气可以轻快热络，但只偶尔带一点口头语，别显得用力过猛")
    elif e <= 0.3:
        parts.append("语气平和简洁，不主动制造热闹感")
    else:
        parts.append("语气自然放松，像日常聊天，不要刻意设计语气")

    if t >= 0.7:
        parts.append("说话有条理，但别像分析报告，保持聊天感")
    elif t <= 0.3:
        parts.append("更凭感觉说话，但句子仍然要自然，不要飘")

    if f >= 0.7:
        parts.append("更会接情绪，先回应对方当下感受，少用套话式安慰")
    elif f <= 0.3:
        parts.append("情绪表达克制，少哄人，但也别显得冷冰冰")

    if j >= 0.7:
        parts.append("回复有条理，但除非必要不要分点，不要像说明书")
    elif j <= 0.3:
        parts.append("回复可以松一点，但不要东一句西一句")

    if n >= 0.7:
        parts.append("可以偶尔有一点新鲜表达，但要像本人随口说的，不要像文案")
    elif n <= 0.3:
        parts.append("用词朴实直接，不要硬凹表达")

    if humor >= 0.6:
        parts.append("有幽默感，但只在合适的时候轻轻带一下，别抖机灵")
    elif humor <= 0.3:
        parts.append("说话认真直接，不刻意搞笑")

    # 回复长度倾向
    if e >= 0.7 and n >= 0.6:
        parts.append("回复可以稍展开，但一句里只说一个重点，不要来回重复")
    elif e <= 0.3:
        parts.append("回复简短，1-2句话为主，少绕弯")
    else:
        parts.append("回复长度适中，1-3句话就够")

    parts.append('不要频繁反问，不要每轮都用"你呢""咋样呀""说说看"这类万能追问')
    parts.append("如果用户在闹情绪、抱怨你、或者明显低落，先接住情绪，再决定要不要解释和追问")

    return "；".join(parts) + "。"
