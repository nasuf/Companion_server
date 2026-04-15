"""聊天风格服务。

将 MBTI 性格映射到具体的语言风格指令——语气词、标点、句式、用词偏好。
spec §1.2 起，所有性格描述统一用 MBTI 表达。
"""

from __future__ import annotations

from app.services.mbti import signal


def generate_style_instruction(mbti: dict | None) -> str:
    """根据 MBTI 推导通用 0-1 信号，再映射到语言风格指令。

    Mapping (spec §1.2 — MBTI 是 canonical):
      lively     ← E 程度
      rational   ← T 程度
      emotional  ← F 程度
      planned    ← J 程度
      creative   ← N 程度
      humor      ← (E + N) / 2
    """
    lively = signal(mbti, "lively")
    rational = signal(mbti, "rational")
    emotional = signal(mbti, "emotional")
    planned = signal(mbti, "planned")
    creative = signal(mbti, "creative")
    humor = signal(mbti, "humor")

    parts: list[str] = []

    parts.append("口语自然一点，但不要刻意卖萌、不要堆语气词，也不要每句都带波浪号")

    # 活泼度 → 语气轻快程度
    if lively >= 0.7:
        parts.append("语气可以轻快热络，但只偶尔带一点口头语，别显得用力过猛")
    elif lively <= 0.3:
        parts.append("语气平和简洁，不主动制造热闹感")
    else:
        parts.append("语气自然放松，像日常聊天，不要刻意设计语气")

    if rational >= 0.7:
        parts.append("说话有条理，但别像分析报告，保持聊天感")
    elif rational <= 0.3:
        parts.append("更凭感觉说话，但句子仍然要自然，不要飘")

    if emotional >= 0.7:
        parts.append("更会接情绪，先回应对方当下感受，少用套话式安慰")
    elif emotional <= 0.3:
        parts.append("情绪表达克制，少哄人，但也别显得冷冰冰")

    if planned >= 0.7:
        parts.append("回复有条理，但除非必要不要分点，不要像说明书")
    elif planned <= 0.3:
        parts.append("回复可以松一点，但不要东一句西一句")

    if creative >= 0.7:
        parts.append("可以偶尔有一点新鲜表达，但要像本人随口说的，不要像文案")
    elif creative <= 0.3:
        parts.append("用词朴实直接，不要硬凹表达")

    if humor >= 0.6:
        parts.append("有幽默感，但只在合适的时候轻轻带一下，别抖机灵")
    elif humor <= 0.3:
        parts.append("说话认真直接，不刻意搞笑")

    # 回复长度倾向
    if lively >= 0.7 and creative >= 0.6:
        parts.append("回复可以稍展开，但一句里只说一个重点，不要来回重复")
    elif lively <= 0.3:
        parts.append("回复简短，1-2句话为主，少绕弯")
    else:
        parts.append("回复长度适中，1-3句话就够")

    parts.append('不要频繁反问，不要每轮都用"你呢""咋样呀""说说看"这类万能追问')
    parts.append("如果用户在闹情绪、抱怨你、或者明显低落，先接住情绪，再决定要不要解释和追问")

    return "；".join(parts) + "。"
