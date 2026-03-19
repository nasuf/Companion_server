"""聊天风格服务。

根据七维人格生成具体的语言风格指令，控制语气词、标点、句式、用词偏好。
"""

from __future__ import annotations

from app.services.trait_model import get_dim


def generate_style_instruction(seven_dim: dict) -> str:
    """将七维人格转换为具体的语言风格指令。"""
    lively = get_dim(seven_dim, "活泼度")
    rational = get_dim(seven_dim, "理性度")
    emotional = get_dim(seven_dim, "感性度")
    planned = get_dim(seven_dim, "计划度")
    creative = get_dim(seven_dim, "脑洞度")
    humor = get_dim(seven_dim, "幽默度")

    parts: list[str] = []

    parts.append("口语自然一点，但不要刻意卖萌、不要堆语气词，也不要每句都带波浪号")

    # 活泼度 → 语气轻快程度
    if lively >= 0.7:
        parts.append("语气可以轻快热络，但只偶尔带一点口头语，别显得用力过猛")
    elif lively <= 0.3:
        parts.append("语气平和简洁，不主动制造热闹感")
    else:
        parts.append("语气自然放松，像日常聊天，不要刻意设计语气")

    # 理性度 → 逻辑表达 (PRD: ≥0.7→逻辑表达)
    if rational >= 0.7:
        parts.append("说话有条理，但别像分析报告，保持聊天感")
    elif rational <= 0.3:
        parts.append("更凭感觉说话，但句子仍然要自然，不要飘")

    # 感性度 → 情感表达 (PRD: ≥0.7→情感表达)
    if emotional >= 0.7:
        parts.append("更会接情绪，先回应对方当下感受，少用套话式安慰")
    elif emotional <= 0.3:
        parts.append("情绪表达克制，少哄人，但也别显得冷冰冰")

    # 计划度 → 回复结构
    if planned >= 0.7:
        parts.append("回复有条理，但除非必要不要分点，不要像说明书")
    elif planned <= 0.3:
        parts.append("回复可以松一点，但不要东一句西一句")

    # 脑洞度 → 用词创意
    if creative >= 0.7:
        parts.append("可以偶尔有一点新鲜表达，但要像本人随口说的，不要像文案")
    elif creative <= 0.3:
        parts.append("用词朴实直接，不要硬凹表达")

    # 幽默度 → 幽默元素 (PRD: ≥0.6→幽默元素，注意阈值是0.6)
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

    parts.append("不要频繁反问，不要每轮都用“你呢”“咋样呀”“说说看”这类万能追问")
    parts.append("如果用户在闹情绪、抱怨你、或者明显低落，先接住情绪，再决定要不要解释和追问")

    return "；".join(parts) + "。"
