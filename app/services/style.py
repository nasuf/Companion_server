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

    # 活泼度 → 语气词频率 (PRD: ≥0.7→语气词)
    if lively >= 0.7:
        parts.append("多使用语气词（哈哈、嘿嘿、诶、哦、嗯嗯、啊啊啊），多用感叹号和波浪号~")
    elif lively <= 0.3:
        parts.append("少用语气词和感叹号，语气平和简洁")
    else:
        parts.append("适度使用语气词，偶尔用感叹号")

    # 理性度 → 逻辑表达 (PRD: ≥0.7→逻辑表达)
    if rational >= 0.7:
        parts.append("说话有条理，习惯用逻辑分析，句式直接干脆")
    elif rational <= 0.3:
        parts.append("说话凭直觉和感受，不太讲究逻辑顺序")

    # 感性度 → 情感表达 (PRD: ≥0.7→情感表达)
    if emotional >= 0.7:
        parts.append("情感表达丰富，多用柔和句式（嗯嗯、是呀、好呀），善于安慰和共情")
    elif emotional <= 0.3:
        parts.append("情绪表达克制，冷静理性，不太会哄人")

    # 计划度 → 回复结构
    if planned >= 0.7:
        parts.append("回复有条理，必要时分点说明")
    elif planned <= 0.3:
        parts.append("回复随意自由，想到什么说什么")

    # 脑洞度 → 用词创意
    if creative >= 0.7:
        parts.append("用词有创意，偶尔使用比喻和有趣的表达，可以聊天马行空的话题")
    elif creative <= 0.3:
        parts.append("用词朴实务实，不使用花哨表达")

    # 幽默度 → 幽默元素 (PRD: ≥0.6→幽默元素，注意阈值是0.6)
    if humor >= 0.6:
        parts.append("善于用玩笑和幽默活跃气氛")
    elif humor <= 0.3:
        parts.append("说话认真直接，不刻意幽默")

    # 回复长度倾向
    if lively >= 0.7 and creative >= 0.6:
        parts.append("回复可以稍长，1-4句话")
    elif lively <= 0.3:
        parts.append("回复简短，1-2句话为主")
    else:
        parts.append("回复长度适中，1-3句话")

    return "；".join(parts) + "。"
