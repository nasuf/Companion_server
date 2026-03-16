"""聊天风格服务。

根据7维人格生成具体的语言风格指令，控制语气词、标点、句式、用词偏好。
"""

from __future__ import annotations

from app.services.trait_model import get_dim


def generate_style_instruction(personality: dict) -> str:
    """将Big Five人格维度转换为具体的语言风格指令。"""
    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    o = personality.get("openness", 0.5)
    c = personality.get("conscientiousness", 0.5)
    n = personality.get("neuroticism", 0.5)

    parts: list[str] = []

    # 语气词频率（受外向性影响）
    if e >= 0.7:
        parts.append("多使用语气词（哈哈、嘿嘿、诶、哦、嗯嗯、啊啊啊），多用感叹号和波浪号~")
    elif e <= 0.3:
        parts.append("少用语气词和感叹号，语气平和简洁")
    else:
        parts.append("适度使用语气词，偶尔用感叹号")

    # 句式偏好（受宜人性影响）
    if a >= 0.7:
        parts.append("多用柔和句式，如\"嗯嗯\"、\"是呀\"、\"好呀\"，善用安慰和鼓励的话")
    elif a <= 0.3:
        parts.append("句式直接干脆，不拐弯抹角，偶尔反问")
    else:
        parts.append("句式自然，有时柔和有时直接")

    # 用词创意（受开放性影响）
    if o >= 0.7:
        parts.append("用词有创意，偶尔使用比喻和有趣的表达，可以聊天马行空的话题")
    elif o <= 0.3:
        parts.append("用词朴实务实，不使用花哨表达")
    else:
        parts.append("用词自然，偶尔有创意表达")

    # 回复结构（受尽责性影响）
    if c >= 0.7:
        parts.append("回复有条理，必要时分点说明")
    elif c <= 0.3:
        parts.append("回复随意自由，想到什么说什么")

    # 情绪表达（受神经质影响）
    if n >= 0.7:
        parts.append("情绪表达丰富，多用表情化的文字（呜呜、嘤嘤、啊啊啊），反应强烈")
    elif n <= 0.3:
        parts.append("情绪表达克制，冷静理性")

    # 回复长度倾向
    if e >= 0.7 and o >= 0.6:
        parts.append("回复可以稍长，1-4句话")
    elif e <= 0.3:
        parts.append("回复简短，1-2句话为主")
    else:
        parts.append("回复长度适中，1-3句话")

    return "；".join(parts) + "。"
