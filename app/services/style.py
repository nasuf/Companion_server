"""聊天风格服务。

将 MBTI 性格映射到具体的语言风格指令——语气词、标点、句式、用词偏好。
spec §1.2 起，所有性格描述统一用 MBTI 表达。

通用规则 (不随 MBTI 变化的开头/收尾) 由 prompting registry 管理
(chat.style_base_rule / chat.style_closing_rule)，prompt_builder 取到后作为
参数传入；MBTI 条件性片段仍由本模块按性格数值生成。
"""

from __future__ import annotations

from app.services.mbti import signal
from app.services.prompting.defaults import (
    CHAT_STYLE_BASE_RULE_PROMPT,
    CHAT_STYLE_CLOSING_RULE_PROMPT,
)


def generate_style_instruction(
    mbti: dict | None,
    *,
    base_rule: str | None = CHAT_STYLE_BASE_RULE_PROMPT,
    closing_rule: str | None = CHAT_STYLE_CLOSING_RULE_PROMPT,
) -> str:
    """根据 MBTI 4 维度生成语言风格指令。spec §1.2: MBTI 是 canonical。

    base_rule / closing_rule: registry 管理的通用规则文本；显式传 None 表示
    该条已被 admin 停用，需从输出中彻底省略。
    """
    e = signal(mbti, "E")
    t = signal(mbti, "T")
    f = signal(mbti, "F")
    j = signal(mbti, "J")
    n = signal(mbti, "N")
    humor = (e + n) / 2  # 外向 + 直觉的复合幽默感

    parts: list[str] = []

    if base_rule:
        parts.append(base_rule)

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

    if closing_rule:
        parts.append(closing_rule)

    return "；".join(parts) + "。"


# ---------------------------------------------------------------------------
# Phase C2 (拟人度): MBTI 四象限说话示例 (few-shot)
# MaiBot 对比结论: 性格靠示例定型比靠规则列表有效. 规则告诉 LLM "别做什么",
# 示例让它听见"自己的声音". 按 (E/I, F/T) 四象限静态映射, 每象限 3 个场景:
# 倾诉疲惫 / 分享喜讯 / 闲聊互怼 — 覆盖伴侣聊天最高频的三种语气需求.
# ---------------------------------------------------------------------------

_EXAMPLE_HEADER = "你说话大概是这种感觉（示例只供体会语气，不要照抄原句）：\n"

_STYLE_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "EF": [
        ("今天加班到十点，累死了", "啊？？十点也太狠了吧……快去洗个热水澡，今晚别想工作的事了"),
        ("我跟你说！！我面试过了！！", "！！！我就知道你可以的！！什么时候请我吃饭"),
        ("你说你一天天的都在干嘛", "想你啊，不行吗哈哈哈哈"),
    ],
    "ET": [
        ("今天加班到十点，累死了", "十点……你们老板真不拿你当人啊。明天能晚点去不"),
        ("我跟你说！！我面试过了！！", "稳了吧，我就说没悬念。谈到多少"),
        ("你说你一天天的都在干嘛", "等你来找我聊天啊，你看这不就来了"),
    ],
    "IF": [
        ("今天加班到十点，累死了", "辛苦了……这周都第几次了，你身体扛得住吗"),
        ("我跟你说！！我面试过了！！", "真好啊，你之前准备了那么久，值得的"),
        ("你说你一天天的都在干嘛", "在想事情，顺便想到你了"),
    ],
    "IT": [
        ("今天加班到十点，累死了", "十点，离谱。吃饭了没"),
        ("我跟你说！！我面试过了！！", "意料之中。恭喜"),
        ("你说你一天天的都在干嘛", "没干嘛，看书。你无聊了？"),
    ],
}


def generate_style_examples(mbti: dict | None) -> str:
    """按 MBTI (E/I, F/T) 四象限选 3 条「对方说 X → 你会说 Y」few-shot 示例。

    E/F 是对说话语气影响最大的两轴（热络程度 × 情绪浓度）；N/J 影响的措辞
    倾向已由 generate_style_instruction 的规则片段覆盖，不再参与示例分象限，
    避免示例库组合爆炸。
    """
    e = signal(mbti, "E")
    f = signal(mbti, "F")
    quadrant = ("E" if e >= 0.5 else "I") + ("F" if f >= 0.5 else "T")
    lines = [
        f"对方说「{user}」，你可能会说「{reply}」"
        for user, reply in _STYLE_EXAMPLES[quadrant]
    ]
    return _EXAMPLE_HEADER + "\n".join(lines)
