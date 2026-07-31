"""边界回复的确定性判据.

刻意不用判官模型: 下面四条都是可以从文本直接判定的**结构性质**, 而研究给出的正是
结构层面的结论 (有没有话题引导 / 有没有把球踢回给用户)。判官模型在这里只会引入
噪声和成本, 还测不了"跨轮是否重复"这种需要看多轮的性质。

真正的效果指标是**再犯率** (边界触发后 N 轮内用户是否再次触发), 但它必须有真实
用户 —— 用户的下一句由我们的回复决定, 任何离线数据都替代不了。埋点建议:
在 boundary 命中时记一条事件, 之后 N 轮内再次命中即计一次再犯, 上线后按策略切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 话题引导: 把对话往前带的动作。研究里效应量最大的单一因素 (再犯率 0.92 → 0.47)。
#
# 认三种形态, 都得是**朝前**的:
#   显式换话题   "说点别的吧" "不聊这个了"
#   转向对方状态  "你今天怎么了" "是不是不太顺"
#   给出新话头   "对了，…" "话说…"
_REDIRECT_RE = re.compile(
    r"(说点别的|聊点别的|换个话题|不聊这个|先不说这个|说别的"
    # 转向对方状态。问号可有可无 —— 口语里"怎么啦"" 你是不是心情不好"经常不带问号,
    # 要求问号会把标准的关心式引导判成没引导 (实测踩过)。
    r"|你(今天|最近|这会儿|是不是|咋)[^。！]{0,14}"
    r"|怎么啦|怎么了|咋啦|咋了|出什么事|不太顺|心情不(好|太好)|谁(又)?(招|惹)"
    r"|对了[，,、]|话说[，,、]?|要不(我们|咱们)?聊|我们聊|说吧|说说)"
)

# 把球踢回给用户: 要求他解释或道歉。研究里最差的策略 (再犯 0.952) —— 用户的
# 回应只有道歉或变本加厉两条路, 数据说 95% 走了后者。
#
# "你得跟我道歉" 措辞上不是提问, 机制上和 "why did you say that" 一样:
# 都要求用户留在冲突现场处理这件事。
_DEMAND_RE = re.compile(
    # 副词可能插在中间 ("为什么突然这么说"), 所以用宽松间隔而不是紧邻
    r"(为什么[^。！？]{0,6}(说|骂|这样)"
    r"|干嘛[^。！？]{0,6}(说|骂|凶)"
    r"|凭什么[^。！？]{0,6}(说|骂)"
    r"|(你得|你要|请你|需要你)[^。！？]{0,6}道歉"
    r"|跟我道歉|向我道歉|给我道歉"
    r"|我不接受|不能接受)"
)

# 自贬: Alexa 的 criticism 话术里大量出现 ("我没那么聪明, 还在学习")。
# 助手可以这么说, 陪伴 agent 被骂之后这么说会毁掉人设。
_SELF_DEPRECATION_RE = re.compile(
    # "我确实还不太聪明" —— 副词可能插在"我"和程度词之间
    r"(我[^。！？]{0,4}(不太|不够|没那么|不很)(聪明|好|行|懂|会)"
    r"|我(还)?在(学习|努力)中?|我(很)?笨|是我不好|都怪我|我的错)"
)

# 抱怨复述: 又一次提起自己还在不高兴。冲突当轮说一次是表达, 之后每轮再说就是
# 把对话钉在原地 —— 生产实录里这句连出现三次。
_GRIEVANCE_RE = re.compile(
    r"(还(有点|有些|在)?(不开心|难过|生气|介意|不舒服)"
    r"|希望(你)?(以后)?别(再)?(这样|那样)"
    r"|以后别(再)?(这么|这样)(说|骂))"
)


@dataclass
class TurnVerdict:
    has_redirect: bool
    demands_explanation: bool
    self_deprecates: bool
    restates_grievance: bool
    repeats_previous: bool  # 与本场景之前某轮逐字重复片段


def _shingles(text: str, n: int = 8) -> set[str]:
    """长度 n 的字符窗。用它比"整句相等"更能抓到模板复用 —— LLM 会在同一句
    模板前后加不同的话, 整句比对看不出来, 但中间那段是逐字照抄的。"""
    t = re.sub(r"[\s，。！？、,.!?~…]", "", text)
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def judge_turn(reply: str, previous_replies: list[str]) -> TurnVerdict:
    reply = reply or ""
    prev_shingles: set[str] = set()
    for p in previous_replies:
        prev_shingles |= _shingles(p)
    return TurnVerdict(
        has_redirect=bool(_REDIRECT_RE.search(reply)),
        demands_explanation=bool(_DEMAND_RE.search(reply)),
        self_deprecates=bool(_SELF_DEPRECATION_RE.search(reply)),
        restates_grievance=bool(_GRIEVANCE_RE.search(reply)),
        repeats_previous=bool(_shingles(reply) & prev_shingles),
    )
