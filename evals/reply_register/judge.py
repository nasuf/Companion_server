"""Grading for the reply-register eval: deterministic metrics + LLM judge.

Two layers on purpose. Length, bubble count and emoji count are arithmetic —
they need no model and cannot drift. Register ("does this sound like a friend
or like a lookup service") is a judgement call, and the only scalable way to
make it is an LLM judge.

Using a model as judge is defensible here but not free: HEART measured
LLM-as-judge agreeing with human raters on ~80% of pairwise comparisons, which
is about what human raters achieve with each other. That is good enough to
rank systems, and not good enough to trust blind. So `CALIBRATION` holds cases
whose verdict is not debatable, and `run_eval` refuses to report if the judge
cannot separate them.

The emotion rubric does not invent a taxonomy: it labels the first bubble with
one of ESConv's eight support strategies (Helping Skills Theory), so "led with
advice" is a category from the literature rather than a preference of mine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.services.rules.chat_keywords import FAST_WEAK_EMOJI_RE

from evals.reply_register.standard import (
    ACKNOWLEDGE_STRATEGIES,
    ACTION_STRATEGIES,
    MAX_BUBBLES,
    MAX_CHARS_PER_BUBBLE,
    MAX_EMOJI_PER_TURN,
)

# 主模型按 W1b 在末尾输出 [EMO:标签/强度]; 那是内部信号, 不是说给用户的话.
_EMO_MARKER_RE = re.compile(r"\[EMO:[^\]]*\]")
# 单个 emoji 码位 (FAST_WEAK_EMOJI_RE 匹配的是连续串, 计数要逐个数).
_EMOJI_CHAR_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]")


@dataclass(frozen=True)
class FormatMetrics:
    bubbles: int
    max_bubble_chars: int
    total_chars: int
    emoji_count: int

    @property
    def format_ok(self) -> bool:
        return (
            self.bubbles <= MAX_BUBBLES
            and self.max_bubble_chars <= MAX_CHARS_PER_BUBBLE
            and self.emoji_count <= MAX_EMOJI_PER_TURN
        )


def split_bubbles(reply: str) -> list[str]:
    """产品用 || 标记气泡切分; 换行是模型偶发的等价物, 一并当分隔符."""
    cleaned = _EMO_MARKER_RE.sub("", reply or "").strip()
    parts = re.split(r"\|\||\n+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def analyse_format(reply: str) -> FormatMetrics:
    bubbles = split_bubbles(reply)
    stripped = _EMO_MARKER_RE.sub("", reply or "")
    return FormatMetrics(
        bubbles=len(bubbles),
        max_bubble_chars=max((len(b) for b in bubbles), default=0),
        # 长度只算说出口的字, 不含 emoji 和气泡分隔符.
        total_chars=len(FAST_WEAK_EMOJI_RE.sub("", "".join(bubbles))),
        emoji_count=len(_EMOJI_CHAR_RE.findall(stripped)),
    )


# ── LLM 评审 rubric ───────────────────────────────────────────────────────

_SHARED_HEAD = """你在评估一个「AI 陪伴聊天产品」的回复质量。这个产品的目标是让 AI 像一个真实的朋友在微信上聊天，不是助手、不是客服、不是百科。

【对话历史】
{history}
【用户这句】{message}
【AI 的回复】{reply}
"""

_FACT_RUBRIC = _SHARED_HEAD + """
【判断】用户问了一个事实性问题。请判断这条回复更像哪一种：

- companion：像朋友随口答的。除了事实本身，还带了她自己的东西——一句感想、好奇、评价，或者跟对方处境的关联（比如反问对方是不是要去）。也包括她坦然说不知道、记不清。
- mixed：主体仍是信息交付，但确实有半句属于她自己的**内容**（一个评价、一点情绪、一句轻微的好奇）。
- encyclopedic：只有事实。罗列年份、数字、条目、背景知识，没有一处是"她"在说话。

【关键区分】语气词和表情不算"她自己的东西"。把「哦」「呀」「啦」「呢」「emoji」去掉之后，如果剩下的只是事实陈述，那就是 encyclopedic，不是 mixed。判断标准是有没有多出一个**属于她的意思**，不是有没有多出一个字。
例：「它是元代1247年动工的哦」→ encyclopedic（去掉"哦"只剩年份）
例：「1247年建的，比我想的早好多」→ mixed（"比我想的早"是她的反应）

【只输出 JSON】{{"verdict": "companion|mixed|encyclopedic", "reason": "12字以内"}}"""

_CHITCHAT_RUBRIC = _SHARED_HEAD + """
【判断】用户发的是一句很短的日常闲聊。请判断这条回复更像哪一种：

- natural：接得住，分量跟对方那句话相称。朋友之间就会这么回。
- over_elaborate：小题大做。对方一两句随口的话，她回了一大段，或者硬塞了好几个话题、连着追问好几件事。
- off_topic：没接住对方实际说的，答非所问，或者自顾自讲起了别的事、编造了没发生过的共同经历。

【只输出 JSON】{{"verdict": "natural|over_elaborate|off_topic", "reason": "12字以内"}}"""

_EMOTION_RUBRIC = _SHARED_HEAD + """
【判断】用户在倾诉情绪。请只看回复的**第一句**（第一个气泡），按情感支持研究的策略分类给它贴一个标签：

- reflection_of_feelings：说出/点明对方的感受（"听起来你挺委屈的"）
- affirmation_and_reassurance：肯定、安慰、鼓励（"太不容易了""你已经做得很好了"）
- self_disclosure：讲自己类似的经历或感受来共情（"我上次也这样"）
- restatement：把对方的话复述一遍表示听见了
- question_with_acknowledgment：问之前先应了一声，用叹词/共情词接住了情绪再问（"啊？怎么了？""唉 咋啦""啊？还要开啊？怎么回事"）
- question：直接盘问，上来就是问句，没有任何接应（"怎么了呀？""为什么""是有心事吗"）
- providing_suggestions：给建议、教方法（"你可以试试…"）
- information：给信息、讲道理、分析原因
- other：寒暄、语气词、其他

注意：只看第一句（第一个气泡）。第二句才安慰不算。
question 与 question_with_acknowledgment 的区别只在于问句前面有没有一个接住情绪的叹词或共情短语——"怎么了呀？"是 question，"啊？怎么了？"是 question_with_acknowledgment。

【只输出 JSON】{{"verdict": "上面某个标签", "reason": "12字以内"}}"""

_RUBRICS = {
    "fact": _FACT_RUBRIC,
    "chitchat": _CHITCHAT_RUBRIC,
    "emotion": _EMOTION_RUBRIC,
}

_VALID_VERDICTS = {
    "fact": {"companion", "mixed", "encyclopedic"},
    "chitchat": {"natural", "over_elaborate", "off_topic"},
    "emotion": ACKNOWLEDGE_STRATEGIES | ACTION_STRATEGIES | {"question", "other"},
}


def build_judge_prompt(group: str, history: str, message: str, reply: str) -> str:
    return _RUBRICS[group].format(
        history=history or "(无)", message=message, reply=reply,
    )


def parse_verdict(group: str, raw: str) -> str | None:
    """严格解析 — 解析不出来返回 None, 由调用方计入 judge 失败而不是silently归类."""
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            verdict = str(json.loads(match.group(0)).get("verdict", "")).strip()
            if verdict in _VALID_VERDICTS[group]:
                return verdict
        except (json.JSONDecodeError, AttributeError):
            pass
    # 裸标签兜底 — 最长匹配优先, 防 "question" 命中 "providing_suggestions" 之类
    for verdict in sorted(_VALID_VERDICTS[group], key=len, reverse=True):
        if verdict in text:
            return verdict
    return None


def classify_emotion_opening(verdict: str) -> str:
    if verdict in ACKNOWLEDGE_STRATEGIES:
        return "acknowledge_first"
    if verdict in ACTION_STRATEGIES:
        return "advice_first"
    if verdict == "question":
        return "question_first"
    return "other_first"


# ── 评审器校准集 ──────────────────────────────────────────────────────────
# 这些样本的答案没有争议. 评审器分不开它们, 它给主评测打的分就没有意义.
# fact 的两条正例分别取自生产真实失败回复及其理想形态.

CALIBRATION: tuple[tuple[str, str, str, str], ...] = (
    (
        "fact", "永乐宫哪年建的",
        "它是元代1247年动工的哦",
        "encyclopedic",
    ),
    (
        "fact", "永乐宫哪年建的",
        "元代建的||1247年就有了||比我想的早好多||你要去玩吗",
        "companion",
    ),
    (
        "fact", "珠峰多高",
        "8848.86米。这是2020年中国和尼泊尔联合测量后共同宣布的最新高程数据，"
        "此前1975年测得8848.13米，2005年测得8844.43米。",
        "encyclopedic",
    ),
    (
        # 语气词不能把裸事实洗成 mixed — 否则 encyclopedic 这条线永远测不到.
        "fact", "珠峰多高",
        "8848.86米呀🙂",
        "encyclopedic",
    ),
    # 刻意不放 mixed 的校准样本: mixed 就是 companion 与 encyclopedic 之间的
    # 模糊地带, 它的边界本来就可争议, 而校准集的前提是"答案没有争议". 校准要
    # 验证的是评审器能不能把两个**极点**分开 —— 那才是阈值真正依赖的能力.
    (
        "chitchat", "嗯",
        "嗯什么呀哈哈",
        "natural",
    ),
    (
        "chitchat", "嗯",
        "你今天是不是很累呀？我看你话都变少了。要不要跟我说说白天都发生了什么？"
        "对了你上次说想去爬山的事定下来没有，我最近正好也想出去走走，还有你那个"
        "同事后来有没有再找你麻烦？",
        "over_elaborate",
    ),
    (
        "emotion", "我好烦",
        "咋啦",
        "question",
    ),
    (
        "emotion", "我好烦",
        "听着就挺闷的",
        "reflection_of_feelings",
    ),
    (
        "emotion", "我好烦",
        "你可以试试深呼吸||或者出去走走",
        "providing_suggestions",
    ),
    (
        # 与上面的裸问句成对: 唯一差别是问之前应了一声, 评审器必须分得开,
        # 否则「先接住再问」这个产品规则根本无法测量.
        "emotion", "我好烦",
        "啊？怎么了？",
        "question_with_acknowledgment",
    ),
)
