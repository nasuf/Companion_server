"""Bounded query classification, not an event calculator or intent router.

These hints describe the evidence a temporal answer needs. They do not prove
that the evidence was retrieved, deduplicated, authorized or computed.
"""
from dataclasses import dataclass
import re
from typing import Literal

Operation = Literal["unspecified", "range", "current", "planned", "as_of",
                    "order", "count", "duration", "comparison", "multiple"]

_QUESTION = re.compile(r"[?？]|几|多少|多久|多长|什么|啥|谁|干嘛|怎么|来着|哪|是否|是不是|有没有|吗|对吧|是吧|记得|查|回顾|总结|统计|算")
_CALCULATIONS = (
    ("count", re.compile(r"几次|多少次|次数|几回|多少回|几场|多少场|几趟|了几天(?![了前后没])|(?<=[上加值请休熬])过几天(?![了前后没再])")),
    ("duration", re.compile(r"多久|多长时间|多少(?:天|小时|分钟|周|个月|年)|"
                            r"隔了几|持续了几|第几天|几天了|几个月了")),
    ("order", re.compile(r"上一次|上次|最近一次|第一次|最后一次|倒数第|第[一二三四五六七八九十\d]{1,3}次")),
    ("comparison", re.compile(r"相比|比较|哪个先|谁先|哪个后|先后顺序")),
)


# 问的是用户自己某段时间里"处于什么状态" (在哪上班/住哪/单身吗), 而不是那天发生的
# 某件事。状态类事实多半没有 occur_time, 按事件日期去筛只会一无所获, 要按有效期回答
# (TIME07)。只认疑问说法: "上周我对象送了我一束花" 是在讲事, 不是在问状态。
_STATE_QUESTION = re.compile(
    r"在哪(?:儿|里)?(?:上班|工作|住|读书|上学|实习)|住(?:在)?哪|在哪个(?:公司|城市|学校)|"
    r"(?:做|干)(?:什么|啥)工作|哪(?:个|家)公司|什么职业|是不是(?:还)?单身|有没有(?:对象|男朋友|女朋友)|"
    r"(?:还|是不是)在(?:做|干|上)什么|还?单身吗|"
    r"还在[^，。？?]{0,8}(?:上班|工作|租房|住|读书|上学|公司|单位|城市)|"
    r"在哪(?:个|家)[^，。？?]{0,4}(?:打工|上班|工作|实习)|换(?:了)?工作|"
    # "是不是还在做销售" "是不是还养着猫" "是不是在休假": 问一段时间里的状态
    # "是不是还要/还有" 问的是事 ("这周我是不是还有个面试"), 不算
    r"是不是(?:还(?![要有得会])|一直|仍然?|在)|还是不是|(?:工作|职业)是(?:什么|啥)"
)
# 问的是别人 ("我哥现在在哪工作") 或 "我们": 不按用户本人的状态回答
OTHER_SUBJECT = re.compile(
    r"我们|咱们|我妈|妈妈|我爸|爸爸|爷爷|奶奶|外公|外婆|男朋友|女朋友|对象|老公|老婆|朋友(?!圈)|"
    r"同事|同学|闺蜜|我哥|哥哥|我姐|姐姐|我弟|弟弟|我妹|妹妹|孩子|儿子|女儿|宝宝|她|(?<![其吉])他|聊天"
)
_AGENT_SUBJECT = re.compile(r"你")
_SELF_SUBJECT = re.compile(r"我")
# 只剩时间词的追问: "那上个月呢" "九月份呢"。去掉时间词后只能剩这些。
_ELLIPSIS_REST = re.compile(r"^[\s那么呢吗啊呀嘛,，。?？!！]*$")


_NOW_FOLLOWUP = re.compile(r"^(?:那|那么)?我?(?:现在|目前|如今)(?:呢|嘞)?[?？]?$")
# 问的是天气、新闻这类公共信息, 不是自己的安排
_PUBLIC_TOPIC = re.compile(r"天气|下雨|下雪|气温|温度|新闻|热搜|股市|比赛|放假安排|节假日")


def is_now_followup(text: str) -> bool:
    """"那现在呢": 接着上一句问现在的状态。"""
    return bool(_NOW_FOLLOWUP.match((text or "").strip()))


def is_agent_question(text: str) -> bool:
    """问的是 "你" (AI) 而不是用户自己: "你下个月打算去哪"。"""
    text = text or ""
    return bool(_AGENT_SUBJECT.search(text) and not _SELF_SUBJECT.search(text))


def is_state_question(text: str) -> bool:
    """问的是用户本人的状态; 问"你"(AI) 的走原来的人设回答。"""
    text = text or ""
    if not _STATE_QUESTION.search(text) or not _QUESTION.search(text):
        return False
    return not (is_agent_question(text) or OTHER_SUBJECT.search(text) or _PUBLIC_TOPIC.search(text))


def asks_own_plan(text: str) -> bool:
    """说到将来时默认是问用户自己的安排 ("明天呢")。排除: 问 AI 的、问别人的、
    问天气新闻的, 以及用户在陈述自己的事 ("我下周要去面试了，好紧张")。"""
    text = text or ""
    if is_agent_question(text) or OTHER_SUBJECT.search(text) or _PUBLIC_TOPIC.search(text):
        return False
    return bool(_QUESTION.search(text) or not _SELF_SUBJECT.search(text))


def is_elliptical_followup(text: str, time_spans: list[tuple[int, int]]) -> bool:
    text = text or ""
    if not 0 < len(text.strip()) <= 12 or not time_spans:
        return False
    rest = "".join(ch for i, ch in enumerate(text) if not any(s <= i < e for s, e in time_spans))
    return bool(_ELLIPSIS_REST.match(rest))


@dataclass(frozen=True)
class TemporalOperation:
    type: Operation = "unspecified"
    # True: 由"问的是过去的状态"推断出的 as_of, 不是用户说了"当时/截至"。
    # 历史问答查不到依据时回到普通回复, 不用兜底句顶替。
    inferred: bool = False

    def diagnostics(self) -> dict:
        # No inferred numbers, dates, identifiers or message text are emitted.
        requirements = {
            "count": ["deduplicated_events", "complete_scope"],
            "order": ["ordered_events", "complete_scope"],
            "duration": ["confirmed_endpoints", "duration_unit"],
            "current": ["current_validity"],
            "as_of": ["historical_validity"],
            "planned": ["plan_status"],
            "comparison": ["comparable_evidence"],
            "multiple": ["separate_queries"],
            "range": ["event_dates"],
            "unspecified": [],
        }
        out = {"version": 1, "type": self.type,
               "execution": "not_executed" if self.type != "unspecified" else "not_requested",
               "requirements": requirements[self.type]}
        if self.inferred:
            out["inferred"] = True
        return out


def classify_temporal_operation(text: str) -> TemporalOperation:
    """Conservative lexical hint; never execute an action based on this alone."""
    if not _QUESTION.search(text):
        return TemporalOperation()
    found = [kind for kind, pattern in _CALCULATIONS if pattern.search(text)]
    if set(found) == {"order", "duration"} and re.search(r"上一次|上次|最近一次|最后一次", text):
        return TemporalOperation("duration")  # "距离上次打球过去几天了": 距最近一次多久
    if len(found) > 1:
        return TemporalOperation("multiple")
    if found:
        return TemporalOperation(found[0])
    if re.search(r"当时|那时候|那时|截至|截止", text):
        return TemporalOperation("as_of")
    if re.search(r"计划|打算|准备|安排", text) and asks_own_plan(text):
        return TemporalOperation("planned")
    if re.search(r"现在|目前|此刻|如今", text) and not OTHER_SUBJECT.search(text):
        return TemporalOperation("current")
    if is_state_question(text) and re.search(r"最近|这阵子|这段时间|这些天|这几天", text):
        return TemporalOperation("current", inferred=True)  # "我最近是不是在减肥"
    return TemporalOperation()
