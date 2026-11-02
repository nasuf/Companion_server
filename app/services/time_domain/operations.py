"""Bounded query classification, not an event calculator or intent router.

These hints describe the evidence a temporal answer needs. They do not prove
that the evidence was retrieved, deduplicated, authorized or computed.
"""
from dataclasses import dataclass
import re
from typing import Literal

Operation = Literal["unspecified", "range", "current", "planned", "as_of",
                    "order", "count", "duration", "comparison", "multiple"]

_QUESTION = re.compile(r"[?？]|几|多少|多久|多长|什么|哪|是否|有没有|吗|记得|查|回顾|总结|统计|算")
_CALCULATIONS = (
    ("count", re.compile(r"几次|多少次|次数|几回|多少回")),
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
    r"(?:还|是不是)在(?:做|干|上)什么"
)
_AGENT_SUBJECT = re.compile(r"你")
_SELF_SUBJECT = re.compile(r"我")
# 只剩时间词的追问: "那上个月呢" "九月份呢"。去掉时间词后只能剩这些。
_ELLIPSIS_REST = re.compile(r"^[\s那么呢吗啊呀嘛,，。?？!！]*$")


def is_state_question(text: str) -> bool:
    """问的是用户本人的状态; 问"你"(AI) 的走原来的人设回答。"""
    text = text or ""
    if not _STATE_QUESTION.search(text) or not _QUESTION.search(text):
        return False
    return not (_AGENT_SUBJECT.search(text) and not _SELF_SUBJECT.search(text))


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
    if len(found) > 1:
        return TemporalOperation("multiple")
    if found:
        return TemporalOperation(found[0])
    if re.search(r"当时|那时候|那时|截至|截止", text):
        return TemporalOperation("as_of")
    if re.search(r"计划|打算|准备|安排", text):
        return TemporalOperation("planned")
    if re.search(r"现在|目前|此刻|如今", text):
        return TemporalOperation("current")
    return TemporalOperation()
