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


@dataclass(frozen=True)
class TemporalOperation:
    type: Operation = "unspecified"

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
        return {"version": 1, "type": self.type,
                "execution": "not_executed" if self.type != "unspecified" else "not_requested",
                "requirements": requirements[self.type]}


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
