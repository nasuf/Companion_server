"""Classify confirmed changes without turning confirmation time into event time.

Only pinned original messages supply time evidence. A later ``yes`` authorizes
the write but cannot make a late proposal newer, or establish its effective date.
The boundary is a possible transition window, never a fabricated exact instant.
"""

from dataclasses import dataclass, replace
from datetime import datetime
import re

from app.services.time_domain.query import resolve_temporal_query
from app.services.time_domain.timestamps import exclusive_end, parse_time

_PLAN = re.compile(
    r"打算|计划(?=在|于|要|去|搬|换|入职|离职|开始|结束|下|明|后|\d)|准备(?=在|于|要|去|搬|换|入职|离职|开始|下|明|后|昨|今|上|这|\d)|将要|预计|想要|希望|等到"
)
_CHANGE = re.compile(
    r"搬到|搬回|搬去|搬离|搬走|迁到|移居|换了|换成|换到|跳槽|辞职|离职|入职|调到|转到|开始|结束|痊愈|取消|改期|改到"
)
_CORRECTION = re.compile(r"说错|记错|弄错|搞错|写错|口误|其实一直|实际上一直")
_NONASSERTIVE = re.compile(
    r"[“”\"「」『』]|如果|假如|假设|万一|也许|可能|据说|听说|[?？]|是不是|是否|不确定|没确定|不清楚|不知道|随便|算了|再说|不用改|^(?:不是|不对|别改|不改)[。！!\s]*$"
)
_OTHER = re.compile(
    r"(?:朋友|同事|家人|妈妈|爸爸|哥哥|姐姐|弟弟|妹妹|(?<!其)他|她).{0,5}(?:搬|住|换|入职|离职|说|告诉)"
)
_DENIED = re.compile(
    r"没(?:有)?(?:搬|换|入职|离职|开始|结束)|并未|尚未|不打算|不准备|不计划|不(?:再)?(?:搬|换|入职|离职)"
)
_OCCURRED_RE = re.compile(
    r"(?:已经|刚刚|已|昨天|前天|上周).{0,8}(?:搬|换|跳槽|辞职|入职|离职|开始|结束|取消|改期)"
    r"|(?:搬到|搬回|换成|换到|跳槽|辞职|调到|入职|离职).{0,12}了"
)
# "马上要搬到上海了" / "我快离职了": 了 marks the imminent, not the done.
_IMMINENT = re.compile(
    r"(?:快要|就要|马上要?|即将|将要|快|要)(?:搬|换|入职|离职|开始|结束)"
)


class _Occurred:
    """Completed-change marker evaluated per clause, excluding imminence."""

    @staticmethod
    def search(text: str):
        for clause in _CLAUSE.findall(text) or [text]:
            if _IMMINENT.search(clause) and not re.search(r"已经|刚刚|已", clause):
                continue
            match = _OCCURRED_RE.search(clause)
            if match:
                return match
        return None


_OCCURRED = _Occurred()
# A change stated about the future is a plan even without 打算/计划. Used only
# when no single date was resolved; it never creates a boundary on its own.
_FUTURE_HINT = re.compile(
    r"明天|后天|下周|下个?月|下礼拜|明年|过几天|过阵子|即将|马上要?|就要|快要"
    r"|(?:快|要|将)(?=搬|换|入职|离职|开始|结束)"
)
# Repeated or mixed acknowledgements ("对对对", "是的是的", "嗯嗯~", "ok") are
# still only an authorization, never a restatement that moves the fact clock.
_ACK = re.compile(
    r"(?:(?:是|是的|对|对的|对呀|对啊|没错|确实|嗯|好|好的|好呀|行|可以|确认|确认修改|就这样|ok|okay|yes)"
    r"[呀啊呢吧哦~～。！!，,、\s]*)+",
    re.IGNORECASE,
)
# Clauses addressed back to the agent or well-wishes carry no fact: "我搬到
# 上海了，你呢？" stays an assertion, and "希望顺利" is not a plan.
_ASIDE = re.compile(
    r"^(?:那|那么)?(?:你|您)(?:呢|那边呢|怎么样|觉得呢)[?？。！!~～\s]*$"
    r"|^(?:希望|祝)[^，,。]{0,10}(?:顺利|开心|一切都好|一切顺利|好运)[呀啊吧。！!~～\s]*$"
)
_CLAUSE = re.compile(r"[^，,。；;！!？?\n]+[，,。；;！!？?]*")


# A confirmation restates the fact only when it names a change, correction,
# plan or time of its own.
_FACT_SIGNAL = re.compile(
    "|".join(p.pattern for p in (_CHANGE, _CORRECTION, _PLAN, _OCCURRED_RE, _FUTURE_HINT))
    + r"|昨天|前天|今天|上周|上个?月|去年|搬|住|入职|离职|在\S{1,8}(?:工作|上班|住)"
)


def _substantive(text: str) -> str:
    """Drop aside clauses before assertion checks; keep everything else verbatim."""
    clauses = [c for c in _CLAUSE.findall(text) if not _ASIDE.match(c.strip())]
    return "".join(clauses) if clauses else text


def select_change_statements(
    original_rows: list[dict], confirmation_rows: list[dict]
) -> list[dict]:
    """A substantive correction is new evidence; an acknowledgement is not.

    Keep a multi-bubble statement together rather than cherry-picking a date.
    This selection changes neither the saved source pins nor the confirmation
    contract: every source is still revalidated before committing.
    """
    for row in confirmation_rows:
        text = row.get("content") or ""
        if not _ACK.fullmatch(text.strip()) and (
            _FACT_SIGNAL.search(text)
            or _NONASSERTIVE.search(text)
            or _DENIED.search(text)
            or _OTHER.search(text)
        ):
            # A restatement, or a hedge/denial ("我还没确定") that must stop
            # the write, both become the statement under evaluation.
            return confirmation_rows
    # "对呀，谢谢" / "是的哈哈" / "对👍" carry no fact of their own: the
    # original statement (and its plan/future markers) stays the evidence.
    return original_rows


@dataclass(frozen=True)
class ChangeSemantics:
    kind: str
    reason: str
    statement_at: datetime | None
    boundary_start: datetime | None = None
    boundary_end: datetime | None = None
    precision: str | None = None
    boundary_source_id: str | None = None
    fact_source_ids: tuple[str, ...] = ()

    def diagnostic(self) -> dict:
        return {
            "version": 1,
            "kind": self.kind,
            "reason": self.reason,
            "statement_at": (
                self.statement_at.isoformat() if self.statement_at else None
            ),
            "fact_source_ids": list(self.fact_source_ids),
            "boundary": {
                "state": "source_window" if self.boundary_start else "unknown",
                "start": (
                    self.boundary_start.isoformat() if self.boundary_start else None
                ),
                "end_exclusive": (
                    self.boundary_end.isoformat() if self.boundary_end else None
                ),
                "precision": self.precision,
                "source_message_id": self.boundary_source_id,
                "meaning": "possible_transition_window_not_exact_valid_from",
            },
        }


def resolve_change_semantics(change: str, original_rows: list[dict]) -> ChangeSemantics:
    value = _resolve_change_semantics(change, original_rows)
    return replace(
        value,
        fact_source_ids=(
            tuple(row["id"] for row in original_rows) if value.kind != "unknown" else ()
        ),
    )


def _resolve_change_semantics(
    change: str, original_rows: list[dict]
) -> ChangeSemantics:
    """Rows must already have passed the SQL scope/hash/source-time checks.

    Unsupported or mixed statements never gain date authority. The existing
    confirmation model can still classify a change with an unknown boundary;
    explicit future plans must not archive the currently true identity fact.
    """
    if change not in {"变化", "错误", "新增"} or not 1 <= len(original_rows) <= 40:
        return ChangeSemantics("unknown", "invalid_contract", None)
    if any(
        not isinstance(row.get("id"), str) or not 1 <= len(row["id"]) <= 200
        for row in original_rows
    ):
        return ChangeSemantics("unknown", "invalid_source_identity", None)
    clocks = [parse_time(row.get("created_at")) for row in original_rows]
    if any(clock is None for clock in clocks):
        return ChangeSemantics("unknown", "source_clock_unavailable", None)
    statement = max(clocks)
    texts = [row.get("content") for row in original_rows]
    if any(not isinstance(text, str) or not 1 <= len(text) <= 12000 for text in texts):
        return ChangeSemantics("unknown", "source_text_unavailable", statement)
    texts = [_substantive(text) for text in texts]
    if any(
        _NONASSERTIVE.search(text) or _OTHER.search(text) or _DENIED.search(text)
        for text in texts
    ):
        return ChangeSemantics("unknown", "non_assertive_or_other_subject", statement)
    planned = any(_PLAN.search(text) for text in texts)
    correction = any(_CORRECTION.search(text) for text in texts)
    occurred = any(_OCCURRED.search(text) for text in texts)
    if planned and occurred:
        return ChangeSemantics("unknown", "mixed_plan_and_completed_change", statement)
    clauses = [c for t in texts for c in re.split(r"[，,。；;！!]", t) if c.strip()]
    if any(_OCCURRED.search(c) for c in clauses) and any(
        _FUTURE_HINT.search(c) and _CHANGE.search(c) and not _OCCURRED.search(c)
        for c in clauses
    ):
        # "前天离职，明天入职" mixes a done change with one that has not
        # happened; one confirmation cannot make both current. Ask instead.
        return ChangeSemantics("unknown", "mixed_completed_and_future_change", statement)
    if planned and correction:
        return ChangeSemantics("unknown", "mixed_plan_and_correction", statement)
    kind = (
        "planned"
        if planned
        else (
            "correction"
            if correction or change == "错误"
            else "addition" if change == "新增" else "transition"
        )
    )
    result = ChangeSemantics(kind, "confirmed_without_proven_boundary", statement)
    if kind == "transition" and not occurred and any(_FUTURE_HINT.search(t) for t in texts):
        # Without a resolved date a future marker still decides plan vs. change:
        # never archive the current fact for something that has not happened.
        result = ChangeSemantics("planned", "future_marker_without_resolved_date", statement)
    # One original utterance and one bound date only. Multi-message confirmation
    # remains useful, but cannot collapse several dates into a single endpoint.
    if len(original_rows) != 1 or result.kind not in {"transition", "planned"}:
        return result
    kind = result.kind
    row, text, anchor = original_rows[0], texts[0], clocks[0]
    if re.search(r"(?:计划|准备工作|面试|会议|安排).{0,8}(?:完成|结束|取消)了?", text):
        return result
    if not re.search(r"我(?!们|的)|本人", text) or not _CHANGE.search(text):
        return result
    query = resolve_temporal_query(
        text, anchor=anchor, anchor_source="persisted_message"
    )
    if (
        query.status != "resolved"
        or query.date_basis != "event_candidate"
        or len(query.expressions) != 1
    ):
        return result
    period = query.expressions[0]
    end = exclusive_end(period.end, period.end_boundary)
    # A bounded window (上周/上个月/昨天晚上) is the possible transition range,
    # not an exact instant. Unknown precisions keep the date unknown.
    window = period.precision in {
        "second", "minute", "hour", "day", "week", "month", "year", "range",
    }
    if period.start > anchor and kind != "planned":
        # Decide future-vs-current before precision, so an unsupported
        # precision can never turn a future change into a current one.
        if not window:
            return ChangeSemantics("planned", "future_change_is_not_current", statement)
        return ChangeSemantics(
            "planned",
            "future_change_is_not_current",
            statement,
            period.start,
            end,
            period.precision,
            row["id"],
        )
    if not window:
        return result
    if kind == "planned" and end <= anchor:
        # A past plan is not evidence it actually happened.
        return ChangeSemantics("planned", "past_plan_not_completion", statement)
    return ChangeSemantics(
        kind,
        "original_source_transition_window",
        statement,
        period.start,
        end,
        period.precision,
        row["id"],
    )
