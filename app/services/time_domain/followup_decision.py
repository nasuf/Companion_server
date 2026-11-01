"""Shared optional follow-up decision: same-event repeat, response and cooldown.

One decision for ordinary replies and proactive A/B messages. Personality may
change tone; it can never turn a suppressed follow-up into an allowed one.

Only *outcome follow-ups* are constrained: a question about how a specific
event turned out (结果/进展/通知/顺利 ...). Ordinary questions inside a live
conversation ("几点的面试呀", "面试前要不要早点睡") are never counted or cut.

Event identity is deliberately narrow: a coarse event family (interview, exam,
check-up ...) plus explicit qualifiers. Sharing a generic word such as 工作 is
not identity, and a different day/round/"new" interview is a different event.
Silence is not dislike: an ignored ask only cools down, and goes quiet once the
user's own mention of the event is no longer recent. Clocks come from the last
real ask or answer, so re-evaluating never extends or resets them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import json
import logging
import re
from uuid import uuid4

from app.services.time_domain.timestamps import parse_time

logger = logging.getLogger(__name__)

POLICY_VERSION = "followup_decision_v1"
# Dense recent dialogue (context, responses) and a sparse, family-matched ledger
# of older rows so an ignored ask is still seen after it leaves the dense page.
HISTORY_DAYS = 7
HISTORY_ROWS = 200
LEDGER_DAYS = 60
LEDGER_ROWS = 40
# Answered "still waiting": ask again after a day. Unanswered: wait out the
# next day too ("yesterday asked, today don't rephrase"), then go quiet.
ANSWERED_COOLDOWN = timedelta(hours=24)
UNANSWERED_COOLDOWN = timedelta(hours=36)
MAX_UNANSWERED = 2
# A user mention this recent means the event is being discussed right now; a
# question right after an ask counts as the user's response.
SESSION_WINDOW = timedelta(hours=3)
REPLY_READ_TIMEOUT_S = 1.5

# Families group different words for the same kind of real-world event.
_FAMILIES: tuple[tuple[str, str], ...] = (
    ("interview", r"面试|笔试|复试|初试|终面|一面|二面|三面|群面|面谈|面了|去面|面完|录用|offer|Offer|OFFER|入职通知|录取通知"),
    ("exam", r"考试|考研|考公|考得|考完|考了|期末|期中|模考|成绩|分数|科目[一二三四]|驾考"),
    ("medical", r"体检|复查|检查结果|化验|报告单|手术|住院|出院|看医生|看病|门诊|牙医|拔牙"),
    ("illness", r"感冒|发烧|退烧|烧退|生病|咳嗽|头疼|头痛|胃疼|肚子疼|过敏|病好|身体好点|身体怎么样"),
    ("move", r"搬家|新家|搬过去|搬完"),
    ("trip", r"出差|旅行|旅游|行程|航班|机票|签证|面签"),
    ("presentation", r"答辩|汇报|述职|演讲|路演|提案"),
    ("competition", r"比赛|决赛|初赛|复赛|选拔"),
    ("date", r"相亲|约会|见家长"),
    ("application", r"申请|报名|审批|投递|简历"),
)
# Words asking how an event turned out. Strong ones point at a result by
# themselves; weak ones only with a named event or an explicit reference.
_STRONG = (r"结果|进展|通知|通过|过了没|过没|过了吗|成绩|分数|录用|有消息|消息了|批下来|批了|好点|好些|康复|出院|定了|落定|搞定"
           r"|出来了吗|出来了没|出来没|出了没|下来了吗|下来没|结束了吗|结束了没|完了吗|完了没|什么时候出|啥时候出"
           r"|办好|弄好|好了吗|好了没|拿到了吗|拿到了没|拿到没|严重吗|好转|退了吗|退了没")
_WEAK = r"怎么样|咋样|如何|顺利|怎么说|聊得"
_OUTCOME = re.compile(_STRONG + "|" + _WEAK)
_STRONG_OUTCOME = re.compile(_STRONG)
# Questions before the event (preparation, nerves, timing, speculation) are not
# outcome follow-ups, even with 怎么样/顺利/好了吗 in them.
_PRE_EVENT = re.compile(
    r"准备|紧张|加油|几点|在哪|哪里|打算|会不会|会顺利|能顺利|会过|能过|期待|担心|信心"
    r"|什么时候(?!出)|啥时候(?!出)"
)
# Result words specific enough to tie a pronoun-less clause to an event named
# elsewhere in the same bubble ("体检累不累？报告什么时候出？").
_EVENT_RESULT = re.compile(r"结果|报告|成绩|分数|通知|出来|录用|offer|批下来|进展")
# Explicit references to an earlier matter. A bare 后来呢 in a story is not one.
# Pure references only: "你说的那家火锅" names its own object and borrows nothing.
_REFERENTIAL = re.compile(r"那件事|这件事|那个事|那事(?![\u4e00-\u9fff])|(?:上次|之前|你)(?:说|提)的(?:那件事|事)")
_LATER = re.compile(r"后来|后续")
# Qualifiers that name a *different* instance of the same family.
_NEW_INSTANCE = re.compile(
    r"新的?(?=(?:一)?(?:轮|个|家|场)?(?:面试|笔试|考试|比赛|体检|offer|申请))|另一个|另外一个|另一家|另外一家|又一|又有|再一次|第[二三四五六七八九十\d]+(?:轮|次|家|场|个)|下一(?:轮|次|家|场|个)|[一二三四]面(?!八方)|科目[一二三四]"
)
_DAY_QUALIFIER = re.compile(r"今天|明天|后天|昨天|前天|这周|下周|上周|周[一二三四五六日天]|\d{1,2}[月号日]")
# Resolution only counts in a message that names the event itself.
_RESOLVED = re.compile(
    r"取消了|不去了|不面了|结束了|已经结束|(?<!准备)过了|通过了|没过|没通过|挂了|被拒"
    r"|(?:拿到|收到)(?:了)?(?:offer|录用|结果|成绩)|出结果了|结果出来了|成绩出来了"
    r"|(?:病|烧|身体|伤|搬|装修|签证)(?:已经|也|都)?好了|痊愈|康复了|搬完了|搞定了|定下来了"
)
# A statement still about the future ("拿到面试通知了，明天去") is not a result.
_FUTURE = re.compile(r"明天|后天|下周|下个?月|过几天|待会|晚点|周[一二三四五六日天]去")
_PENDING = re.compile(r"还没|没消息|没出|在等|等通知|等结果|不知道|不清楚|再说")


def event_family(text: str) -> str | None:
    """First family named in text; None when only generic words appear."""
    best = None
    for family, pattern in _FAMILIES:
        match = re.search(pattern, text)
        if match and (best is None or match.start() < best[1]):
            best = (family, match.start())
    return best[0] if best else None


def _qualifiers(text: str) -> frozenset[tuple[str, str]]:
    return frozenset(("new", m[0]) for m in _NEW_INSTANCE.finditer(text)) | frozenset(
        ("day", m[0]) for m in _DAY_QUALIFIER.finditer(text)
    )


@dataclass(frozen=True)
class EventRef:
    family: str
    qualifiers: frozenset[tuple[str, str]]
    resolved_from: str  # "text" | "context"


def named_event(text: str) -> EventRef | None:
    family = event_family(text)
    return EventRef(family, _qualifiers(text), "text") if family else None


def outcome_event(question: str, recent_user_texts: list[str]) -> EventRef | None:
    """Event an outcome follow-up asks about, or None if it is not one.

    A named event needs any outcome word. A question without one borrows the
    event the user named most recently (``recent_user_texts`` newest first) only
    with an explicit reference (那件事怎么样了) or 后来 + a strong result word.
    """
    if not _OUTCOME.search(question) or _PRE_EVENT.search(question):
        return None
    ref = named_event(question)
    if ref:
        return ref
    if not (_REFERENTIAL.search(question)
            or (_LATER.search(question) and _STRONG_OUTCOME.search(question))):
        return None
    for text in recent_user_texts:
        family = event_family(text)
        if family:
            return EventRef(family, frozenset(), "context")
    return None


def same_event(left: EventRef, right: EventRef) -> bool:
    """Same family and no conflicting instance qualifier.

    A new-instance marker on only one side ("今天的新面试", "二面") marks a
    different event; day words must agree when both sides carry one.
    """
    if left.family != right.family:
        return False
    if any(kind == "new" for kind, _ in left.qualifiers ^ right.qualifiers):
        return False
    left_days = {q for kind, q in left.qualifiers if kind == "day"}
    right_days = {q for kind, q in right.qualifiers if kind == "day"}
    return not (left_days and right_days and left_days != right_days)


@dataclass(frozen=True)
class FollowupDecision:
    decision: str  # allow | suppress
    reason: str
    next_eligible_at: datetime | None = None
    evidence: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def diagnostic(self) -> dict:
        return {
            "policy_version": POLICY_VERSION,
            "decision": self.decision,
            "reason": self.reason,
            "next_eligible_at": self.next_eligible_at.isoformat() if self.next_eligible_at else None,
            "evidence": self.evidence,
            "model_assist": "not_enabled",
        }


def _sentence_spans(text: str):
    from app.services.time_domain.followup import _SENTENCE

    return [m for m in _SENTENCE.finditer(text or "") if m.group(0).strip()]


def _dated(rows: list[dict], reference: datetime) -> list[tuple[datetime, dict]]:
    out, seen = [], set()
    for row in rows:
        stamp = parse_time(row.get("created_at"))
        if (stamp is None or stamp > reference or row.get("role") not in {"user", "assistant"}
                or row.get("id") in seen):
            continue
        seen.add(row.get("id"))
        out.append((stamp, row))
    out.sort(key=lambda item: (item[0], str(item[1].get("id"))))
    return out


def _oldest(rows: list[dict] | None) -> datetime:
    stamps = [parse_time(r.get("created_at")) for r in rows or []]
    return min((s for s in stamps if s is not None), default=datetime.max.replace(tzinfo=timezone.utc))


def resolve_target(question: str, history: list[dict], *, now: datetime,
                   current_user_message: str = "") -> EventRef | None:
    """The event an optional question follows up on, from recent user turns."""
    from app.services.time_domain.followup import _is_question

    reference = parse_time(now)
    if reference is None or not _is_question(question):
        return None
    users = [str(r.get("content") or "") for _, r in _dated(history, reference) if r.get("role") == "user"]
    recent = ([current_user_message] if current_user_message else []) + users[::-1][:2]
    return outcome_event(question, recent)


def decide_followup(
    question: str,
    history: list[dict],
    *,
    now: datetime,
    current_user_message: str = "",
    history_truncated: bool = False,
    ledger: list[dict] | None = None,
    ledger_truncated: bool = False,
) -> FollowupDecision:
    """Decide one generated optional question against bounded, dated history.

    ``history`` is the dense recent page (HISTORY_DAYS/HISTORY_ROWS); ``ledger``
    adds older rows naming the same event family (LEDGER_DAYS/LEDGER_ROWS).
    Required confirmations and crisis care never reach this function.
    """
    from app.services.time_domain.followup import _is_question

    reference = parse_time(now)
    if reference is None:
        return FollowupDecision("suppress", "clock_unavailable")
    if not _is_question(question):
        return FollowupDecision("allow", "not_a_question")
    target = resolve_target(question, history, now=reference, current_user_message=current_user_message)
    if target is None:
        return FollowupDecision("allow", "not_an_outcome_followup")
    rows = _dated(list(history) + list(ledger or []), reference)
    evidence = {"event_family": target.family, "event_resolved_from": target.resolved_from,
                "history_rows": len(rows), "history_truncated": history_truncated or ledger_truncated}

    def names_target(text: str) -> bool:
        ref = named_event(text)
        return ref is not None and same_event(ref, target)

    # The user is talking about this event right now: questions are part of
    # the conversation, not a follow-up.
    if names_target(current_user_message):
        return FollowupDecision("allow", "user_discussing_event", evidence=evidence)
    mentions = [stamp for stamp, r in rows if r.get("role") == "user"
                and names_target(str(r.get("content") or ""))]
    # Outcome asks about this event. Detail questions ("几点的面试呀") are
    # not outcome asks and never count.
    asks: list[tuple[datetime, dict]] = []
    recent_users: list[str] = []
    for stamp, row in rows:
        text = str(row.get("content") or "")
        if row.get("role") == "user":
            recent_users.insert(0, text)
            del recent_users[2:]
            continue
        bubble = named_event(text)
        for part in _sentence_spans(text):
            clause = part.group(0)
            if not _is_question(clause):
                continue
            ref = outcome_event(clause, recent_users)
            if (ref is None and bubble and _EVENT_RESULT.search(clause)
                    and not _PRE_EVENT.search(clause)):
                # "体检累不累？报告什么时候出？": the same bubble names the event.
                ref = bubble
            if ref is not None and same_event(ref, target):
                asks.append((stamp, row))
                break
    if (mentions and reference - mentions[-1] <= SESSION_WINDOW
            and (not asks or mentions[-1] > asks[-1][0])):
        # The user brought the event up again after the last ask: live talk.
        return FollowupDecision("allow", "user_discussing_event", evidence=evidence)
    if not asks:
        if ledger_truncated and _oldest(ledger) > reference - timedelta(days=HISTORY_DAYS):
            # A capped ledger that only reaches back a few days cannot prove the
            # event was never asked about; one reaching past a week can.
            return FollowupDecision("suppress", "history_truncated", None, evidence)
        return FollowupDecision("allow", "first_ask", evidence=evidence)
    last_ask_at = asks[-1][0]
    evidence.update(last_ask_at=last_ask_at.isoformat(), same_event_asks=len(asks))
    # A response: a user message naming the event after the ask, or else the
    # user's first message within the session window after it.
    response = None
    for stamp, row in rows:
        if stamp <= last_ask_at or row.get("role") != "user":
            continue
        if names_target(str(row.get("content") or "")):
            response = (stamp, row, True)
            break
        if response is None and stamp - last_ask_at <= SESSION_WINDOW:
            response = (stamp, row, False)
    if response is not None:
        answered_at, answer, named = response
        evidence["answered_at"] = answered_at.isoformat()
        text = str(answer.get("content") or "")
        if named and _RESOLVED.search(text) and not _PENDING.search(text) and not _FUTURE.search(text):
            return FollowupDecision("suppress", "event_resolved_no_new_info", evidence=evidence)
        eligible = answered_at + ANSWERED_COOLDOWN
        if reference < eligible:
            return FollowupDecision("suppress", "answered_recently", eligible, evidence)
        return FollowupDecision("allow", "answered_cooldown_elapsed", evidence=evidence)
    # Unanswered. Without the user's own recent mention, stay quiet: asks aging
    # out of a bounded read must never re-arm a question every few days.
    recent_mention = [m for m in mentions if reference - m <= timedelta(days=HISTORY_DAYS)]
    since = recent_mention[-1] if recent_mention else None
    unanswered = [a for a in asks if since is None or a[0] > since]
    evidence["unanswered_asks"] = len(unanswered)
    if since is None or len(unanswered) >= MAX_UNANSWERED:
        return FollowupDecision("suppress", "silent_until_new_evidence", None, evidence)
    eligible = last_ask_at + UNANSWERED_COOLDOWN
    if reference < eligible:
        return FollowupDecision("suppress", "unanswered_cooldown", eligible, evidence)
    if history_truncated or ledger_truncated:
        return FollowupDecision("suppress", "history_truncated", None, evidence)
    return FollowupDecision("allow", "unanswered_cooldown_elapsed", evidence=evidence)


def may_be_event_followup(text: str) -> bool:
    """Cheap prefilter: only these replies pay for the bounded history reads."""
    from app.services.time_domain.followup import _is_question

    return any(_is_candidate(m.group(0)) for m in _sentence_spans(text))


def _is_candidate(clause: str) -> bool:
    """Could be an outcome follow-up: a named event, an explicit reference, or
    后来 + a strong result word. 今天过得怎么样 is never one."""
    from app.services.time_domain.followup import _is_question

    return bool(_is_question(clause) and _OUTCOME.search(clause) and not _PRE_EVENT.search(clause) and (
        event_family(clause) or _REFERENTIAL.search(clause)
        or (_LATER.search(clause) and _STRONG_OUTCOME.search(clause))))


_RECENT_SQL = """SELECT m.id,m.role,left(m.content,500) AS content,m.created_at
    FROM messages m JOIN conversations c ON c.id=m.conversation_id
    WHERE c.user_id=$1 AND c.workspace_id=$2 AND NOT c.is_deleted
      AND m.role IN ('user','assistant')
      AND m.created_at > ($3::timestamptz AT TIME ZONE 'UTC') - make_interval(days => $4::int)
      AND m.created_at <= ($3::timestamptz AT TIME ZONE 'UTC')
      AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
      AND NOT COALESCE(m.metadata,'{{}}'::jsonb) ? 'component_card'
      {extra}
    ORDER BY m.created_at DESC,m.id DESC LIMIT $5"""


async def load_followup_history(database, *, user_id: str, workspace_id: str,
                                now: datetime) -> tuple[list[dict], bool]:
    """Last HISTORY_DAYS of this workspace's dialogue, newest HISTORY_ROWS kept.

    Human takeover and component cards are not the agent asking or the user
    answering. Content is clipped; only short clauses are ever compared.
    """
    rows = await database.query_raw(
        _RECENT_SQL.format(extra=""), user_id, workspace_id, now.isoformat(),
        HISTORY_DAYS, HISTORY_ROWS + 1)
    return list(reversed(rows[:HISTORY_ROWS])), len(rows) > HISTORY_ROWS


async def load_family_ledger(database, *, user_id: str, workspace_id: str, now: datetime,
                             family: str) -> tuple[list[dict], bool]:
    """Older rows naming one event family, newest LEDGER_ROWS kept."""
    pattern = dict(_FAMILIES)[family]
    rows = await database.query_raw(
        _RECENT_SQL.format(extra="AND left(m.content,500) ~ $6"), user_id, workspace_id, now.isoformat(),
        LEDGER_DAYS, LEDGER_ROWS + 1, pattern)
    return list(reversed(rows[:LEDGER_ROWS])), len(rows) > LEDGER_ROWS


async def decide_reply(text: str, *, database, user_id: str, workspace_id: str, now: datetime,
                       current_user_message: str = "", history=None,
                       ) -> tuple[str, list[FollowupDecision], dict]:
    """Remove suppressed outcome follow-up clauses from one bubble.

    The bubble is returned byte-for-byte unchanged when nothing is suppressed;
    otherwise only the removed clauses are cut, keeping separators and layout.
    ``history`` caches the dense read across bubbles of one reply.
    """
    from app.services.time_domain.followup import _is_question

    if history is None:
        history = {}
    cuts, decisions = [], []
    for match in _sentence_spans(text):
        clause = match.group(0)
        if not _is_candidate(clause):
            continue
        if "recent" not in history:
            history["recent"] = await load_followup_history(
                database, user_id=user_id, workspace_id=workspace_id, now=now)
        recent, truncated = history["recent"]
        target = resolve_target(clause, recent, now=now, current_user_message=current_user_message)
        ledger, ledger_truncated = [], False
        if target is not None:
            key = "ledger:" + target.family
            if key not in history:
                history[key] = await load_family_ledger(
                    database, user_id=user_id, workspace_id=workspace_id, now=now,
                    family=target.family)
            ledger, ledger_truncated = history[key]
        decision = decide_followup(clause, recent, now=now, current_user_message=current_user_message,
                                   history_truncated=truncated, ledger=ledger,
                                   ledger_truncated=ledger_truncated)
        decisions.append(decision)
        if not decision.allowed:
            cuts.append(match.span())
    if not cuts:
        return text, decisions, history
    kept, start = [], 0
    for begin, end in cuts:
        kept.append(text[start:begin])
        start = end
    kept.append(text[start:])
    return "".join(kept).strip().rstrip("，,、"), decisions, history


def summarize(decisions: list[FollowupDecision], *, status: str, **extra) -> dict:
    """Bounded Trace payload. ``status`` separates not-run, out of scope,
    unavailable and aborted from a real check that found nothing."""
    suppressed = [d for d in decisions if not d.allowed]
    primary = suppressed[0] if suppressed else (decisions[0] if decisions else None)
    return {
        "version": 1,
        "policy_version": POLICY_VERSION,
        "status": status,
        "checked_count": len(decisions),
        "suppressed_count": len(suppressed),
        "decision": primary.diagnostic() if primary else None,
        **extra,
    }


def _strip_candidates(text: str) -> str:
    from app.services.time_domain.followup import _is_question

    cuts = [m.span() for m in _sentence_spans(text) if _is_candidate(m.group(0))]
    kept, start = [], 0
    for begin, end in cuts:
        kept.append(text[start:begin])
        start = end
    kept.append(text[start:])
    return "".join(kept).strip().rstrip("，,、")


async def guard_reply_followups(replies: list[str], *, user_id: str, workspace_id: str | None,
                                now: datetime, current_user_message: str,
                                fallback: str = "好，我听着。", database=None,
                                ) -> tuple[list[str], dict]:
    """Ordinary-reply entry. Required confirmations and crisis care never call it.

    Without a scope or history, only the optional outcome follow-up clauses are
    dropped; the rest of the reply is still sent.
    """
    if not any(may_be_event_followup(reply) for reply in replies):
        return replies, summarize([], status="not_requested")
    if not workspace_id:
        status = "scope_unavailable"
    else:
        from app.db import db

        database = database if database is not None else db

        async def run():
            out, decisions, cache = [], [], {}
            for reply in replies:
                text, found, cache = await decide_reply(
                    reply, database=database, user_id=user_id, workspace_id=workspace_id,
                    now=now, current_user_message=current_user_message, history=cache)
                decisions.extend(found)
                if text:
                    out.append(text)
            recent = cache.get("recent", ([], False))
            return out or [fallback], summarize(
                decisions, status="checked", history_rows=len(recent[0]),
                history_truncated=recent[1])

        try:
            # On the reply hot path: a slow read costs only the optional question.
            return await asyncio.wait_for(run(), timeout=REPLY_READ_TIMEOUT_S)
        except Exception as exc:  # includes TimeoutError; CancelledError propagates
            logger.warning("[FOLLOWUP-DECISION] history unavailable error=%s", type(exc).__name__)
            status = "unavailable"
    stripped = sum(1 for r in replies for m in _sentence_spans(r) if _is_candidate(m.group(0)))
    kept = [t for t in (_strip_candidates(r) if may_be_event_followup(r) else r for r in replies) if t]
    return kept or [fallback], summarize([], status=status, suppressed_count=stripped)


async def checked_proactive_insert(database, *, conversation_id: str, user_id: str,
                                   workspace_id: str, message: str, metadata: dict,
                                   since: datetime | None, now: datetime,
                                   ) -> tuple[str | None, datetime | None, dict]:
    """Final send eligibility for an optional proactive outcome follow-up.

    Decision and insert share one transaction under a per-workspace lock, so two
    workers cannot both pass: the second sees the first's committed question.
    A newer user message, an active human takeover or a suppressed decision all
    abort without writing. The decision is stored in the message's Trace
    diagnostics. Returns (id, created_at, diagnostics).
    """

    def naive(ts: datetime) -> str:
        return ts.astimezone(timezone.utc).replace(tzinfo=None).isoformat()

    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended('followup-send:'||$1,0))",
            workspace_id)
        # Serialize with an admin claiming the conversation, as other writers do.
        takeovers = await tx.query_raw(
            "SELECT status FROM human_takeovers WHERE user_id=$1 ORDER BY id FOR SHARE", user_id)
        if any(row["status"] == "active" for row in takeovers):
            return None, None, summarize([], status="aborted", aborted="human_takeover")
        # Read the clock after the lock: a competing worker's question was
        # committed before we got here and must be inside our history read.
        now = max(now, datetime.now(timezone.utc))
        _, decisions, cache = await decide_reply(
            message, database=tx, user_id=user_id, workspace_id=workspace_id, now=now)
        recent = cache.get("recent", ([], False))
        diagnostics = summarize(decisions, status="checked", history_rows=len(recent[0]),
                                history_truncated=recent[1])
        if any(not d.allowed for d in decisions):
            return None, None, diagnostics
        stored = dict(metadata)
        response = dict(stored.get("response_diagnostics") or {})
        response["temporal_context"] = {**(response.get("temporal_context") or {}),
                                        "followup_decision": diagnostics}
        stored["response_diagnostics"] = response
        rows = await tx.query_raw(
            """INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
            SELECT $1, $2, 'assistant', $3, $4::jsonb, $5::timestamp
            WHERE $6::timestamp IS NULL OR NOT EXISTS (
                SELECT 1 FROM messages u WHERE u.conversation_id=$2 AND u.role='user'
                  AND u.created_at > $6::timestamp)
            RETURNING id""",
            str(uuid4()), conversation_id, message,
            json.dumps(stored, ensure_ascii=False, default=str),
            naive(now), naive(since) if since else None,
        )
        if not rows:
            return None, None, {**diagnostics, "status": "aborted", "aborted": "user_replied"}
        return str(rows[0]["id"]), now, diagnostics
