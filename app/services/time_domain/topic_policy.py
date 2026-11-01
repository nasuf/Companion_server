"""Source-local topic preference commands, independent of persistence and models.

A conversational pause is not evidence that an event happened or ended. Only
explicit speaker requests create preferences. Deictic or ambiguous targets pause
this session; they never invent a permanent subject or a global preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
import unicodedata
from zoneinfo import ZoneInfo

from app.services.time_domain.followup import (
    _direct_request,
    _DENY_PERMISSION,
    _RESUME as _SESSION_RESUME,
)
from app.services.time_domain.limits import SESSION_SECONDS
from app.services.time_domain.timestamps import parse_time

POLICY_VERSION = "topic_policy_v1"
MAX_COMMANDS = 8
MAX_SOURCE_IDS = 256
MAX_LABEL = 64
MAX_TOPICS = 128

_CONDITIONAL = re.compile(
    r"(?:让|叫|要求|嘱咐|希望)(?:我|他|她)|假如|假设|如果|要是|比如|举例|开玩笑|不是让你|并非让你|没(?:有)?(?:说|让)|不(?:是|要)不|不希望你不|为什么|什么意思|怎么理解|翻译"
)
_STOP = re.compile(
    r"^(?P<time>暂时|现在|先|这会儿|今天|今晚|以后|今后|往后|永远|从今以后|再也)?(?:我)?(?:都)?(?:请你|你)?(?:不要|别|不用|不许|不想)(?:再|一直|老)?(?:追问|问(?:我)?|提起|提|聊|讨论)(?P<topic>[^。！？!?；;，,\n]{0,80})$"
)
_TOPIC_FIRST = re.compile(
    r"^(?:关于)?(?P<topic>[^。！？!?；;，,\n]{1,64}?)(?:这个话题|这件事)?(?P<time>暂时|现在|先|今天|今晚|以后|今后|往后|永远)?(?:请你|你)?(?:不要|别|不用|不许)(?:再|一直|老)?(?:追问|问(?:我)?|提起|提|聊|讨论)(?:了|吧|啦|呀|哦|呢)*$"
)
_RESUME = re.compile(
    r"^(?:现在|今天|这会儿|那|好|好了|以后)?(?:你|我们|我想)?(?:可以(?:继续)?问(?:我)?|继续聊|聊聊|继续讨论|可以聊|可以提)(?P<topic>[^。！？!?；;，,\n]{0,80})$"
)
_GLOBAL = re.compile(
    r"^(?:我)?(?:一直|向来|本来)?(?:不喜欢|讨厌|不爱)(?:你)?(?:一直|老是|老|总是|不停)?(?:被(?:人|你)?问|被追问|问我|追问我)(?:问题|了|的感觉)*$"
)
_GLOBAL_TARGET = {
    "任何问题",
    "问题",
    "什么",
    "任何事",
    "所有事",
    "所有问题",
    "任何事情",
}
_DEICTIC = {
    "",
    "这个",
    "这件事",
    "那个",
    "那件事",
    "这些",
    "这事",
    "这个话题",
    "这方面",
    "它",
    "我",
    "这",
    "那",
}
_SUFFIX = re.compile(
    r"(?:这个话题|这件事|这事|的事|方面|的事情)?(?:了|吧|啦|呀|啊|哦|呢|好不好|行不行)*$"
)
_DURABLE = {"以后", "今后", "往后", "永远", "从今以后", "再也"}


@dataclass(frozen=True)
class TopicCommand:
    scope: str
    topic: str
    action: str
    expires_at: datetime | None
    source_quote: str
    reason: str
    ordinal: int

    @property
    def key(self) -> str:
        return (
            self.scope
            + ":"
            + (topic_key(self.topic) if self.scope == "topic" else "all")
        )


def topic_key(label: str) -> str:
    text = unicodedata.normalize("NFKC", label).lower()
    return re.sub(r"[\W_]+", "", text)


def topic_id(workspace_id: str, generation: str, key: str) -> str:
    return hashlib.sha256(
        (workspace_id + "\0" + generation + "\0" + key).encode()
    ).hexdigest()


def _label(value: str) -> str:
    text = value.strip()
    while text and (unicodedata.category(text[-1])[0] in "PSM" or text[-1] == "\u200d"):
        text = text[:-1].rstrip()
    return _SUFFIX.sub("", text).strip().strip('“”「」‘’"').strip()


def _preference_clauses(text: str) -> list[str]:
    """Keep quoted noun labels, but never turn quoted instructions into requests.

    The original spelling is restored before evidence is recorded. Only a
    short noun immediately following a direct verb is eligible, so an entire
    quoted request (or a quoted negation) still follows the shared quote guard.
    """
    labels: dict[str, str] = {}
    quote = re.compile(r'“[^”]*”|「[^」]*」|‘[^’]*’|"[^"\n]*"')

    def protect(match):
        label = match[0][1:-1]
        prefix = text[: match.start()]
        if (
            re.search(r"(?:问我|追问|问|提起|提|聊聊|聊|讨论)$", prefix)
            and 1 <= len(label) <= MAX_LABEL
            and re.fullmatch(r"[\w\s·-]+", label)
            and not re.search(r"别|不要|不许|可以|不能|不想|如果|假设", label)
        ):
            token = f"TOPICQUOTEDLABEL{len(labels)}TOKEN"
            if token in text:
                return match[0]
            labels[token] = match[0]
            return token
        return match[0]

    clauses = _direct_request(quote.sub(protect, text)).split("。")
    for token, value in labels.items():
        clauses = [clause.replace(token, value) for clause in clauses]
    return clauses


def parse_commands(text: str, *, statement_at: datetime) -> tuple[TopicCommand, ...]:
    anchor = parse_time(statement_at)
    if anchor is None or len(text) > 12000:
        return ()
    commands = []
    for ordinal, clause in enumerate(_preference_clauses(text)):
        clause = clause.strip()
        if not clause or _CONDITIONAL.search(clause) or len(commands) >= MAX_COMMANDS:
            continue
        original_clause = clause
        explicit_expiry = None
        expiry_uncertain = False
        until = re.fullmatch(
            r"(?:在|到)?(.{1,32}?)(之前|以前|前|为止)(?:先)?((?:不要|别|不用|不许).+)",
            clause,
        )
        if until:
            from app.services.time_domain.query import resolve_temporal_query
            from app.services.time_domain.timestamps import exclusive_end

            parsed = resolve_temporal_query(
                until[1], anchor=anchor, anchor_source="persisted_message"
            )
            if parsed.status == "resolved" and len(parsed.expressions) == 1:
                expression = parsed.expressions[0]
                explicit_expiry = (
                    exclusive_end(expression.end, expression.end_boundary)
                    if until[2] == "为止"
                    else expression.start
                )
                expiry_uncertain = explicit_expiry <= anchor
            else:
                expiry_uncertain = True
            clause = until[3]
        matched = _STOP.fullmatch(clause) or _TOPIC_FIRST.fullmatch(clause)
        resume = _RESUME.fullmatch(clause)
        global_stop = bool(_GLOBAL.fullmatch(clause))
        if not matched and not resume and not global_stop:
            permission = (
                False
                if _DENY_PERMISSION.fullmatch(clause)
                else True if _SESSION_RESUME.fullmatch(clause) else None
            )
            if permission is None:
                continue
            commands.append(
                TopicCommand(
                    "session",
                    "",
                    "resume" if permission else "pause",
                    anchor + timedelta(seconds=SESSION_SECONDS),
                    clause,
                    "explicit_session_request",
                    ordinal,
                )
            )
            continue
        target = _label((matched or resume).group("topic")) if not global_stop else ""
        marker = matched.group("time") if matched else ""
        if global_stop or (target in _GLOBAL_TARGET and (marker in _DURABLE or resume)):
            scope, target, reason = "global", "", "explicit_general_preference"
        elif target in _DEICTIC:
            scope, target, reason = "session", "", "ambiguous_target_session_only"
        elif len(target) > MAX_LABEL or re.search(
            r"(?:是不是|是否|会不会|能不能|可不可以|不想|不要|你别|但|不过)", target
        ):
            scope, target, reason = "session", "", "ambiguous_target_session_only"
        else:
            scope, reason = "topic", "explicit_named_topic"
        action = (
            "resume"
            if resume
            else (
                "stop"
                if global_stop or (marker in _DURABLE and scope != "session")
                else "pause"
            )
        )
        expiry = None
        if action == "pause" or scope == "session":
            expiry = anchor + timedelta(seconds=SESSION_SECONDS)
            if marker in {"今天", "今晚"}:
                from app.config import settings

                local = anchor.astimezone(ZoneInfo(settings.schedule_timezone))
                expiry = (
                    (local + timedelta(days=1))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .astimezone(timezone.utc)
                )
        if until:
            action = "pause"
            if expiry_uncertain:
                scope, target, expiry = (
                    "session",
                    "",
                    anchor + timedelta(seconds=SESSION_SECONDS),
                )
                reason = "ambiguous_expiry_session_only"
            else:
                expiry = explicit_expiry
                reason = "explicit_expiry"
        commands.append(
            TopicCommand(
                scope, target, action, expiry, original_clause, reason, ordinal
            )
        )
    return tuple(commands)


def topic_matches(label: str, text: str) -> bool:
    """Literal topic identity only; semantic event/question matching is TIME06.

    Never use loose edit distance: identical surrounding words do not make two
    interviews or two people the same event. Pronoun-only clauses are resolved
    by the caller's concrete topic context, not guessed here.
    """
    key = topic_key(label)
    value = topic_key(text)
    if key and key in value:
        return True
    # Presentation suffixes do not establish a different event identity.
    # Do not strip people, dates or substantive qualifiers (纠纷/亏损/etc.).
    core = re.sub(r"(?:相关情况|相关安排|的安排|的情况|安排|情况|进展)$", "", key)
    return len(core) >= 2 and core != key and core in value


@dataclass(frozen=True)
class TopicRestriction:
    id: str
    topic: str
    scope: str
    action: str
    expires_at: datetime | None
    source_id: str | None
    version: int
    state: str = "paused"
    last_user_mention: datetime | None = None
    last_agent_question: datetime | None = None
    answered_at: datetime | None = None
    asked_count: int = 0

    def active(self, now: datetime) -> bool:
        return self.action in {"pause", "stop"} and (
            self.expires_at is None or now < self.expires_at
        )

    def diagnostic(self) -> dict:
        def iso(value):
            return value.isoformat() if value is not None else None

        return {
            "id": self.id,
            "scope": self.scope,
            "topic": self.topic,
            "state": self.state,
            "action": self.action,
            "version": self.version,
            "expires_at": iso(self.expires_at),
            "source_message_id": self.source_id,
            "last_user_mention": iso(self.last_user_mention),
            "last_agent_question": iso(self.last_agent_question),
            "answered_at": iso(self.answered_at),
            "asked_count": self.asked_count,
        }
