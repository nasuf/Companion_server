"""Conversation-local optional follow-up policy. Safety/confirmation is exempt.

No topic is inferred to be finished merely because time elapsed. Explicit
requests to stop questions expire after a session, or an explicit invitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import re
import unicodedata

from app.services.time_domain.limits import HISTORY_LIMIT, SESSION_SECONDS
from app.services.time_domain.timestamps import parse_time

_STOP = re.compile(r"(?:别|不要|不用|不想|停止|不许|别再|不要再).{0,8}(?:追问|问我|一直问|再问|问了)|(?:能不能|可不可以).{0,5}(?:别|不要).{0,6}问|不要一直问|别一直问|别老问")
# Restoring permission requires a complete, affirmative request. A substring
# such as 允许你问 in 不允许你问, or 可以问 in a conditional, is not consent.
_REQUEST_LEAD = r"(?:(?:好的|好了|好吧|好|行|嗯|那|现在|这会儿|其实)\s*)*"
_REQUEST_END = r"(?:了|吧|呀|啊|啦|呢|哦|~|～|…|\s)*"
_RESUME = re.compile(
    _REQUEST_LEAD
    + r"(?:(?:你(?:现在|这会儿)?可以|你继续|可以继续|允许你)(?:再|继续)?问(?:我)?"
    r"|问我吧|你想问什么)" + _REQUEST_END
)
_DENY_PERMISSION = re.compile(
    _REQUEST_LEAD
    + r"(?:(?:我)?不允许你|(?:你(?:现在|这会儿)?)?(?:还)?(?:不可以|不能|不许))"
    r"(?:再|继续)?问(?:我)?" + _REQUEST_END
)
_QUESTION = re.compile(
    r"[?？]|(?:吗|怎么样|好不好)[呀啊嘛呢哇~～…\s]*$"
    r"|(?:你|您).*(?:呢|没有|咋样|干嘛|做什么|忙什么|忙啥|在哪|去哪|吃什么|吃了什么)[呀啊哇~～…\s]*$"
    # Short colloquial questions often omit both a pronoun and question mark.
    # Match complete interrogative clauses, not statements such as 什么都行
    # or 有啥事都可以说. This remains a conservative lexical guard.
    r"|^(?:(?:那|所以|然后)\s*)?(?:你|您)?"
    r"(?:(?:今天|最近|这会儿|现在|刚才|这两天)\s*)?(?:是有|有|是)?(?:啥|什么)"
    r"(?:(?:(?:开心|有趣|特别|新鲜|好|烦心)的?)?事(?:情|儿)?|好消息|情况|打算|安排|想法|原因|意思)"
    r"[呀啊呢嘛哇不~～…\s]*$"
    # Standalone colloquial questions can omit the subject and question mark.
    # Anchor the whole clause so statements such as 忙什么都可以 stay intact.
    r"|^(?:(?:那|所以|然后)\s*)?(?:(?:今天|昨天|最近|刚才|现在|这会儿)\s*)?"
    r"(?:是)?(?:忙(?:啥|什么)|干(?:啥|什么)|做(?:啥|什么)|吃(?:啥|什么)|去哪|怎么了|咋了|进展怎么样)"
    r"(?:了|啦)?[呀啊呢嘛哇~～…\s]*$"
    # Short care questions remain interrogative without 吗 or a question mark.
    r"|^(?:(?:那|所以|然后|今天|昨天|最近|刚才|现在|这会儿)\s*)?"
    r"(?:(?:是)?忙(?:啥|什么)(?:累到|累着|忙成这样)(?:了|啦)?|怎么(?:了|啦)|咋(?:了|啦)|(?:怎么|咋)这么(?:累|困|难过|烦|忙))"
    r"[呀啊呢嘛哇~～…\s]*$"
    # Modal + sentence-final negation is another colloquial yes/no form.
    # Exclude declarative endings such as 勇敢说不 and conditional 或不/就不.
    r"|^(?:(?:那|所以|然后)\s*)?(?:你|您)?"
    r"(?:(?:今天|最近|这会儿|现在|刚才|这两天)\s*)?(?:你|您)?"
    r"(?:可以|能|要|想|会|方便)[^。！？!?；;，,\n]{0,24}(?<!说)(?<!或)(?<!就)(?:不|没)$"
    # Affirmative-negative questions need no final 吗 or question mark.
    # Keep reported/conditional statements outside this complete-clause form.
    r"|^(?:(?:那|所以|然后)\s*)?(?:你|您)?"
    r"(?:(?:今天|现在|接下来|这会儿)\s*)?(?:你|您)?"
    r"(?:是不是|能不能|可不可以|会不会|要不要|想不想)"
    r"(?![^。！？!?；;，,\n]*(?:都可以|都行|都没关系|都无所谓)[呀啊啦呢~～…\s]*$)"
    r"[^。！？!?；;，,\n]{1,24}$"
)
_SENTENCE = re.compile(r"[^。！？!?；;，,\n|]+[。！？!?；;，,]*")
_NON_WORD = re.compile(r"[\W_]+")


def _is_question(text: str) -> bool:
    # Informal generated replies may end an interrogative clause with a full
    # stop. Keep question marks, but ignore whitespace/declarative terminators.
    if "?" in text or "？" in text:
        return True
    cleaned = text.strip()
    while cleaned and (unicodedata.category(cleaned[-1])[0] in "PSM" or cleaned[-1] == "\u200d" or cleaned[-1].isspace()):
        cleaned = cleaned[:-1]
    while cleaned and (unicodedata.category(cleaned[0])[0] in "PSM" or cleaned[0] == "\u200d" or cleaned[0].isspace()):
        cleaned = cleaned[1:]
    return bool(_QUESTION.search(cleaned))


def _key(text: str) -> str:
    return _NON_WORD.sub("", text).strip("嗯啊呀啦呢哦嘛吧哇")


def similar_question(left: str, right: str) -> bool:
    a, b = _key(left), _key(right)
    return min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.72


@dataclass(frozen=True)
class FollowupPolicy:
    optional_questions_allowed: bool = True
    reason: str = "no_recent_opt_out"
    recent_questions: tuple[str, ...] = ()
    topic_snapshot: object | None = None
    topic_context: str = ""

    def prompt_value(self) -> str:
        import json
        restrictions = getattr(self.topic_snapshot, "items", ())
        status = getattr(self.topic_snapshot, "status", "not_requested")
        topic_value = ""
        if restrictions or status in {"unavailable", "scope_unavailable", "capacity_exceeded", "source_unavailable"}:
            values = [{"scope": item.scope, "topic": item.topic, "action": item.action}
                      for item in restrictions[:20]]
            payload = {"status": status, "restrictions": values}
            rendered = json.dumps(payload, ensure_ascii=False)
            # Bound added prompt bytes, never truncate a label or JSON token.
            # The output guard still checks every stored restriction. This
            # affects crowded ledgers only; ordinary/no-policy inputs stay exact.
            while len(rendered.encode("utf-8")) > 256 and values:
                values.pop()
                payload["omitted_restriction_count"] = len(restrictions) - len(values)
                rendered = json.dumps(payload, ensure_ascii=False)
            topic_value = "; topic_preferences=" + rendered
        return ("optional_questions_allowed=" + str(self.optional_questions_allowed).lower()
                + f"; reason={self.reason}; recent_questions="
                + " / ".join(self.recent_questions) + topic_value)

    def diagnostics(self) -> dict:
        result = {"optional_questions_allowed": self.optional_questions_allowed,
                  "followup_reason": self.reason,
                  "recent_question_count": len(self.recent_questions)}
        if self.topic_snapshot is not None:
            result["topic_preferences"] = self.topic_snapshot.diagnostics()
        return result


def _direct_request(text: str) -> str:
    # Quoted/reported instructions are data, not a request directed at the agent.
    text = re.sub(r"""“[^”]*”|「[^」]*」|‘[^’]*’|"[^"]*"|'[^']*'""", "", text)
    clauses = re.split(r"[。！？!?；;，,\n]", text)
    return "。".join(clause for clause in clauses if not re.search(
        r"(?:他说|她说|同事说|朋友说|老板说|别人说|我没说|我没有说|不是让你|并不是让你|并非让你)", clause))


def _requested_permission(text: str) -> bool | None:
    """Apply explicit clauses in order; ambiguity never grants permission."""
    decision = None
    for clause in _direct_request(text).split("。"):
        clause = clause.strip()
        # Emoji and presentation selectors after an explicit permission do not
        # change its meaning. Do not trim words (notably negations/questions).
        end = len(clause)
        while end and (unicodedata.category(clause[end - 1])[0] in "PSM"
                       or clause[end - 1] == "\u200d" or clause[end - 1].isspace()):
            end -= 1
        clause = clause[:end]
        if _STOP.search(clause) or _DENY_PERMISSION.fullmatch(clause):
            decision = False
        elif _RESUME.fullmatch(clause):
            decision = True
    return decision


def derive_followup_policy(messages: list[dict], *, now: datetime,
                           current_message: str = "") -> FollowupPolicy:
    from app.services.time_domain.topic_policy import parse_commands

    def permission(text):
        commands = parse_commands(text, statement_at=now)
        if commands and all(command.scope == "topic" for command in commands):
            return None
        return _requested_permission(text)

    reference = parse_time(now)
    allowed = True
    reason = "no_recent_opt_out"
    questions = []
    recent = []
    for msg in messages[-HISTORY_LIMIT:]:
        ts = parse_time(msg.get("createdAt"))
        if reference is None or ts is None or not 0 <= (reference - ts).total_seconds() <= SESSION_SECONDS:
            continue
        recent.append((ts, msg))
    # Retrieval order must not let an old invitation undo a newer opt-out.
    # Stable sorting preserves the supplied order for equal timestamps; never
    # guess an ordering from opaque message IDs, nor reorder caller-owned data.
    for _, msg in sorted(recent, key=lambda row: row[0]):
        text = str(msg.get("content") or "")
        if msg.get("role") == "user":
            decision = permission(text)
            if decision is not None:
                allowed = decision
                reason = "user_reopened" if decision else "user_opted_out"
        elif msg.get("role") == "assistant":
            questions.extend(s.strip() for s in _SENTENCE.findall(text) if _is_question(s))
    decision = permission(current_message)
    if decision is not None:
        allowed = decision
        reason = "user_reopened" if decision else "user_opted_out"
    return FollowupPolicy(allowed, reason, tuple(q[:100] for q in questions[-6:]),
                          topic_context=(" ".join(str(msg.get("content") or "") for _, msg in sorted(recent, key=lambda row: row[0])[-2:]) + " " + current_message)[-1000:])


def guard_optional_questions(replies: list[str], policy: FollowupPolicy, *, fallback: str | None = "好，我听着。") -> tuple[list[str], int]:
    """Respect explicit opt-out after generation; call only on ordinary replies.

    Similarity alone never deletes a reply. Required confirmations, crisis care,
    and fallback errors must bypass this guard at the call site.
    """
    snapshot = policy.topic_snapshot
    restrictions = getattr(snapshot, "items", ())
    unavailable = getattr(snapshot, "status", "") in {"unavailable", "scope_unavailable", "capacity_exceeded", "source_unavailable"}
    if unavailable:
        # Unknown/over-capacity storage cannot authorize an unobserved topic,
        # including declarative rementions. Required replies bypass this guard.
        return ([fallback] if fallback else []), len(replies)
    if policy.optional_questions_allowed and not restrictions and not unavailable:
        return replies, 0
    from app.services.time_domain.topic_policy import topic_matches
    now = getattr(snapshot, "sampled_at", None)
    restrictions = tuple(item for item in restrictions if now is not None and item.active(now))
    global_stop = unavailable or not policy.optional_questions_allowed or any(item.scope in {"session", "global"} for item in restrictions)
    kept, removed = [], 0
    for reply in replies:
        parts = _SENTENCE.findall(reply)
        accepted = []
        for part in parts:
            named = any(item.scope == "topic" and topic_matches(item.topic, part) for item in restrictions)
            contextual = any(item.scope == "topic" and topic_matches(item.topic, policy.topic_context) for item in restrictions)
            # A generated question that points back to a prior issue while
            # repeating a substantial part of a restricted label is ambiguous.
            # Suppress that optional question, without merging topic identities
            # or treating the partial reference as new evidence about the event.
            partial_reference = False
            if _is_question(part) and re.search(r"之前|上次|原来|那件|这件|后来", part):
                for item in restrictions:
                    if item.scope != "topic":
                        continue
                    label, clause = _key(item.topic), _key(part)
                    overlap = SequenceMatcher(None, label, clause).find_longest_match().size
                    if len(label) >= 4 and overlap >= max(2, (len(label) + 1) // 2):
                        partial_reference = True
                        break
            if (global_stop and _is_question(part)) or named or partial_reference or (contextual and re.search(r"(?:那件事|这件事|那个|这个|后来|结果|进展|后续|怎么样|如何)", part) and _is_question(part)):
                removed += 1
            else:
                accepted.append(part)
        text = "".join(accepted).strip().rstrip("，,")
        if text:
            kept.append(text)
    return (kept or ([fallback] if fallback else [])), removed


def repeats_recent_question(text: str, policy: FollowupPolicy) -> bool:
    return any(similar_question(part, old)
               for part in _SENTENCE.findall(text) if _is_question(part)
               for old in policy.recent_questions)
