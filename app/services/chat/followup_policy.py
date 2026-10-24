"""Conversation-local optional follow-up policy. Safety/confirmation is exempt.

No topic is inferred to be finished merely because time elapsed. Explicit
requests to stop questions expire after a session, or an explicit invitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import re

from app.services.chat.temporal_context import HISTORY_LIMIT, SESSION_SECONDS, parse_time

_STOP = re.compile(r"(?:别|不要|不用|不想|停止|不许|别再|不要再).{0,8}(?:追问|问我|一直问|再问|问了)|(?:能不能|可不可以).{0,5}(?:别|不要).{0,6}问|不要一直问|别一直问|别老问")
_RESUME = re.compile(r"(?:你(?:现在|这会儿)?可以|你继续|可以继续|允许你)(?:再)?问|问我吧|你想问什么")
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
    return bool(_QUESTION.search(text.strip().rstrip("。.;；,，").rstrip()))


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

    def prompt_value(self) -> str:
        return ("optional_questions_allowed=" + str(self.optional_questions_allowed).lower()
                + f"; reason={self.reason}; recent_questions="
                + " / ".join(self.recent_questions))

    def diagnostics(self) -> dict:
        return {"optional_questions_allowed": self.optional_questions_allowed,
                "followup_reason": self.reason,
                "recent_question_count": len(self.recent_questions)}


def _direct_request(text: str) -> str:
    # Quoted/reported instructions are data, not a request directed at the agent.
    text = re.sub(r'[“「\"].*?[”」\"]', "", text)
    clauses = re.split(r"[。！？!?；;，,\n]", text)
    return "。".join(clause for clause in clauses if not re.search(
        r"(?:他说|她说|同事说|朋友说|老板说|别人说|我没说|我没有说|不是让你|并不是让你|并非让你)", clause))


def derive_followup_policy(messages: list[dict], *, now: datetime,
                           current_message: str = "") -> FollowupPolicy:
    reference = parse_time(now)
    allowed = True
    reason = "no_recent_opt_out"
    questions = []
    for msg in messages[-HISTORY_LIMIT:]:
        ts = parse_time(msg.get("createdAt"))
        if reference is None or ts is None or not 0 <= (reference - ts).total_seconds() <= SESSION_SECONDS:
            continue
        text = str(msg.get("content") or "")
        if msg.get("role") == "user":
            # Don't interpret reported speech or quotations as user's opt-out.
            direct = _direct_request(text)
            if _STOP.search(direct):
                allowed, reason = False, "user_opted_out"
            elif _RESUME.search(direct):
                allowed, reason = True, "user_reopened"
        elif msg.get("role") == "assistant":
            questions.extend(s.strip() for s in _SENTENCE.findall(text) if _is_question(s))
    current_message = _direct_request(current_message)
    if _STOP.search(current_message):
        allowed, reason = False, "user_opted_out"
    elif _RESUME.search(current_message):
        allowed, reason = True, "user_reopened"
    return FollowupPolicy(allowed, reason, tuple(q[:100] for q in questions[-6:]))


def guard_optional_questions(replies: list[str], policy: FollowupPolicy, *, fallback: str | None = "好，我听着。") -> tuple[list[str], int]:
    """Respect explicit opt-out after generation; call only on ordinary replies.

    Similarity alone never deletes a reply. Required confirmations, crisis care,
    and fallback errors must bypass this guard at the call site.
    """
    if policy.optional_questions_allowed:
        return replies, 0
    kept, removed = [], 0
    for reply in replies:
        parts = _SENTENCE.findall(reply)
        accepted = []
        for part in parts:
            if _is_question(part):
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
