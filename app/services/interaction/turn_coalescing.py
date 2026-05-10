"""Turn-level semantic coalescing for aggregated user messages.

This module sits after the Redis quiet-window aggregation and before the chat
orchestrator. Its job is narrowly scoped: when the same user turn contains
multiple rewritten versions of the same read-only question, keep one
representative message so downstream reply logic does not answer the same
question twice.

The rules are intentionally conservative. We only coalesce high-confidence
read-only queries with the same semantic signature. Write actions and safety
related commands are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.chat.intent_dispatcher import (
    infer_schedule_query_type,
    is_explicit_current_state_query,
)
from app.services.rules.chat_keywords import (
    HIGH_CONFIDENCE_CANCEL_KEYWORDS,
    RECORD_MEMORY_CUES,
    REMINDER_ACTION_CUES,
    REMINDER_CONTENT_CUES,
    UNDO_CANCEL_KEYWORDS,
)
from app.services.rules.keyword_policy import compact_chat_text
from app.services.schedule_domain.schedule import resolve_schedule_query_scope
from app.services.schedule_domain.time_parser import has_explicit_time


_WRITE_ACTION_CUES = (
    *HIGH_CONFIDENCE_CANCEL_KEYWORDS,
    *RECORD_MEMORY_CUES,
    *REMINDER_ACTION_CUES,
    *REMINDER_CONTENT_CUES,
    *UNDO_CANCEL_KEYWORDS,
)

_QUESTION_CUES = (
    "吗", "呢", "？", "?", "什么", "几", "多少", "哪", "多大", "叫",
)

_STATE_FOLLOWUP_CUES = (
    "忙啥", "忙什么", "干嘛", "干啥", "做什么", "做啥", "有空", "忙吗", "不忙",
)


@dataclass(frozen=True)
class TurnSemanticSignature:
    intent: str
    subject: str
    domain: str
    slot: str
    scope: str = ""


@dataclass(frozen=True)
class CoalescedMessage:
    text: str
    signature: str
    duplicate_of: str


@dataclass(frozen=True)
class FactQueryRule:
    domain: str
    slot: str
    cues: tuple[str, ...]


@dataclass(frozen=True)
class TurnCoalesceResult:
    texts: list[str]
    coalesced: list[CoalescedMessage]

    @property
    def combined_text(self) -> str | None:
        return "\n".join(self.texts).strip() if self.texts else None

    @property
    def metadata(self) -> dict | None:
        if not self.coalesced:
            return None
        return {
            "original_count": len(self.texts) + len(self.coalesced),
            "kept_count": len(self.texts),
            "coalesced_count": len(self.coalesced),
            "dropped": [
                {
                    "text": item.text,
                    "signature": item.signature,
                    "duplicate_of": item.duplicate_of,
                }
                for item in self.coalesced
            ],
        }


_FACT_QUERY_RULES: tuple[FactQueryRule, ...] = (
    FactQueryRule("identity", "age", ("多大", "几岁", "多少岁", "年龄")),
    FactQueryRule("identity", "name", ("叫什么", "叫啥", "名字", "称呼")),
    FactQueryRule(
        "identity",
        "birthday",
        ("生日", "出生日期", "哪天出生", "什么时候出生", "几月几号"),
    ),
    FactQueryRule("identity", "gender", ("男生", "女生", "性别", "男的", "女的")),
    FactQueryRule("identity", "zodiac", ("星座", "生肖", "属什么")),
    FactQueryRule("identity", "birthplace", ("出生地", "哪里出生", "在哪出生")),
    FactQueryRule("identity", "hometown", ("哪里人", "哪的人", "老家", "家乡")),
    FactQueryRule("identity", "residence", ("住哪", "住在哪里", "现居", "在哪住")),
    FactQueryRule(
        "identity", "education",
        ("学校", "就读", "读书", "高中", "初中", "大学", "毕业"),
    ),
    FactQueryRule("identity", "job", ("职业", "工作", "上班", "做什么的")),
    FactQueryRule("identity", "family", ("家人", "父母", "爸爸", "妈妈", "兄弟", "姐妹")),
    FactQueryRule("identity", "relationship", ("朋友", "对象", "男朋友", "女朋友", "伴侣")),
    FactQueryRule("identity", "pet", ("宠物", "养猫", "养狗", "猫叫什么", "狗叫什么")),
    FactQueryRule("preference", "food", ("喜欢吃", "爱吃", "吃什么", "喝什么", "口味")),
    FactQueryRule(
        "preference", "music",
        ("喜欢听", "爱听", "喜欢什么音乐", "喜欢什么歌", "喜欢的歌"),
    ),
    FactQueryRule(
        "preference", "movie",
        ("喜欢看什么电影", "爱看什么电影", "喜欢的电影", "最喜欢的电影"),
    ),
    FactQueryRule(
        "preference", "drama",
        ("喜欢看什么剧", "爱看什么剧", "喜欢的剧", "最喜欢的剧"),
    ),
    FactQueryRule(
        "preference", "book",
        ("喜欢看什么书", "爱看什么书", "喜欢的书", "最喜欢的书"),
    ),
    FactQueryRule(
        "preference", "game",
        ("喜欢玩什么游戏", "爱玩什么游戏", "喜欢的游戏", "最喜欢的游戏"),
    ),
    FactQueryRule("preference", "color", ("喜欢什么颜色", "喜欢的颜色", "最喜欢的颜色", "色系")),
    FactQueryRule("preference", "habit", ("习惯", "作息", "几点睡", "几点起")),
)


def _signature_key(signature: TurnSemanticSignature) -> str:
    return "|".join((
        signature.intent,
        signature.subject,
        signature.domain,
        signature.slot,
        signature.scope,
    ))


def _has_write_action(text: str) -> bool:
    return any(cue in text for cue in _WRITE_ACTION_CUES)


def _is_question_like(text: str) -> bool:
    return any(cue in text for cue in _QUESTION_CUES)


def _fact_rule(normalized: str) -> FactQueryRule | None:
    for rule in _FACT_QUERY_RULES:
        if any(cue in normalized for cue in rule.cues):
            return rule
    return None


def _fact_subject(
    normalized: str,
    *,
    rule: FactQueryRule,
    previous: TurnSemanticSignature | None,
) -> str | None:
    if "我" in normalized:
        return "user"
    if "你" in normalized:
        return "ai"
    if (
        previous
        and previous.intent == "fact_query"
        and previous.domain == rule.domain
        and previous.slot == rule.slot
        and previous.subject in {"ai", "user"}
    ):
        return previous.subject
    return None


def _fact_signature(
    text: str,
    *,
    previous: TurnSemanticSignature | None,
) -> TurnSemanticSignature | None:
    normalized = compact_chat_text(text)
    if not normalized or not _is_question_like(normalized):
        return None
    rule = _fact_rule(normalized)
    if rule is None:
        return None
    subject = _fact_subject(normalized, rule=rule, previous=previous)
    if subject is None:
        return None
    return TurnSemanticSignature(
        intent="fact_query",
        subject=subject,
        domain=rule.domain,
        slot=rule.slot,
    )


def _schedule_signature(
    text: str,
    *,
    previous: TurnSemanticSignature | None,
) -> TurnSemanticSignature | None:
    normalized = compact_chat_text(text)
    if (
        previous
        and previous.intent == "schedule_query"
        and previous.scope == "current"
        and _is_question_like(normalized)
        and any(cue in normalized for cue in _STATE_FOLLOWUP_CUES)
    ):
        return previous

    query_type = infer_schedule_query_type(text, require_query_cue=False)
    if query_type is None:
        return None
    has_time_scope = has_explicit_time(text)
    if not has_time_scope and not (
        previous and previous.intent == "schedule_query"
    ):
        return None
    scope = resolve_schedule_query_scope(text, require_query_cue=False)
    scope_key = (
        scope.target_date.date().isoformat()
        if scope and scope.target_date is not None
        else (scope.date_label if scope else query_type)
    )
    if (
        previous
        and previous.intent == "schedule_query"
        and not has_time_scope
        and query_type == "current"
    ):
        scope_key = previous.scope
    return TurnSemanticSignature(
        intent="schedule_query",
        subject="ai",
        domain="schedule",
        slot="availability",
        scope=scope_key,
    )


def _current_state_signature(
    text: str,
    *,
    previous: TurnSemanticSignature | None,
) -> TurnSemanticSignature | None:
    normalized = compact_chat_text(text)
    if (
        previous
        and previous.intent == "current_state"
        and _is_question_like(normalized)
        and any(cue in normalized for cue in _STATE_FOLLOWUP_CUES)
    ):
        return previous
    if not is_explicit_current_state_query(text):
        return None
    return TurnSemanticSignature(
        intent="current_state",
        subject="ai",
        domain="state",
        slot="current_activity",
        scope="current",
    )


def semantic_signature_for_turn_message(
    text: str,
    *,
    previous: TurnSemanticSignature | None = None,
) -> TurnSemanticSignature | None:
    """Return a conservative semantic signature for one turn message."""
    cleaned = (text or "").strip()
    if not cleaned or _has_write_action(cleaned):
        return None

    return (
        _schedule_signature(cleaned, previous=previous)
        or _current_state_signature(cleaned, previous=previous)
        or _fact_signature(cleaned, previous=previous)
    )


def coalesce_turn_messages(texts: list[str]) -> TurnCoalesceResult:
    """Drop repeated rewrites of the same read-only query inside one turn."""
    kept: list[str] = []
    coalesced: list[CoalescedMessage] = []
    seen: dict[str, str] = {}
    previous_signature: TurnSemanticSignature | None = None

    for raw in texts:
        text = (raw or "").strip()
        if not text:
            continue
        signature = semantic_signature_for_turn_message(
            text,
            previous=previous_signature,
        )
        if signature is None:
            kept.append(text)
            previous_signature = None
            continue

        key = _signature_key(signature)
        if key in seen:
            coalesced.append(CoalescedMessage(
                text=text,
                signature=key,
                duplicate_of=seen[key],
            ))
            previous_signature = signature
            continue

        kept.append(text)
        seen[key] = text
        previous_signature = signature

    return TurnCoalesceResult(texts=kept, coalesced=coalesced)
