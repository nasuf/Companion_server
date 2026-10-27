"""聊天消息上下文的无副作用工具函数（orchestrator 拆分 R1）。

从 orchestrator.py 提取：这些函数只操作消息 dict/记录对象，无 DB/Redis/LLM
副作用，是主编排流中最独立的一层。orchestrator 通过 re-export 保持既有
导入路径兼容（tests 仍可 from orchestrator import）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

_T = TypeVar("_T")

# Bubbles of one reply land seconds apart (the reply delay is capped at 30s).
# A wider gap means a separate AI turn: an offer made hours ago must not ground
# today's "好".
TURN_BUBBLE_MAX_GAP = timedelta(minutes=10)


def message_field(message: Any, key: str, default: Any = None) -> Any:
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)


def _is_retrieval_snapshot(message: Any) -> bool:
    metadata = message_field(message, "metadata")
    return isinstance(metadata, dict) and bool(
        metadata.get("memory_retrievals") or metadata.get("memory_retrieval_analysis")
    )


@dataclass(frozen=True)
class RetrievalTurnView:
    """Whole-turn text addressed to the bubble holding the retrieval snapshot."""

    id: Any
    conversationId: Any
    metadata: Any
    content: str


@dataclass(frozen=True)
class AssistantTurn:
    """The AI's previous turn, read-only; never mutate the rows behind it.

    Two bubbles matter and they are usually different rows:
    - ``last`` is the turn's identity. The topic-completion verdict, the
      B-followup marker and the reply time all belong to the last AI message,
      so ``id`` / ``metadata`` / ``createdAt`` read from it.
    - ``retrieval_anchor`` holds the memory retrieval snapshot (normally the
      first bubble). Only retrieval feedback targets it, via ``retrieval_view``.
    ``content`` joins every bubble with newlines so guards see each bubble.
    """

    last: Any
    retrieval_anchor: Any
    content: str
    source_message_ids: tuple[str, ...]
    previous_user_text: str | None
    bubble_count: int = 1

    @property
    def id(self) -> Any:
        return message_field(self.last, "id")

    @property
    def conversationId(self) -> Any:
        return message_field(self.last, "conversationId")

    @property
    def metadata(self) -> Any:
        return message_field(self.last, "metadata")

    @property
    def createdAt(self) -> Any:
        return message_field(self.last, "createdAt")

    def retrieval_view(self) -> RetrievalTurnView:
        return RetrievalTurnView(
            id=message_field(self.retrieval_anchor, "id"),
            conversationId=message_field(self.retrieval_anchor, "conversationId"),
            metadata=message_field(self.retrieval_anchor, "metadata"),
            content=self.content,
        )


def _is_proactive(message: Any) -> bool:
    metadata = message_field(message, "metadata")
    return isinstance(metadata, dict) and bool(metadata.get("proactive"))


def _same_assistant_turn(earlier: Any, later: Any) -> bool:
    """A proactive message never joins a reply, and a long silence splits turns."""
    if _is_proactive(earlier) != _is_proactive(later):
        return False
    earlier_at = _parse_message_created_at(message_field(earlier, "createdAt"))
    later_at = _parse_message_created_at(message_field(later, "createdAt"))
    if earlier_at is None or later_at is None:
        return True
    return later_at - earlier_at <= TURN_BUBBLE_MAX_GAP


def group_dialogue_turns(messages: list[_T], *, user_max_gap: timedelta | None = None) -> list[list[_T]]:
    """Group chronological rows without changing their identity or filtering them.

    Callers own eligibility and limits. AI groups share the previous-reply
    proactive/time boundary. A user gap of None preserves template dialogue's
    continuous-fragment policy; memory passes a finite gap to avoid treating
    statements hours apart as one extraction-context turn. Unknown/missing
    times split finite-gap user groups. Original objects and IDs are retained.
    """
    groups: list[list[_T]] = []
    for row in messages:
        previous = groups[-1][-1] if groups else None
        role = message_field(row, "role")
        merge = previous is not None and role == message_field(previous, "role")
        if merge and role == "assistant":
            merge = _same_assistant_turn(previous, row)
        elif merge and role == "user":
            if user_max_gap is not None:
                before = _parse_message_created_at(message_field(previous, "createdAt"))
                now = _parse_message_created_at(message_field(row, "createdAt"))
                merge = before is not None and now is not None and timedelta(0) <= now - before <= user_max_gap
        else:
            merge = False
        if merge:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def previous_assistant_turn(recent_messages: list[Any], current_user_message_id: str | None) -> AssistantTurn | None:
    """Merge the AI bubbles of the turn right before this user message, excluding notices.

    Like the main history window, a user row separates turns; so do a switch
    between proactive and reply messages and a gap over ``TURN_BUBBLE_MAX_GAP``.
    Current/later rows are excluded when the current ID is present; missing IDs
    use the fetched window.
    """
    current_index = len(recent_messages)
    if current_user_message_id:
        for idx, message in enumerate(recent_messages):
            if message_field(message, "id") == current_user_message_id:
                current_index = idx
                break
        else:
            # Some ingress paths save after fetching history. They can still
            # use this window, provided it has no current row to bound it.
            current_index = len(recent_messages)
    bubbles = []
    previous_user_text = None
    for message in reversed(recent_messages[:current_index]):
        if is_offering_received_notice(message_field(message, "metadata")):
            continue
        role = message_field(message, "role")
        if role == "assistant":
            if bubbles and not _same_assistant_turn(message, bubbles[-1]):
                break
            bubbles.append(message)
            continue
        if bubbles:
            if role == "user":
                previous_user_text = message_field(message, "content")
            break
    if not bubbles:
        return None
    ordered = bubbles[::-1]
    return AssistantTurn(
        last=ordered[-1],
        # Retrieval snapshots sit on the first bubble; point feedback at that real
        # row instead of copying its metadata onto another one.
        retrieval_anchor=next((m for m in ordered if _is_retrieval_snapshot(m)), ordered[-1]),
        content="\n".join(str(message_field(m, "content") or "") for m in ordered),
        source_message_ids=tuple(str(message_field(m, "id")) for m in ordered if message_field(m, "id")),
        previous_user_text=previous_user_text,
        bubble_count=len(ordered),
    )


def _previous_assistant_message(recent_messages: list[Any], current_user_message_id: str | None) -> AssistantTurn | None:
    """Compatibility name for callers; every consumer sees the whole turn."""
    return previous_assistant_turn(recent_messages, current_user_message_id)


def _current_turn_message_ids(
    reply_context: dict | None,
    current_user_message_id: str | None,
) -> set[str]:
    ids: set[str] = set()
    raw_ids = (reply_context or {}).get("turn_message_ids")
    if isinstance(raw_ids, list):
        ids.update(item for item in raw_ids if isinstance(item, str) and item)
    if current_user_message_id:
        ids.add(current_user_message_id)
    return ids


def _achievement_turn_id(turn_message_ids: set[str] | list[str]) -> str | None:
    if not turn_message_ids:
        return None
    return "user-turn:" + ",".join(sorted(turn_message_ids))


def _ensure_current_user_message(
    messages: list[dict],
    *,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
) -> list[dict]:
    """Ensure the prompt context contains the current user turn.

    Normal chat saves the user message before fetching history, but delayed /
    aggregated delivery can hit visibility or payload races. If the current
    turn is absent, the main LLM sees the previous user turn as latest and can
    regenerate the prior reply. Add a synthetic current turn as a final guard.
    """
    if user_message_id:
        for idx, message in enumerate(messages):
            if message.get("id") != user_message_id:
                continue
            # Card / delayed paths persist an empty (or display-only) row, then
            # pass a rewritten prompt text as user_message. Keep the row, swap
            # in the text the main LLM is supposed to answer.
            if (message.get("content") or "") == (user_message or ""):
                return messages
            updated = dict(message)
            updated["content"] = user_message
            return [*messages[:idx], updated, *messages[idx + 1 :]]
    elif messages:
        last = messages[-1]
        if last.get("role") == "user" and last.get("content") == user_message:
            return messages

    received_at = None
    if user_message_id and reply_context:
        received_at = reply_context.get("latest_received_at") or reply_context.get("received_at")
    if user_message_id and (not isinstance(received_at, str) or not received_at):
        received_at = datetime.now(UTC).isoformat()

    return [
        *messages,
        {
            "id": user_message_id,
            "role": "user",
            "content": user_message,
            "createdAt": received_at,
            "synthetic_current": True,
        },
    ]


def collapse_turn_fragments(
    messages: list[dict],
    *,
    turn_message_ids: set[str],
    combined_text: str,
    combined_id: str | None,
) -> list[dict]:
    """Fold the current turn's fragment rows into one coherent user message.

    Fragment aggregation persists each piece ("我" / "喜欢" / "你") as a separate
    DB row, then processes them as one combined message ("我喜欢你"). Left as-is,
    the reply prompt shows the fragments as N separate user turns and the model
    may answer only the last one. This rebuilds the tail as a single user turn
    with the combined text so the LLM sees the message the user actually meant.

    Only affects the reply prompt — the memory pipeline still reads the original
    per-row messages (its watermark tracks message ids individually).

    No-op when there is 0/1 fragment or nothing combined to inject.
    """
    if len(turn_message_ids) <= 1 or not combined_text.strip():
        return messages

    kept = [m for m in messages if m.get("id") not in turn_message_ids]
    if len(kept) == len(messages):
        return messages  # none of the turn ids are in this window

    fragment_times = [
        _parse_message_created_at(m.get("createdAt"))
        for m in messages
        if m.get("id") in turn_message_ids
    ]
    latest = max((t for t in fragment_times if t is not None), default=None)
    kept.append({
        "id": combined_id,
        "role": "user",
        "content": combined_text,
        "createdAt": latest.isoformat() if latest else None,
        "coalesced_turn": True,
    })
    return kept


def _parse_message_created_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def is_offering_received_notice(metadata: Any) -> bool:
    """True for WeChat-style 小灰条 rows that must not enter the reply LLM."""
    if isinstance(metadata, dict):
        return bool(metadata.get("offering_received"))
    return False


def last_user_turns(
    items: list[_T], turns: int, role_of: Callable[[_T], str | None],
) -> list[_T]:
    """Keep the newest ``turns`` user turns: from the turns-th last user row to the end.

    Windows are counted in user turns, not rows: one AI reply is split into 1-4
    bubbles stored as separate rows (user:assistant ≈ 1:2.56), so a row-count
    window gets eaten by bubbles. With fewer user rows than ``turns`` the list is
    returned whole, including any leading AI opener.
    """
    if turns <= 0:
        return []
    seen = 0
    for index in range(len(items) - 1, -1, -1):
        if role_of(items[index]) == "user":
            seen += 1
            if seen == turns:
                return items[index:]
    return list(items)


def _max_user_created_at(messages: list[dict]) -> datetime | None:
    times = [
        ts for ts in (
            _parse_message_created_at(m.get("createdAt"))
            for m in messages
            if m.get("role") == "user"
        )
        if ts is not None
    ]
    return max(times, default=None)
