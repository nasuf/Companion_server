"""AI 寄出的现实礼物写进记忆：寄出 / 签收 / 道谢 / 撤销。

与用户送 AI 的礼物（offerings.py）对称：结构化事件直接写 生活/馈赠，双侧保证落库，不经
聊天抽取的「记/不记」预筛——那一关可能把"我给用户寄了礼物"判成不值得记。也正因为这里已经
写过，礼物相关的聊天消息（real_world_type=gift）不再进通用抽取（chat/post_process.py），
否则同一件事会被记两遍。

记忆写的是带日期的既成事实，不写"签收前要保密"这类指令：指令进了长期记忆，揭晓之后还会
一直生效。签收前的记忆一律不含礼物名——记忆会注入聊天，AI 可能说漏嘴。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.observability.events import EVT_GIFT_MEMORY
from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.retrieval.context_selector import exceeds_injection_limit
from app.services.memory.storage.persistence import store_memory
from app.services.offline.chat_emit import single_bubble_text
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

# 与 offerings 的馈赠记忆同档：礼物稀少，是关系里的大事；没人再提起时照常由惰性衰减降下去。
RECEIVED_IMPORTANCE = 0.72
EVENT_IMPORTANCE = 0.6
_QUOTE_LIMIT = 60


def _date_prefix(moment: datetime | None = None) -> str:
    local = (moment or datetime.now(UTC)).astimezone(ZoneInfo(settings.schedule_timezone))
    return local.strftime("%Y-%m-%d ")


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _quote(text: str | None) -> str:
    cleaned = single_bubble_text(text or "")
    return cleaned if len(cleaned) <= _QUOTE_LIMIT else cleaned[:_QUOTE_LIMIT] + "…"


def _fit(base: str, *extras: str) -> str:
    """按顺序追加补充句，只加放得下的：超过单条注入上限的记忆检索时会被整条跳过。"""
    text = base
    for extra in extras:
        if extra and not exceeds_injection_limit(text + extra):
            text += extra
    return text


def _store(*, stage: str, gift: dict[str, Any], user_id: str, workspace_id: str | None, side: str,
           content: str, importance: float) -> None:
    async def _write() -> None:
        try:
            memory_id = await store_memory(
                user_id,
                content,
                level=2,
                importance=importance,
                main_category="生活",
                sub_category="馈赠",
                source=side,
                workspace_id=workspace_id,
                provenance=AI_AUTHORED if side == "ai" else USER_STATED,
                # 结构化事件各自独立成行，不能被 reconciliation 并进别的记忆。
                skip_reconciliation=True,
            )
            outcome = "stored" if memory_id else "deduplicated"
        except Exception as exc:
            outcome = "failed"
            logger.warning("[gift-memory] %s write failed gift_id=%s side=%s: %s", stage, gift["id"], side, exc)
        logger.info(
            "[gift-memory] %s side=%s outcome=%s gift_id=%s",
            stage, side, outcome, gift["id"],
            extra={"event": EVT_GIFT_MEMORY, "stage": stage, "side": side, "outcome": outcome,
                   "gift_id": gift["id"], "user_id": user_id},
        )

    fire_background(_write())


def remember_gift_sent(user_id: str, workspace_id: str | None, gift: dict[str, Any]) -> None:
    """寄出：AI 记得自己寄了礼物、在等用户签收揭晓。不含礼物名。"""
    _store(
        stage="sent", gift=gift, user_id=user_id, workspace_id=workspace_id, side="ai",
        content=f"{_date_prefix()}我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么",
        importance=EVENT_IMPORTANCE,
    )


def remember_gift_received(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
    *,
    agent_name: str | None,
    reveal_message: str | None,
) -> None:
    """签收：双方都记住礼物是什么；AI 侧再带上揭晓时说的话（为什么送）和随礼寄语。"""
    name = gift.get("gift_name") or "一份小礼物"
    day = _date_prefix()
    reveal = _quote(reveal_message)
    note = _quote(gift.get("gift_note"))
    _store(
        stage="received", gift=gift, user_id=user_id, workspace_id=workspace_id, side="ai",
        content=_fit(
            f"{day}用户签收了我寄的礼物：{name}",
            f"，揭晓时我跟用户说：「{reveal}」" if reveal else "",
            f"，随礼写的是：「{note}」" if note else "",
        ),
        importance=RECEIVED_IMPORTANCE,
    )
    _store(
        stage="received", gift=gift, user_id=user_id, workspace_id=workspace_id, side="user",
        content=f"{day}我收到并签收了{agent_name or '对方'}寄来的礼物：{name}",
        importance=RECEIVED_IMPORTANCE,
    )


def remember_gift_thanks(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
    *,
    message: str,
    message_id: str | None,
    message_content: str | None,
) -> None:
    """道谢：AI 侧保证记住用户的原话；用户侧照常抽取，看这句话里有没有值得记的喜好。"""
    name = gift.get("gift_name") or "礼物"
    _store(
        stage="thanks", gift=gift, user_id=user_id, workspace_id=workspace_id, side="ai",
        content=f"{_date_prefix()}用户收到我送的{name}后特意跟我道谢：「{_quote(message)}」",
        importance=EVENT_IMPORTANCE,
    )
    if not message_content:
        return

    async def _extract() -> None:
        try:
            await process_memory_pipeline(
                user_id,
                f"user: {message_content}",
                # 签收那条已经单独记过；放进历史上下文，抽取只看这句道谢里新的信息（比如喜不喜欢）。
                context_conversation=f"user: 我签收了你寄来的「{name}」",
                side="user",
                workspace_id=workspace_id,
                statement_time=datetime.now(UTC),
                evidence_message_ids=[message_id] if message_id else None,
            )
        except Exception as exc:
            logger.warning("[gift-memory] thanks extraction failed gift_id=%s: %s", gift["id"], exc)

    fire_background(_extract())


def remember_gift_cancelled(user_id: str, workspace_id: str | None, gift: dict[str, Any]) -> None:
    """撤销一份已经告诉用户在路上的礼物：否则 AI 会一直以为它还在路上。"""
    today = _date_prefix()
    sent_at = _parse_time(gift.get("ordered_at")) or _parse_time(gift.get("created_at"))
    sent_day = _date_prefix(sent_at) if sent_at else today
    sent_hint = "" if sent_day == today else f"（{sent_day.strip()}寄出）"
    _store(
        stage="cancelled", gift=gift, user_id=user_id, workspace_id=workspace_id, side="ai",
        content=f"{today}我寄给用户的那份礼物{sent_hint}中途取消了，没能送到用户手上",
        importance=EVENT_IMPORTANCE,
    )
