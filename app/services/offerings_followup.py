"""Link post-offering chat (purchase / use) to 生活/馈赠 memories."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.db import db
from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.memory.storage.persistence import store_memory
from app.services.offerings import (
    KIND_GIFT,
    KIND_RED_PACKET,
    MEMORY_IMPORTANCE,
    MEMORY_LEVEL,
    STATUS_RECEIVED,
    _offering_from_row,
)
from app.services.offerings_memory_text import (
    _agent_name,
    _ticket_amount,
    _yuan_amount,
)
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

FOLLOWUP_WINDOW_HOURS = 168
_FOLLOWUP_REDIS_PREFIX = "offering:followup_mem:"
_FOLLOWUP_REDIS_TTL_SECONDS = 30 * 24 * 3600

_PURCHASE_HINTS = (
    "买了", "买好了", "买好啦", "去买了", "刚买", "购了", "下单", "挑到", "挑好了",
    "入手", "搞定", "购物", "出门买",
)
_GIFT_USE_HINTS = (
    "用了", "喝了", "吃了", "尝了", "试了", "打开", "泡了", "带上", "穿上",
    "戴上", "摆好", "收好", "喜欢这",
)

_PURCHASE_DETAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"买(?:好|到|完|了|好啦)?(?:了)?(.{2,28}?)(?:[。！？!?~\n]|$)"),
    re.compile(r"挑(?:到|选)(?:了)?(.{2,28}?)(?:[。！？!?~\n]|$)"),
    re.compile(r"入手(?:了)?(.{2,28}?)(?:[。！？!?~\n]|$)"),
)


def _normalize(text: str) -> str:
    return "".join((text or "").split())


def detect_spending_followup(ai_text: str, user_text: str = "") -> bool:
    """Whether the assistant turn looks like spending/using a recent offering."""
    blob = _normalize(f"{ai_text} {user_text}")
    if not blob:
        return False
    if any(h in blob for h in _PURCHASE_HINTS):
        return True
    if any(h in blob for h in _GIFT_USE_HINTS):
        return True
    if "买" in blob and any(w in blob for w in ("咖啡", "东西", "礼物", "件", "罐", "杯")):
        return True
    return False


def _clean_detail(raw: str) -> str:
    text = (raw or "").strip(" ，,。！？!?~的了呢啊呀吧 ")
    text = re.sub(r"^(?:了|的|一些|点)", "", text)
    return text.strip()[:28]


def extract_followup_detail(
    ai_text: str,
    user_text: str,
    offering: dict[str, Any],
) -> str:
    """Best-effort noun phrase for what was bought or used."""
    for pattern in _PURCHASE_DETAIL_PATTERNS:
        match = pattern.search(ai_text or "")
        if match:
            detail = _clean_detail(match.group(1))
            if len(detail) >= 2:
                return detail

    if offering.get("kind") == KIND_GIFT:
        title = str(offering.get("product_title") or "").strip()
        blob = f"{ai_text} {user_text}"
        if title and title in blob:
            return title

    for source in (user_text, ai_text):
        if not source:
            continue
        for token in ("挂耳咖啡", "咖啡", "咖啡罐", "小挂饰", "礼物"):
            if token in source:
                return token

    if offering.get("kind") == KIND_GIFT:
        title = str(offering.get("product_title") or "礼物").strip()
        return title or "礼物"

    if any(h in _normalize(ai_text) for h in _PURCHASE_HINTS):
        return "东西"
    return ""


def build_followup_memory_texts(
    offering: dict[str, Any],
    detail: str,
) -> tuple[str, str] | None:
    agent = _agent_name(offering)
    detail = _clean_detail(detail)
    if not detail:
        return None

    if offering.get("kind") == KIND_GIFT:
        title = str(offering.get("product_title") or "礼物").strip() or "礼物"
        if detail == title or detail in title or title in detail:
            user_text = f"{agent}用了我送的{title}"
            ai_text = f"我用了用户送的{title}"
        else:
            user_text = f"{agent}用了我送的{title}（{detail}）"
            ai_text = f"我用了用户送的{title}（{detail}）"
        return user_text, ai_text

    yuan = _yuan_amount(offering)
    tickets = _ticket_amount(offering)
    amount_part = f"{yuan}元红包（{tickets}钞票）" if tickets != yuan else f"{yuan}元红包"
    user_text = f"{agent}用我发的{amount_part}买了{detail}"
    ai_text = f"我用用户发的{amount_part}买了{detail}"
    return user_text, ai_text


async def find_recent_received_offering(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT id, user_id, agent_id, conversation_id, message_id, kind,
               ticket_amount, agent_value_yuan, status, blessing, metadata,
               created_at, received_at
        FROM user_offerings
        WHERE conversation_id = $1
          AND user_id = $2
          AND status = $3
          AND received_at IS NOT NULL
          AND received_at >= NOW() - ($4::text || ' hours')::interval
          AND ($5::text IS NULL OR agent_id = $5)
        ORDER BY received_at DESC
        LIMIT 1
        """,
        conversation_id,
        user_id,
        STATUS_RECEIVED,
        str(FOLLOWUP_WINDOW_HOURS),
        agent_id,
    )
    if not rows:
        return None
    return _offering_from_row(rows[0])


async def _followup_already_recorded(offering_id: str) -> bool:
    try:
        redis = get_redis()
        return bool(await redis.get(f"{_FOLLOWUP_REDIS_PREFIX}{offering_id}"))
    except Exception:
        return False


async def _mark_followup_recorded(offering_id: str) -> None:
    try:
        redis = get_redis()
        await redis.set(
            f"{_FOLLOWUP_REDIS_PREFIX}{offering_id}",
            "1",
            ex=_FOLLOWUP_REDIS_TTL_SECONDS,
        )
    except Exception:
        pass


async def maybe_record_offering_followup(
    *,
    user_id: str,
    agent_id: str | None,
    conversation_id: str,
    workspace_id: str | None,
    user_message: str,
    ai_response: str,
) -> bool:
    """Write purchase/use memories tied to the latest received offering, once."""
    if not (conversation_id and user_id and (ai_response or "").strip()):
        return False
    if not detect_spending_followup(ai_response, user_message):
        return False

    offering = await find_recent_received_offering(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
    )
    if not offering:
        return False

    offering_id = str(offering.get("id") or "")
    if not offering_id or await _followup_already_recorded(offering_id):
        return False

    detail = extract_followup_detail(ai_response, user_message, offering)
    texts = build_followup_memory_texts(offering, detail)
    if not texts:
        return False

    user_text, ai_text = texts
    ws = workspace_id or offering.get("workspace_id")
    try:
        await store_memory(
            user_id,
            user_text,
            level=MEMORY_LEVEL,
            importance=MEMORY_IMPORTANCE,
            main_category="生活",
            sub_category="馈赠",
            source="user",
            workspace_id=ws,
            provenance=USER_STATED,
            skip_reconciliation=True,
        )
        await store_memory(
            user_id,
            ai_text,
            level=MEMORY_LEVEL,
            importance=MEMORY_IMPORTANCE,
            main_category="生活",
            sub_category="馈赠",
            source="ai",
            workspace_id=ws,
            provenance=AI_AUTHORED,
            skip_reconciliation=True,
        )
    except Exception:
        logger.exception(
            "offering followup memory write failed offering=%s",
            offering_id[:8],
        )
        return False

    await _mark_followup_recorded(offering_id)
    logger.info(
        "offering followup memory recorded kind=%s offering=%s detail=%s",
        offering.get("kind"),
        offering_id[:8],
        detail[:20],
    )
    return True


async def record_offering_followup_from_chat(
    *,
    user_id: str,
    agent_id: str | None,
    conversation_id: str,
    workspace_id: str | None,
    user_message: str,
    ai_response: str,
) -> None:
    """Background-safe wrapper; never raises."""
    try:
        await maybe_record_offering_followup(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            user_message=user_message,
            ai_response=ai_response,
        )
    except Exception:
        logger.exception("offering followup memory skipped")
