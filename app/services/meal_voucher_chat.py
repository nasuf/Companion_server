"""Meal-voucher chat card trigger, payload, and once-per-conversation state."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import db
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

MEAL_VOUCHER_CARD_TYPE = "meal_voucher"
MEAL_VOUCHER_CARD_FIRST = "first"
MEAL_VOUCHER_CARD_REPEAT = "repeat"

_PENDING_TTL_S = 180
_SENT_TTL_S = 365 * 24 * 60 * 60
_CAMPAIGN_TZ = ZoneInfo(settings.schedule_timezone)
_CAMPAIGN_END = datetime(2026, 8, 24, 0, 0, tzinfo=_CAMPAIGN_TZ)
_FOLLOWUP_TERMS = (
    "霸王餐",
    "霸王餐券",
    "券码",
    "二维码",
    "激活",
    "核销",
    "扫码",
    "怎么领",
    "咋领",
    "怎么用",
    "过期",
    "有效期",
    "几次",
    "工作人员",
    "合作商家",
)

_RELEASE_CLAIM_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class MealVoucherCardDecision:
    state: Literal["none", "first", "repeat"]
    component_card: dict[str, Any] | None = None
    claim_token: str | None = None

    @property
    def should_send(self) -> bool:
        return self.state == MEAL_VOUCHER_CARD_FIRST and self.component_card is not None


def build_meal_voucher_component_card() -> dict[str, Any]:
    """Build the canonical card persisted in assistant message metadata."""
    return {
        "version": 1,
        "type": MEAL_VOUCHER_CARD_TYPE,
        "title": "霸王餐券",
        "subtitle": "现场通关后 · 工作人员扫码激活",
        "body": "点击进入「我的」，出示你的霸王餐券二维码",
        "footer": "去「我的」查看",
        "accent": "#FF7A1A",
        "payload": {
            "target_tab": "profile",
            "target_section": "meal_voucher",
            "fallback_text": "霸王餐券入口：前往「我的」查看二维码",
            "native_status": "ended",
            "campaign_ends_at": _CAMPAIGN_END.isoformat(),
            "native_message": (
                "佛山“西甲”霸王餐活动已结束，"
                "这张卡片仅作为历史消息保留。"
            ),
        },
    }


def is_meal_voucher_campaign_active(now: datetime | None = None) -> bool:
    """Stop creating new cards after the Foshan football campaign ends."""
    current = now or datetime.now(_CAMPAIGN_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=_CAMPAIGN_TZ)
    return current.astimezone(_CAMPAIGN_TZ) < _CAMPAIGN_END


def is_meal_voucher_turn(
    user_message: str,
    classified_memories: list[Any] | None,
) -> bool:
    """Require both a meal-related question and retrieved template knowledge."""
    text = (user_message or "").strip()
    if not text or not any(term in text for term in _FOLLOWUP_TERMS):
        return False
    return any(
        "霸王餐" in str(getattr(memory, "text", "") or "")
        for memory in (classified_memories or [])
    )


def _state_key(conversation_id: str) -> str:
    return f"chat:meal_voucher_card:{conversation_id}"


async def _history_has_card(conversation_id: str) -> bool:
    """DB backstop keeps the once-only guarantee across Redis loss/restarts."""
    try:
        rows = await db.query_raw(
            """
            SELECT EXISTS (
                SELECT 1
                FROM messages
                WHERE conversation_id = $1
                  AND role = 'assistant'
                  AND metadata->'component_card'->>'type' = 'meal_voucher'
            ) AS sent
            """,
            conversation_id,
        )
        return bool(rows and rows[0].get("sent"))
    except Exception as exc:
        logger.warning("[MEAL-CARD] history lookup failed: %s", exc)
        return False


async def prepare_meal_voucher_card(
    *,
    conversation_id: str,
    user_message: str,
    classified_memories: list[Any] | None,
) -> MealVoucherCardDecision:
    """Return first/repeat/none and atomically reserve the first card send."""
    if not is_meal_voucher_campaign_active():
        return MealVoucherCardDecision(state="none")
    if not is_meal_voucher_turn(user_message, classified_memories):
        return MealVoucherCardDecision(state="none")

    if await _history_has_card(conversation_id):
        return MealVoucherCardDecision(state=MEAL_VOUCHER_CARD_REPEAT)

    token = uuid.uuid4().hex
    try:
        redis = await get_redis()
        acquired = bool(
            await redis.set(
                _state_key(conversation_id),
                token,
                nx=True,
                ex=_PENDING_TTL_S,
            )
        )
        if not acquired:
            return MealVoucherCardDecision(state=MEAL_VOUCHER_CARD_REPEAT)
    except Exception as exc:
        # The DB backstop already found no card. Degrade to a first send rather
        # than hiding the entry point; only concurrent Redis outages can duplicate.
        logger.warning("[MEAL-CARD] Redis claim unavailable: %s", exc)
        token = None

    return MealVoucherCardDecision(
        state=MEAL_VOUCHER_CARD_FIRST,
        component_card=build_meal_voucher_component_card(),
        claim_token=token,
    )


async def finalize_meal_voucher_card(
    *,
    conversation_id: str,
    decision: MealVoucherCardDecision,
    emitted: bool,
) -> None:
    """Commit the sent marker, or release an unused pending claim."""
    if decision.state != MEAL_VOUCHER_CARD_FIRST:
        return
    try:
        redis = await get_redis()
        key = _state_key(conversation_id)
        if emitted:
            await redis.set(key, "sent", ex=_SENT_TTL_S)
        elif decision.claim_token:
            await redis.eval(
                _RELEASE_CLAIM_LUA,
                1,
                key,
                decision.claim_token,
            )
    except Exception as exc:
        logger.warning("[MEAL-CARD] state finalize failed: %s", exc)
