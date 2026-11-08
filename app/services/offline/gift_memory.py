"""AI 寄给用户的现实礼物的状态记忆 (B01.08)：寄出 → 签收 → 道谢，或中途取消。

一份礼物双方最多各一条记忆 (offline_gift_memories 记着是哪两条)，按礼物**当前**状态改写
同一对，不追加 (写入、并发、用户删过的记忆怎么处理见 record_memory.py，活动记忆共用同一套)：
取消后"礼物在路上"不会和"已签收"并存，签收后"等签收再揭晓"也不再残留。

保密规则不变：签收 (delivered) 之前记忆一律不含礼物名——记忆会注入聊天，AI 可能说漏嘴。
只有用户真的被告知过「有礼物在路上」(发过 gift_sent 卡片) 才记；失败/跳过/从没通知的不记。
揭晓时说的话和用户的道谢从消息表/礼物行读回，state_key 变了才改写。

与用户送 AI 的礼物（offerings.py）对称。礼物相关的聊天消息（real_world_type=gift）仍不进
通用抽取（chat/post_process.py），道谢里用户表达的喜好另走抽取 (remember_gift_thanks)。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.memory.provenance import GIFT_RECORD
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.retrieval.context_selector import exceeds_injection_limit
from app.services.offline import record_memory
from app.services.offline.chat_emit import single_bubble_text
from app.services.offline.gift_status import IN_TRANSIT_STATUSES, REVEALED_STATUS
from app.services.offline.record_memory import Kind
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

# 与 offerings 的馈赠记忆同档：礼物稀少，是关系里的大事；没人再提起时照常由惰性衰减降下去。
RECEIVED_IMPORTANCE = 0.72
EVENT_IMPORTANCE = 0.6
MAIN, SUB = "生活", "馈赠"
_DELIVERED = (2, RECEIVED_IMPORTANCE)
_EVENT = (2, EVENT_IMPORTANCE)
_QUOTE_LIMIT = 60
_day = record_memory.day


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


def _date_prefix(moment: datetime | None = None) -> str:
    local = (moment or datetime.now(UTC)).astimezone(ZoneInfo(settings.schedule_timezone))
    return local.strftime("%Y-%m-%d ")


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


def _notified_sent(gift: dict[str, Any]) -> bool:
    """用户是否被告知过「有礼物在路上」。调用方传入时才写记忆。"""
    return bool(gift.get("notified_sent"))


def render(gift: dict[str, Any]) -> dict[str, Any] | None:
    """按礼物当前状态写出双方记忆；不该有记忆时返回 None。

    用户从没收到「礼物在路上」通知的不记。签收 (delivered) 前不写礼物名。
    """
    if not _notified_sent(gift):
        return None
    status = str(gift.get("status") or "")
    name = str(gift.get("gift_name") or "").strip() or "一份小礼物"
    agent = (gift.get("agent_name") or "").strip() or "对方"
    sent_at = _parse_time(gift.get("ordered_at")) or _parse_time(gift.get("created_at"))
    sent_day = _day(sent_at)
    sent_prefix = f"{sent_day} " if sent_day else ""

    if status == REVEALED_STATUS:
        day = _day(gift.get("delivered_at")) or sent_day
        prefix = f"{day} " if day else ""
        reveal = _quote(gift.get("reveal_message"))
        note = _quote(gift.get("gift_note"))
        thanks = _quote(gift.get("thanks_message"))
        return {
            "user": _fit(
                f"{prefix}我收到并签收了{agent}寄来的礼物：{name}",
                f"，我跟{agent}说：「{thanks}」" if thanks else "",
            ),
            "ai": _fit(
                f"{prefix}用户签收了我寄的礼物：{name}",
                f"，揭晓时我跟用户说：「{reveal}」" if reveal else "",
                f"，随礼写的是：「{note}」" if note else "",
                f"，用户收到后跟我道谢：「{thanks}」" if thanks else "",
            ),
            "level": _DELIVERED[0], "importance": _DELIVERED[1],
        }
    if status in IN_TRANSIT_STATUSES:
        # 在途：AI 记得自己寄了、在等签收揭晓；用户侧不记（礼物还没揭晓，用户也没确认收到）。
        return {
            "user": None,
            "ai": f"{sent_prefix}我给用户寄了一份礼物，想等用户签收时再揭晓，所以没提前说是什么",
            "level": _EVENT[0], "importance": _EVENT[1],
        }
    if status == "cancelled":
        # 取消日取礼物行自己的 updated_at: 文字要稳定, 不能随重算的那天变 (state_key 会跟着变)。
        cancelled = _day(gift.get("updated_at")) or sent_day
        hint = "" if not sent_day or sent_day == cancelled else f"（{sent_day}寄出）"
        return {
            "user": None,
            "ai": f"{cancelled + ' ' if cancelled else ''}我寄给用户的那份礼物{hint}中途取消了，没能送到用户手上",
            "level": _EVENT[0], "importance": _EVENT[1],
        }
    # failed / skipped / pending_address / selecting：没有值得长期记住的既成事实。
    return None


async def _load(database, gift_id: str, user_id: str) -> dict[str, Any] | None:
    rows = await database.query_raw(
        """SELECT g.id, g.user_id, g.workspace_id, g.status, g.gift_name, g.gift_note,
                  g.ordered_at, g.created_at, g.updated_at, g.delivered_at, g.thanks_message,
                  a.name AS agent_name,
                  EXISTS (SELECT 1 FROM messages m JOIN conversations c ON c.id = m.conversation_id
                          WHERE c.user_id = g.user_id AND m.metadata->>'real_world_type' = 'gift'
                            AND m.metadata->>'source_id' = g.id
                            AND m.metadata->>'trigger_type' = 'gift_sent') AS notified_sent,
                  (SELECT m.content FROM messages m JOIN conversations c ON c.id = m.conversation_id
                   WHERE c.user_id = g.user_id AND m.metadata->>'real_world_type' = 'gift'
                     AND m.metadata->>'source_id' = g.id AND m.metadata->>'trigger_type' = 'gift_received_reply'
                   ORDER BY m.created_at LIMIT 1) AS reveal_message
           FROM real_world_gifts g LEFT JOIN ai_agents a ON a.id = g.agent_id
           WHERE g.id = $1 AND g.user_id = $2""",
        gift_id, user_id,
    )
    return dict(rows[0]) if rows else None


KIND = Kind(
    name="gift", link_table="offline_gift_memories", key_column="gift_id",
    provenance=GIFT_RECORD, main_category=MAIN, sub_category=SUB,
    load=_load, render=render,
)


async def sync(gift_id: str, user_id: str) -> str:
    """把这份礼物的记忆对改成与礼物当前状态一致。返回 unchanged/written/archived/missing/raced。"""
    return await record_memory.sync(KIND, gift_id, user_id)


def schedule_sync(gift_id: str | None, user_id: str | None) -> None:
    """礼物状态变化后调用 (不阻塞接口); 同一礼物多次触发只会收敛到最新状态。"""
    record_memory.schedule_sync(KIND, gift_id, user_id)


async def archive_for_user(user_id: str) -> int:
    """清空某用户礼物前调用: 对应的礼物记忆一起归档 (链接随礼物级联删除)。"""
    return await record_memory.archive_for_user(KIND, user_id)


def remember_gift_thanks(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
    *,
    message: str,
    message_id: str | None,
    message_content: str | None,
) -> None:
    """道谢：道谢内容进礼物状态记忆由 sync 负责；这里只抽取用户这句话里的喜好。"""
    schedule_sync(gift.get("id"), user_id)
    if not message_content:
        return
    name = gift.get("gift_name") or "礼物"

    async def _extract() -> None:
        try:
            await process_memory_pipeline(
                user_id,
                f"user: {message_content}",
                # 签收/道谢已由状态记忆记过；这里只看这句里新的信息（比如喜不喜欢）。
                context_conversation=f"user: 我签收了你寄来的「{name}」",
                side="user",
                workspace_id=workspace_id,
                statement_time=datetime.now(UTC),
                evidence_message_ids=[message_id] if message_id else None,
            )
        except Exception as exc:
            logger.warning("[gift-memory] thanks extraction failed gift_id=%s: %s", gift.get("id"), exc)

    fire_background(_extract())
