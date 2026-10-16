from __future__ import annotations

import re
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text


def _single_line_friend_message(reply: str, *, fallback: str, limit: int) -> str:
    line = reply.strip().split("\n", 1)[0].strip(" 「」\"'")
    if not line:
        return fallback
    # LLM 偶尔会把 agent 名字当成被称呼的人，产出“小芜，给你寄了...”。
    # 删除句首称呼后再补回“我”，确保说话视角是 AI -> 用户。
    line = re.sub(r"^[^，,]{1,16}[，,]\s*(?=(?:给你|我给你|寄|这次|有))", "", line)
    if line.startswith("给你"):
        line = "我" + line
    if any(bad in line for bad in ("你给我", "寄给我", "送给我")):
        return fallback
    return line[:limit]


async def first_address_request_message(user_id: str, workspace_id: str | None) -> str:
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    try:
        prompt_text = (await get_prompt_text("offline.gift_first_address_request")).format(
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return reply.split("\n", 1)[0][:120]
    except Exception:
        return "我有一点现实里的小心意想寄给你。先去「我的礼物」里补一下收货地址吧，我会把它稳稳放好。"


async def gift_sent_message(user_id: str, workspace_id: str | None) -> str:
    """寄出提示。礼物在签收前保密：prompt 拿不到任何礼物信息，从源头杜绝泄露。"""
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = "我给你寄了一份小惊喜，已经在路上了。先不告诉你是什么，等你签收那天再揭晓～"
    try:
        prompt_text = (await get_prompt_text("offline.gift_sent_message")).format(
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return _single_line_friend_message(reply, fallback=fallback, limit=120)
    except Exception:
        return fallback


async def gift_shipped_message(user_id: str, workspace_id: str | None) -> str:
    """发货了、在运输途中。同样不知道礼物是什么，也还没到签收的时候。"""
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = "给你寄的礼物已经发货啦，正在路上，先别猜是什么，到了我再提醒你～"
    try:
        prompt_text = (await get_prompt_text("offline.gift_shipped_message")).format(
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return _single_line_friend_message(reply, fallback=fallback, limit=120)
    except Exception:
        return fallback


async def gift_out_for_delivery_message(user_id: str, workspace_id: str | None) -> str:
    """快递派送中，提醒用户收到后去礼物页确认签收。同样不知道礼物是什么。"""
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = "快递说礼物今天在派送啦，收到以后记得去「我的礼物」点一下签收，才能拆开看看是什么～"
    try:
        prompt_text = (await get_prompt_text("offline.gift_out_for_delivery_message")).format(
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return _single_line_friend_message(reply, fallback=fallback, limit=120)
    except Exception:
        return fallback


async def gift_received_message(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
) -> str:
    """用户签收后揭晓：说出礼物是什么、为什么想送它。"""
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = f"你签收啦！其实是「{gift.get('gift_name') or '一份小礼物'}」，希望你会喜欢。"
    try:
        prompt_text = (await get_prompt_text("offline.gift_received_message")).format(
            gift_name=gift.get("gift_name") or "小礼物",
            gift_reason=gift.get("gift_reason") or "",
            gift_note=gift.get("gift_note") or "",
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return _single_line_friend_message(reply, fallback=fallback, limit=120)
    except Exception:
        return fallback


async def gift_thanks_reply(gift: dict[str, Any], message: str) -> str:
    try:
        prompt_text = (await get_prompt_text("offline.gift_thanks_reply")).format(
            gift_name=gift.get("gift_name") or "礼物",
            message=message,
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return reply.split("\n", 1)[0][:80]
    except Exception:
        return "收到你的谢谢，我会偷偷开心很久。"
