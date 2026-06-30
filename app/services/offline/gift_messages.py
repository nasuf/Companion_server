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


async def gift_sent_message(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
) -> str:
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = f"我给你寄出了一份「{gift['gift_name']}」。不用立刻做什么，它现在已经在路上了。"
    try:
        prompt_text = (await get_prompt_text("offline.gift_sent_message")).format(
            gift_name=gift.get("gift_name") or "小礼物",
            gift_reason=gift.get("gift_reason") or "",
            gift_note=gift.get("gift_note") or "",
            memory=memory or "暂无",
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return _single_line_friend_message(reply, fallback=fallback, limit=120)
    except Exception:
        return fallback


async def gift_delivered_message(
    user_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
) -> str:
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = f"我看见「{gift['gift_name']}」已经送到了，记得方便的时候查收一下。"
    try:
        prompt_text = (await get_prompt_text("offline.gift_delivered_message")).format(
            gift_name=gift.get("gift_name") or "小礼物",
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
