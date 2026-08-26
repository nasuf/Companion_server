"""Searchable text for offering cards (red packets / gifts).

These strings are persisted to messages.content, injected into chat history,
and written to 生活/馈赠 memories. Wording uses 元 (agent_value_yuan) plus
钞票/ticket_amount so recall matches how users talk ("500块/元/钱").
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

OfferingEvent = Literal["sent", "received"]


def _agent_name(offering: dict[str, Any]) -> str:
    return str(offering.get("agent_name") or "对方").strip() or "对方"


def _yuan_amount(offering: dict[str, Any]) -> int:
    yuan = offering.get("agent_value_yuan")
    if yuan is not None:
        return int(yuan)
    return int(offering.get("ticket_amount") or 0)


def _ticket_amount(offering: dict[str, Any]) -> int:
    return int(offering.get("ticket_amount") or 0)


def _format_date_hint(offering: dict[str, Any]) -> str:
    raw = offering.get("created_at") or offering.get("received_at") or ""
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _blessing_suffix(offering: dict[str, Any]) -> str:
    blessing = str(offering.get("blessing") or "").strip()
    if not blessing:
        return ""
    return f"，祝福：{blessing}"


def build_offering_memory_texts(
    offering: dict[str, Any],
    *,
    event: OfferingEvent = "sent",
) -> tuple[str, str]:
    """Return (user-side, ai-side) memory text for store_memory."""
    agent = _agent_name(offering)
    date_hint = _format_date_hint(offering)
    date_prefix = f"{date_hint} " if date_hint else ""

    if offering.get("kind") == "gift":
        title = str(offering.get("product_title") or "礼物").strip() or "礼物"
        sub = str(offering.get("product_subcategory") or "").strip()
        detail = f"{title}（{sub}）" if sub else title
        if event == "received":
            user_text = f"{date_prefix}我送给{agent}的{detail}已被{agent}收下"
            ai_text = f"{date_prefix}我收下了用户送的{detail}"
        else:
            user_text = f"{date_prefix}我给{agent}送了礼物：{detail}"
            ai_text = f"{date_prefix}用户送给我礼物：{detail}"
        return user_text, ai_text

    yuan = _yuan_amount(offering)
    tickets = _ticket_amount(offering)
    blessing = _blessing_suffix(offering)
    amount_part = f"{yuan}元红包（{tickets}钞票）" if tickets != yuan else f"{yuan}元红包"

    if event == "received":
        user_text = f"{date_prefix}{agent}领取了我发的{amount_part}{blessing}"
        ai_text = f"{date_prefix}我领取了用户发的{amount_part}{blessing}"
    else:
        user_text = f"{date_prefix}我给{agent}发了{amount_part}{blessing}"
        ai_text = f"{date_prefix}用户给我发了{amount_part}{blessing}"
    return user_text, ai_text


def build_offering_history_text(
    offering: dict[str, Any],
    *,
    component_card: dict[str, Any] | None = None,
) -> str:
    """Line persisted on messages.content and shown in LLM history."""
    user_text, _ = build_offering_memory_texts(offering, event="sent")
    if offering.get("status") == "received":
        agent = _agent_name(offering)
        if offering.get("kind") == "gift":
            suffix = f"{agent}已收下礼物"
        else:
            suffix = f"{agent}已领取红包"
        if suffix not in user_text:
            return f"{user_text}（{suffix}）"
    if component_card and not user_text:
        return render_component_card_line("", component_card)
    return user_text


def render_component_card_line(content: str, card: dict[str, Any]) -> str:
    """Render a component card into natural language for prompts."""
    card_type = str(card.get("type") or "")
    payload = card.get("payload") if isinstance(card.get("payload"), dict) else {}
    text = (content or "").strip()

    if card_type == "red_packet":
        yuan = payload.get("agent_value_yuan") or payload.get("ticket_amount") or ""
        tickets = payload.get("ticket_amount")
        status = str(payload.get("status_label") or "").strip()
        body = str(card.get("body") or "").strip()
        amount = f"{yuan}元"
        if tickets and tickets != yuan:
            amount = f"{yuan}元（{tickets}钞票）"
        line = f"用户发了红包，金额{amount}"
        if body:
            line += f"，{body}"
        if status:
            line += f"（{status}）"
    elif card_type == "gift":
        title = str(payload.get("product_title") or card.get("title") or "礼物").strip()
        sub = str(payload.get("product_subcategory") or card.get("body") or "").strip()
        status = str(payload.get("status_label") or "").strip()
        line = f"用户送了礼物：{title}"
        if sub and sub != title:
            line += f"（{sub}）"
        if status:
            line += f"（{status}）"
    else:
        return text

    if text and line not in text:
        return f"{text}\n{line}"
    return line or text


def offering_recall_search_query(message: str) -> str | None:
    """Deterministic enhanced_query when the user asks about past gifts/red packets."""
    text = "".join(message.split())
    if not text:
        return None
    recall_hints = ("上次", "记得", "还记得", "之前", "给我", "送我", "转账", "那次")
    offering_terms = ("红包", "礼物", "钱", "元", "钞票", "心意", "给过", "发过", "送过")
    if not any(h in text for h in recall_hints):
        return None
    if not any(t in text for t in offering_terms):
        return None
    parts = ["用户给AI的红包或礼物馈赠 给钱 转账 元 买了 购买"]
    if "咖啡" in text:
        parts.append("挂耳咖啡 咖啡")
    if "礼物" in text:
        parts.append("礼物")
    if "红包" in text or "钱" in text:
        parts.append("红包 元")
    return " ".join(dict.fromkeys(" ".join(parts).split()))
