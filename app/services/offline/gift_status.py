"""礼物状态机与用户侧可见性规则（单一来源）。

    pending_address → selecting → ordered → shipping → out_for_delivery → delivered
                                      └─────────── failed / skipped / cancelled

- ``delivered`` 表示用户在 App 里确认签收，即「已收到」。它是唯一会向用户
  揭晓礼物名称、图片、寄语的状态——在途期间礼物保持神秘。
- 物流侧的「已送达 / 已签收」（快递员、驿站代收）只推进到 ``out_for_delivery``，
  最终签收必须由用户在礼物页点「确认签收」完成。
- 轨迹事件里 ``status='delivered'`` 只由签收动作写入，物流同步不会写入它。
"""

from __future__ import annotations

from typing import Any

IN_TRANSIT_STATUSES: tuple[str, ...] = ("ordered", "shipping", "out_for_delivery")
# 占着「当前礼物」名额：自动触发看到这些状态就不再新建礼物。
ACTIVE_STATUSES: tuple[str, ...] = ("pending_address", "selecting", *IN_TRANSIT_STATUSES)
# 已经真实花出去的钱（计入礼物预算）。
SPENT_STATUSES: tuple[str, ...] = (*IN_TRANSIT_STATUSES, "delivered")
REVEALED_STATUS = "delivered"
CONFIRMABLE_STATUS = "out_for_delivery"
CLOSED_STATUSES: tuple[str, ...] = ("delivered", "failed", "skipped", "cancelled")

MANUAL_PROVIDER = "manual"
MYSTERY_GIFT_NAME = "神秘礼物"

_TRANSIT_RANK = {status: rank for rank, status in enumerate(IN_TRANSIT_STATUSES)}


def transit_rank(status: str) -> int:
    return _TRANSIT_RANK[status]


def gift_status_for_event(event_status: str | None) -> str | None:
    """物流节点 → 礼物状态。packed 归入 shipping；签收与未知状态不映射。"""
    value = (event_status or "").strip().lower()
    if value == "ordered":
        return "ordered"
    if value in {"packed", "shipping"}:
        return "shipping"
    if value == "out_for_delivery":
        return "out_for_delivery"
    return None


def normalize_provider_event_status(status: str | None) -> str:
    """把 provider 的轨迹状态收敛到四个用户可见节点：下单 / 运输 / 派送 / 签收。

    揽收归入运输；快递侧「已送达」只代表包裹到了，不代表用户确认收到，记为派送。
    """
    value = (status or "").strip().lower()
    if value == "delivered":
        return "out_for_delivery"
    if value == "packed":
        return "shipping"
    return value or "shipping"


def advances_transit(current: str | None, target: str | None) -> bool:
    """物流同步只能让在途状态前进，不能倒退，也不能越过签收。"""
    if current not in _TRANSIT_RANK or target not in _TRANSIT_RANK:
        return False
    return _TRANSIT_RANK[target] > _TRANSIT_RANK[current]


def is_revealed(gift: dict[str, Any]) -> bool:
    return gift.get("status") == REVEALED_STATUS


def user_view(gift: dict[str, Any]) -> dict[str, Any]:
    """返回给用户的礼物视图：签收前抹掉一切能猜出礼物是什么的字段。

    在服务端脱敏而不是让客户端隐藏，抓包也看不到；旧版客户端同样只能看到神秘礼物。
    """
    revealed = is_revealed(gift)
    view = {
        **gift,
        "revealed": revealed,
        "can_confirm_receipt": gift.get("status") == CONFIRMABLE_STATUS,
    }
    if revealed:
        if gift.get("provider") == MANUAL_PROVIDER:
            # 后台手动礼物的「送礼原因」是写给运营的分析（第三人称、引用记忆），
            # 只给签收揭晓的 AI 消息参考；用户看到的是随礼寄语。
            view["gift_reason"] = None
        return view
    view.update(
        gift_name=MYSTERY_GIFT_NAME,
        gift_reason=None,
        gift_note=None,
        product_image_url=None,
        product_url=None,
        provider_product_id=None,
        paid_amount_cents=0,
        # 未签收时 delivered_at 可能是供应商给的预计送达时间，不对用户展示。
        delivered_at=None,
    )
    return view
