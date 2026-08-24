"""VIP 权益计量的单一常量源 + UTC+8 周期键工具。

改这些常量会直接改变计费/发放口径，请同步 Flutter 端与后台系统设置文案。
"""

from __future__ import annotations

from datetime import datetime

# ── 对话额度（权益项 1）──────────────────────────────────────────────
FREE_DAILY_MESSAGES = 20            # 非 VIP 每日免费句数
VIP_MONTHLY_MESSAGES = 5200         # VIP 每月免费句数
# 超额扣费（钞票/句）：非 VIP 0.5，VIP 0.3
OVERAGE_TICKET_PER_MSG = {"free": 0.5, "vip": 0.3}

# ── 音乐陪伴时长（权益项 6）──────────────────────────────────────────
FREE_DAILY_MUSIC_SECONDS = 1800     # 每日免费 0.5h
MUSIC_HALF_HOUR_SECONDS = 1800      # 计费/计券单元 0.5h
MUSIC_COUPON_UNIT_SECONDS = 3600    # 1 张音乐畅听券 = 1 小时（礼包清单）
# 超额扣费（钞票/0.5h）：非 VIP 10，VIP 5
MUSIC_TICKET_PER_HALF_HOUR = {"free": 10, "vip": 5}

# ── VIP 每月发放（权益项 3/4/6）─────────────────────────────────────
VIP_MONTHLY_GIFT_TICKETS = 40       # 限时钞票，随 VIP 存续结转，到期清零
VIP_MONTHLY_MUSIC_COUPONS = 20      # 音乐畅听券，当月有效不结转
VIP_MONTHLY_MAKEUP_CARDS = 2        # 补签卡，1 个月有效
VIP_GRANT_PERIOD_DAYS = 30          # 发放周期锚点（vip_last_grant_at + 30d）
VIP_GIFT_VALID_DAYS = 30            # VIP 赠送券/卡有效期

# ── 游戏积分（权益项 5）─────────────────────────────────────────────
GAME_VIP_MULTIPLIER = 1.5           # VIP 对局正向结算积分加成

# ── 消耗品 kind（与 store_catalog 对齐）──────────────────────────────
MUSIC_COUPON_KIND = "music_hour_coupon"
MAKEUP_CARD_KIND = "makeup_card"


def _now_utc8() -> datetime:
    # 集中在这里，测试可 monkeypatch；生产走 NTP 修正后的北京时间。
    from app.services.schedule_domain.time_service import get_current_time

    return get_current_time().now


def day_key(now: datetime | None = None) -> str:
    """UTC+8 自然日键 'YYYY-MM-DD'。"""
    return (now or _now_utc8()).strftime("%Y-%m-%d")


def month_key(now: datetime | None = None) -> str:
    """UTC+8 自然月键 'YYYY-MM'。"""
    return (now or _now_utc8()).strftime("%Y-%m")


def message_period(is_vip: bool) -> tuple[str, str, int]:
    """按 VIP 状态返回 (period_scope, period_key, free_limit)。"""
    if is_vip:
        return "month", month_key(), VIP_MONTHLY_MESSAGES
    return "day", day_key(), FREE_DAILY_MESSAGES


def overage_per_msg(is_vip: bool) -> float:
    return OVERAGE_TICKET_PER_MSG["vip" if is_vip else "free"]


def music_ticket_per_half_hour(is_vip: bool) -> int:
    return MUSIC_TICKET_PER_HALF_HOUR["vip" if is_vip else "free"]
