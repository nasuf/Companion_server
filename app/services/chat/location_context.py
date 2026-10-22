"""双方所在地: 主回复 prompt、联网检索和地点校验共用同一份位置事实。

2026-10 生产: 用户在西安, 主回复 prompt 里却只有 AI 档案里的杭州转塘这一个地点,
于是模型拿转塘去联网搜餐厅推荐给用户, 被问"我们在一个城市吗"也顺着答"对啊"。
用户的设备定位早就在 users 表里, 只有线下活动模块在用。

只用到城市/区县粒度, 经纬度不出库; 定位权限撤销后不再使用旧值。读失败时退回
"不知道对方在哪" —— 那仍然带着"不要假设同城"的约束, 比没有这段安全。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from app.db import db
from app.services.schedule_domain.sql_time import LOCAL_UTC_OFFSET_HOURS

logger = logging.getLogger(__name__)

Relation = Literal["same", "different", "unknown"]

# 超过这个天数的设备定位只展示、不拿来判断同城, 也不交给联网检索。
STALE_AFTER_DAYS = 30
_GRANTED_PERMISSIONS = frozenset({"whileInUse", "always"})
_PROVINCE_LEVEL = re.compile(r"(?:省|自治区|特别行政区)$")
# 档案城市是"浙江省 杭州市 西湖区 转塘街道"这类全称, 取地级那一段做同城判断。
_CITY_LEVEL = re.compile(r"[^\s省区县镇]{1,8}?(?:市|自治州|地区|盟)")
_CITY_SUFFIX = re.compile(r"(?:市|自治州|地区|盟)$")


def city_core(name: str | None) -> str:
    """"西安市" → "西安", 用于同城比较与简短称呼。"""
    return _CITY_SUFFIX.sub("", (name or "").strip())


def agent_city(agent: Any) -> str | None:
    """从档案全称里取地级城市; 取不到就返回 None, 不猜。"""
    place = getattr(agent, "city", None)
    if not isinstance(place, str) or not place.strip():
        return None
    match = _CITY_LEVEL.search(place)
    return match.group(0) if match else None


def _local_day(stamp: datetime) -> date:
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (stamp.astimezone(UTC) + timedelta(hours=LOCAL_UTC_OFFSET_HOURS)).date()


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class LocationContext:
    ai_place: str | None = None
    ai_city: str | None = None
    user_city: str | None = None
    user_area: str | None = None          # 省级或区县级, 取决于客户端上报
    user_country: str | None = None
    age_days: int | None = None
    stated_residence: str | None = None   # 用户说过的现居地记忆原文

    @property
    def device_known(self) -> bool:
        return bool(self.user_city)

    @property
    def stale(self) -> bool:
        return self.device_known and (self.age_days is None or self.age_days > STALE_AFTER_DAYS)

    @property
    def relation(self) -> Relation:
        if not self.device_known or self.stale or not self.ai_city:
            return "unknown"
        return "same" if city_core(self.user_city) == city_core(self.ai_city) else "different"

    @property
    def user_source(self) -> str:
        sources = [name for name, present in (
            ("device", self.device_known), ("stated", bool(self.stated_residence)),
        ) if present]
        return "+".join(sources) or "unknown"

    def user_place(self) -> str | None:
        if not self.user_city:
            return None
        area = self.user_area or ""
        if _PROVINCE_LEVEL.search(area):
            return f"{area}{self.user_city}"
        return f"{self.user_city}{area}" if area != self.user_city else self.user_city

    def describe_user(self) -> str:
        """注入 prompt 的"对方在哪"一句; 只有数据, 规则在模板里。"""
        parts: list[str] = []
        place = self.user_place()
        if place:
            if self.age_days is None:
                age = "更新时间未知"
            elif self.age_days == 0:
                age = "今天更新"
            elif self.age_days == 1:
                age = "昨天更新"
            elif self.age_days <= STALE_AFTER_DAYS:
                age = f"{self.age_days} 天前更新"
            else:
                age = "一个多月前更新，可能已经不准"
            parts.append(f"{place}（对方手机定位，{age}）")
        if self.stated_residence:
            parts.append(f"对方说过：{self.stated_residence}")
        return "；".join(parts) or "还不知道"

    def verifier_facts(self) -> dict[str, Any]:
        """地点校验器用的对方位置; 不含坐标。"""
        if not self.device_known and not self.stated_residence:
            return {"known": False}
        facts: dict[str, Any] = {"known": True}
        if self.device_known:
            facts.update(city=self.user_place(), source="手机定位", stale=self.stale)
        if self.stated_residence:
            facts["stated"] = self.stated_residence
        return facts

    def web_search_location(self) -> dict[str, str] | None:
        """Ark web_search 的 user_location; 过期或不知道时不传, 绝不拿 AI 的城市充数。"""
        if not self.device_known or self.stale:
            return None
        location = {"type": "approximate", "country": self.user_country or "中国",
                    "city": self.user_city}
        if self.user_area and _PROVINCE_LEVEL.search(self.user_area):
            location["region"] = self.user_area
        return location

    def diagnostics(self) -> dict[str, Any]:
        """Trace 用: 只记来源和新鲜度, 不落具体地名。"""
        return {
            "ai_city_known": bool(self.ai_city),
            "user_source": self.user_source,
            "user_age_days": self.age_days,
            "user_stale": self.stale,
            "relation": self.relation,
            "web_search_location": self.web_search_location() is not None,
        }


async def load_location_context(
    agent: Any, user_id: str, workspace_id: str | None, *, now: datetime | None = None,
) -> LocationContext:
    ai = LocationContext(
        ai_place=(getattr(agent, "city", None) or "").strip() or None,
        ai_city=agent_city(agent),
    )
    try:
        users = await db.query_raw(
            """SELECT location_city, location_region, location_country,
                      location_permission_status, location_updated_at
               FROM users WHERE id = $1""",
            user_id,
        )
        stated = await db.query_raw(
            """SELECT content FROM memories_user
               WHERE user_id = $1 AND workspace_id IS NOT DISTINCT FROM $2::text
                 AND NOT is_archived AND main_category = '身份' AND sub_category = '现居地'
               ORDER BY level, updated_at DESC LIMIT 1""",
            user_id, workspace_id,
        )
    except Exception as exc:  # noqa: BLE001 — 位置是增强信息, 不能卡住主回复
        logger.warning("[location-ctx] load failed: %s", type(exc).__name__)
        return ai
    stated_text = str(stated[0]["content"]).strip()[:80] if stated else None
    row = users[0] if users else {}
    city = (row.get("location_city") or "").strip()
    if not city or row.get("location_permission_status") not in _GRANTED_PERMISSIONS:
        return LocationContext(ai_place=ai.ai_place, ai_city=ai.ai_city,
                               stated_residence=stated_text or None)
    updated = _as_datetime(row.get("location_updated_at"))
    today = _local_day(now or datetime.now(UTC))
    return LocationContext(
        ai_place=ai.ai_place, ai_city=ai.ai_city,
        user_city=city,
        user_area=(row.get("location_region") or "").strip() or None,
        user_country=(row.get("location_country") or "").strip() or None,
        age_days=max(0, (today - _local_day(updated)).days) if updated else None,
        stated_residence=stated_text or None,
    )


# ── 回复里的对方位置断言 (校验器资格闸门, 不是语义判定) ─────────────────────

_SAME_PLACE_QUESTION = re.compile(
    r"(?:我们|咱们|咱俩|你我|我俩).{0,6}(?:一个|同一个?|同)(?:地方|城市|城|地儿?|个地方)",
)
# 回复开头就附和 —— 对"我们在一个城市吗"答"对啊…"。
_AFFIRMATIVE_OPENING = re.compile(r"^\s*(?:对|是|嗯|没错|当然|可不|那必须|必须的)")
_CO_LOCATION_QUESTION = re.compile(
    r"(?:我们|咱们|咱俩|你我|我俩).{0,6}(?:一个|同一个?|同)(?:地方|城市|城|地儿?|个地方)|"
    r"我(?:是)?(?:哪里|哪儿|哪个城市)(?:人|的)|我(?:现在)?(?:人)?在哪(?:儿|里|个城市)?(?=[？?吗呢啊呀]|$)|"
    r"你知道我在哪|"
    r"(?:离|跟|和)我.{0,4}(?:远|近)|我(?:这边|这儿|这里)?附近",
)
_USER_LOCATION_CLAIM = re.compile(
    r"(?:我们|咱们|咱俩|我俩)(?:都|也|就|是|还)?(?:在|住)|同(?:一个)?城|一个城市|一个地方|"
    r"你(?:也|就|那边|那儿|那里|现在|应该|肯定)?(?:在|住在?)(?!干|忙|做|吃|玩|想|看|听|写|学|睡|休息|上班|加班)|"
    r"离你(?:很|挺|不|这)?(?:近|远)|你(?:家)?附近|就在我(?:家|这)?附近|(?:去|到)(?:这|那)家|要不要去|"
    r"我(?:这边|这儿|这里)附近",
)


def affirms_same_place_wrongly(location: LocationContext | None, question: str, reply: str) -> bool:
    """确知不同城时, 对"我们在一个地方/城市吗"开口就附和 —— 不用模型也能判错。

    线上核验模型对这一句时放时拦 (2026-10-22 三轮里放过一次); 事实齐全时直接
    判定, 更正文本只陈述已知事实, 误判的代价只是多说一句真话。
    """
    return bool(
        location is not None and location.relation == "different"
        and _SAME_PLACE_QUESTION.search(question or "")
        and _AFFIRMATIVE_OPENING.search(reply or "")
    )


def asks_about_user_location(question: str) -> bool:
    return bool(_CO_LOCATION_QUESTION.search(question or ""))


def claims_user_location(text: str) -> bool:
    return bool(_USER_LOCATION_CLAIM.search(text or ""))
