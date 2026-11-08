"""线下活动的状态记忆 (VOICE01.10): 用户回应过的活动, 双方各一条记忆, 始终与活动当前状态一致。

用户接受、忽略、到达、完成、取消, 或活动过期时, 按活动记录的**当前**状态重写同一对记忆
(offline_activity_memories 记着是哪两条), 不追加新条目: 取消后"用户打算去 X"不会残留
(40cc0d69 删掉接受记忆就是因为这个)。写入、并发和用户删过的记忆怎么处理见 record_memory.py
(礼物共用同一套)。

用户从没回应过的推荐不写 (生产上大多数推荐属于这类, 写进去只会冲淡记忆), 用户问起"推荐过
什么"由 recommendation_history 直接读表。行的 provenance 是 activity_record: 写入期合并、
去重整理和 L3 压缩都不碰它们, 否则下一次状态变化就找不到、改不到了。
"""

from __future__ import annotations

from typing import Any

from app.services.memory.provenance import ACTIVITY_RECORD
from app.services.offline import record_memory
from app.services.offline.record_memory import Kind

MAIN, SUB = "生活", "其他特殊事件"
# 去过 / 正在去的是关系里的事, 进 L2; 没去成的落 L3, 像真人一样慢慢淡出
_KEPT = (2, 0.55)
_FADED = (3, 0.4)
_day = record_memory.day


def responded(activity: dict[str, Any]) -> bool:
    """用户对这条推荐做过选择 (接受/不考虑/到过/完成); 只是收到推荐的不算。"""
    return bool(activity.get("accepted_at") or activity.get("ignored_at") or activity.get("reached")
                or activity.get("status") == "completed")


def _outcome(a: dict[str, Any]) -> tuple[str, tuple[int, float]] | None:
    """(用户这边做了什么, (层级, 重要度)); None = 不该有记忆。"""
    status, reached = a.get("status"), bool(a.get("reached"))
    arrived = _day(a.get("arrival_confirmed_at"))
    if status == "completed":
        return (f"{arrived}到过现场，这次活动已经完成" if reached else "这次活动已经完成"), _KEPT
    if status == "accepted":
        if reached:
            return f"{arrived}到了现场", _KEPT
        return f"{_day(a.get('accepted_at'))}接受了，放进了待出行，还没去", _KEPT
    if status == "ignored":
        return f"{_day(a.get('ignored_at'))}说暂时不考虑", _FADED
    if status == "cancelled" and a.get("accepted_at"):
        return f"接受后又取消了（{_day(a.get('updated_at'))}），没去", _FADED
    if status == "expired" and (a.get("accepted_at") or a.get("ignored_at")):
        return f"活动过期了（{_day(a.get('updated_at'))}），没去", _FADED
    return None


def render(activity: dict[str, Any], agent_name: str | None) -> dict[str, Any] | None:
    """按活动当前状态写出双方的记忆; 不该有记忆时返回 None。"""
    if not responded(activity):
        return None
    outcome = _outcome(activity)
    if outcome is None:
        return None
    what, (level, importance) = outcome
    title = str(activity.get("title") or "").strip()
    if not title:
        return None
    place = str(activity.get("location_name") or "").strip()
    if place and place in title:
        place = ""  # 标题已经是这个地方, 不再重复一遍
    city = str(activity.get("city") or "").strip()
    if city and city not in place:
        place = f"{city}·{place}" if place else city
    where = f"（{place}）" if place else ""
    day = _day(activity.get("created_at"))
    agent = (agent_name or "").strip() or "对方"
    return {
        # 用户侧写用户的经历; AI 侧写清"是我推荐给用户的、在用户那边", 不会被当成自己去过
        "user": f"{day} {agent}给用户推荐了线下活动「{title}」{where}，用户{what}",
        "ai": f"{day} 我给用户推荐了用户那边的线下活动「{title}」{where}，用户{what}",
        "level": level,
        "importance": importance,
    }


async def _load(database, activity_id: str, user_id: str) -> dict[str, Any] | None:
    rows = await database.query_raw(
        """SELECT a.id, a.user_id, a.workspace_id, a.title, a.location_name, a.city, a.status, a.reached,
                  a.accepted_at, a.ignored_at, a.arrival_confirmed_at, a.completed_at, a.created_at,
                  a.updated_at, g.name AS agent_name
           FROM offline_activity_recommendations a LEFT JOIN ai_agents g ON g.id = a.agent_id
           WHERE a.id = $1 AND a.user_id = $2""",
        activity_id, user_id,
    )
    return dict(rows[0]) if rows else None



KIND = Kind(
    name="activity", link_table="offline_activity_memories", key_column="activity_id",
    provenance=ACTIVITY_RECORD, main_category=MAIN, sub_category=SUB,
    load=_load, render=lambda activity: render(activity, activity.get("agent_name")),
)


async def sync(activity_id: str, user_id: str) -> str:
    """把这个活动的记忆对改成与活动当前状态一致。返回 unchanged/written/archived/missing/raced。"""
    return await record_memory.sync(KIND, activity_id, user_id)


def schedule_sync(activity_id: str | None, user_id: str | None) -> None:
    """状态变化后调用 (不阻塞接口); 同一活动多次触发只会收敛到最新状态。"""
    record_memory.schedule_sync(KIND, activity_id, user_id)


async def archive_for_user(user_id: str) -> int:
    """用户清空活动列表前调用: 对应的活动记忆一起归档 (链接随活动级联删除)。"""
    return await record_memory.archive_for_user(KIND, user_id)
