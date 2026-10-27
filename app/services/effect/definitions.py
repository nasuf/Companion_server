"""内测统计口径的唯一定义: 人群、样本门槛、比率状态和指标字典.

任何统计接口 (数据监控 / 回复效果 / 后续日报和派生表) 都从这里取人群条件和指标
定义, 不在各自的 SQL 里另写一份。改动口径时提升 DEFINITION_VERSION —— 缓存键带
版本号, 旧口径算出的数不会被新页面读到。

人群: 内部账号 = role=admin, 或后台勾选了 users.is_internal, 或模板系统账号。
统计默认只看外部用户 (cohort="external"), 页面可切到 "all"。
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME

DEFINITION_VERSION = "beta-metrics-v1"

Cohort = Literal["external", "all"]
COHORTS: tuple[Cohort, ...] = ("external", "all")
DEFAULT_COHORT: Cohort = "external"

# 比率至少要有这么多样本才报百分比; 不足时只给分子/分母 (如 7/15)。
MIN_SAMPLE = 20

# SQL 里直接内联的常量必须是安全字面量; 它来自代码而非用户输入, 这里再钉一次。
assert "'" not in TEMPLATE_SYSTEM_USERNAME


def internal_user_sql(alias: str) -> str:
    """SQL 布尔表达式: users 行 `alias` 是内部账号."""
    return (
        f"({alias}.role = 'admin' OR {alias}.is_internal"
        f" OR {alias}.username = '{TEMPLATE_SYSTEM_USERNAME}')"
    )


def _check(cohort: Cohort) -> bool:
    """True 表示需要过滤; 未知人群直接报错, 不能静默退回全量."""
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}")
    return cohort == "external"


def user_row_population_sql(alias: str, cohort: Cohort) -> str:
    """SQL 布尔表达式: users 行 `alias` 属于该人群."""
    return f"NOT {internal_user_sql(alias)}" if _check(cohort) else "TRUE"


def population_sql(user_id_column: str, cohort: Cohort) -> str:
    """SQL 布尔表达式: `user_id_column` 指向的用户属于该人群.

    用 NOT EXISTS 而不是 JOIN: 调用方的行数和去重语义保持不变。用户行已删除
    (外键为空) 时算外部 —— 与此前的全量统计一致。
    """
    if not _check(cohort):
        return "TRUE"
    return (
        f"NOT EXISTS (SELECT 1 FROM users pop WHERE pop.id = {user_id_column}"
        f" AND {internal_user_sql('pop')})"
    )


def chat_activity_sql(cohort: Cohort) -> str:
    """FROM/WHERE 片段: 每一行是一条计入"聊天活跃"的用户消息 (别名 m / c).

    聊天活跃用户、次日回访、数据监控的日活/时段/句子区间都从这一个片段派生,
    调用方只追加时间条件和分组。
    """
    return (
        "messages m JOIN conversations c ON c.id = m.conversation_id"
        f" WHERE m.role = 'user' AND {population_sql('c.user_id', cohort)}"
    )


def conversation_population_sql(conversation_id_column: str, cohort: Cohort) -> str:
    """SQL 布尔表达式: 会话的拥有者属于该人群 (messages 只有 conversation_id)."""
    if not _check(cohort):
        return "TRUE"
    return (
        "NOT EXISTS (SELECT 1 FROM conversations popc JOIN users pop ON pop.id = popc.user_id"
        f" WHERE popc.id = {conversation_id_column} AND {internal_user_sql('pop')})"
    )


def local_day_bounds(day: date_cls) -> tuple[datetime, datetime]:
    """本地日的 [起, 止), 以库里的 naive UTC 表示.

    库里的时间戳是 `timestamp without time zone` 且按 UTC 存, 直接用带时区的值
    比较会静默偏 8 小时 —— 作息落库那个 bug 就是这么来的。
    """
    tz = ZoneInfo(settings.schedule_timezone)
    start_local = datetime(day.year, day.month, day.day, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
    )


def local_today(now: datetime | None = None) -> date_cls:
    tz = ZoneInfo(settings.schedule_timezone)
    return (now.astimezone(tz) if now else datetime.now(tz)).date()


async def population_fingerprint() -> str:
    """当前内部账号集合的指纹, 用作长期缓存键的一部分.

    有人被勾选为内部 (或改了 admin 角色) 后指纹变化, 已定型日子的旧缓存自然失效,
    不需要在每个改动入口记得清缓存。
    """
    from app.db import db

    rows = await db.query_raw(
        f"""
        SELECT md5(COALESCE(string_agg(id, ',' ORDER BY id), '')) AS fingerprint
        FROM users u WHERE {internal_user_sql('u')}
        """
    )
    return str(rows[0]["fingerprint"])[:12] if rows else "none"


def ratio(numerator: int, denominator: int | None, *, settled: bool = True) -> dict[str, Any]:
    """比率加上它的状态; 页面按 state 渲染, 不自己判断.

    state: no_sample (分母为 0) / unsettled (窗口未走完) / insufficient_sample
    (分母不足 MIN_SAMPLE, 只报分子分母) / ok (value 可以是真实的 0)。
    """
    base = {"numerator": int(numerator or 0), "denominator": int(denominator or 0),
            "value": None, "min_sample": MIN_SAMPLE}
    if not settled or denominator is None:
        return {**base, "state": "unsettled"}
    if not denominator:
        return {**base, "state": "no_sample"}
    if denominator < MIN_SAMPLE:
        return {**base, "state": "insufficient_sample"}
    return {**base, "value": round(numerator / denominator, 4), "state": "ok"}


# 指标字典: 页面悬停说明、日报和派生表都引用这里, 不各写一份。
METRICS: dict[str, dict[str, str]] = {
    "chat_active_users": {
        "label": "聊天活跃用户",
        "definition": "本地日 (UTC+8) 内至少发过 1 条用户消息的用户数",
        "denominator": "人",
        "source": "messages ⋈ conversations",
    },
    "next_day_return": {
        "label": "次日回访率",
        "definition": "D 日活跃的用户中, D+1 本地日仍发过用户消息的比例; D+1 过完才定型",
        "denominator": "D 日活跃用户数",
        "source": "messages ⋈ conversations",
    },
    "continuation": {
        "label": "对话延续率",
        "definition": "AI 回复后 5 分钟内用户再次开口的回合占比",
        "denominator": "有回复的用户回合数",
        "source": "messages",
    },
    "proactive_response": {
        "label": "主动消息回应率",
        "definition": (
            "一次主动发送算一个事件 (按 proactive_event_id 去重, 卡片和文字只算一次),"
            " 发出后 6 小时内用户开口即算回应; 无事件 ID 的旧消息按条计"
        ),
        "denominator": "主动发送事件数",
        "source": "messages.metadata.proactive_event_id",
    },
}


def describe(keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    """供接口随数据一起返回的定义说明."""
    selected = keys or tuple(METRICS)
    return {
        "definition_version": DEFINITION_VERSION,
        "min_sample": MIN_SAMPLE,
        "internal_rule": "role=admin、后台勾选的内部账号、模板系统账号",
        "metrics": {key: METRICS[key] for key in selected},
    }
