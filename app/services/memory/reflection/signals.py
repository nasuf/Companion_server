"""从互动历史里算出**确定性**的行为事实, 供反思使用.

## 为什么把这一层单独拆出来

反思最大的风险不是模型胡说, 是模型**忠实地总结了错误的数据**。设计这套东西的时候
我自己就踩了一次: `messages.created_at` 是 `timestamp without time zone` 存 UTC,
`AT TIME ZONE 'Asia/Shanghai'` 会把它**减** 8 小时。据此算出的活跃时段是"凌晨 5 点
是峰值", 真实是 21 点。要是这个数字喂给 LLM, 它会产出"用户总在凌晨找我, 可能失眠",
写进记忆永久生效, AI 从此照着它说话 —— 而整条链路不会报任何错。

所以事实计算完全不含 LLM, 每条都能用构造数据断言精确值。LLM 只在下一层做归纳,
且只看这里产出的事实, 看不到原始消息。

## 三条硬约束

    时区    一律用 UTC+8 的固定偏移常量, 禁止在 SQL 里写 AT TIME ZONE
            (有静态测试扫这个字面量)
    样本量  低于下限直接不产出, 而不是产出一个基于三条数据的"趋势"
    证据    每条事实带样本量和取值区间, 下游可追溯、可复核
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from app.db import db

logger = logging.getLogger(__name__)

# 数据库里的时间戳是不带时区的 UTC。转本地时间只用这一个偏移量, 不要在 SQL 里写
# AT TIME ZONE —— 对 `timestamp without time zone` 它的方向跟直觉相反 (会减而不是
# 加), 而算错的时段会变成一条"用户总在凌晨找我"的永久记忆。
LOCAL_UTC_OFFSET_HOURS = 8

# 样本量下限。低于它就不产出对应事实 —— 三条消息看不出"作息", 五次对话看不出"趋势"。
MIN_MESSAGES_FOR_TIMING = 20
MIN_MESSAGES_FOR_EMOTION = 15
MIN_DAYS_FOR_RHYTHM = 5
MIN_PROACTIVE_FOR_RESPONSE_RATE = 4

# 主动消息发出后多久之内的用户消息算"回应了"。取 6 小时而不是几分钟: 用户可能在
# 忙, 隔几小时回来接上话仍然是回应; 但隔一天再来就是新的一次对话了。
PROACTIVE_REPLY_WINDOW_HOURS = 6

# 一次反思最多看多久的历史。太长会把早已过去的状态当成"现在的他"。
DEFAULT_WINDOW_DAYS = 14


@dataclass(frozen=True)
class BehaviouralFact:
    """一条可追溯的行为事实。

    `statement` 是给 LLM 看的自然语言; `evidence` 是它的出处, 用于人工复核和
    下游的"引用不上就丢弃"检查。
    """

    key: str
    statement: str
    sample_size: int
    evidence: dict = field(default_factory=dict)


def _local_hour_expr(column: str) -> str:
    """把 UTC 列换算成本地小时。

    刻意用加固定偏移而不是 `AT TIME ZONE`: 后者对 `timestamp without time zone`
    的语义是"把这个值当作该时区的时间, 转成 UTC", 方向跟这里要的正好相反。
    """
    return (
        f"EXTRACT(HOUR FROM {column} + INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours')::int"
    )


def _local_date_expr(column: str) -> str:
    return f"({column} + INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours')::date"


def _as_date(value) -> date | None:
    """query_raw 取回 date 列时给的是字符串, 需要显式转。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _describe_hour_band(hour: int) -> str:
    if hour < 6:
        return "凌晨"
    if hour < 9:
        return "清晨"
    if hour < 12:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    if hour < 23:
        return "晚上"
    return "深夜"


async def _timing_fact(
    user_id: str, agent_id: str, since: datetime, workspace_id: str | None,
) -> BehaviouralFact | None:
    """用户通常在一天的什么时候来说话。"""
    rows = await db.query_raw(
        f"""
        SELECT {_local_hour_expr('m.created_at')} AS hour, COUNT(*)::int AS n
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1 AND c.agent_id = $2
          AND c.workspace_id IS NOT DISTINCT FROM $4
          AND c.is_deleted = false
          AND m.role = 'user' AND m.created_at >= $3::timestamp
        GROUP BY 1 ORDER BY n DESC
        """,
        user_id, agent_id, since, workspace_id,
    )
    total = sum(r["n"] for r in rows)
    if total < MIN_MESSAGES_FOR_TIMING:
        return None

    top = rows[:3]
    covered = sum(r["n"] for r in top)
    bands = sorted({_describe_hour_band(r["hour"]) for r in top})
    hours = sorted(r["hour"] for r in top)
    return BehaviouralFact(
        key="timing",
        statement=(
            f"最近 {total} 条消息里有 {covered} 条集中在 "
            f"{'、'.join(f'{h}点' for h in hours)}（{'/'.join(bands)}）"
        ),
        sample_size=total,
        evidence={"top_hours": hours, "covered": covered, "total": total},
    )


async def _emotion_fact(
    user_id: str, agent_id: str, since: datetime, workspace_id: str | None,
) -> BehaviouralFact | None:
    """情绪构成。用逐条持久化的 12 类标签, 不重新推断。"""
    rows = await db.query_raw(
        """
        SELECT m.metadata->'emotion'->>'emotion' AS emotion, COUNT(*)::int AS n
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1 AND c.agent_id = $2
          AND c.workspace_id IS NOT DISTINCT FROM $4
          AND c.is_deleted = false
          AND m.role = 'user' AND m.created_at >= $3::timestamp
          AND m.metadata->'emotion'->>'emotion' IS NOT NULL
        GROUP BY 1 ORDER BY n DESC
        """,
        user_id, agent_id, since, workspace_id,
    )
    total = sum(r["n"] for r in rows)
    if total < MIN_MESSAGES_FOR_EMOTION:
        return None

    # 中性占绝大多数是常态, 单独列出来没有信息量; 真正有意思的是非中性的构成。
    non_neutral = [r for r in rows if r["emotion"] != "中性"]
    if not non_neutral:
        return BehaviouralFact(
            key="emotion",
            statement=f"最近 {total} 条消息的情绪几乎都是中性",
            sample_size=total,
            evidence={"total": total, "non_neutral": 0},
        )

    shown = non_neutral[:3]
    parts = [f"{r['emotion']} {r['n']} 次" for r in shown]
    return BehaviouralFact(
        key="emotion",
        statement=(
            f"最近 {total} 条消息里, 非中性情绪有 "
            f"{sum(r['n'] for r in non_neutral)} 条, 主要是 {'、'.join(parts)}"
        ),
        sample_size=total,
        evidence={
            "total": total,
            "distribution": {r["emotion"]: r["n"] for r in non_neutral},
        },
    )


async def _rhythm_fact(
    user_id: str, agent_id: str, since: datetime, workspace_id: str | None,
) -> BehaviouralFact | None:
    """来的频率与连续性 —— 天天来还是隔很久来一次。"""
    rows = await db.query_raw(
        f"""
        SELECT {_local_date_expr('m.created_at')} AS day, COUNT(*)::int AS n
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1 AND c.agent_id = $2
          AND c.workspace_id IS NOT DISTINCT FROM $4
          AND c.is_deleted = false
          AND m.role = 'user' AND m.created_at >= $3::timestamp
        GROUP BY 1 ORDER BY 1
        """,
        user_id, agent_id, since, workspace_id,
    )
    if len(rows) < MIN_DAYS_FOR_RHYTHM:
        return None

    # query_raw 把 date 列取回来是字符串, 直接相减会抛 TypeError。初版就是这样,
    # 而调用方吞异常, 于是这条事实一直静默缺席。
    days = [_as_date(r["day"]) for r in rows]
    if any(d is None for d in days):
        return None
    span = (days[-1] - days[0]).days + 1
    active = len(days)
    biggest_gap = max(
        ((days[i + 1] - days[i]).days for i in range(len(days) - 1)), default=0,
    )
    per_active_day = sum(r["n"] for r in rows) / active

    statement = (
        f"{span} 天里有 {active} 天来说过话, 平均每次聊 {per_active_day:.0f} 条"
    )
    if biggest_gap >= 3:
        statement += f"; 中间最长断过 {biggest_gap} 天"
    return BehaviouralFact(
        key="rhythm",
        statement=statement,
        sample_size=active,
        evidence={
            "span_days": span, "active_days": active,
            "biggest_gap_days": biggest_gap,
        },
    )


async def _proactive_response_fact(
    user_id: str, agent_id: str, since: datetime, workspace_id: str | None,
) -> BehaviouralFact | None:
    """AI 主动搭话时, 用户理不理。

    回应必须从消息流里判断: 一条标了 proactive 的 AI 消息之后, 窗口内有没有用户
    消息。**不能**数 proactive_event_logs 的 user_replied —— 那个事件是每条用户
    消息都会写的 (mark_user_replied_for_conversation 挂在 ws/chat 的入口上, 用来
    重置沉默衰减), 跟主动消息没有对应关系。

    初版就是这么错的, 在生产数据上算出"我主动找了 4 次, 他回了 102 次"。数字荒谬
    所以一眼看穿, 但如果比例落在合理区间, 它会变成一条"用户对我的主动搭话很热情"
    的洞见写进记忆 —— 完全虚假, 且没有任何报错。
    """
    rows = await db.query_raw(
        f"""
        WITH proactive AS (
          SELECT m.id, m.created_at
          FROM messages m
          JOIN conversations c ON c.id = m.conversation_id
          WHERE c.user_id = $1 AND c.agent_id = $2
            AND c.workspace_id IS NOT DISTINCT FROM $4
            AND c.is_deleted = false
            AND m.role = 'assistant'
            AND (m.metadata->>'proactive')::boolean IS TRUE
            AND m.created_at >= $3::timestamp
        )
        SELECT
          COUNT(*)::int AS sent,
          COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM messages r
            JOIN conversations rc ON rc.id = r.conversation_id
            WHERE rc.user_id = $1 AND rc.agent_id = $2
              AND rc.workspace_id IS NOT DISTINCT FROM $4
              AND rc.is_deleted = false AND r.role = 'user'
              AND r.created_at > proactive.created_at
              AND r.created_at <= proactive.created_at
                  + INTERVAL '{PROACTIVE_REPLY_WINDOW_HOURS} hours'
          ))::int AS answered
        FROM proactive
        """,
        user_id, agent_id, since, workspace_id,
    )
    if not rows:
        return None
    sent = int(rows[0].get("sent") or 0)
    answered = int(rows[0].get("answered") or 0)
    if sent < MIN_PROACTIVE_FOR_RESPONSE_RATE:
        return None
    if answered > sent:
        # 自洽性兜底: 回应数不可能超过发送数。真出现了说明查询语义又错了,
        # 宁可少一条事实, 也不要拿它去归纳。
        logger.error(
            f"proactive response fact inconsistent (answered={answered} > sent={sent}); "
            "dropping"
        )
        return None

    return BehaviouralFact(
        key="proactive_response",
        statement=(
            f"我主动找了 {sent} 次, 其中 {answered} 次他在 "
            f"{PROACTIVE_REPLY_WINDOW_HOURS} 小时内回了"
        ),
        sample_size=sent,
        evidence={"sent": sent, "answered": answered},
    )


async def _length_fact(
    user_id: str, agent_id: str, since: datetime, workspace_id: str | None,
) -> BehaviouralFact | None:
    """消息长度 —— 一句话打发还是愿意展开。"""
    rows = await db.query_raw(
        """
        SELECT COUNT(*)::int AS n,
               AVG(LENGTH(m.content))::float AS avg_len,
               COUNT(*) FILTER (WHERE LENGTH(m.content) <= 4)::int AS very_short
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1 AND c.agent_id = $2
          AND c.workspace_id IS NOT DISTINCT FROM $4
          AND c.is_deleted = false
          AND m.role = 'user' AND m.created_at >= $3::timestamp
          AND m.content IS NOT NULL
        """,
        user_id, agent_id, since, workspace_id,
    )
    if not rows or int(rows[0].get("n") or 0) < MIN_MESSAGES_FOR_TIMING:
        return None
    row = rows[0]
    total = int(row["n"])
    avg_len = float(row["avg_len"] or 0)
    very_short = int(row["very_short"] or 0)
    return BehaviouralFact(
        key="message_length",
        statement=(
            f"平均每条消息 {avg_len:.0f} 字, 其中 {very_short} 条在 4 字以内"
        ),
        sample_size=total,
        evidence={"avg_length": round(avg_len, 1),
                  "very_short": very_short, "total": total},
    )


async def collect_behavioural_facts(
    *, user_id: str, agent_id: str, workspace_id: str | None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[BehaviouralFact]:
    """算出这段时间里可验证的行为事实。样本量不足的项会缺席, 不会被编出来。

    workspace_id 刻意不给默认值。给了 None 作默认的话, 漏传就变成"只匹配
    workspace 为 NULL 的会话" —— 结果是静默返回空, 而不是报错。开发时就这么中过
    一次: 预览脚本没传, 所有用户的事实全部归零, 看起来像"数据量还不够"。

    按 workspace 收口: 会话是按 workspace 分的, 而洞见只会写进其中一个。不过滤的话
    统计会把别的 workspace (比如重建 agent 前的旧会话) 的消息混进来, 得出的"他最近
    很活跃"其实说的是另一段关系。同时排除软删除的会话 —— 用户删掉的对话不该继续
    影响 AI 对他的判断。

    单项失败不影响其余 —— 少一条事实只是让归纳少一点依据, 而让整轮反思因为一个
    聚合查询出错就失败是不值得的。
    """
    since = datetime.now(UTC) - timedelta(days=window_days)
    producers = (
        _timing_fact, _emotion_fact, _rhythm_fact,
        _proactive_response_fact, _length_fact,
    )

    facts: list[BehaviouralFact] = []
    for produce in producers:
        try:
            fact = await produce(user_id, agent_id, since, workspace_id)
        except Exception as e:
            logger.warning(f"behavioural fact {produce.__name__} failed: {e}")
            continue
        if fact is not None:
            facts.append(fact)
    return facts


def format_facts_for_prompt(facts: list[BehaviouralFact]) -> str:
    """渲染成给 LLM 的清单。带编号是为了让它能引用, 引用不上的洞见会被丢弃。"""
    return "\n".join(
        f"[{i + 1}] {fact.statement}" for i, fact in enumerate(facts)
    )
