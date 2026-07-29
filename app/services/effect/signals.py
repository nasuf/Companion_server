"""从已落库的行为数据算出"这些回复接得住吗".

不要求用户点赞: 显式反馈的采样率低到没有统计意义, 而且打断沉浸感。用行为信号 ——
用户有没有接着说话、隔多久说、一次聊几轮、主动搭话理不理、第二天还回不回来。

**热路径零改动**: 全部指标是对 messages / conversations / proactive_chat_logs 的
只读聚合, 由每日 cron 触发。不新增 LLM 调用, 不在回复链路上加任何 await。

时间窗的取法有依据。实测 (2026-07, 685 个回合) 「AI 回复 → 用户下一条」的间隔:

    中位数 17 秒；2 分钟内 77%，5 分钟 82%，15 分钟 84%，1 小时 87%，3 小时 89%

曲线在 2 分钟处就基本走平, 窗口越大越接近饱和、越不敏感。取 5 分钟: 过了最陡段,
又没到失去分辨力的地方。同时报间隔中位数 —— 连续量比一个接近饱和的比率更能反映
细微变化。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings

logger = logging.getLogger(__name__)

# "这条回复接住了" 的判定窗口。取值依据见模块 docstring。
CONTINUATION_WINDOW = timedelta(minutes=5)

# 会话边界。跟话题重置 / 重逢感知同线 (topic.TOPIC_RESET_GAP_SECONDS),
# 三处用同一个语义: 隔了这么久再开口, 算新的一段对话。
SESSION_GAP = timedelta(hours=3)

# 主动消息发出后, 多久之内回应算"理了"。比聊天中的 5 分钟宽得多 —— 主动消息本就
# 是在用户没在看的时候发的, 用聊天的尺度衡量等于必然判负。
PROACTIVE_RESPONSE_WINDOW = timedelta(hours=6)

# 按 response_diagnostics 切片的维度。只放取值有限且语义稳定的键 —— 高基数字段
# 切出来每格样本太少, 比率会剧烈抖动, 反而误导。
SLICE_DIMENSIONS = (
    "reply_path",
    "memory_relevance",
    "needs_web_search",
    "reply_emotion_source",
)

# 一个切片至少要有这么多回合才报比率。低于此只报样本数 —— 3 个回合里 2 个延续
# 是 67%, 跟 300 个回合里 200 个延续的 67% 完全不是一回事。
MIN_SLICE_TURNS = 20

# 跨日界往后多看一段, 供 LEAD 取"下一条用户消息"。要覆盖「回复耗时 + 延续窗口」,
# 1 小时留足余量 (回复 p90 约 9 秒, 窗口 5 分钟)。
_BOUNDARY_LOOKAHEAD = timedelta(hours=1)


@dataclass
class SliceMetric:
    dimension: str
    value: str
    turns: int
    continued: int
    median_gap_s: int | None = None

    @property
    def continuation_rate(self) -> float | None:
        if self.turns < MIN_SLICE_TURNS:
            return None
        return round(self.continued / self.turns, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "continuation_rate": self.continuation_rate,
            "sufficient_sample": self.turns >= MIN_SLICE_TURNS,
        }


@dataclass
class EffectMetrics:
    """某一天的效果快照."""

    date: str
    # 对话延续
    turns: int = 0
    continued: int = 0
    median_gap_s: int | None = None
    # 会话深度
    sessions: int = 0
    median_session_turns: float | None = None
    # 主动交流
    proactive_sent: int = 0
    proactive_answered: int = 0
    # 回访
    active_users: int = 0
    returned_next_day: int | None = None      # 次日数据未到齐时为 None
    # 切片
    slices: list[SliceMetric] = field(default_factory=list)

    @property
    def continuation_rate(self) -> float | None:
        return round(self.continued / self.turns, 4) if self.turns else None

    @property
    def proactive_response_rate(self) -> float | None:
        if not self.proactive_sent:
            return None
        return round(self.proactive_answered / self.proactive_sent, 4)

    @property
    def next_day_return_rate(self) -> float | None:
        if not self.active_users or self.returned_next_day is None:
            return None
        return round(self.returned_next_day / self.active_users, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "turns": self.turns,
            "continued": self.continued,
            "continuation_rate": self.continuation_rate,
            "median_gap_s": self.median_gap_s,
            "sessions": self.sessions,
            "median_session_turns": self.median_session_turns,
            "proactive_sent": self.proactive_sent,
            "proactive_answered": self.proactive_answered,
            "proactive_response_rate": self.proactive_response_rate,
            "active_users": self.active_users,
            "returned_next_day": self.returned_next_day,
            "next_day_return_rate": self.next_day_return_rate,
            "slices": [s.as_dict() for s in self.slices],
        }


def _day_bounds(day: date_cls) -> tuple[datetime, datetime]:
    """本地日的 [起, 止)。

    naive datetime: 库里的时间戳是 `timestamp without time zone` 且按 UTC 存,
    直接用带时区的值比较会静默偏 8 小时 —— 作息落库那个 bug 就是这么来的。
    """
    tz = ZoneInfo(settings.schedule_timezone)
    start_local = datetime(day.year, day.month, day.day, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
    )


# 每个用户回合一行: 何时问的、何时回的、下一次开口在何时、这次回复的生成条件。
# 所有指标都从这一份派生, 避免每个指标各写一套"什么算一个回合"。
_TURNS_CTE = """
WITH asked AS (
    SELECT
        m.conversation_id,
        m.created_at AS asked_at,
        LEAD(m.created_at) OVER (
            PARTITION BY m.conversation_id ORDER BY m.created_at
        ) AS next_ask_at,
        (m.created_at < $2::timestamp) AS in_window
    FROM messages m
    WHERE m.role = 'user'
      AND m.created_at >= $1::timestamp
      -- 多看一段到次日: LEAD 只在窗口内取值的话, 每天最后一个回合的 next_ask
      -- 必为 NULL, 于是被系统性判成"没接住" —— 哪怕用户过了两分钟就回了, 只是
      -- 那两分钟跨过了午夜。这些多出来的行只用于提供 LEAD 值, 下面按 in_window
      -- 过滤掉。
      AND m.created_at <  $2::timestamp + $4::interval
),
turns AS (
    SELECT
        a.conversation_id,
        a.asked_at,
        a.next_ask_at,
        r.created_at AS replied_at,
        r.metadata -> 'response_diagnostics' AS diag
    FROM asked a
    LEFT JOIN LATERAL (
        SELECT created_at, metadata
        FROM messages x
        WHERE x.conversation_id = a.conversation_id
          AND x.role = 'assistant'
          AND x.created_at > a.asked_at
        ORDER BY x.created_at
        LIMIT 1
    ) r ON TRUE
    WHERE a.in_window
),
scored AS (
    SELECT
        t.*,
        -- 用户连发多条 (碎片聚合) 时 next_ask 可能早于回复, 那不算"被回复接住"。
        (t.next_ask_at IS NOT NULL
         AND t.replied_at IS NOT NULL
         AND t.next_ask_at > t.replied_at
         AND t.next_ask_at - t.replied_at <= $3::interval) AS continued,
        CASE
            WHEN t.replied_at IS NOT NULL
             AND t.next_ask_at IS NOT NULL
             AND t.next_ask_at > t.replied_at
            THEN EXTRACT(EPOCH FROM (t.next_ask_at - t.replied_at))
        END AS gap_s
    FROM turns t
    WHERE t.replied_at IS NOT NULL
)
"""


async def _fetch_core(start: datetime, end: datetime) -> dict[str, Any]:
    from app.db import db

    rows = await db.query_raw(
        _TURNS_CTE + """
        SELECT
            COUNT(*)::int AS turns,
            COUNT(*) FILTER (WHERE continued)::int AS continued,
            percentile_disc(0.5) WITHIN GROUP (ORDER BY gap_s)::int AS median_gap_s
        FROM scored
        """,
        start.isoformat(), end.isoformat(),
        f"{int(CONTINUATION_WINDOW.total_seconds())} seconds",
        f"{int(_BOUNDARY_LOOKAHEAD.total_seconds())} seconds",
    )
    return rows[0] if rows else {}


async def _fetch_slices(start: datetime, end: datetime) -> list[SliceMetric]:
    from app.db import db

    out: list[SliceMetric] = []
    for dim in SLICE_DIMENSIONS:
        rows = await db.query_raw(
            _TURNS_CTE + f"""
            SELECT
                diag ->> '{dim}' AS value,
                COUNT(*)::int AS turns,
                COUNT(*) FILTER (WHERE continued)::int AS continued,
                percentile_disc(0.5) WITHIN GROUP (ORDER BY gap_s)::int AS median_gap_s
            FROM scored
            -- 键存在但值是 JSON null 时 ->> 给出 SQL NULL, 切出来会是一个叫
            -- "None" 的格子。它不携带任何分组信息, 只会占位误导。
            WHERE diag ->> '{dim}' IS NOT NULL
            GROUP BY 1
            ORDER BY turns DESC
            """,
            start.isoformat(), end.isoformat(),
            f"{int(CONTINUATION_WINDOW.total_seconds())} seconds",
            f"{int(_BOUNDARY_LOOKAHEAD.total_seconds())} seconds",
        )
        for r in rows:
            out.append(SliceMetric(
                dimension=dim,
                value=str(r.get("value")),
                turns=int(r.get("turns") or 0),
                continued=int(r.get("continued") or 0),
                median_gap_s=r.get("median_gap_s"),
            ))
    return out


async def _fetch_sessions(start: datetime, end: datetime) -> dict[str, Any]:
    """会话数与每次会话的轮数中位数。会话边界 = 用户消息间隔超过 SESSION_GAP."""
    from app.db import db

    rows = await db.query_raw(
        """
        WITH u AS (
            SELECT conversation_id, created_at,
                   created_at - LAG(created_at) OVER (
                       PARTITION BY conversation_id ORDER BY created_at
                   ) AS since_prev
            FROM messages
            WHERE role = 'user'
              AND created_at >= $1::timestamp
              AND created_at <  $2::timestamp
        ),
        marked AS (
            SELECT conversation_id,
                   SUM(CASE WHEN since_prev IS NULL OR since_prev > $3::interval
                            THEN 1 ELSE 0 END)
                       OVER (PARTITION BY conversation_id ORDER BY created_at
                             ROWS UNBOUNDED PRECEDING) AS session_no
            FROM u
        ),
        per_session AS (
            SELECT conversation_id, session_no, COUNT(*)::int AS turns
            FROM marked GROUP BY 1, 2
        )
        SELECT COUNT(*)::int AS sessions,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY turns)::float8 AS median_turns
        FROM per_session
        """,
        start.isoformat(), end.isoformat(), f"{int(SESSION_GAP.total_seconds())} seconds",
    )
    return rows[0] if rows else {}


async def _fetch_proactive(start: datetime, end: datetime) -> dict[str, Any]:
    """主动消息发出后, 用户在窗口内有没有开口."""
    from app.db import db

    rows = await db.query_raw(
        """
        WITH sent AS (
            SELECT conversation_id, created_at
            FROM messages
            WHERE role = 'assistant'
              AND metadata ? 'proactive'
              AND created_at >= $1::timestamp
              AND created_at <  $2::timestamp
        )
        SELECT
            COUNT(*)::int AS sent,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM messages u
                WHERE u.conversation_id = s.conversation_id
                  AND u.role = 'user'
                  AND u.created_at > s.created_at
                  AND u.created_at <= s.created_at + $3::interval
            ))::int AS answered
        FROM sent s
        """,
        start.isoformat(), end.isoformat(),
        f"{int(PROACTIVE_RESPONSE_WINDOW.total_seconds())} seconds",
    )
    return rows[0] if rows else {}


async def _fetch_retention(start: datetime, end: datetime) -> dict[str, Any]:
    """当日活跃用户, 以及其中次日还回来的.

    次日窗口尚未走完时 returned 会偏低 —— 调用方据此判断要不要展示 (见 collect)。
    """
    from app.db import db

    rows = await db.query_raw(
        """
        WITH active AS (
            SELECT DISTINCT c.user_id
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE m.role = 'user'
              AND m.created_at >= $1::timestamp AND m.created_at < $2::timestamp
        ),
        nextday AS (
            SELECT DISTINCT c.user_id
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE m.role = 'user'
              AND m.created_at >= $2::timestamp
              AND m.created_at <  $2::timestamp + INTERVAL '1 day'
        )
        SELECT (SELECT COUNT(*) FROM active)::int AS active_users,
               (SELECT COUNT(*) FROM active a JOIN nextday n ON n.user_id = a.user_id)::int
                   AS returned
        """,
        start.isoformat(), end.isoformat(),
    )
    return rows[0] if rows else {}


async def collect(day: date_cls, *, now: datetime | None = None) -> EffectMetrics:
    """算出某一天的效果快照。单项失败不影响其余 —— 半份数据也比没有强."""
    start, end = _day_bounds(day)
    metrics = EffectMetrics(date=day.isoformat())

    try:
        core = await _fetch_core(start, end)
        metrics.turns = int(core.get("turns") or 0)
        metrics.continued = int(core.get("continued") or 0)
        metrics.median_gap_s = core.get("median_gap_s")
    except Exception as exc:
        logger.warning(f"effect: core metrics failed for {day}: {exc}")

    try:
        metrics.slices = await _fetch_slices(start, end)
    except Exception as exc:
        logger.warning(f"effect: slice metrics failed for {day}: {exc}")

    try:
        sess = await _fetch_sessions(start, end)
        metrics.sessions = int(sess.get("sessions") or 0)
        metrics.median_session_turns = sess.get("median_turns")
    except Exception as exc:
        logger.warning(f"effect: session metrics failed for {day}: {exc}")

    try:
        pro = await _fetch_proactive(start, end)
        metrics.proactive_sent = int(pro.get("sent") or 0)
        metrics.proactive_answered = int(pro.get("answered") or 0)
    except Exception as exc:
        logger.warning(f"effect: proactive metrics failed for {day}: {exc}")

    try:
        ret = await _fetch_retention(start, end)
        metrics.active_users = int(ret.get("active_users") or 0)
        # 次日还没过完就报回访率, 会得到一个必然偏低且每小时都在变的数。
        reference = now or datetime.now(ZoneInfo(settings.schedule_timezone))
        next_day_over = reference.astimezone(ZoneInfo("UTC")).replace(tzinfo=None) >= (
            end + timedelta(days=1)
        )
        metrics.returned_next_day = int(ret.get("returned") or 0) if next_day_over else None
    except Exception as exc:
        logger.warning(f"effect: retention metrics failed for {day}: {exc}")

    return metrics
