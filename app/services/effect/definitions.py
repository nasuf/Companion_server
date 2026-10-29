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

# v3: a recorded model fee stays priced after the model leaves model_registry.
DEFINITION_VERSION = "beta-metrics-v3"

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


async def population_fingerprint(client: Any = None) -> str:
    """当前内部账号集合的指纹, 用作长期缓存键的一部分.

    有人被勾选为内部 (或改了 admin 角色) 后指纹变化, 已定型日子的旧缓存自然失效,
    不需要在每个改动入口记得清缓存。`client` 可传事务, 在同一快照里读。
    """
    if client is None:
        from app.db import db as client

    rows = await client.query_raw(
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


def proactive_events_sql(start: str, end: str, reply_window: str, cohort: Cohort) -> str:
    """主动发送事件的唯一 SQL: [start, end) 内发出的事件, 每行一个.

    一次发送算一个事件: 同一 proactive_event_id 的卡片和文字只算一次, 以最早一条
    的时间为发出时间, 并按这一条归日 (卡片和文字跨过午夜时不会两天各算一次)。同一
    ID 在前一天内已发过的 (重试) 不再算新事件。早于事件 ID 上线的旧消息没有 ID,
    只能按条计 (identified=false)。first_reply_at 是发出后 reply_window 内的第一条
    用户消息, 没有则为 NULL; 更短的回应窗口由调用方在它上面再判。

    列: user_id, conversation_id, event_key, trigger_type, identified, bubbles,
    sent_at, first_reply_at。回复效果页与每日派生表都从这里取数。
    """
    return f"""
        SELECT e.*, (
            SELECT MIN(r.created_at) FROM messages r
            WHERE r.conversation_id = e.conversation_id AND r.role = 'user'
              AND r.created_at > e.sent_at AND r.created_at <= e.sent_at + {reply_window}
        ) AS first_reply_at
        FROM (
            SELECT c.user_id, b.conversation_id,
                   COALESCE(b.metadata->>'proactive_event_id', 'legacy:' || b.id) AS event_key,
                   -- 卡片的 trigger_type 带 _card 后缀, MIN 取到文字那条的类型
                   MIN(COALESCE(b.metadata->>'trigger_type', 'unknown')) AS trigger_type,
                   bool_or(b.metadata->>'proactive_event_id' IS NOT NULL) AS identified,
                   COUNT(*)::int AS bubbles, MIN(b.created_at) AS sent_at
            FROM messages b JOIN conversations c ON c.id = b.conversation_id
            WHERE b.role = 'assistant' AND b.metadata->>'proactive' = 'true'
              AND {population_sql('c.user_id', cohort)}
              AND b.created_at >= {start} - INTERVAL '1 hour' AND b.created_at < {end} + INTERVAL '1 hour'
            GROUP BY c.user_id, b.conversation_id,
                     COALESCE(b.metadata->>'proactive_event_id', 'legacy:' || b.id)
        ) e
        WHERE e.sent_at >= {start} AND e.sent_at < {end}
          AND (NOT e.identified OR NOT EXISTS (
              SELECT 1 FROM messages p
              WHERE p.conversation_id = e.conversation_id AND p.role = 'assistant'
                AND p.created_at >= {start} - INTERVAL '1 day' AND p.created_at < {start} - INTERVAL '1 hour'
                AND p.metadata->>'proactive_event_id' = e.event_key))
    """


# 每日派生指标的类型, 决定页面怎么画、样本门槛怎么用:
#   count     计数, value 即人数/条数; 0 是真实为零
#   rate      比率 numerator/denominator (0..1); 分母少于 MIN_SAMPLE 或未定型时不给值,
#             只给分子分母 —— 小样本的百分比会被读成差异
#   mean      人均值 numerator/denominator; 分母为 0 时不给值, 其余照给并附分母
#   quantile  分位数 (毫秒), sample_size 为样本数; 少于 MIN_SAMPLE 不给值
# settle_after_s: 本地日结束后再过多久这个数不再变化 (定型)。
_H = 3600

# 无回复判定窗口: 用户一串消息的最后一条之后多久没有 AI 回复算"没回"。
NO_REPLY_WINDOW_S = 600

# 主动消息短回应窗口 (秒); 长窗口沿用 effect.signals.PROACTIVE_RESPONSE_WINDOW (6 小时)。
PROACTIVE_SHORT_WINDOW_S = 3 * _H

# 指标字典: 页面悬停说明、日报和派生表都引用这里, 不各写一份。
METRICS: dict[str, dict[str, Any]] = {
    "chat_active_users": {
        "label": "聊天活跃用户",
        "definition": "本地日 (UTC+8) 内至少发过 1 条用户消息的用户数",
        "denominator": "人",
        "source": "messages ⋈ conversations",
        "kind": "count", "unit": "人", "settle_after_s": 0,
    },
    "opened_users": {
        "label": "打开 App 的用户",
        "definition": "本地日内有过一次登录/心跳记录的用户数 (user_daily_activity)",
        "denominator": "人",
        "source": "user_daily_activity",
        "kind": "count", "unit": "人", "settle_after_s": 0, "collected_since": "2026-05-29",
    },
    "new_users": {
        "label": "新注册用户",
        "definition": "注册时间落在该本地日的用户数",
        "denominator": "人",
        "source": "users.created_at",
        "kind": "count", "unit": "人", "settle_after_s": 0,
    },
    "user_messages": {
        "label": "用户消息数",
        "definition": "本地日内用户发出的消息条数",
        "denominator": "条",
        "source": "messages",
        "kind": "count", "unit": "条", "settle_after_s": 0,
    },
    "turns": {
        "label": "回合数",
        "definition": "用户一串连续消息在 10 分钟内得到 AI 的非主动回复, 算 1 个回合; 不含人工接管与游戏中",
        "denominator": "回合",
        "source": "messages",
        "kind": "count", "unit": "回合", "settle_after_s": NO_REPLY_WINDOW_S,
    },
    "turns_per_active_user": {
        "label": "人均日回合",
        "definition": "回合数 / 聊天活跃用户数",
        "denominator": "聊天活跃用户数",
        "source": "user_daily_facts",
        "kind": "mean", "unit": "回合/人", "settle_after_s": NO_REPLY_WINDOW_S,
    },
    "next_day_return": {
        "label": "次日回访率",
        "definition": "D 日活跃的用户中, D+1 本地日仍发过用户消息的比例; D+1 过完才定型",
        "denominator": "D 日活跃用户数",
        "source": "messages ⋈ conversations",
        "kind": "rate", "unit": "%", "settle_after_s": 24 * _H,
    },
    "continuation": {
        "label": "对话延续率",
        "definition": "AI 回复后 5 分钟内用户再次开口的占比; 按每条得到回复的用户消息计 (与回合口径不同)",
        "denominator": "得到 AI 回复的用户消息数",
        "source": "messages",
        "kind": "rate", "unit": "%", "settle_after_s": _H,
    },
    "proactive_events": {
        "label": "主动发送事件",
        "definition": "一次主动发送算一个事件 (卡片和文字只算一次); 无事件 ID 的旧消息按条计",
        "denominator": "事件",
        "source": "messages.metadata.proactive_event_id",
        "kind": "count", "unit": "事件", "settle_after_s": 0,
    },
    "proactive_response": {
        "label": "主动消息回应率",
        "definition": (
            "一次主动发送算一个事件 (按 proactive_event_id 去重, 卡片和文字只算一次),"
            " 发出后 6 小时内用户开口即算回应; 无事件 ID 的旧消息按条计"
        ),
        "denominator": "主动发送事件数",
        "source": "messages.metadata.proactive_event_id",
        "kind": "rate", "unit": "%", "settle_after_s": 6 * _H,
    },
    "proactive_response_3h": {
        "label": "主动消息 3 小时回应率",
        "definition": "同主动消息回应率, 回应窗口为发出后 3 小时",
        "denominator": "主动发送事件数",
        "source": "messages.metadata.proactive_event_id",
        "kind": "rate", "unit": "%", "settle_after_s": PROACTIVE_SHORT_WINDOW_S,
    },
    "no_reply_10m": {
        "label": "无回复率",
        "definition": (
            "用户一串消息的最后一条之后 10 分钟内没有 AI 的非主动回复;"
            " 排除人工接管期间和游戏中的消息"
        ),
        "denominator": "用户消息串",
        "source": "messages",
        "kind": "rate", "unit": "%", "settle_after_s": NO_REPLY_WINDOW_S,
    },
    "reply_first_bubble_ms_p50": {
        "label": "首气泡耗时 p50",
        "definition": (
            "服务端收到一串用户消息的最后一条 → 首个回复气泡就绪 (server-turn-timing-v1),"
            " 含排队与刻意延迟; 不是客户端收到时间"
        ),
        "denominator": "带计时的回合",
        "source": "messages.metadata.response_diagnostics",
        "kind": "quantile", "unit": "ms", "settle_after_s": 0, "collected_since": "2026-10-27",
    },
    "reply_first_bubble_ms_p90": {
        "label": "首气泡耗时 p90",
        "definition": "同首气泡耗时 p50, 取第 90 百分位",
        "denominator": "带计时的回合",
        "source": "messages.metadata.response_diagnostics",
        "kind": "quantile", "unit": "ms", "settle_after_s": 0, "collected_since": "2026-10-27",
    },
    "reply_processing_ms_p50": {
        "label": "回复处理耗时 p50",
        "definition": "生成至首气泡减去回复前拟人停顿 (generation_ms − in_graph_delay_ms), 不含排队与刻意延迟",
        "denominator": "带计时的回合",
        "source": "messages.metadata.response_diagnostics",
        "kind": "quantile", "unit": "ms", "settle_after_s": 0, "collected_since": "2026-10-28",
    },
    "reply_processing_ms_p90": {
        "label": "回复处理耗时 p90",
        "definition": "同回复处理耗时 p50, 取第 90 百分位",
        "denominator": "带计时的回合",
        "source": "messages.metadata.response_diagnostics",
        "kind": "quantile", "unit": "ms", "settle_after_s": 0, "collected_since": "2026-10-28",
    },
    "cost_per_active_user": {
        "label": "每活跃用户日模型费用",
        "definition": "该人群用户模型费用 / 聊天活跃人日；不含评测、TTS、联网搜索和 ASR",
        "denominator": "聊天活跃用户数",
        "source": "llm_usage",
        "kind": "mean", "unit": "CNY/人", "settle_after_s": 0, "collected_since": "2026-04-29",
    },

    "feedback_per_100_turns": {
        "label": "每百回合意见反馈", "definition": "App 意见反馈条数 / 已回复回合数 × 100；不是逐句负反馈",
        "denominator": "已回复回合数", "source": "user_feedback → user_daily_facts",
        "kind": "rate", "unit": "条/百回合", "scale": 100, "settle_after_s": NO_REPLY_WINDOW_S,
    },
    "cost_per_turn": {
        "label": "每回合模型费用", "definition": "该人群模型费用 / 已回复回合数；不含评测、TTS、联网搜索和 ASR",
        "denominator": "已回复回合数", "source": "user_daily_facts",
        "kind": "mean", "unit": "CNY/回合", "settle_after_s": NO_REPLY_WINDOW_S, "collected_since": "2026-04-29",
    },

}


# Versioned implicit signals: these are review cues, not negative feedback labels.
META_COMMENT_PHRASES = ("答非所问", "你没听懂", "你是机器人", "怎么又", "你记错了", "又来了", "你刚说过")
METRICS.update({
    "resend_60s": {"label":"60 秒内重发", "definition":"AI 回复后 60 秒内用户重发同一句文本；去空白标点，排除空消息、图片、表情、接管和游戏", "denominator":"已回复回合数", "source":"messages → user_daily_facts", "kind":"rate", "unit":"%", "settle_after_s":NO_REPLY_WINDOW_S+60},
    "meta_comment": {"label":"对回复的元评论", "definition":"用户消息命中版本化元评论词表的条数占比；只是疑似不满意信号", "denominator":"用户消息数", "source":"messages → user_daily_facts", "kind":"rate", "unit":"%", "settle_after_s":0},
    "opened_not_chatted": {"label":"打开未聊天", "definition":"当天打开 App 但没有聊天的用户人日占打开人日的比例", "denominator":"打开 App 的人日", "source":"user_daily_facts", "kind":"rate", "unit":"%", "settle_after_s":0, "collected_since":"2026-05-29"},
    "hourly_messages": {"label":"分时聊天消息", "definition":"按北京时间小时分桶的用户消息数", "denominator":"条", "source":"messages → metrics_daily", "kind":"count", "unit":"条", "settle_after_s":0},
    "short_session": {"label":"短会话", "definition":"结束 3 小时、完整观察且不含游戏/接管的会话中，已回复回合不超过 3 的比例；不是用户负反馈", "denominator":"已结束的完整会话", "source":"messages → user_daily_facts", "kind":"rate", "unit":"%", "settle_after_s":3*_H},
    "session_depth": {"label":"会话深度", "definition":"用户间隔超过 3 小时分开会话；按结束日统计已回复消息串数；气泡不重复计数", "denominator":"完整会话", "source":"messages → metrics_daily", "kind":"count", "unit":"次", "settle_after_s":3*_H},
    "session_ending": {"label":"会话收尾方", "definition":"完整会话最后一条非主动消息的角色；AI 提问由问号或版本化疑问词判定", "denominator":"完整会话", "source":"messages → metrics_daily", "kind":"count", "unit":"次", "settle_after_s":3*_H},
    "session_unknown": {"label":"会话左边界不完整", "definition":"连续 92 天无 3 小时中断、无法在有界回看中确认开始的会话，排除深度与收尾统计", "denominator":"会话", "source":"messages → metrics_daily", "kind":"count", "unit":"次", "settle_after_s":3*_H},
    "feedback_count": {"label":"意见反馈条数", "definition":"App 意见反馈，不是逐句反馈，也不是管理员标注", "denominator":"条", "source":"user_feedback → metrics_daily", "kind":"count", "unit":"条", "settle_after_s":0},
    "admin_bug_labels": {"label":"管理员问题标注", "definition":"管理员多选错误类型按标签计数；同一报告可有多个标签，不等于用户负反馈人数", "denominator":"标签", "source":"bug_reports.error_types → metrics_daily", "kind":"count", "unit":"个", "settle_after_s":0},
})


# Reconciliation is a system-wide ledger; it deliberately has no external cohort.
RECONCILIATION_METRICS = {
    'token_exact_match': ('用量与 Trace 完全一致', 'rate', '%'),
    'token_missing_usage': ('有已计量 LLM 节点但缺用量记录', 'count', '回合'),
    'token_unmetered_calls': ('供应商未返回用量的调用', 'count', '次'),
    'token_unpriced_models': ('未定价模型', 'count', '种'),
    'token_unlinked_rows': ('未关联 Trace 的用量记录', 'count', '行'),
    'token_max_input': ('单次最大输入', 'max', 'token'),
}
METRICS.update({key: dict(label=label, definition="按 Trace 精确对账；仅全部用户，供应商未返回用量单列",
    denominator="可核对 Trace", source="trace_runs + llm_usage → metrics_daily", kind=kind, unit=unit,
    settle_after_s=3600, cohorts=("all",), retention_days=30 if key in
    ("token_exact_match", "token_missing_usage", "token_max_input") else None)
    for key,(label,kind,unit) in RECONCILIATION_METRICS.items()})


for _name,_label in {
    'sent':'已发出（有事件 ID）','legacy':'历史未识别事件','push_accepted':'推送供应商已接受',
    'answered_3h':'3 小时内回应','answered_6h':'6 小时内回应','three_turns':'回应后聊满 3 回合',
    'silent_2d':'已走完 24 小时窗口的沉默用户发送事件','awakened_24h':'24 小时内唤醒',
    'pending_24h':'24 小时观察未结束','logs':'主动状态日志','skip_reasons':'主动跳过原因'
}.items():
    METRICS['proactive_'+_name]=dict(label=_label,definition='按主动事件 ID 去重；状态日志独立计数，不构造虚假转化率',
        denominator='事件',source='proactive_event_logs + messages + notification_events → metrics_daily',
        kind='count',unit='次',settle_after_s=27*_H)


def describe(keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    """供接口随数据一起返回的定义说明."""
    selected = keys or tuple(METRICS)
    return {
        "definition_version": DEFINITION_VERSION,
        "min_sample": MIN_SAMPLE,
        "internal_rule": "role=admin、后台勾选的内部账号、模板系统账号",
        "metrics": {key: METRICS[key] for key in selected},
    }
