"""聚合类时间问题的事件时间线.

## 解决什么

LongMemEval 时间推理子集实测 (2026-07-30, 10 题):

    普通时间题 ("X 和 Y 哪个先" / "距今几天")   k=5 就全中     8/10
    聚合题     ("上次…是几个月前" / "几次…")     需要 k=26~42   2/10

八成的时间题现在就是好的。失败的两道都要**先穷举某个主题下的所有事件**才能作答:

    "上次和朋友去博物馆是几个月前" —— 要拿到全部博物馆相关事件, 才能判断哪些算
    "和朋友参观"、哪次是最近的
    "参加两次连续两天的慈善活动至今几个月" —— 要拿到全部慈善活动, 才能识别出哪
    两次是连着的

普通检索取"最相似的一条", 这类题要的是"这个主题下的全部"。

## 为什么不是简单加大 k

40 条记忆 × 中位 52 token ≈ 2080, 而注入预算是 900。但这类题**不需要记忆全文**,
只需要"事件 + 日期"。压成 `[2023-02-14] 参加24小时骑行慈善活动` 约 15 token, 40 条
只要 600 —— 比现在还省。

## 日期从哪来

`occur_time` 只有 12%(用户侧)/30%(AI侧) 填了, 光靠它时间线是空的。降级到
`statement_time`: 实测两者都有的记忆里中位只差 2 天 —— 用户讲刚发生的事时不带时间
词, 说话时刻就约等于事件时刻。

已知偏差: "我小时候在苏州长大" 这种明确指向久远过去的, statement_time 会把它标成
今天。这类内容有明显词面特征, 直接排除掉而不是标错日期 —— 时间线里一条错误日期
会让 LLM 算出完全错误的间隔, 比少一条严重得多。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# 聚合类提问的词面特征。刻意用规则而不是 LLM: 这是热路径上的一个开关, 判错的代价
# 只是多注入/少注入一段时间线, 不值得为它加一次模型调用。
_AGGREGATE_PATTERNS = (
    # 次数
    r"几次|多少次|几回|多少回",
    # 时长/间隔
    r"多久|多长时间|几个月|几年|几周|几天前|有多少天|隔了多久",
    # 序数/极值 —— 需要在多个同类事件里挑一个
    r"上一?次|最近一次|最后一次|第一次|头一次|最早|最晚",
    # 先后关系
    r"哪个先|哪个在先|先还是后|之前还是之后|谁先|先做的",
    # 频率
    r"多久一次|经常|每隔",
)
_AGGREGATE_RE = re.compile("|".join(_AGGREGATE_PATTERNS))

# 明确指向久远过去的词。这些记忆的 statement_time 不能当事件时间用 ——
# 标错日期比缺一条更糟。
_VAGUE_PAST_RE = re.compile(r"小时候|童年|少年时|以前|从前|当年|那时候|很久以前|上学时|读书时")


def is_aggregate_time_question(text: str) -> bool:
    """这句话是不是需要"把某主题下的事件全找出来"才能答.

    宁可漏判不可误判: 命中就会多注入一段时间线, 挤占其他记忆的预算。
    """
    if not text:
        return False
    return bool(_AGGREGATE_RE.search(text))


@dataclass(frozen=True)
class TimelineEntry:
    at: datetime
    text: str
    dated_by: str  # occur_time | statement_time


def _event_time(row: dict) -> tuple[datetime, str] | None:
    for field in ("occur_time", "statement_time"):
        v = row.get(field)
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                v = None
        if isinstance(v, datetime):
            return v, field
    return None


def build_timeline(rows: list[dict], *, limit: int = 40) -> list[TimelineEntry]:
    """把候选记忆压成按时间排序的事件条目.

    只保留能定出日期的; 明确讲久远过去的排除 (见模块 docstring)。
    """
    entries: list[TimelineEntry] = []
    for row in rows:
        content = (row.get("content") or "").strip()
        if not content or _VAGUE_PAST_RE.search(content):
            continue
        stamped = _event_time(row)
        if stamped is None:
            continue
        at, source = stamped
        entries.append(TimelineEntry(at=at, text=content, dated_by=source))

    entries.sort(key=lambda e: e.at)
    # 超限时保留**最近的** limit 条: 聚合题问的多是"上次""最近几次", 久远条目
    # 的边际价值低于近期。
    return entries[-limit:]


# 时间线整段的 token 上限。比记忆区的 900 小: 时间线是**额外**注入的一段, 不该
# 反过来把正常记忆挤没。
TIMELINE_TOKEN_BUDGET = 400

# 单条内容截到多少字。聚合题要的是"哪天发生了什么事", 不是事件细节 —— 20 字够
# 认出是哪件事 ("参加24小时骑行慈善活动"), 再长只是摊薄条数。
_ENTRY_MAX_CHARS = 20


def format_timeline(
    entries: list[TimelineEntry],
    *,
    max_chars: int = _ENTRY_MAX_CHARS,
    token_budget: int = TIMELINE_TOKEN_BUDGET,
) -> str:
    """渲染成紧凑的日期行, 按 token 预算从最近往前收.

    按预算裁而不是按条数裁: 中文 40 字一行约 56 token, 40 行就是 2500+ —— 光按
    "取 40 条"排版, 预算会翻三倍 (这个错第一版就犯了, 被测试挡下)。

    超预算时丢最久远的: 聚合题问的多是"上次""最近几次"。
    """
    from app.services.memory.retrieval.context_selector import estimate_tokens

    if not entries:
        return ""
    lines: list[str] = []
    used = 0
    for e in reversed(entries):  # 从最近往前
        text = e.text if len(e.text) <= max_chars else e.text[: max_chars - 1] + "…"
        # statement_time 定的日期加个"约", 让 LLM 知道这个日期是推断的不是确证的
        mark = "" if e.dated_by == "occur_time" else "约"
        line = f"{mark}{e.at:%Y-%m-%d} {text}"
        cost = estimate_tokens(line) + 1  # +1 为换行
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    lines.reverse()
    return "\n".join(lines)
