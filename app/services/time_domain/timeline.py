"""Bounded memory date evidence, never an exhaustive event ledger.

occur_time is a stored event assertion, not independent verification. A
statement_time labels only when something was mentioned. Neither record/access
clocks nor a short candidate list prove an event date, exact count, or first/last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.services.time_domain.timestamps import parse_time, local_clock


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
    owner: str  # user | ai —— 这是谁的经历


def _event_time(row: dict) -> tuple[datetime, str] | None:
    for field in ("occur_time", "statement_time"):
        value = parse_time(row.get(field))
        if value is not None:
            return value, field
    return None


def build_timeline(rows: list[dict], *, limit: int = 40) -> list[TimelineEntry]:
    """把候选记忆压成按时间排序的事件条目.

    保留可标注的事件或提及日期；提及时间不会变成事件日期。
    """
    if limit <= 0:
        return []
    entries: list[TimelineEntry] = []
    for row in rows:
        content = " ".join((row.get("content") or "").split())
        if not content:
            continue
        stamped = _event_time(row)
        if stamped is None:
            continue
        at, source = stamped
        entries.append(
            TimelineEntry(
                at=at, text=content, dated_by=source,
                owner=row.get("source") if row.get("source") in {"ai", "user"} else "unknown",
            )
        )

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
    timezone_name: str | None = None,
) -> str:
    """渲染成紧凑的日期行, 按 token 预算从最近往前收.

    按预算裁而不是按条数裁: 中文 40 字一行约 56 token, 40 行就是 2500+ —— 光按
    "取 40 条"排版, 预算会翻三倍 (这个错第一版就犯了, 被测试挡下)。

    超预算时丢最久远的: 聚合题问的多是"上次""最近几次"。
    """
    from app.services.memory.retrieval.context_selector import estimate_tokens

    if not entries:
        return ""
    if max_chars < 2 or token_budget <= 0:
        return ""
    header = "【候选日期摘录，非完整历史；不能据此确定总次数或首次/末次】"
    lines: list[str] = []
    used = estimate_tokens(header) + 1
    for e in reversed(entries):  # 从最近往前
        text = e.text if len(e.text) <= max_chars else e.text[: max_chars - 1] + "…"
        label = "事件记录" if e.dated_by == "occur_time" else "提及（事件时间未知）"
        # 归属前缀是必须的, 不是装饰: 时间线同时含用户经历和 AI 自己的经历, 不标
        # 归属的话"我上次去博物馆是几个月前"可能被 AI 自己的经历答掉。
        who = "我" if e.owner == "ai" else "用户" if e.owner == "user" else "归属未知"
        line = f"[{who}] {label} {local_clock(e.at, timezone_name=timezone_name)[:10]} {text}"
        cost = estimate_tokens(line) + 1  # +1 为换行
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    lines.reverse()
    return "\n".join([header, *lines]) if lines else ""


def timeline_evidence(rows: list[dict], *, timezone_name: str | None = None) -> tuple[str | None, dict]:
    """Count only this bounded retrieval candidate set, never the memory bank."""
    entries = build_timeline(rows, limit=max(1, len(rows)))
    selected = entries[-40:]
    zone = timezone_name or settings.schedule_timezone
    try:
        ZoneInfo(zone)
        valid_zone = True
    except (ValueError, ZoneInfoNotFoundError):
        valid_zone = False
    text = format_timeline(selected, timezone_name=zone) if valid_zone else ""
    shown = max(0, len(text.splitlines()) - 1)
    return text or None, {
        "version": 2, "status": "rendered" if valid_zone else "unavailable",
        "timezone": zone if valid_zone else None,
        "reason": "candidate_dates" if valid_zone else "invalid_timezone",
        "candidate_count": len(rows),
        "dated_count": len(entries), "shown_count": shown,
        "event_date_count": sum(e.dated_by == "occur_time" for e in entries),
        "mention_only_count": sum(e.dated_by == "statement_time" for e in entries),
        "undated_count": len(rows) - len(entries),
        "truncated": shown < len(entries), "complete_history": False,
        "token_budget": TIMELINE_TOKEN_BUDGET,
    }
