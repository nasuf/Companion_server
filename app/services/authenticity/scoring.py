"""Per-user features and a transparent "AI relay / script" probability.

There is no labeled data yet. The probability is a hand-weighted logistic sum
calibrated on production users of 2026-10/11 (one suspected relay against the
other active external users), so it ranks and explains; it does not prove.

Feature reference (30 days, active external users vs the suspected relay):
- compose speed, chars/second from first keystroke to send: 0.24-2.2 vs 12.75
- messages of 3 chars or fewer: 14%-40% vs 0%
- messages with Chinese punctuation: 1%-72% vs 100%
- median length: 3-17 vs 34 chars
- replies within 10 s of an AI message: 0%-53% vs 0%
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.effect.definitions import internal_user_sql

MODEL_VERSION = "authenticity-heuristic-v1"
PRIOR_LOGIT = -2.2
MIN_TEXT_MESSAGES = 20
MIN_REPLIES = 10
MIN_TIMED = 5


@dataclass(frozen=True, slots=True)
class Contribution:
    key: str
    label: str
    value: float | None
    samples: int
    logit: float
    applied: bool
    note: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _num(row: dict, key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def contributions(row: dict) -> list[Contribution]:
    """Log-odds terms; a term with too few samples is shown but not applied."""
    messages = int(row.get("messages") or 0)
    replies = int(row.get("replies") or 0)
    timed = int(row.get("timed") or 0)
    client_samples = int(row.get("client_samples") or 0)
    hits = int(row.get("network_hits") or 0)
    out: list[Contribution] = []

    def add(key, label, value, samples, minimum, logit, note):
        applied = value is not None and samples >= minimum
        out.append(Contribution(key, label, value, samples,
                                round(logit(value), 3) if applied else 0.0, applied, note))

    add("compose_cps", "输入速度（字/秒，中位数）", _num(row, "compose_cps"), timed, MIN_TIMED,
        lambda v: _clamp((math.log(max(v, 0.05)) - math.log(4.0)) * 1.6, -1.5, 3.0),
        "从开始输入到发送；手打一般 0.3–3 字/秒，粘贴会远超 8；语音输入也偏快")
    add("paste_share", "整段粘贴的消息占比", _num(row, "paste_share"), client_samples, MIN_TIMED,
        lambda v: _clamp(3.0 * (v - 0.2), -0.8, 1.5),
        "App 上报：单次插入的字数占整条消息 80% 以上；输入法语音转文字也会这样")
    add("punct_share", "带中文标点的消息占比", _num(row, "punct_share"), messages, MIN_TEXT_MESSAGES,
        lambda v: _clamp(3.0 * (v - 0.7), -1.0, 1.0),
        "聊天里手打常省略标点；大模型几乎每句都有")
    add("short_share", "3 字以内短消息占比", _num(row, "short_share"), messages, MIN_TEXT_MESSAGES,
        lambda v: _clamp(1.0 - 8.0 * v, -1.0, 1.0),
        "真人常发“嗯”“哈哈”这类短句；转贴大模型输出几乎没有")
    add("median_chars", "消息长度中位数（字）", _num(row, "median_chars"), messages, MIN_TEXT_MESSAGES,
        lambda v: _clamp((v - 15.0) / 10.0, -0.8, 1.0),
        "真人多在 3–17 字")
    add("fast_reply_share", "AI 回复后 10 秒内接话占比", _num(row, "fast_reply_share"), replies, MIN_REPLIES,
        lambda v: _clamp(0.8 - 6.0 * v, -1.0, 0.8),
        "转给另一个模型再贴回来，做不到秒回")
    add("burst_share", "连发多条的占比", _num(row, "burst_share"), messages, MIN_TEXT_MESSAGES,
        lambda v: _clamp(0.5 - 4.0 * v, -1.0, 0.5),
        "真人会把一句话拆成几条连发")
    add("datacenter_share", "来自机房/云服务器 IP 的请求占比", _num(row, "datacenter_share"), hits, 1,
        lambda v: _clamp(1.5 * v, 0.0, 1.5),
        "手机走运营商网络；脚本多跑在云服务器上（开 VPN/加速器也会显示为机房）")
    shared = _num(row, "shared_accounts")
    add("shared_accounts", "同一非运营商 IP 下的其他账号数", shared, 1 if shared is not None else 0, 1,
        lambda v: _clamp(0.5 * v, 0.0, 1.5),
        "运营商出口 IP 多人共用，不计入")
    return out


def probability(terms: list[Contribution]) -> float:
    logit = PRIOR_LOGIT + sum(t.logit for t in terms)
    return 1.0 / (1.0 + math.exp(-logit))


def verdict(p: float, terms: list[Contribution]) -> str:
    if not any(t.applied for t in terms if t.key not in {"datacenter_share", "shared_accounts"}):
        return "insufficient"
    if p >= 0.8:
        return "likely_ai"
    if p >= 0.5:
        return "suspicious"
    if p >= 0.2:
        return "likely_human"
    return "human"


_FEATURES_SQL = """
WITH m AS (
  SELECT c.user_id, m.role, m.created_at, btrim(m.content) AS content, m.metadata,
         lag(m.created_at) OVER w AS prev_at, lag(m.role) OVER w AS prev_role
  FROM messages m JOIN conversations c ON c.id = m.conversation_id
  WHERE m.created_at >= $1::timestamp {user_filter}
  WINDOW w AS (PARTITION BY m.conversation_id ORDER BY m.created_at)
), text AS (
  SELECT user_id, count(*) AS messages,
    avg((char_length(content) <= 3)::int)::float AS short_share,
    avg((content ~ '[，。！？]')::int)::float AS punct_share,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY char_length(content))::float AS median_chars,
    count(*) FILTER (WHERE prev_role = 'assistant') AS replies,
    (avg((created_at - prev_at < interval '10 seconds')::int)
       FILTER (WHERE prev_role = 'assistant'))::float AS fast_reply_share,
    avg((prev_role = 'user')::int)::float AS burst_share,
    max(created_at) AS last_message_at
  FROM m
  WHERE role = 'user' AND content <> ''
    AND NOT COALESCE(metadata, '{{}}'::jsonb) ? 'component_card'
  GROUP BY user_id
), timing AS (
  SELECT user_id,
    count(*) FILTER (WHERE ms IS NOT NULL) AS timed,
    (percentile_cont(0.5) WITHIN GROUP (ORDER BY chars * 1000.0 / GREATEST(ms, 1000))
       FILTER (WHERE ms IS NOT NULL))::float AS compose_cps,
    count(*) FILTER (WHERE max_insert_chars IS NOT NULL) AS client_samples,
    (avg((max_insert_chars >= 0.8 * chars)::int) FILTER (WHERE max_insert_chars IS NOT NULL))::float
      AS paste_share
  FROM (SELECT user_id, chars, max_insert_chars,
               COALESCE(client_compose_ms, server_compose_ms) AS ms
        FROM user_input_signals WHERE created_at >= $1::timestamp AND chars >= 10 {user_filter_s}) s
  GROUP BY user_id
), net AS (
  SELECT user_id, sum(hits) AS network_hits,
    COALESCE(sum(hits) FILTER (WHERE network_kind = 'datacenter'), 0)::float
      / NULLIF(sum(hits), 0) AS datacenter_share
  FROM user_network_observations WHERE last_seen_at >= $1::timestamp {user_filter_s}
  GROUP BY user_id
), shared AS (
  SELECT a.user_id, count(DISTINCT b.user_id) AS shared_accounts
  FROM user_network_observations a
  JOIN user_network_observations b ON b.ip_hash = a.ip_hash AND b.user_id <> a.user_id
    AND b.last_seen_at >= $1::timestamp
  WHERE a.last_seen_at >= $1::timestamp AND a.network_kind NOT IN ('carrier', 'campus') {user_filter_a}
  GROUP BY a.user_id
)
SELECT u.id AS user_id, u.username, u.display_name, u.role, u.is_internal,
  {internal} AS internal,
  t.messages, t.short_share, t.punct_share, t.median_chars, t.replies, t.fast_reply_share,
  t.burst_share, t.last_message_at,
  g.timed, g.compose_cps, g.client_samples, g.paste_share,
  n.network_hits, n.datacenter_share, COALESCE(s.shared_accounts, 0) AS shared_accounts
FROM users u
JOIN text t ON t.user_id = u.id
LEFT JOIN timing g ON g.user_id = u.id
LEFT JOIN net n ON n.user_id = u.id
LEFT JOIN shared s ON s.user_id = u.id
"""


def _since(days: int, now: datetime | None) -> str:
    at = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    return at.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


async def feature_rows(days: int, *, user_id: str | None = None,
                       now: datetime | None = None, database=None) -> list[dict]:
    from app.db import db

    params: list[Any] = [_since(days, now)]
    filters = {"user_filter": "", "user_filter_s": "", "user_filter_a": ""}
    if user_id is not None:
        params.append(user_id)
        filters = {"user_filter": "AND c.user_id = $2", "user_filter_s": "AND user_id = $2",
                   "user_filter_a": "AND a.user_id = $2"}
    sql = _FEATURES_SQL.format(internal=internal_user_sql("u"), **filters)
    return await (database or db).query_raw(sql, *params)


_REFERENCE_KEYS = ("compose_cps", "punct_share", "short_share", "median_chars",
                   "fast_reply_share", "burst_share")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def reference_ranges(rows: list[dict], *, exclude: str | None = None) -> dict[str, dict[str, float | int]]:
    """p10/p50/p90 over other active external users (enough messages for text features)."""
    population = [r for r in rows if not r.get("internal") and (exclude is None or r.get("user_id") != exclude)
                  and int(r.get("messages") or 0) >= MIN_TEXT_MESSAGES]
    ranges: dict[str, dict[str, float | int]] = {}
    for key in _REFERENCE_KEYS:
        values = [float(r[key]) for r in population if r.get(key) is not None
                  and (key != "compose_cps" or int(r.get("timed") or 0) >= MIN_TIMED)]
        if len(values) >= 3:
            ranges[key] = {"p10": round(_quantile(values, 0.1), 3), "p50": round(_quantile(values, 0.5), 3),
                           "p90": round(_quantile(values, 0.9), 3), "users": len(values)}
    return ranges


def assess(row: dict) -> dict:
    terms = contributions(row)
    p = probability(terms)
    return {
        "probability": round(p, 4),
        "verdict": verdict(p, terms),
        "contributions": [asdict(t) | {"value": None if t.value is None else round(t.value, 3)}
                          for t in terms],
    }
