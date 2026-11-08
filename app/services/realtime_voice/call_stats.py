"""每通语音通话的诊断计数 (VOICE01.01), 通话结束时落一行 realtime_call_sessions。

之前只有日志里的零散行 (memory block / usage / mic), 容器重建后就没了, 也无法按会话看、
无法和 StepFun 时期比较。这里在一通电话内累计:

- 检索: 次数、耗时、候选数、门槛后选中数、超时次数;
- 注入: 每次回复带出的记忆条数, 以及寒暄轮次跳过次数;
- 轮次: 用户说完到第一声音频的耗时 (首音延迟), 打断、客户端打断、复读检测、工具调用;
- 环境: 平台、模型、VAD 门槛与静音时长、麦克风电平。

只存计数、耗时和分位数, 不存任何通话原文。某项从没发生过就记 null, 后台显示"未采集",
不当作 0。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# 单通电话最多记这么多个样本, 防一通很长的电话把一行撑得很大
_MAX_SAMPLES = 500


def _bounded_append(values: list[int], value: float) -> None:
    if len(values) < _MAX_SAMPLES:
        values.append(max(0, round(value)))


def percentiles(values: list[int]) -> dict[str, int] | None:
    """p50 / p90 / max (最近秩法); 没有样本返回 None。"""
    if not values:
        return None
    ordered = sorted(values)

    def rank(p: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]

    return {"p50": rank(0.5), "p90": rank(0.9), "max": ordered[-1], "n": len(ordered)}


@dataclass
class CallStats:
    platform: str = "unknown"
    model: str = ""
    vad_threshold: float | None = None
    silence_ms: int | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    retrieval_ms: list[int] = field(default_factory=list)
    retrieval_timeouts: int = 0
    candidates: list[int] = field(default_factory=list)
    selected: list[int] = field(default_factory=list)
    injected: list[int] = field(default_factory=list)
    first_audio_ms: list[int] = field(default_factory=list)
    barge_ins: int = 0
    client_interrupts: int = 0
    repeats: int = 0
    tool_calls: int = 0
    _turn_ended_at: float | None = None

    # ── 由 bridge 和端点在通话中调用 ─────────────────────────────────────
    def retrieval(self, elapsed_s: float, *, timed_out: bool = False) -> None:
        """一次预取检索的耗时 (bridge 计); 超时也计耗时。"""
        _bounded_append(self.retrieval_ms, elapsed_s * 1000)
        if timed_out:
            self.retrieval_timeouts += 1

    def retrieval_result(self, *, candidates: int, selected: int) -> None:
        """检索返回时 (端点计): 候选数和门槛后选中数成对记, 两列永远对齐。"""
        if len(self.candidates) < _MAX_SAMPLES:
            self.candidates.append(max(0, candidates))
            self.selected.append(max(0, selected))

    def reply_note(self, memory_lines: int) -> None:
        """一次回复带出的记忆条数 (含 0: 寒暄轮次和没有相关记忆的回复)。"""
        _bounded_append(self.injected, memory_lines)

    def turn_ended(self, at: float | None = None) -> None:
        """用户这一轮说完了 (monotonic 秒); 下一段音频的首包据此计首音延迟。"""
        self._turn_ended_at = time.monotonic() if at is None else at

    def first_audio(self, at: float | None = None) -> None:
        if self._turn_ended_at is None:
            return  # 开场问候、或同一轮的后续回复: 不计
        now = time.monotonic() if at is None else at
        _bounded_append(self.first_audio_ms, (now - self._turn_ended_at) * 1000)
        self._turn_ended_at = None

    # ── 通话结束时汇总 ─────────────────────────────────────────────────
    def summary(self, *, turns: int, speech_turns: int, echo_dropped: int, memory_skipped: int,
                mic: dict[str, Any] | None, usage: dict[str, int] | None) -> dict[str, Any]:
        # 候选里没被选中的 (相关度门槛 / 每侧条数上限 / 去重)
        rejected = sum(c - s for c, s in zip(self.candidates, self.selected)) if self.candidates else None
        return {
            "schema_version": SCHEMA_VERSION,
            "environment": {"platform": self.platform, "model": self.model,
                            "vad_threshold": self.vad_threshold, "silence_ms": self.silence_ms, "mic": mic},
            "turns": {"answered": turns, "speech_segments": speech_turns, "echo_dropped": echo_dropped,
                      "barge_ins": self.barge_ins, "client_interrupts": self.client_interrupts,
                      "repeats": self.repeats, "tool_calls": self.tool_calls},
            "first_audio_ms": percentiles(self.first_audio_ms),
            "retrieval": {
                "count": len(self.retrieval_ms), "timeouts": self.retrieval_timeouts,
                "latency_ms": percentiles(self.retrieval_ms),
                "candidates": percentiles(self.candidates), "selected": percentiles(self.selected),
                "rejected_total": rejected,
            },
            "injection": {"replies": len(self.injected),
                          "replies_with_memories": sum(1 for n in self.injected if n > 0),
                          "lines": percentiles(self.injected), "small_talk_skipped": memory_skipped},
            "usage": usage,
            # 原始样本 (每项最多 _MAX_SAMPLES 个): 后台按窗口合并多通电话算基线分位数
            "samples": {"first_audio_ms": self.first_audio_ms, "retrieval_ms": self.retrieval_ms,
                        "injected": self.injected},
        }


async def persist(stats: CallStats, summary: dict[str, Any], *, user_id: str, workspace_id: str | None,
                  conversation_id: str, agent_id: str) -> None:
    """落一行; 失败只记日志, 绝不影响通话收尾。"""
    import json

    from app.db import db

    ended = datetime.now(UTC)
    try:
        await db.execute_raw(
            """INSERT INTO realtime_call_sessions(id,user_id,workspace_id,conversation_id,agent_id,platform,model,
                 started_at,ended_at,duration_s,stats)
               VALUES($1,$2,$3,$4,$5,$6,$7,$8::timestamptz,$9::timestamptz,$10,$11::jsonb)""",
            str(uuid4()), user_id, workspace_id, conversation_id, agent_id, stats.platform, stats.model,
            stats.started_at.isoformat(), ended.isoformat(),
            round((ended - stats.started_at).total_seconds(), 1), json.dumps(summary),
        )
    except Exception as exc:
        logger.warning("[realtime-voice] call stats not stored: %s", type(exc).__name__)


_MAX_WINDOW_DAYS = 90
_MAX_CALLS = 2000


async def baseline(*, user_id: str | None = None, days: int = 7, limit: int = 50) -> dict:
    """近 days 天通话的基线: 逐通列表 + 合并各通原始样本后的分位数 (VOICE01.01)。"""
    from app.db import db

    days = max(1, min(days, _MAX_WINDOW_DAYS))
    limit = max(1, min(limit, 200))
    since = datetime.now(UTC) - timedelta(days=days)
    params: list = [since.isoformat()]
    scope = ""
    if user_id:
        params.append(user_id)
        scope = " AND user_id = $2"
    rows = await db.query_raw(
        f"""SELECT id, platform, model, started_at, duration_s, stats
            FROM realtime_call_sessions
            WHERE started_at >= ($1::timestamptz) {scope}
            ORDER BY started_at DESC LIMIT {_MAX_CALLS}""",
        *params,
    )
    merged: dict[str, list[int]] = {"first_audio_ms": [], "retrieval_ms": [], "injected": []}
    calls = []
    for row in rows:
        stats = row["stats"] if isinstance(row["stats"], dict) else {}
        for key, bucket in merged.items():
            if len(bucket) < _MAX_CALLS * 4:
                bucket.extend((stats.get("samples") or {}).get(key) or [])
        if len(calls) < limit:
            retr = stats.get("retrieval") or {}
            calls.append({
                "id": row["id"], "platform": row["platform"], "model": row["model"],
                "started_at": row["started_at"].isoformat() if hasattr(row["started_at"], "isoformat")
                else str(row["started_at"]),
                "duration_s": row["duration_s"],
                "turns": (stats.get("turns") or {}).get("answered"),
                "first_audio_ms": stats.get("first_audio_ms"),
                "retrieval_count": retr.get("count"), "retrieval_timeouts": retr.get("timeouts"),
                "rejected_total": retr.get("rejected_total"),
                "injection": stats.get("injection"),
            })
    return {
        "scope": {"user_id": user_id},
        "window": {"days": days, "since": since.isoformat()},
        "call_count": len(rows),
        "baseline": {key: percentiles(values) for key, values in merged.items()},
        "calls": calls,
        "sampled_at": datetime.now(UTC).isoformat(),
    }
