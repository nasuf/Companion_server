"""主动聊天策略层。"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from app.services.proactive.state import ProactiveStateRecord, PROACTIVE_WINDOWS, is_same_local_day

UTC = timezone.utc

# spec §1.2 base hit rates 已经在 PROACTIVE_WINDOWS 表里: 0/5/12/25/35%
# stage multiplier 保留 (spec 未禁止, 产品决策允许按阶段做 ±15% 调节),
# silence_penalty 用于衰减阶段 n≤4 期间渐降命中率 (spec §8.3 "最多 4 次"隐含).
STAGE_HIT_MULTIPLIER = {
    "cold_start": 0.85,
    "warming": 1.0,
    "intimate": 1.15,
}


def should_hit_window(state: ProactiveStateRecord) -> tuple[bool, float]:
    base_rate = 0.0
    for window in PROACTIVE_WINDOWS:
        if int(window["index"]) == int(state.current_window_index or 0):
            base_rate = float(window["hit_rate"])
            break

    multiplier = STAGE_HIT_MULTIPLIER.get(state.stage, 1.0)
    silence_penalty = max(0.5, 1.0 - 0.08 * max(0, state.silence_level_n))
    final_rate = max(0.0, min(1.0, base_rate * multiplier * silence_penalty))
    return random.random() < final_rate, final_rate


def scene_candidate_available(
    state: ProactiveStateRecord,
    schedule_status: dict[str, Any] | None,
    now: datetime | None = None,
) -> bool:
    now_ts = now or datetime.now(UTC)
    status = str((schedule_status or {}).get("status") or "idle")
    if status == "sleep":
        return False
    if is_same_local_day(state.daily_scene_triggered_at, now_ts):
        return False
    return True


def select_trigger_type(
    state: ProactiveStateRecord,
    *,
    scene_available: bool,
) -> str:
    """spec §1.3 触发类型基础配比: 沉默唤醒 30% / 记忆主动 30% / 定时情景 40%.

    - scene 不可用时按 spec §1.3 兜底规则重抽 (沉默 50% / 记忆 50%)
    - 不再按 stage 做动态倾斜, stage 只影响 hit_rate multiplier 和话题来源
    """
    if not scene_available:
        return fallback_trigger_type()

    r = random.random()
    if r < 0.30:
        return "silence_wakeup"
    if r < 0.60:
        return "memory_proactive"
    return "scheduled_scene"


def fallback_trigger_type() -> str:
    """spec §1.3 兜底: 定时情景不可用或抽取失败时, 在沉默/记忆之间 50/50."""
    return random.choice(["silence_wakeup", "memory_proactive"])


# ── spec §3.4.7 话题范围分级库 (P1-P5) ──
# 权威源: 每个 P-level 列出该等级 *新增* 的话题类型; 抽取时按
# `allowed_topics(stage_code)` 累加当级及以下所有话题 (spec "高等级
# 话题范围包含低等级话题").
TOPIC_RANGE_BY_STAGE: dict[str, tuple[str, ...]] = {
    "P1": ("天气", "问候", "兴趣爱好试探", "公共话题"),
    "P2": ("日常琐事", "AI的简单生活", "询问近况"),
    "P3": ("共同兴趣", "回忆之前聊过的事", "分享有趣见闻"),
    "P4": ("关心用户情绪", "分享内心感受", "讨论人生话题"),
    "P5": ("深入价值观探讨", "回忆共同经历", "随意调侃"),
}

_P_LEVELS = ("P1", "P2", "P3", "P4", "P5")

# proactive 3-archetype → spec P-code. cold_start 覆盖 P1-P2; warming
# 覆盖到 P4; intimate 放开全 5 级。保留 3-stage 是因为 state.stage 由
# memory 数量 + topic_intimacy 联合决定, 不能纯用 topic_intimacy 推。
_STAGE_TO_P: dict[str, str] = {
    "cold_start": "P2",
    "warming": "P4",
    "intimate": "P5",
}


def allowed_topics(stage_code: str) -> tuple[str, ...]:
    """spec §3.4.7: 返回 `stage_code` 及以下所有 P-level 的累加话题白名单。

    `stage_code` 接受 P1-P5 或 proactive 3-archetype (cold_start/warming/intimate)。
    """
    p_code = _STAGE_TO_P.get(stage_code, stage_code)
    if p_code not in _P_LEVELS:
        p_code = "P1"
    idx = _P_LEVELS.index(p_code)
    result: list[str] = []
    for p in _P_LEVELS[: idx + 1]:
        result.extend(TOPIC_RANGE_BY_STAGE[p])
    return tuple(result)


def select_topic_theme(stage: str) -> str:
    """spec §3.2 + §3.4.7: 根据 stage 抽取允许范围内的话题方向。"""
    themes = allowed_topics(stage)
    return random.choice(themes) if themes else "问候"


# ── spec §4.1 沉默唤醒 话题来源配比 ──
# key: 话题来源枚举, value: {stage: probability}
# 来源: ai_l1 / ai_l2 / ai_schedule / user_l1 / user_l2 / greeting
SILENCE_SOURCE_DIST: dict[str, dict[str, float]] = {
    "cold_start": {
        "ai_l1": 0.20, "ai_l2": 0.00, "ai_schedule": 0.10,
        "user_l1": 0.00, "user_l2": 0.00, "greeting": 0.70,
    },
    "warming": {
        "ai_l1": 0.10, "ai_l2": 0.00, "ai_schedule": 0.10,
        "user_l1": 0.05, "user_l2": 0.05, "greeting": 0.70,
    },
    "intimate": {
        "ai_l1": 0.05, "ai_l2": 0.05, "ai_schedule": 0.10,
        "user_l1": 0.10, "user_l2": 0.10, "greeting": 0.60,
    },
}

# ── spec §4.2 记忆主动触发 话题来源配比 (PAD 情绪融合独立 100% 不参与抽签) ──
MEMORY_SOURCE_DIST: dict[str, dict[str, float]] = {
    "cold_start": {
        "ai_l1": 1.00, "ai_l2": 0.00,
        "user_l1": 0.00, "user_l2": 0.00,
    },
    "warming": {
        "ai_l1": 0.75, "ai_l2": 0.05,
        "user_l1": 0.10, "user_l2": 0.10,
    },
    "intimate": {
        "ai_l1": 0.50, "ai_l2": 0.10,
        "user_l1": 0.15, "user_l2": 0.25,
    },
}


def _weighted_choice(weights: dict[str, float]) -> str:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return "greeting"
    r = random.random() * total
    acc = 0.0
    for name, w in weights.items():
        if w <= 0:
            continue
        acc += w
        if r <= acc:
            return name
    return next(iter(weights))


def select_topic_source(stage: str, trigger_type: str) -> str:
    """spec §4.1/§4.2: 根据 stage + trigger_type 抽一个话题来源.

    Returns one of:
      - silence_wakeup:  ai_l1 / ai_l2 / ai_schedule / user_l1 / user_l2 / greeting
      - memory_proactive: ai_l1 / ai_l2 / user_l1 / user_l2
      - scheduled_scene:  ai_schedule (固定)
    """
    if trigger_type == "scheduled_scene":
        return "ai_schedule"
    if trigger_type == "silence_wakeup":
        dist = SILENCE_SOURCE_DIST.get(stage) or SILENCE_SOURCE_DIST["cold_start"]
        return _weighted_choice(dist)
    if trigger_type == "memory_proactive":
        dist = MEMORY_SOURCE_DIST.get(stage) or MEMORY_SOURCE_DIST["cold_start"]
        return _weighted_choice(dist)
    return "greeting"
