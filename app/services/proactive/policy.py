"""主动聊天策略层。"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from app.services.proactive import tuning
from app.services.proactive.state import ProactiveStateRecord, is_same_local_day

UTC = timezone.utc

# spec §1.2 base hit rates 默认 0/5/12/25/35% (后台「聊天管理 → 主动交流」可调).
# spec §8.3 第一阶段 n=1-4 要求 "正常概率, 最多 4 次", 4 次上限由
# _escalate_silence_level 的 n+1 升级机制保证 (n=5 后走强制触发路径,
# 不再进此函数), 频率维度不做额外折扣.


def should_hit_window(
    state: ProactiveStateRecord,
    *,
    rate_multiplier: float = 1.0,
) -> tuple[bool, float]:
    base_rate = tuning.window_rate(int(state.current_window_index or 0))
    cap = float(tuning.value("a_window_rate_cap"))
    final_rate = max(0.0, min(cap, base_rate * rate_multiplier))
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


_TRIGGER_TYPES = ("silence_wakeup", "memory_proactive", "scheduled_scene")
_FALLBACK_TRIGGER_TYPES = ("silence_wakeup", "memory_proactive")


def select_trigger_type() -> str:
    """spec §1.3 触发类型基础配比: 默认沉默唤醒 30% / 记忆主动 30% / 定时情景 40%.

    抽中 scheduled_scene 后, **调用方**再校验 scene 可用性 (AI 睡眠 / 当日已用),
    不可用走 `fallback_trigger_type()` 兜底. 不在此函数预校验 — 否则在 scene
    不可用时段会偏到 50/50, 错失 30/30 配比.
    """
    values = tuning.current()
    return _weighted_choice(
        {name: float(values[f"trigger_{name}"]) for name in _TRIGGER_TYPES},
        default="silence_wakeup",
    )


def fallback_trigger_type() -> str:
    """spec §1.3 兜底: 定时情景不可用或抽取失败时, 在沉默/记忆之间 (默认 50/50)."""
    values = tuning.current()
    return _weighted_choice(
        {name: float(values[f"fallback_{name}"]) for name in _FALLBACK_TRIGGER_TYPES},
        default="silence_wakeup",
    )


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

# spec §3.4.7 P-level 映射. cold_start 保留 — 老 state.stage 行 (pre-4-tier 拆分)
# 可能仍存此值.
_STAGE_TO_P: dict[str, str] = {
    "p1_cold": "P1",
    "p2_cold": "P2",
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


# ── spec §4.1 沉默唤醒 / §4.2 记忆主动 话题来源配比 ──
# 实际生效值由后台「聊天管理 → 主动交流」调整, 默认值与说明见 proactive/tuning.py:
#   - spec §5 把 P1+P2 合并成一档配比 → p1_cold / p2_cold / cold_start 共用「冷启动」表
#   - 记忆主动 Phase 2 关系记忆: warming/intimate 给 "relationship" (我们之间的共同经历)
#     分一份权重; 冷启动没有共同历史, 不分
#   - 《主动交流提示词》提示词3「带时间感知历史对话搭话」= recent_dialogue (近两天带
#     时间戳的聊天记录): 沉默唤醒从 greeting 匀出, 记忆主动从 ai_l1 匀出; 冷启动不分 ——
#     还不熟就追问对方的事, 像查户口; 近两天没聊过时 context 兜底回落
# 下面两张是规格默认值 (只读视图), 运行时请走 select_topic_source。
_STAGES = ("p1_cold", "p2_cold", "cold_start", "warming", "intimate")
SILENCE_SOURCE_DIST: dict[str, dict[str, float]] = {
    stage: tuning.default_distribution("silence", tuning.stage_tier(stage)) for stage in _STAGES
}
MEMORY_SOURCE_DIST: dict[str, dict[str, float]] = {
    stage: tuning.default_distribution("memory", tuning.stage_tier(stage)) for stage in _STAGES
}


def _weighted_choice(weights: dict[str, float], *, default: str = "greeting") -> str:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return default
    r = random.random() * total
    acc = 0.0
    last = default
    for name, w in weights.items():
        if w <= 0:
            continue
        acc += w
        last = name
        if r <= acc:
            return name
    # 浮点累加误差让 r 落在最后一点点外: 归给最后一个有权重的项 (不能是权重 0 的项)
    return last


def select_topic_source(stage: str, trigger_type: str) -> str:
    """spec §4.1/§4.2: 根据 stage + trigger_type 抽一个话题来源.

    Returns one of:
      - silence_wakeup:  ai_l1 / ai_l2 / ai_schedule / user_l1 / user_l2 / greeting / recent_dialogue
      - memory_proactive: ai_l1 / ai_l2 / user_l1 / user_l2 / relationship (Phase 2 共同经历)
                          / recent_dialogue (近两天带时间戳的聊天记录)
      - scheduled_scene:  ai_schedule (固定)
    """
    if trigger_type == "scheduled_scene":
        return "ai_schedule"
    if trigger_type == "silence_wakeup":
        return _weighted_choice(tuning.distribution("silence", stage))
    if trigger_type == "memory_proactive":
        return _weighted_choice(tuning.distribution("memory", stage), default="ai_l1")
    return "greeting"
