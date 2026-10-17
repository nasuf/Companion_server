"""主动交流可调参数: 后台「聊天管理 → 主动交流」读写的唯一登记表。

每个参数的默认值、取值范围和给运营看的说明只在这里定义一次:
  - 后台页面按这份表自动生成 (新增参数只改这里)
  - 主动交流各模块运行时经 `value()` / `distribution()` 读取生效值
  - 后台改过的值存在 system_config.proactive_policy (只存与默认不同的项),
    经 runtime_config 的版本号刷新同步到所有 worker (≤2s)

默认值即上线前的硬编码值 (第四部分主动交流规格 + 《主动聊天机制（新增）》),
没有任何覆盖时行为与改造前完全一致。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Kind = Literal["percent", "minutes", "count", "hour", "bool"]

_SUM_TOLERANCE = 0.0005
_PERCENT_DECIMALS = 4  # 存小数, 精度 0.01%


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    description: str


@dataclass(frozen=True)
class SumGroup:
    """同一张配比表: 成员合计必须为 100%."""
    key: str
    group: str
    title: str


@dataclass(frozen=True)
class Param:
    key: str
    group: str
    label: str
    description: str
    default: float | int | bool
    kind: Kind
    minimum: float = 0.0
    maximum: float = 1.0
    sum_group: str | None = None


GROUPS: tuple[Group, ...] = (
    Group(
        "a_windows", "长时间没聊后的主动搭话（A 模式）· 时间窗命中率",
        "话题判定为已聊完（或本次会话已追问过）后进入 A 模式。AI 说完最后一句后，"
        "每个时间段到点时按命中率决定是否主动发一条；发出后本轮停止，用户回复后重新计时。"
        "实际命中率 = 设定值 × 用户作息节律系数（0.55–1.25，近 30 天同一时段的回复情况），"
        "再受上限封顶。",
    ),
    Group(
        "a_trigger", "A 模式 · 触发类型配比",
        "时间窗命中后，这一条主动消息属于哪一类。",
    ),
    Group(
        "silence_sources", "沉默唤醒 · 话题来源",
        "按话题亲密度分三张表：0–40 冷启动 / 41–80 升温 / 81–100 亲密，每张表合计必须为 100%。"
        "抽到的来源没有素材时改为打招呼；近两天对话里出现过危机照护时也视为没有素材。",
    ),
    Group(
        "memory_sources", "记忆主动 · 话题来源",
        "按话题亲密度分三张表，每张表合计必须为 100%。抽到的记忆为空时本次取消发送；"
        "近两天对话为空时改用 AI 核心记忆。",
    ),
    Group(
        "a_extras", "A 模式 · 其他随机项",
        "叠加在话题来源之上的几个随机选择。",
    ),
    Group(
        "b_mode", "话题没聊完的追问（B 模式）与被动承接",
        "AI 说完最后一句后用户一段时间没回，小模型判定话题是否聊完：没聊完且过了门槛就追问一句；"
        "用户回来时按判定结果决定要不要先说一句承接 / 过渡。B 模式本身没有概率，满足条件即发送。",
    ),
    Group(
        "general", "通用频率与时段",
        "所有主动消息共用的防打扰规则（用户自己设的提醒不受这些限制）。",
    ),
)

SUM_GROUPS: tuple[SumGroup, ...] = (
    SumGroup("trigger_split", "a_trigger", "时间窗命中后"),
    SumGroup("trigger_fallback", "a_trigger", "定时情景不可用时（AI 在睡觉或今天已发过），以及连续多次没回复后的强制发送"),
    SumGroup("silence_cold", "silence_sources", "冷启动（亲密度 0–40）"),
    SumGroup("silence_warming", "silence_sources", "升温（亲密度 41–80）"),
    SumGroup("silence_intimate", "silence_sources", "亲密（亲密度 81–100）"),
    SumGroup("memory_cold", "memory_sources", "冷启动（亲密度 0–40）"),
    SumGroup("memory_warming", "memory_sources", "升温（亲密度 41–80）"),
    SumGroup("memory_intimate", "memory_sources", "亲密（亲密度 81–100）"),
)

# (来源 key, 标签, 说明)
_SILENCE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("greeting", "打招呼", "不带特定话题，结合当下时段轻松问候"),
    ("ai_l1", "AI 核心记忆", "从 AI 自己的核心人设 / 经历里找话题"),
    ("ai_l2", "AI 次要记忆", "从 AI 自己的近期经历里找话题"),
    ("ai_schedule", "AI 当前作息", "分享 AI 此刻正在做的事"),
    ("user_l1", "用户核心记忆", "从用户的核心事实（身份、长期偏好）里找话题"),
    ("user_l2", "用户次要记忆", "从用户近期提到的事里找话题"),
    ("recent_dialogue", "近两天对话", "接着近 48 小时聊过的内容，关心后续或轻度延伸"),
)
_MEMORY_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("ai_l1", "AI 核心记忆", "从 AI 自己的核心人设 / 经历里找话题"),
    ("ai_l2", "AI 次要记忆", "从 AI 自己的近期经历里找话题"),
    ("user_l1", "用户核心记忆", "从用户的核心事实（身份、长期偏好）里找话题"),
    ("user_l2", "用户次要记忆", "从用户近期提到的事里找话题"),
    ("relationship", "共同经历", "你们之间发生过的事"),
    ("recent_dialogue", "近两天对话", "接着近 48 小时聊过的内容，关心后续或轻度延伸"),
)
_SILENCE_DEFAULTS: dict[str, dict[str, float]] = {
    "cold": {"greeting": 0.70, "ai_l1": 0.20, "ai_schedule": 0.10},
    "warming": {
        "greeting": 0.60, "ai_l1": 0.10, "ai_schedule": 0.10,
        "user_l1": 0.05, "user_l2": 0.05, "recent_dialogue": 0.10,
    },
    "intimate": {
        "greeting": 0.50, "ai_l1": 0.05, "ai_l2": 0.05, "ai_schedule": 0.10,
        "user_l1": 0.10, "user_l2": 0.10, "recent_dialogue": 0.10,
    },
}
# 冷启动不追问对方的事 (还不熟就问东问西像查户口), 也没有共同经历可聊
_MEMORY_DEFAULTS: dict[str, dict[str, float]] = {
    "cold": {"ai_l1": 1.00},
    "warming": {
        "ai_l1": 0.50, "ai_l2": 0.05, "user_l1": 0.10, "user_l2": 0.10,
        "relationship": 0.10, "recent_dialogue": 0.15,
    },
    "intimate": {
        "ai_l1": 0.30, "ai_l2": 0.10, "user_l1": 0.15, "user_l2": 0.15,
        "relationship": 0.15, "recent_dialogue": 0.15,
    },
}
STAGE_TIERS = ("cold", "warming", "intimate")


def _source_params(prefix: str, group: str, sources, defaults) -> list[Param]:
    return [
        Param(
            f"{prefix}_{tier}_{source}", group, label, description,
            defaults[tier].get(source, 0.0), "percent", sum_group=f"{prefix}_{tier}",
        )
        for tier in STAGE_TIERS
        for source, label, description in sources
    ]


PARAMS: tuple[Param, ...] = (
    Param("a_window_1_rate", "a_windows", "30 分钟–1 小时",
          "AI 说完后 30 分钟到 1 小时之间到点时，发一条的概率", 0.05, "percent"),
    Param("a_window_2_rate", "a_windows", "1–2 小时",
          "1 到 2 小时之间到点时，发一条的概率", 0.12, "percent"),
    Param("a_window_3_rate", "a_windows", "2–4 小时",
          "2 到 4 小时之间到点时，发一条的概率", 0.25, "percent"),
    Param("a_window_4_rate", "a_windows", "4–6 小时",
          "4 到 6 小时之间到点时，发一条的概率；6 小时后从头开始新一轮", 0.35, "percent"),
    Param("a_window_rate_cap", "a_windows", "单个时间窗命中率上限",
          "按作息节律放大后的封顶值", 0.50, "percent"),

    Param("trigger_silence_wakeup", "a_trigger", "沉默唤醒",
          "没有特定由头，随口搭话或分享", 0.30, "percent", sum_group="trigger_split"),
    Param("trigger_memory_proactive", "a_trigger", "记忆主动",
          "从 AI 或用户的记忆里找一个话题", 0.30, "percent", sum_group="trigger_split"),
    Param("trigger_scheduled_scene", "a_trigger", "定时情景",
          "分享 AI 此刻作息里的状态（每天最多一次）", 0.40, "percent", sum_group="trigger_split"),
    Param("fallback_silence_wakeup", "a_trigger", "沉默唤醒",
          "改发沉默唤醒的比例", 0.50, "percent", sum_group="trigger_fallback"),
    Param("fallback_memory_proactive", "a_trigger", "记忆主动",
          "改发记忆主动的比例", 0.50, "percent", sum_group="trigger_fallback"),

    *_source_params("silence", "silence_sources", _SILENCE_SOURCES, _SILENCE_DEFAULTS),
    *_source_params("memory", "memory_sources", _MEMORY_SOURCES, _MEMORY_DEFAULTS),

    Param("music_probability", "a_extras", "音乐推荐",
          "沉默唤醒时改为推荐一首歌的概率（不分亲密阶段；AI 此刻不空闲时本次取消）", 0.04, "percent"),
    Param("trending_interest_first", "a_extras", "热点：爱好优先",
          "命中热点且用户有爱好记忆时，先按爱好挑热点的概率（挑不中再随机挑一条）", 0.50, "percent"),
    Param("recent_dialogue_care_followup", "a_extras", "近两天对话：关心追问后续",
          "来源为近两天对话时，「关心追问后续」的占比，其余为「话题轻度延伸」", 0.50, "percent"),

    Param("followup_judge_delay_minutes", "b_mode", "判定等待时间",
          "AI 说完最后一句后，用户多久没回复就开始判定话题有没有聊完", 5, "minutes", 1, 60),
    Param("followup_max_per_session", "b_mode", "单次会话最多追问次数",
          "规格要求为 1（单会话仅一次）；设为 0 即关闭 B 模式追问（判定照跑，供被动承接使用）",
          1, "count", 0, 3),
    Param("followup_respect_active_hours", "b_mode", "追问遵守主动时段",
          "开启时深夜不追问；关闭后用户深夜还在聊、话题没聊完时也会追问一句（A 模式仍遵守时段）",
          True, "bool"),
    Param("return_line_min_gap_minutes", "b_mode", "回归承接的间隔阈值",
          "话题没聊完、本次会话没追问过时，用户隔多久回来，AI 先说一句承接时间差的话", 10, "minutes", 1, 180),

    Param("active_hour_start", "general", "主动时段开始",
          "每天几点开始允许发主动消息（北京时间）", 8, "hour", 0, 23),
    Param("active_hour_end", "general", "主动时段结束",
          "到几点为止（不含这个整点），之后到次日开始前都不发", 22, "hour", 1, 24),
    Param("cooldown_minutes", "general", "全局冷却",
          "任意一条主动消息发出后，多久之内不再发下一条", 30, "minutes", 0, 240),
    Param("daily_max", "general", "每日上限",
          "每个用户每天最多收到几条 A 模式主动消息（B 追问不计入）", 3, "count", 0, 10),
)

PARAM_MAP: dict[str, Param] = {p.key: p for p in PARAMS}
DEFAULTS: dict[str, Any] = {p.key: p.default for p in PARAMS}
_SUM_MEMBERS: dict[str, tuple[Param, ...]] = {
    g.key: tuple(p for p in PARAMS if p.sum_group == g.key) for g in SUM_GROUPS
}


class TuningError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("；".join(errors))
        self.errors = errors


# ────────────────────────────────────────────────────────────────────
# 校验
# ────────────────────────────────────────────────────────────────────

def _coerce(param: Param, raw: Any) -> float | int | bool:
    """取值合法化; 不合法抛 ValueError (消息直接给后台看)."""
    if param.kind == "bool":
        if not isinstance(raw, bool):
            raise ValueError(f"「{param.label}」必须是开关值")
        return raw
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"「{param.label}」必须是数字")
    if not (param.minimum <= raw <= param.maximum):
        if param.kind == "percent":
            raise ValueError(
                f"「{param.label}」须在 {param.minimum:.0%}–{param.maximum:.0%} 之间",
            )
        raise ValueError(f"「{param.label}」须在 {param.minimum:g}–{param.maximum:g} 之间")
    if param.kind == "percent":
        return round(float(raw), _PERCENT_DECIMALS)
    if float(raw) != int(raw):
        raise ValueError(f"「{param.label}」必须是整数")
    return int(raw)


def _check_consistency(values: Mapping[str, Any]) -> list[str]:
    errors = []
    for group in SUM_GROUPS:
        total = sum(float(values[p.key]) for p in _SUM_MEMBERS[group.key])
        if abs(total - 1.0) > _SUM_TOLERANCE:
            title = next(g.title for g in GROUPS if g.key == group.group)
            errors.append(f"「{title} · {group.title}」合计为 {total:.1%}，必须为 100%")
    if values["active_hour_start"] >= values["active_hour_end"]:
        errors.append("主动时段开始必须早于结束")
    return errors


def apply_patch(current: Mapping[str, Any] | None, patch: Mapping[str, Any]) -> dict[str, Any]:
    """后台保存: 把 patch (key → 新值, None = 恢复默认) 合并进已存覆盖, 返回新的覆盖.

    整体校验 (配比合计 / 时段先后) 在合并后进行; 与默认相同的值不存, 保持覆盖最小。
    """
    errors: list[str] = []
    merged = {k: v for k, v in (current or {}).items() if k in PARAM_MAP}
    for key, raw in patch.items():
        param = PARAM_MAP.get(key)
        if param is None:
            errors.append(f"未知参数：{key}")
            continue
        if raw is None:
            merged.pop(key, None)
            continue
        try:
            merged[key] = _coerce(param, raw)
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise TuningError(errors)
    errors = _check_consistency({**DEFAULTS, **merged})
    if errors:
        raise TuningError(errors)
    return {k: v for k, v in merged.items() if v != DEFAULTS[k]}


def resolve(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """已存覆盖 → 全量生效值. 只信任通过校验的部分, 绝不因为一份坏数据让主动交流停摆.

    逐项校验失败的回落默认; 校验后某张配比表合计不为 100% (比如登记表新增了来源)
    整张表回落默认; 时段先后矛盾两项一起回落。
    """
    values = dict(DEFAULTS)
    for key, raw in (overrides or {}).items():
        param = PARAM_MAP.get(key)
        if param is None:
            continue
        try:
            values[key] = _coerce(param, raw)
        except ValueError as e:
            logger.warning(f"[PROACTIVE-TUNING] ignored stored value: {e}")
    for group in SUM_GROUPS:
        members = _SUM_MEMBERS[group.key]
        if abs(sum(float(values[p.key]) for p in members) - 1.0) > _SUM_TOLERANCE:
            logger.warning(f"[PROACTIVE-TUNING] {group.key} does not sum to 100%; using defaults")
            for p in members:
                values[p.key] = p.default
    if values["active_hour_start"] >= values["active_hour_end"]:
        values["active_hour_start"] = DEFAULTS["active_hour_start"]
        values["active_hour_end"] = DEFAULTS["active_hour_end"]
    return values


# ────────────────────────────────────────────────────────────────────
# 运行时读取
# ────────────────────────────────────────────────────────────────────

_memo: tuple[object, dict[str, Any]] | None = None


def current() -> dict[str, Any]:
    """当前生效的全量参数 (随 runtime_config 快照刷新)."""
    global _memo
    from app.services.runtime_config import proactive_policy_overrides

    raw = proactive_policy_overrides()
    memo = _memo
    if memo is not None and memo[0] is raw:
        return memo[1]
    values = resolve(raw)
    _memo = (raw, values)
    return values


def value(key: str) -> Any:
    return current()[key]


def stage_tier(stage: str) -> str:
    """主动交流阶段 (p1_cold / p2_cold / cold_start / warming / intimate) → 配比表档位."""
    return stage if stage in ("warming", "intimate") else "cold"


def distribution(prefix: str, stage: str) -> dict[str, float]:
    """某张话题来源配比表: {来源: 权重}."""
    group = f"{prefix}_{stage_tier(stage)}"
    values = current()
    return {p.key.removeprefix(f"{group}_"): float(values[p.key]) for p in _SUM_MEMBERS[group]}


def default_distribution(prefix: str, tier: str) -> dict[str, float]:
    group = f"{prefix}_{tier}"
    return {p.key.removeprefix(f"{group}_"): float(p.default) for p in _SUM_MEMBERS[group]}


def window_rate(index: int) -> float:
    if index < 1 or index > 4:
        return 0.0  # window 0 是 B 模式判定窗, 不参与概率
    return float(value(f"a_window_{index}_rate"))
