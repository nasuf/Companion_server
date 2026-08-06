"""把一局游戏的过程数据整理成「能讲的素材」.

## 为什么需要它

结束文案原来是一串硬编码 if/else (`if count <= 2: return "《X》才刚开头…"`), 不带
人设、不带记忆、不带情绪 —— 用户说它"机械"是字面意义上的准确。

而讽刺的是, **最丰富的数据喂给了最贫瘠的生成路径**。库里每一步 AI 决策都存着:

    {"move": {"coordinate": "C17"}, "reason": "territory_and_influence",
     "algorithm": "uct_mcts_pattern_capture_rollout", "simulations": 220,
     "top_candidates": [{"move": {"coordinate": "M3"}, "prior": 1.98, ...}]}

agent 明明"想"过 220 种可能、权衡过别的选点, 嘴上却只会说"先停在这里吧"。这份数据
是我们相对其他陪伴产品的**结构性优势** —— 它们的游戏 AI 是黑箱, 说不出"我本来想走
M3 抢右边, 后来觉得先把 C17 围起来更稳"。

## 这里只做整理, 不做措辞

输出的是事实清单, 让 LLM 自己组织语言。在这里拼句子等于又造一个模板 —— 逐局记忆
的模板化就是这么来的 (实测那批记忆两两相似度中位 0.710, 普通记忆才 0.361)。

## 数据实况 (2026-08 生产盘点)

745 局里只有 233 局完局 (31%), 中位时长 11-85 秒。而且**中断局大多没有过程数据**
(actions 为空 —— 只有完局才回填)。所以叙事素材只对完局有意义, 中断局保持轻量。

各游戏丰富度差别很大: gomoku 251 局/57% 完局率最富, chess 和 ludo 基本是死的
(39 局 0 完局、1 个关键时刻)。素材组装对所有游戏统一处理, 但空就是空, 不硬凑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 认得的决策理由代码。**只做白名单, 不在这里翻译成人话** —— 那些措辞放在
# game.finish_reply 的词表里, 后台可编辑。
#
# 理由: 这十条是"AI 决策理由变成人话"的唯一出口, 也是这个功能最有价值的部分
# (别家的游戏 AI 是黑箱, 说不出"我本来想走 M3 抢右边")。文案写死在代码里, 后台
# 调不了; 而且预写短语会让模型照抄, 反倒失去结合上下文措辞的机会。
#
# 白名单仍留在代码里: 引擎随时会加新 reason, 没在名单里的直接丢比塞给模型一个
# 它看不懂的 raw code 好。
_KNOWN_REASONS = frozenset({
    "territory_and_influence", "capture_threat", "block_win", "double_threat",
    "extend_line", "center_control", "defend", "safe_merge", "open_space",
    "clear_lines",
})

# 一局最多给几条素材。给太多 LLM 会平铺直叙念清单, 反而更像报告。
_MAX_MOMENTS = 3
_MAX_DECISIONS = 2


@dataclass
class GameNarrative:
    """一局的叙事素材。字段空就是空 —— 不硬凑。"""

    title: str = ""
    outcome: str = ""          # 用户视角: win / lose / draw / aborted
    is_cooperative: bool = False
    action_count: int = 0
    minutes: int = 0
    #: 引擎标出的高光时刻 (人话)
    moments: list[str] = field(default_factory=list)
    #: AI 自己的决策理由 (人话), 这是别家给不出的东西
    decisions: list[str] = field(default_factory=list)
    #: 客观稀有性 (来自 rarity.py)
    rarity_notes: list[str] = field(default_factory=list)

    @property
    def has_substance(self) -> bool:
        """有没有值得让 LLM 讲一讲的东西。全空时不值得调模型。"""
        return bool(self.moments or self.decisions or self.rarity_notes)


def _decision_texts(snapshots: list[Any]) -> list[str]:
    """从 AI 决策快照里挑出"它当时在想什么".

    输出的是 `reason 代码（坐标）`, 由 prompt 里的词表翻译成人话 —— 见
    `_KNOWN_REASONS` 的说明。同一个 reason 不重复 (下十步都是"想围地"讲一次就够)。
    """
    out: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        if not isinstance(snap, dict) or snap.get("event_type") != "ai_move_decided":
            continue
        reason = str(snap.get("reason") or "")
        if reason not in _KNOWN_REASONS or reason in seen:
            continue
        seen.add(reason)
        # 有坐标就带上 —— "我本来想走 M3" 比"我本来想走别处"具体得多
        move = snap.get("move")
        coord = ""
        if isinstance(move, dict) and move.get("coordinate"):
            coord = f"（{move['coordinate']}）"
        out.append(f"{reason}{coord}")
        if len(out) >= _MAX_DECISIONS:
            break
    return out


def build_narrative(
    *,
    title: str,
    outcome: str,
    is_cooperative: bool,
    action_count: int,
    duration_seconds: int | None,
    moment_texts: list[str],
    snapshots: list[Any] | None,
    rarity_notes: list[str],
) -> GameNarrative:
    return GameNarrative(
        title=title,
        outcome=outcome,
        is_cooperative=is_cooperative,
        action_count=action_count,
        minutes=(duration_seconds or 0) // 60,
        moments=[m for m in moment_texts if m][:_MAX_MOMENTS],
        decisions=_decision_texts(snapshots or []),
        rarity_notes=list(rarity_notes),
    )


_OUTCOME_TEXT = {
    "win": "用户赢了",
    "lose": "我赢了",
    "draw": "平局",
    "aborted": "中途停下了",
}
_COOP_OUTCOME_TEXT = {
    "win": "一起过关了",
    "lose": "没能过关",
    "aborted": "中途停下了",
}


def render_material(n: GameNarrative) -> str:
    """渲染成给 LLM 的事实清单 —— 只给事实, 不给措辞."""
    lines = [f"- 游戏：《{n.title}》"]
    table = _COOP_OUTCOME_TEXT if n.is_cooperative else _OUTCOME_TEXT
    lines.append(f"- 结果：{table.get(n.outcome, '结束了')}")
    if n.action_count:
        lines.append(f"- 走了 {n.action_count} 步" + (f"，约 {n.minutes} 分钟" if n.minutes else ""))
    if n.rarity_notes:
        lines.append("- 特别之处：" + "；".join(n.rarity_notes))
    if n.moments:
        lines.append("- 局中发生过：" + "；".join(n.moments))
    if n.decisions:
        # 这条是 agent 的主观视角, 单独标出来让 LLM 知道可以用第一人称讲
        lines.append("- 我当时的判断：" + "；".join(n.decisions))
    return "\n".join(lines)
