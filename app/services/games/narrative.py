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

# AI 决策快照里真正有的字段 (2026-08 生产实测 761 条 ai_move_decided):
#
#     algorithm / score / depth|search_depth / nodes_searched /
#     principal_variation / candidates_considered / move|swap|algebraic
#
# **没有 reason 字段** —— 第一版按一条围棋样本假设有 reason 代号 (如
# territory_and_influence), 实测 740/761 条为 None, 剩下的是扫雷的整句中文。
# 按代号做白名单等于只覆盖 2/60 局。
#
# 现在改用 score + depth: 这两个才是 AI "当时怎么想"的通用表达 —— 局面评分说明
# 它觉得自己占优还是吃紧, 搜索深度说明它算了多远。所有对弈类引擎都产出这两个,
# 而且它们是**数值**, 措辞完全交给 prompt, 代码里一个面向用户的字都没有。

# score 的量纲各游戏不同 (黑白棋是子差, 象棋是兵值), 所以只取符号不取绝对值 ——
# 跨游戏比较数值毫无意义, 但"它当时觉得自己占优/吃紧"是通用的。
_SCORE_ADVANTAGE = 30
_SCORE_TROUBLE = -30

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
    """从 AI 决策快照里挑出"它当时怎么想的".

    输出的是**信号**不是措辞 (`占优 / 吃紧 / 算了 6 步`), 措辞交给 prompt。

    只取局面**转折**的那两步: 每步都报评分会变成流水账, 而"从吃紧翻到占优"那一刻
    才是有故事的地方。
    """
    decided = [
        s for s in snapshots
        if isinstance(s, dict) and s.get("event_type") == "ai_move_decided"
    ]
    if not decided:
        return []

    out: list[str] = []
    prev_sign: int | None = None
    for snap in decided:
        raw = snap.get("score")
        if not isinstance(raw, (int, float)):
            continue
        sign = 1 if raw >= _SCORE_ADVANTAGE else (-1 if raw <= _SCORE_TROUBLE else 0)
        if prev_sign is not None and sign != prev_sign and sign != 0:
            # 刻意**不给坐标**。原本想让"我本来想走 M3"更具体, 实测适得其反:
            #   1. 模型会把 AI 自己的落子说成用户的 ("你 c8b6 那步")
            #   2. 棋谱坐标对用户没有意义 —— 没人聊天时说"你 c8b6 那步"
            # 转折发生在第几步、往后算了多远, 这两个才是能讲成人话的。
            depth = snap.get("depth") or snap.get("search_depth")
            parts = ["中局开始感觉稳了" if sign > 0 else "中途一度感觉吃紧"]
            if isinstance(depth, int) and depth >= 3:
                parts.append(f"当时往后算了 {depth} 步")
            out.append("，".join(parts))
            if len(out) >= _MAX_DECISIONS:
                break
        if sign != 0:
            prev_sign = sign
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
