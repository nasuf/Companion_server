"""一局游戏有没有"真的玩起来".

## 为什么需要这个判断

中途退出也会给用户判负, 于是那些局在库里是 `status='settled'` —— 跟真下完一局
无法区分。实测 (2026-08):

| 游戏 | settled 局数 | 步数中位 | 最少步 |
|---|---|---|---|
| 国际象棋 | 9 | **0** | 0 |
| 中国象棋 | 23 | **0** | 0 |
| 围棋 | 10 | **0** | 0 |
| 五子棋 | 143 | 20 | 9 |
| 黑白棋 | 30 | 60 | 58 |

象棋/围棋一局真棋都没下完过 —— 全部是开局就走。而这些局此前会走完整的完局链路:
调 LLM 生成走心复盘、算稀有性、写记忆。生产上真实发生过一局 **4 步 18 秒**的棋
产出「本来以为几步就能定局, 结果那步关键的交换把局势彻底搅活了, 直到最后才敢松
口气」—— 一场不存在的对局。

`ended_reason` 恒为 "settled"、`terminal_state.result` 恒为 null, 没有任何字段能
区分"真赢"和"退出判负"。所以只能看**做了多少步**。

## 为什么阈值按游戏分

扫雷 3 步扫完小盘是真的赢, 象棋 4 步不可能。这不是主观的"精彩程度"判断 (那种该
交给 LLM), 而是每个游戏规则决定的客观下限, 所以适合写成表。

阈值取自"理论最快结束"与实测最小值中较可靠的那个:
五子棋 9 = 先手第 5 子成五 (用户 5 手 + AI 4 手), 恰好等于实测最小值。

**已知取舍**: 国际象棋的"愚人杀"理论上 4 手就能将杀, 会被这里判成没玩起来。
接受这个漏判 —— 它极其罕见, 而现状是 100% 的退出局都在编故事。
"""

from __future__ import annotations

# 每个游戏"算是玩过一局"的最少动作数。缺省的游戏按 _DEFAULT_FLOOR 处理。
_ACTION_FLOOR = {
    "gomoku": 9,             # 先手第 5 子成五的理论最小值, 与实测最小值一致
    "reversi": 20,           # 天然要填到 58+ 手, 设低位留出"提前无子可下"的余地
    "chess": 10,             # 见 docstring 的愚人杀取舍
    "xiangqi": 10,
    "go": 20,                # 围棋真局是几十手起
    "chinese_checkers": 20,  # 棋子要走完全场
    "match3": 4,             # 实测完成目标的最小步数
    "minesweeper": 3,        # 小盘 3 步扫完是真的
    "number_merge": 8,
    "tetris_duel": 10,
}
_DEFAULT_FLOOR = 8


def action_floor(game_key: str) -> int:
    return _ACTION_FLOOR.get(game_key, _DEFAULT_FLOOR)


def played_enough(game_key: str, action_count: int | None) -> bool:
    """这局是不是真的玩起来了 —— 而不是开局就走 (退出同样判负落 settled)."""
    return (action_count or 0) >= action_floor(game_key)
