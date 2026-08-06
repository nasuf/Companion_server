"""一局游戏在这个用户的历史里有多稀有.

## 为什么需要它

"这局值不值得记住"本质是**相对**的: 用户第一次赢你, 值得记; 第二十次赢, 不值得。
同样是连跳五格, 新手做到和老手做到完全两回事。

这种判断交给 LLM 会算错 —— 它看不到历史, 只能看这一局。而"是不是首次/破纪录/
连胜"恰恰是 SQL 一句话的事, 而且免费、准确。所以分工是: **这里算客观稀有性,
LLM 拿着它判断说什么、要不要记**。

## 刻意不做的

不在这里判断"精彩不精彩"。那是主观的、跨游戏的, 写成规则要给 10 个游戏各配一套
阈值, 每加一个新游戏重写一遍 —— 交给 LLM 更合适。这里只回答可以被计算的问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.db import db
from app.services.games.substance import action_floor


@dataclass
class GameRarity:
    """一局在用户历史里的位置。字段都可能为 None —— 数据不足时不猜。"""

    #: 这个游戏此前玩过多少局 (不含本局)
    prior_games: int = 0
    #: 此前赢过多少局
    prior_wins: int = 0
    #: 本局是不是该游戏的第一局
    is_first_ever: bool = False
    #: 本局是不是第一次赢这个游戏
    is_first_win: bool = False
    #: 当前连胜/连败 (含本局, 正数连胜负数连败; 0 表示打断了)
    streak: int = 0
    #: 本局步数是不是历史最少 (只对完局有意义)
    is_fewest_moves: bool = False
    #: 本局时长是不是历史最长
    is_longest: bool = False
    #: 人类可读的稀有性描述, 供 prompt 使用; 没有任何稀有点时为空
    notes: list[str] = field(default_factory=list)

    @property
    def is_notable(self) -> bool:
        """有没有任何客观上值得一提的地方."""
        return bool(self.notes)


def _describe(r: GameRarity, title: str) -> list[str]:
    notes: list[str] = []
    if r.is_first_ever:
        notes.append(f"这是你们第一次一起玩《{title}》")
    if r.is_first_win:
        notes.append(f"这是用户第一次在《{title}》赢")
    if r.streak >= 3:
        notes.append(f"用户已经连赢 {r.streak} 局")
    elif r.streak <= -3:
        notes.append(f"用户已经连输 {abs(r.streak)} 局")
    if r.is_fewest_moves and r.prior_games >= 3:
        notes.append("这是步数最少的一局")
    if r.is_longest and r.prior_games >= 3:
        notes.append("这是玩得最久的一局")
    return notes


async def compute_rarity(
    *,
    workspace_id: str | None,
    game_key: str,
    game_title: str,
    session_id: str,
    user_outcome: str | None,
    action_count: int,
    duration_seconds: int | None,
) -> GameRarity:
    """算这一局在该 workspace 的历史里有多特别.

    全链路吞异常: 稀有性是锦上添花, 算不出来不该让游戏结束流程失败。
    """
    rarity = GameRarity()
    if not workspace_id:
        return rarity
    try:
        rows = await db.query_raw(
            """
            SELECT user_outcome, action_count, duration_seconds, ended_at
            FROM (
                SELECT
                    (result->>'user_outcome') AS user_outcome,
                    COALESCE(
                        (result->'process'->game_key->>'action_count')::int,
                        (result->'gomoku'->>'move_count')::int,
                        0
                    ) AS action_count,
                    duration_seconds,
                    COALESCE(ended_at, created_at) AS ended_at
                FROM game_sessions
                WHERE workspace_id = $1
                  AND game_key = $2
                  AND id <> $3
                  AND status = 'settled'
                  -- 只跟"真的玩起来过"的局比。中途退出也判负 → 也是 settled,
                  -- 实测象棋的 settled 步数中位数是 0。不筛的话比较池全是 2 秒的
                  -- 退出局, 于是一局 18 秒会被评成"这是玩得最久的一局"(生产实例)。
                  AND COALESCE(
                        (result->'process'->game_key->>'action_count')::int,
                        (result->'gomoku'->>'move_count')::int,
                        0
                      ) >= $4
                -- 只看最近这些局。这个查询跑在**每局结束的热路径**上, 而
                -- game_sessions 上没有 workspace_id 索引 (现有索引是 user_id /
                -- agent_id / conversation_id), 走的是扫表。当前数据量下 60ms 左右,
                -- 瓶颈还在网络往返, 但表会随游戏量线性增长。
                --
                -- 限行不影响判定: 连胜最多看几局、首次/首胜只要有一条就能否定,
                -- "步数最少/时长最长"取近 200 局的极值与全历史极值实际差别极小 ——
                -- 而且"这半年最快的一局"本来就比"有史以来"更接近人的感受。
                ORDER BY COALESCE(ended_at, created_at) DESC
                LIMIT 200
            ) prior
            ORDER BY ended_at DESC
            """,
            workspace_id, game_key, session_id, action_floor(game_key),
        )
    except Exception:
        return rarity

    rarity.prior_games = len(rows)
    rarity.prior_wins = sum(1 for r in rows if r.get("user_outcome") == "win")
    rarity.is_first_ever = rarity.prior_games == 0
    rarity.is_first_win = user_outcome == "win" and rarity.prior_wins == 0

    # 连胜/连败: 从本局往前数, 遇到不同结果就停。平局和中断打断连续性 ——
    # "连赢三局"里夹一局平局就不该再算连赢。
    if user_outcome in ("win", "lose"):
        streak = 1
        for r in rows:  # 已按时间倒序
            if r.get("user_outcome") != user_outcome:
                break
            streak += 1
        rarity.streak = streak if user_outcome == "win" else -streak

    if user_outcome in ("win", "lose", "draw") and action_count > 0:
        prior_counts = [
            int(r["action_count"]) for r in rows if (r.get("action_count") or 0) > 0
        ]
        rarity.is_fewest_moves = bool(prior_counts) and action_count < min(prior_counts)
    if duration_seconds:
        prior_dur = [int(r["duration_seconds"]) for r in rows if r.get("duration_seconds")]
        rarity.is_longest = bool(prior_dur) and duration_seconds > max(prior_dur)

    rarity.notes = _describe(rarity, game_title)
    return rarity
