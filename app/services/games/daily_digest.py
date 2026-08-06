"""当天一起玩了什么 —— 供每日自我总结取用.

## 为什么放进每日总结, 而不是单开一个 job

`review_daily_schedule` 每天已经在做的事就是"把今天发生的事拼成素材, 让 LLM 写一段
自我回顾, 再拆成记忆条目"。游戏是今天发生的事之一, 并进去是自然的:

  - 不额外花 LLM 调用 (那次调用本来就要发生)
  - 产出走 `provenance="daily_summary"`, 语义正好是"日常琐事, 优先被整合压缩"
  - 让"今天陪他下了三盘棋, 他赢了两盘"和"今天下午在做皮具"出现在同一段回顾里 ——
    这才是一个人回想一天的方式, 而不是游戏单独一份台账

## 为什么是"当天聚合"而不是"逐局"

逐局写记忆已经在 games/native.py 收紧了 (只留客观稀有的那几局)。这里补的是另一个
维度: 单局不值得记, 但"今天陪他玩了三盘"值得 —— 那是**陪伴的密度**, 跟单局的
精彩程度是两回事。

聚合还顺带解决了模板化: 三局各写一条会得到三句结构雷同的话, 合成一句就没有这个
问题。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.db import db

logger = logging.getLogger(__name__)


@dataclass
class GameDayDigest:
    total: int = 0
    finished: int = 0
    user_wins: int = 0
    ai_wins: int = 0
    draws: int = 0
    aborted: int = 0
    #: 按局数排序的游戏名
    titles: list[str] = field(default_factory=list)
    #: 总时长 (分钟)
    minutes: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total == 0


def _titles_text(titles: list[str]) -> str:
    if not titles:
        return ""
    if len(titles) == 1:
        return f"《{titles[0]}》"
    if len(titles) == 2:
        return f"《{titles[0]}》和《{titles[1]}》"
    return "、".join(f"《{t}》" for t in titles[:2]) + f"等 {len(titles)} 种"


def render_digest(digest: GameDayDigest) -> str:
    """渲染成一句素材, 空则返回空串让调用方跳过整段.

    刻意只给**事实**不给措辞: 这段会喂进 daily_summary 的 LLM, 让它自己组织语言。
    在这里写好句子等于又造一个模板 —— 逐局记忆的模板化就是这么来的。
    """
    if digest.is_empty:
        return ""
    parts = [f"今天和用户一起玩了 {digest.total} 局{_titles_text(digest.titles)}"]
    if digest.minutes >= 1:
        parts.append(f"合计约 {digest.minutes} 分钟")
    if digest.finished:
        outcome = []
        if digest.user_wins:
            outcome.append(f"用户赢 {digest.user_wins} 局")
        if digest.ai_wins:
            outcome.append(f"我赢 {digest.ai_wins} 局")
        if digest.draws:
            outcome.append(f"平局 {digest.draws} 局")
        if outcome:
            parts.append("、".join(outcome))
    # 中断只在占比高时才提 —— 偶尔退出很正常, 但"开了五局跑了四局"是个信号
    if digest.aborted and digest.aborted >= max(2, digest.total * 0.6):
        parts.append(f"其中 {digest.aborted} 局中途退出了")
    return "，".join(parts) + "。"


async def collect_today_games(
    *, workspace_id: str | None, local_day_start, local_day_end=None
) -> GameDayDigest:
    """汇总一天的对局.

    `local_day_end` 不传时取 start + 24h。必须有上界: 调用方 (每日回顾) 在凌晨 4:00
    跑、回顾的是**前一天**, 没有上界会把今天凌晨那几个小时也算进昨天的账。

    失败返回空 digest —— 总结不该因为游戏查询挂掉而失败, 但会记日志 (见下)。
    """
    from datetime import timedelta
    digest = GameDayDigest()
    if not workspace_id:
        return digest
    try:
        rows = await db.query_raw(
            """
            SELECT
                COALESCE(result->>'game_title', game_key) AS title,
                status,
                (result->>'user_outcome') AS outcome,
                COALESCE(duration_seconds, 0) AS dur
            FROM game_sessions
            WHERE workspace_id = $1
              AND COALESCE(ended_at, created_at) >= $2::timestamptz AT TIME ZONE 'UTC'
              AND COALESCE(ended_at, created_at) <  $3::timestamptz AT TIME ZONE 'UTC'
            """,
            workspace_id,
            local_day_start.isoformat(),
            (local_day_end or (local_day_start + timedelta(days=1))).isoformat(),
        )
    except Exception as exc:
        # 显式记日志而不是静默吞。第一版没有 ::timestamptz 转换, Prisma 把 datetime
        # 序列化成 text 与 timestamp 列比较直接报错 —— 而 except 把它吃掉了, 表现
        # 成"今天没玩游戏"。静默失败比报错危险得多: 没人会发现总结里少了一块。
        logger.warning("collect_today_games failed for %s: %s", workspace_id, exc)
        return digest

    seen: dict[str, int] = {}
    total_seconds = 0
    for r in rows:
        digest.total += 1
        total_seconds += int(r.get("dur") or 0)
        title = r.get("title") or "游戏"
        seen[title] = seen.get(title, 0) + 1
        if r.get("status") == "settled":
            digest.finished += 1
            outcome = r.get("outcome")
            if outcome == "win":
                digest.user_wins += 1
            elif outcome == "lose":
                digest.ai_wins += 1
            elif outcome == "draw":
                digest.draws += 1
        else:
            digest.aborted += 1

    digest.titles = [t for t, _ in sorted(seen.items(), key=lambda x: -x[1])]
    digest.minutes = total_seconds // 60
    return digest
