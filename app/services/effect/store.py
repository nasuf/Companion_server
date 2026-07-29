"""效果指标的缓存与批量读取.

不建新表: 指标全部由 messages / conversations 现有数据算出, 任何一天都能重算。
再存一份等于多一处需要跟原始数据保持一致的状态, 而它唯一的好处只是省几次聚合。

改用 Redis 缓存: 一天过完之后数值就不再变 —— 唯一的例外是次日回访率, 它要等次日
也过完才定型。所以缓存 TTL 分两档, 定型的长存, 未定型的短存等下次重算。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.effect.signals import EffectMetrics, collect

logger = logging.getLogger(__name__)

_KEY_PREFIX = "effect:daily:"
# 已定型: 存够看年度趋势。未定型: 十分钟, 让当天的数随着聊天推进而刷新。
_TTL_SETTLED_S = 400 * 24 * 3600
_TTL_PENDING_S = 600


def _key(day: date_cls) -> str:
    return f"{_KEY_PREFIX}{day.isoformat()}"


def _is_settled(day: date_cls, now: datetime) -> bool:
    """这一天的数值定型了没有.

    要等次日也过完 —— 次日回访率在那之前一直在涨, 缓存下来就固化成一个偏低的错值。
    """
    return day <= (now.date() - timedelta(days=2))


@dataclass
class EffectRange:
    days: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"days": self.days, "summary": summarise(self.days)}


async def get_day(day: date_cls, *, force: bool = False) -> dict[str, Any]:
    """取一天的指标, 优先读缓存."""
    from app.redis_client import get_redis

    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    redis = None
    try:
        redis = await get_redis()
        if not force:
            cached = await redis.get(_key(day))
            if cached:
                return json.loads(cached)
    except Exception as exc:  # Redis 挂了也要能出数, 只是每次都重算
        logger.warning(f"effect: cache read failed for {day}: {exc}")

    payload = (await collect(day, now=now)).as_dict()

    if redis is not None:
        try:
            await redis.set(
                _key(day), json.dumps(payload, ensure_ascii=False),
                ex=_TTL_SETTLED_S if _is_settled(day, now) else _TTL_PENDING_S,
            )
        except Exception as exc:
            logger.warning(f"effect: cache write failed for {day}: {exc}")
    return payload


async def get_range(days: int = 14, *, force: bool = False) -> EffectRange:
    """最近 N 天 (含今天), 按日期升序."""
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    today = now.date()
    out: list[dict[str, Any]] = []
    for back in range(days - 1, -1, -1):
        out.append(await get_day(today - timedelta(days=back), force=force))
    return EffectRange(days=out)


def summarise(days: list[dict[str, Any]]) -> dict[str, Any]:
    """把一段时间汇总成几个可比的数.

    比率按**总量**算而不是按天平均: 只有 3 个回合的那天不该跟有 300 个回合的那天
    在均值里等权, 否则冷清的一天能把整段结论带偏。
    """
    turns = sum(int(d.get("turns") or 0) for d in days)
    continued = sum(int(d.get("continued") or 0) for d in days)
    pro_sent = sum(int(d.get("proactive_sent") or 0) for d in days)
    pro_ans = sum(int(d.get("proactive_answered") or 0) for d in days)

    # 回访只统计已定型的日子 —— 未定型的 returned_next_day 是 None。
    ret_days = [d for d in days if d.get("returned_next_day") is not None]
    ret_active = sum(int(d.get("active_users") or 0) for d in ret_days)
    ret_back = sum(int(d.get("returned_next_day") or 0) for d in ret_days)

    gaps = [d["median_gap_s"] for d in days if d.get("median_gap_s") is not None]
    gaps.sort()

    return {
        "turns": turns,
        "continuation_rate": round(continued / turns, 4) if turns else None,
        "median_gap_s": gaps[len(gaps) // 2] if gaps else None,
        "proactive_sent": pro_sent,
        "proactive_response_rate": round(pro_ans / pro_sent, 4) if pro_sent else None,
        "next_day_return_rate": round(ret_back / ret_active, 4) if ret_active else None,
        "settled_days": len(ret_days),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """比率的 95% 置信区间 (Wilson score).

    切片一定要带区间。只给"weak 78% / medium 83%"这样两个数, 人会立刻读成"medium
    更好" —— 而 100 个样本下这点差距完全在噪声里。用 Wilson 而不是正态近似, 是因为
    样本少或比率贴近 0/1 时后者会给出越界的区间 (比如 102%)。
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def merge_slices(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把每天的切片按 (维度, 取值) 累加.

    单日样本普遍不够判比率 (实测一天几十个回合, 切完每格个位数), 累计起来才有可比
    性。这也是切片必须在这里合并、而不是在页面上逐日展示的原因。
    """
    from app.services.effect.signals import MIN_SLICE_TURNS

    bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for d in days:
        for s in d.get("slices") or []:
            key = (s.get("dimension") or "", str(s.get("value")))
            acc = bucket.setdefault(key, {
                "dimension": key[0], "value": key[1], "turns": 0, "continued": 0,
            })
            acc["turns"] += int(s.get("turns") or 0)
            acc["continued"] += int(s.get("continued") or 0)

    out = []
    for acc in bucket.values():
        enough = acc["turns"] >= MIN_SLICE_TURNS
        lo, hi = wilson_interval(acc["continued"], acc["turns"])
        out.append({
            **acc,
            "continuation_rate": (
                round(acc["continued"] / acc["turns"], 4) if enough else None
            ),
            "ci_low": round(lo, 4) if enough else None,
            "ci_high": round(hi, 4) if enough else None,
            "sufficient_sample": enough,
        })
    out.sort(key=lambda x: (x["dimension"], -x["turns"]))
    return out


async def refresh_recent(days: int = 3) -> None:
    """预热最近几天 —— 由每日 cron 调用.

    重算最近 3 天而不只是昨天: 前天的次日回访率到今天才定型, 不重算就会把那个偏低
    的中间值一直缓存下去。
    """
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    for back in range(days):
        await get_day(now.date() - timedelta(days=back), force=True)
