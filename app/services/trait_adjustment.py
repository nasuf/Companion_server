"""自适应性格调整服务 (MBTI 版)。

spec §1.2 起 MBTI 是 canonical personality. 调整对象从 7 维改为
MBTI 的 4 个百分比维度 (EI / NS / TF / JP)。隐式反馈关键词、直接反馈
正则模式都映射到 MBTI 维度。

规则:
- 单次 delta ≤ ±2
- 每日 ≤ 5 次调整
- 每周 ≤ 10 次调整
- 100 天后变化范围 ±10（相对初始 mbti）
- 值域 0-100
- 调整结果写入 agent.currentMbti，同时重算 type 字符串
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from prisma import Json

from app.db import db
from app.redis_client import get_redis
from app.services.mbti import MBTI_DIMS, get_initial_mbti, get_mbti

logger = logging.getLogger(__name__)


# 关键词 → (维度, delta) 映射。
# delta > 0 表示往该维度的"首字母"端推 (EI=+E, NS=+N, TF=+T, JP=+J)
KEYWORD_RULES: dict[str, dict[str, list[str]]] = {
    "EI": {  # +E (外向) / -I (内向)
        "positive": ["好开心", "太有趣了", "哈哈哈", "好嗨", "笑死", "活泼", "热闹"],
        "negative": ["安静点", "别闹了", "太吵了", "消停点", "稳重点", "别那么兴奋"],
    },
    "NS": {  # +N (直觉/想象) / -S (感觉/务实)
        "positive": ["好有创意", "脑洞大", "有想象力", "奇思妙想", "天马行空", "妙啊"],
        "negative": ["说正经的", "别瞎想", "务实点", "实际一点", "接地气", "别幻想"],
    },
    "TF": {  # +T (理性) / -F (感性)
        "positive": ["有道理", "说得对", "分析得好", "很理性", "逻辑清晰", "客观"],
        "negative": ["好感动", "好温柔", "暖心", "贴心", "懂我", "体贴", "细腻"],
    },
    "JP": {  # +J (计划) / -P (随性)
        "positive": ["好有条理", "安排得好", "计划周全", "靠谱", "井井有条", "严谨"],
        "negative": ["随性", "自在", "随意", "洒脱", "随遇而安", "怎样都行"],
    },
}


_DIRECT_FEEDBACK_PATTERNS: list[tuple[str, str, int]] = [
    # (regex, dimension, delta)
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(活泼|开朗|热情|外向)", "EI", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(安静|内向|稳重)", "EI", -2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(理性|理智|客观)", "TF", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(感性|温柔|温暖)", "TF", -2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(有条理|计划|严谨)", "JP", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(随性|随意|放松)", "JP", -2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(有创意|有想象力|脑洞)", "NS", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(实际|务实|接地气)", "NS", -2),
]


def detect_direct_feedback(message: str) -> list[dict] | None:
    """检测显式调整诉求 (e.g. "我希望你更活泼")。"""
    results = []
    for pattern, dimension, delta in _DIRECT_FEEDBACK_PATTERNS:
        if re.search(pattern, message):
            results.append({"dimension": dimension, "delta": delta, "source": "direct"})
    return results if results else None


def infer_feedback(message: str, mbti: dict | None = None) -> list[dict]:
    """从消息中推断隐式 MBTI 调整。关键词匹配 → ±1 微调。

    `mbti` 留作未来基于当前性格做加权（暂未使用，为接口兼容保留）。
    """
    _ = mbti
    results = []
    for dim, rules in KEYWORD_RULES.items():
        for kw in rules["positive"]:
            if kw in message:
                results.append({"dimension": dim, "delta": +1, "source": "implicit"})
                break
        for kw in rules["negative"]:
            if kw in message:
                results.append({"dimension": dim, "delta": -1, "source": "implicit"})
                break
    return results


def _derive_type(percentages: dict) -> str:
    return (
        ("E" if percentages.get("EI", 50) > 50 else "I")
        + ("N" if percentages.get("NS", 50) > 50 else "S")
        + ("T" if percentages.get("TF", 50) > 50 else "F")
        + ("J" if percentages.get("JP", 50) > 50 else "P")
    )


async def apply_trait_adjustment(agent_id: str, adjustments: list[dict]) -> dict | None:
    """Apply MBTI 调整，返回调整后的 mbti dict 或 None。"""
    if not adjustments:
        return None

    redis = await get_redis()
    now = datetime.now(UTC)
    day_key = f"trait_adj:{agent_id}:{now.strftime('%Y%m%d')}"
    week_key = f"trait_adj_week:{agent_id}:{now.strftime('%Y%W')}"

    # 频率限制
    daily_count = int(await redis.get(day_key) or 0)
    weekly_count = int(await redis.get(week_key) or 0)
    if daily_count >= 5:
        logger.debug(f"Daily trait adjustment limit reached for agent {agent_id}")
        return None
    if weekly_count >= 10:
        logger.debug(f"Weekly trait adjustment limit reached for agent {agent_id}")
        return None

    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return None

    current = get_mbti(agent)
    if not current:
        logger.debug(f"No MBTI on agent {agent_id}, skip adjustment")
        return None
    initial = get_initial_mbti(agent) or current

    days_since_creation = (
        now - agent.createdAt.replace(
            tzinfo=UTC if agent.createdAt.tzinfo is None else agent.createdAt.tzinfo
        )
    ).days

    # 复制百分比部分用于修改
    updated = {k: int(current.get(k, 50)) for k in MBTI_DIMS}
    applied_count = 0

    for adj in adjustments:
        dim = adj["dimension"]
        if dim not in MBTI_DIMS:
            continue

        delta = max(-2, min(2, int(adj["delta"])))

        # 100 天后限制 ±10 偏离初始
        if days_since_creation >= 100:
            init_val = initial.get(dim, 50)
            tentative = updated[dim] + delta
            if abs(tentative - init_val) > 10:
                continue

        updated[dim] = max(0, min(100, updated[dim] + delta))
        applied_count += 1

        try:
            await db.traitfeedbacklog.create(
                data={
                    "agent": {"connect": {"id": agent_id}},
                    "dimension": dim,
                    "delta": delta,
                    "source": adj.get("source", "implicit"),
                    "reason": adj.get("reason"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log trait feedback: {e}")

    if applied_count == 0:
        return None

    new_mbti = {
        **updated,
        "type": _derive_type(updated),
        "summary": current.get("summary", ""),
    }
    await db.aiagent.update(
        where={"id": agent_id},
        data={"currentMbti": Json(new_mbti)},
    )

    await redis.incr(day_key)
    await redis.expire(day_key, 86400)
    await redis.incr(week_key)
    await redis.expire(week_key, 86400 * 7)

    logger.info(
        f"Applied {applied_count} MBTI adjustments for agent {agent_id} → "
        f"type={new_mbti['type']}"
    )
    return new_mbti
