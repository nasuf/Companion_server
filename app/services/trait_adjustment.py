"""自适应性格调整服务。

隐式反馈推断 + 直接反馈捕获 + 频率控制 + 调整算法。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from prisma import Json

from app.db import db
from app.redis_client import get_redis
from app.services.trait_model import SEVEN_DIMS

logger = logging.getLogger(__name__)

# --- 0.5.3 关键词规则库 ---

KEYWORD_RULES: dict[str, dict[str, list[str]]] = {
    "活泼度": {
        "positive": ["好开心", "太有趣了", "哈哈哈", "好好玩", "好嗨", "太搞笑", "笑死", "乐死", "有意思", "好玩"],
        "negative": ["安静点", "别闹了", "太吵了", "消停点", "冷静", "别那么兴奋", "淡定", "稳重点", "别跳了", "烦死了"],
    },
    "理性度": {
        "positive": ["有道理", "说得对", "分析得好", "很理性", "逻辑清晰", "客观", "你说的对", "有见解", "思路清晰", "分析到位"],
        "negative": ["太理性了", "别讲道理了", "不要分析", "少说废话", "别那么死板", "变通一下", "太教条", "别计较", "别钻牛角尖", "别抬杠"],
    },
    "感性度": {
        "positive": ["好感动", "好温柔", "暖心", "贴心", "懂我", "体贴", "好细腻", "好温暖", "有共情", "善解人意"],
        "negative": ["别矫情", "太敏感了", "想太多", "别多想", "太脆弱", "坚强点", "没必要", "太玻璃心", "太作了", "别小题大做"],
    },
    "计划度": {
        "positive": ["好有条理", "安排得好", "计划周全", "很细心", "考虑周到", "靠谱", "井井有条", "有章法", "规划到位", "严谨"],
        "negative": ["别那么死板", "放松点", "太严谨了", "不用那么计划", "随便一点", "别那么认真", "太较真", "灵活点", "别框太死", "松散点"],
    },
    "随性度": {
        "positive": ["随性", "自在", "随意", "洒脱", "不拘束", "率真", "随遇而安", "无所谓", "怎样都行", "随你"],
        "negative": ["认真点", "上心点", "别太随意", "正经一点", "负责点", "别敷衍", "用心一点", "严肃点", "别开玩笑", "靠谱点"],
    },
    "脑洞度": {
        "positive": ["好有创意", "脑洞大", "有想象力", "好有趣的想法", "太有才了", "奇思妙想", "独特", "新颖", "妙啊", "天马行空"],
        "negative": ["说正经的", "别瞎想", "务实点", "太离谱了", "不切实际", "别做梦了", "实际一点", "接地气点", "别幻想", "正常点"],
    },
    "幽默度": {
        "positive": ["好好笑", "太幽默了", "笑死我了", "段子手", "太逗了", "好有梗", "幽默感", "搞笑", "机智", "风趣"],
        "negative": ["不好笑", "别开玩笑", "严肃点", "正经一点", "冷笑话", "尬", "太冷了", "别逗了", "无聊", "幼稚"],
    },
}

# --- 0.5.6 直接反馈模式 ---

_DIRECT_FEEDBACK_PATTERNS: list[tuple[str, str, float]] = [
    # (regex pattern, dimension, delta)
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(活泼|开朗|热情)", "活泼度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(安静|冷静|稳重)", "活泼度", -2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(理性|理智|客观)", "理性度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(感性|温柔|温暖)", "感性度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(有条理|计划|严谨)", "计划度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(随性|随意|放松)", "随性度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(有创意|有想象力|脑洞)", "脑洞度", +2),
    (r"(我希望你|你能不能|你可以).*(更|多一点).*(幽默|搞笑|有趣)", "幽默度", +2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(活泼|闹|吵)", "活泼度", -2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(理性|死板|教条)", "理性度", -2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(感性|矫情|敏感)", "感性度", -2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(严谨|死板)", "计划度", -2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(随意|敷衍)", "随性度", -2),
    (r"(我希望你|你能不能|你可以).*(少一点|不要那么).*(幽默|搞笑|冷笑话)", "幽默度", -2),
]


def detect_direct_feedback(message: str) -> list[dict] | None:
    """检测用户直接反馈。如"我希望你更活泼" → delta=+2。"""
    results = []
    for pattern, dimension, delta in _DIRECT_FEEDBACK_PATTERNS:
        if re.search(pattern, message):
            results.append({"dimension": dimension, "delta": delta, "source": "direct"})
    return results if results else None


# --- 0.5.4 隐式反馈推断 ---

def infer_feedback(message: str, personality: dict | None = None) -> list[dict]:
    """从消息中推断隐式性格反馈。关键词匹配 + 情感倾向。"""
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


# --- 0.5.5 调整算法 + 频率控制 ---

async def apply_trait_adjustment(agent_id: str, adjustments: list[dict]) -> dict | None:
    """应用性格调整。

    约束:
    - 单次 delta ≤ ±2
    - 每日 ≤ 5 次调整
    - 每周 ≤ 10 次调整
    - 100天后变化范围 ±10
    - 值域 0-100
    """
    if not adjustments:
        return None

    redis = await get_redis()
    now = datetime.now(UTC)
    day_key = f"trait_adj:{agent_id}:{now.strftime('%Y%m%d')}"
    week_key = f"trait_adj_week:{agent_id}:{now.strftime('%Y%W')}"

    # 检查频率限制
    daily_count = int(await redis.get(day_key) or 0)
    weekly_count = int(await redis.get(week_key) or 0)

    if daily_count >= 5:
        logger.debug(f"Daily trait adjustment limit reached for agent {agent_id}")
        return None
    if weekly_count >= 10:
        logger.debug(f"Weekly trait adjustment limit reached for agent {agent_id}")
        return None

    # 获取当前性格
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return None

    from app.services.trait_model import get_seven_dim
    current = get_seven_dim(agent)
    initial = agent.sevenDimTraits if isinstance(agent.sevenDimTraits, dict) else current.copy()

    # 计算创建天数（用于100天后的±10范围限制）
    days_since_creation = (now - agent.createdAt.replace(tzinfo=UTC if agent.createdAt.tzinfo is None else agent.createdAt.tzinfo)).days

    updated = current.copy()
    applied_count = 0

    for adj in adjustments:
        dim = adj["dimension"]
        if dim not in SEVEN_DIMS:
            continue

        # 单次 delta 限制 ±2
        delta = max(-2, min(2, adj["delta"]))

        # 100天后限制 ±10 范围
        if days_since_creation >= 100:
            init_val = initial.get(dim, 50)
            new_val = updated.get(dim, 50) + delta
            if abs(new_val - init_val) > 10:
                continue

        # 应用调整
        updated[dim] = max(0, min(100, updated.get(dim, 50) + delta))
        applied_count += 1

        # 记录反馈日志
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

    # 更新 currentTraits
    await db.aiagent.update(
        where={"id": agent_id},
        data={"currentTraits": Json(updated)},
    )

    # 更新频率计数
    await redis.incr(day_key)
    await redis.expire(day_key, 86400)
    await redis.incr(week_key)
    await redis.expire(week_key, 86400 * 7)

    logger.info(f"Applied {applied_count} trait adjustments for agent {agent_id}")
    return updated
