"""表达学习 MVP（Phase E3，借鉴 MaiBot expression_learner）。

从与用户的对话中后台学习 situation→style 表达方式（含用户的梗、口头禅、
网络用语），Redis 存储；回复时按使用频次加权随机抽取几条注入
「表达习惯参考」段——让 AI 越聊越像用户圈子里的人，打破"人设建号时
一次生成、永久静态"的 AI 感。

MVP 边界（与 MaiBot 完整版的差距，留待 V2）：
- Redis List 存储，无 embedding 语义检索（V2 需要向量表 + 按当前消息检索）
- 注入选择：count 加权随机（高频表达更常被"想起"），非语义匹配
- 每 (agent, user) 上限 MAX_EXPRESSIONS 条，满了淘汰 count 最低的
- 学习节流：每 LEARN_EVERY_N 条用户消息触发一次批量提取（后台，不阻塞热路径）
"""

from __future__ import annotations

import json
import logging
import random

from app.observability.events import EVT_EXPR_LEARN
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import safe_format

logger = logging.getLogger(__name__)

_EXPR_KEY = "expression:{agent_id}:{user_id}"
_COUNTER_KEY = "expression:msgcount:{agent_id}:{user_id}"

LEARN_EVERY_N = 20          # 每 N 条用户消息学习一批
MAX_EXPRESSIONS = 50        # 每 (agent, user) 表达上限
INJECT_COUNT = 3            # 每轮注入条数
_MAX_FIELD_LEN = 30         # situation/style 单字段长度上限 (防 LLM 跑偏)
_KEY_TTL_S = 90 * 86400     # 90 天无互动自然过期


async def bump_message_counter(agent_id: str, user_id: str) -> bool:
    """用户消息计数 +1；到达学习批次阈值时归零并返回 True。"""
    try:
        redis = await get_redis()
        key = _COUNTER_KEY.format(agent_id=agent_id, user_id=user_id)
        n = await redis.incr(key)
        await redis.expire(key, _KEY_TTL_S)
        if int(n) >= LEARN_EVERY_N:
            await redis.delete(key)
            return True
        return False
    except Exception as e:
        logger.debug(f"expression counter bump failed: {e}")
        return False


async def load_expressions(agent_id: str, user_id: str) -> list[dict]:
    """读全部已学表达。Redis 不可用 / 无数据时返回空列表。"""
    try:
        redis = await get_redis()
        raw = await redis.get(_EXPR_KEY.format(agent_id=agent_id, user_id=user_id))
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.debug(f"expression load failed: {e}")
        return []


async def _save_expressions(
    agent_id: str, user_id: str, expressions: list[dict],
) -> None:
    redis = await get_redis()
    await redis.set(
        _EXPR_KEY.format(agent_id=agent_id, user_id=user_id),
        json.dumps(expressions, ensure_ascii=False),
        ex=_KEY_TTL_S,
    )


def _validate_items(items: object) -> list[dict]:
    """LLM 输出清洗：只收 situation/style 均为非空短字符串的条目。"""
    if isinstance(items, dict):
        items = items.get("expressions", [])
    if not isinstance(items, list):
        return []
    valid: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        if (
            situation and style
            and len(situation) <= _MAX_FIELD_LEN
            and len(style) <= _MAX_FIELD_LEN
        ):
            valid.append({"situation": situation, "style": style})
    return valid


def merge_expressions(
    existing: list[dict], new_items: list[dict], cap: int = MAX_EXPRESSIONS,
) -> list[dict]:
    """按 style 文本去重合并：重复学到 → count+1（该表达确实高频）；
    超上限时按 (count, seq) 升序淘汰 — count 平局时新鲜的赢
    （刚学到的表达比同频旧表达更该留下，seq 是单调递增的学习序号）。"""
    by_style = {e.get("style"): dict(e) for e in existing if e.get("style")}
    next_seq = max((int(e.get("seq", 0)) for e in by_style.values()), default=0)
    for item in new_items:
        style = item["style"]
        next_seq += 1
        if style in by_style:
            by_style[style]["count"] = int(by_style[style].get("count", 1)) + 1
            by_style[style]["situation"] = item["situation"]  # 场景描述取最新
            by_style[style]["seq"] = next_seq
        else:
            by_style[style] = {**item, "count": 1, "seq": next_seq}
    merged = sorted(
        by_style.values(),
        key=lambda e: (int(e.get("count", 1)), int(e.get("seq", 0))),
        reverse=True,
    )
    return merged[:cap]


async def learn_expressions(
    agent_id: str, user_id: str, messages: list[dict],
) -> int:
    """批量提取一次表达方式并合并入库。返回新学到（含加权）的条数。

    只喂最近 20 条消息文本；prompt 明确只学**用户**的说话方式，
    不学 AI 自己的（防自我强化循环，MaiBot 同款规则）。
    """
    lines = [
        f"{'用户' if m.get('role') == 'user' else 'AI'}: {str(m.get('content', ''))[:200]}"
        for m in messages[-20:]
        if str(m.get("content", "")).strip()
    ]
    if len([ln for ln in lines if ln.startswith("用户")]) < 3:
        return 0  # 用户发言太少, 不值得学
    tpl = await get_prompt_text("expression.learn_style")
    prompt = safe_format(tpl, {"conversation": "\n".join(lines)})
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"expression learn LLM failed: {e}")
        return 0
    new_items = _validate_items(result)
    if not new_items:
        return 0
    existing = await load_expressions(agent_id, user_id)
    merged = merge_expressions(existing, new_items)
    try:
        await _save_expressions(agent_id, user_id, merged)
    except Exception as e:
        logger.warning(f"expression save failed: {e}")
        return 0
    logger.info(
        f"[EXPR-LEARN] learned {len(new_items)} expressions (total {len(merged)})",
        extra={
            "event": EVT_EXPR_LEARN,
            "n_new": len(new_items),
            "n_total": len(merged),
        },
    )
    return len(new_items)


def weighted_sample(
    expressions: list[dict], k: int, rng: random.Random | None = None,
) -> list[dict]:
    """count 加权无放回抽样：高频表达更常被"想起"，但低频也有机会。"""
    r = rng or random
    pool = list(expressions)
    picked: list[dict] = []
    while pool and len(picked) < k:
        weights = [max(1, int(e.get("count", 1))) for e in pool]
        chosen = r.choices(pool, weights=weights, k=1)[0]
        picked.append(chosen)
        pool.remove(chosen)
    return picked


async def sample_expression_habits(
    agent_id: str | None, user_id: str, k: int = INJECT_COUNT,
) -> list[str]:
    """热路径入口：抽 k 条已学表达，渲染成「当 X 时，可以 Y」行。"""
    if not agent_id:
        return []
    expressions = await load_expressions(agent_id, user_id)
    if not expressions:
        return []
    return [
        f"当「{e['situation']}」时，可以「{e['style']}」"
        for e in weighted_sample(expressions, k)
    ]
