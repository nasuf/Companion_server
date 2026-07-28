"""每周从行为事实归纳出对相处有帮助的判断.

这是记忆系统里唯一会写入**推断**的路径 —— 其他所有记忆都源自某人真的说过的话,
而这些是从统计规律推出来的, 没人说过。这个差别决定了三处设计:

    落 L2 不落 L1   L1 永不衰减, 一条错误推断进去就是永久的人设污染。让它们进 L2,
                    有用的会因为被反复检索而自己升上去 (值驱动晋升), 没用的淡出。
    豁免有损压缩    压缩会抹掉证据边界, 而没有证据的推断无法复核。
    每条带证据      引用不到具体事实的判断直接丢弃, 不进库。

上游 (signals.py) 保证喂进来的数字是对的; 这里保证模型没有超出这些数字乱推。

## 为什么写记忆而不是写画像

画像是一整块会被整体重写的文字, 每轮对话都注入。判断放进去有两个问题: 一是每条
都常驻 prompt, 三五条之后就开始挤占预算; 二是整块重写意味着没法单独撤销某一条。
写成记忆则各自独立 —— 相关时才被检索到, 没用的会衰减, 错的能单独撤。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.config import settings
from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.provenance import REFLECTED
from app.services.memory.reflection.signals import (
    BehaviouralFact,
    collect_behavioural_facts,
    format_facts_for_prompt,
)
from app.services.memory.storage.persistence import store_memory
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

MAX_INSIGHTS_PER_RUN = 3
MIN_FACTS_TO_REFLECT = 3

# 判断落在 L2 的中段。不给高分是因为它们尚未被证明有用 —— 让检索使用情况把有用的
# 抬上去, 比一上来就假定它重要更诚实。
REFLECTION_IMPORTANCE = 0.62

# 归入「思维」而不是「身份」: 这是对相处方式的判断, 不是关于用户是谁的事实。
REFLECTION_MAIN_CATEGORY = "思维"
REFLECTION_SUB_CATEGORY = "相处方式"


@dataclass(frozen=True)
class Insight:
    text: str
    based_on: list[int]


def _parse_insights(raw: object, fact_count: int) -> list[Insight]:
    """解析模型输出, 丢掉引用不上事实的条目。

    引用检查是这一层的主要防线: 模型可以编出一句听起来很有道理的判断, 但它编不出
    一个指向具体观察的有效引用 —— 除非那条判断真的是从观察来的。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, dict):
        return []

    out: list[Insight] = []
    for item in raw.get("insights") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not (8 <= len(text) <= 80):
            continue
        refs = [
            n for n in (item.get("based_on") or [])
            if isinstance(n, int) and 1 <= n <= fact_count
        ]
        if not refs:
            logger.info(f"reflection insight dropped (no valid citation): {text[:30]}")
            continue
        out.append(Insight(text=text, based_on=sorted(set(refs))))
    return out[:MAX_INSIGHTS_PER_RUN]


async def generate_insights(facts: list[BehaviouralFact]) -> list[Insight]:
    """调模型做归纳。事实不足时不调 —— 三条观察推不出东西, 只会逼它编。"""
    if len(facts) < MIN_FACTS_TO_REFLECT:
        return []
    try:
        template = await get_prompt_text("memory.reflection")
    except PromptDisabledError:
        logger.info("memory.reflection prompt disabled; skipping")
        return []

    prompt = template.format(
        facts=format_facts_for_prompt(facts),
        max_insights=MAX_INSIGHTS_PER_RUN,
    )
    try:
        raw = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"reflection LLM failed: {e}")
        return []
    return _parse_insights(raw, len(facts))


async def reflect_for_user(
    *, user_id: str, agent_id: str, workspace_id: str | None,
    dry_run: bool = False,
) -> dict:
    """给一个 (user, agent) 跑一轮反思。

    dry_run 只算事实和洞见, 不写库 —— 开 flag 前先看它会产出什么。
    """
    stats: dict = {"facts": 0, "insights": 0, "stored": 0, "dry_run": dry_run}

    facts = await collect_behavioural_facts(
        user_id=user_id, agent_id=agent_id, workspace_id=workspace_id,
    )
    stats["facts"] = len(facts)
    if len(facts) < MIN_FACTS_TO_REFLECT:
        return stats

    insights = await generate_insights(facts)
    stats["insights"] = len(insights)
    stats["preview"] = [
        {"text": i.text, "based_on": [facts[n - 1].key for n in i.based_on]}
        for i in insights
    ]
    if dry_run or not insights:
        return stats

    existing = await _existing_reflection_texts(
        user_id=user_id, workspace_id=workspace_id,
    )
    for insight in insights:
        if insight.text in existing:
            # 同一个判断周复一周地生成是预期内的, 跳过即可。
            continue
        cited = "; ".join(facts[n - 1].statement for n in insight.based_on)
        try:
            memory_id = await store_memory(
                user_id=user_id,
                content=insight.text,
                level=2,
                importance=REFLECTION_IMPORTANCE,
                main_category=REFLECTION_MAIN_CATEGORY,
                sub_category=REFLECTION_SUB_CATEGORY,
                source="user",
                workspace_id=workspace_id,
                provenance=REFLECTED,
                # 必须跳过 reconciliation。它只写保护 profile_seed / knowledge_seed,
                # 一条 user_stated 的 L2 记忆会被 update_existing / merge_existing
                # 直接改写内容 —— 那等于让 LLM 的推断覆盖掉用户真的说过的话, 而且
                # 原文没有留存。重复问题改用上面基于 provenance=reflected 的精确
                # 文本比对处理, 只在自己产出的行之间去重。
                skip_reconciliation=True,
            )
        except Exception as e:
            logger.warning(f"reflection store failed: {e}")
            continue
        if memory_id:
            existing.add(insight.text)
            stats["stored"] += 1
            logger.info(
                f"[REFLECTION] user={user_id[:8]} stored «{insight.text[:40]}» "
                f"based_on=[{cited[:80]}]"
            )
    return stats


async def _existing_reflection_texts(
    *, user_id: str, workspace_id: str | None,
) -> set[str]:
    """本 workspace 已有的反思判断原文, 用于避免逐周堆积同一条。

    只比自己产出的行 —— 反思不该因为跟某条用户陈述相似就放弃写入, 更不该反过来
    去改那条陈述。
    """
    try:
        rows = await db.query_raw(
            """
            SELECT content FROM memories_user
            WHERE user_id = $1
              AND workspace_id IS NOT DISTINCT FROM $2
              AND provenance = $3 AND is_archived = false
            """,
            user_id, workspace_id, REFLECTED,
        )
    except Exception as e:
        logger.warning(f"reflection dedup lookup failed: {e}")
        return set()
    return {str(r.get("content") or "") for r in rows}


def reflection_enabled_for(workspace_id: str | None) -> bool:
    """总闸 + 灰度白名单。与整合同样的形状。

    默认关闭: 这是唯一会写入推断的路径, 而它的产出质量取决于积累了多少互动数据。
    数据量到了、按 evals 的闸门验过再开。
    """
    if not settings.memory_reflection_enabled:
        return False
    allowlist = {
        w.strip() for w in settings.memory_reflection_workspaces.split(",")
        if w.strip()
    }
    return not allowlist or (workspace_id or "") in allowlist
