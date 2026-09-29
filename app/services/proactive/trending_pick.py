"""A 模式「全网热点内容搭话」的热点筛选 (《主动交流提示词（新增）》4-1 / 4-2).

热榜抓取 (trending_context.resolve_trending_context) 给出 48 小时候选后:

  用户有爱好记忆 → 一半概率先走 4-2「用户爱好匹配热点」, 无匹配自动切 4-1
  其余           → 4-1「随机热点」

选中那条的 100 字摘要交给 4-3 (proactive.trending_chat) 生成闲聊消息; 候选原样
挂到链接卡片 —— 消息和卡片必须是同一件事 (之前出现过"文本说银锁骨链, 卡片是机场")。

规则黑名单在 LLM 之前再挡一层: 死讯/事故/政治类即使模型看漏也不该进候选。
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Literal

from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.storage import repo as memory_repo
from app.services.prompting.utils import render_prompt

logger = logging.getLogger(__name__)

PickMode = Literal["interest", "random"]

INTEREST_FIRST_PROBABILITY = 0.5
_PICK_TIMEOUT_S = 10.0
_MAX_CANDIDATES = 12
_SUMMARY_MAX_CHARS = 100
# 「用户爱好记忆库」: 偏好类里的喜好 (厌恶 / 雷区不是能拿来聊的爱好)
_HOBBY_SUBCATEGORIES = frozenset({"饮食喜好", "审美爱好", "人际喜好", "生活习惯", "其他"})
_MAX_HOBBIES = 12

# 明确拒绝的内容 (4-1 / 4-2 的屏蔽范围之外再加一道确定性防线): 名人死讯 / 事故 /
# 凶杀 / 自杀 / 政治敏感 / 性 / 歧视都不是朋友主动谈资; 另挡热榜页面的 UI 残余文本
_HOT_CONTENT_BLOCKLIST: tuple[str, ...] = (
    "去世", "身故", "遇难", "自杀", "死亡", "殉职", "溺亡",
    "车祸", "事故", "凶杀", "命案", "被杀", "谋杀",
    "性侵", "强奸", "猥亵",
    "抗议", "游行", "起义", "冲突", "占领", "封控", "抓捕",
    "战争", "导弹", "军演", "空袭",
    "歧视", "辱骂",
    "keyword_pinyin", "热榜聚合", "更多作品推荐",
)


@dataclass(frozen=True)
class TrendingPick:
    summary: str
    item: dict[str, Any]
    mode: PickMode


def _is_chatworthy(candidate: dict) -> bool:
    title = str(candidate.get("title") or "").strip()
    if not title:
        return False
    text = f"{title} {candidate.get('snippet') or ''}"
    return not any(word in text for word in _HOT_CONTENT_BLOCKLIST)


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        platform = str(c.get("platform") or "").strip()
        prefix = f"[{platform}] " if platform else ""
        snippet = str(c.get("snippet") or "").strip()[:120]
        title = str(c.get("title") or "").strip()
        lines.append(f"{i}. {prefix}{title}" + (f" —— {snippet}" if snippet else ""))
    return "\n".join(lines)


def _parse_pick(raw: Any, candidates: list[dict], mode: PickMode) -> TrendingPick | None:
    if not isinstance(raw, dict):
        return None
    try:
        index = int(raw.get("index", -1))
    except (TypeError, ValueError):
        return None
    summary = str(raw.get("summary") or "").strip()[:_SUMMARY_MAX_CHARS]
    if not (0 <= index < len(candidates)) or not summary:
        return None
    return TrendingPick(summary=summary, item=candidates[index], mode=mode)


async def _load_user_hobbies(user_id: str, workspace_id: str) -> list[str]:
    try:
        rows = await memory_repo.find_many(
            source="user",
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "isArchived": False,
                "mainCategory": "偏好",
            },
            order={"importance": "desc"},
            take=30,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[TRENDING] hobby load failed: {e}")
        return []
    hobbies = [r.content for r in rows if r.content and r.subCategory in _HOBBY_SUBCATEGORIES]
    return hobbies[:_MAX_HOBBIES]


async def _run_pick(key: str, params: dict[str, Any]) -> Any:
    try:
        return await asyncio.wait_for(
            render_prompt(key, params, lambda p: invoke_json(get_utility_model(), p)),
            timeout=_PICK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.info(f"[TRENDING] {key} timed out")
        return None


async def pick_trending(
    candidates: list[dict],
    *,
    user_id: str,
    workspace_id: str,
    exclude_titles: set[str] | None = None,
) -> TrendingPick | None:
    """从热榜候选里挑一条能聊的, 返回摘要 + 原候选; 一条都不合适返回 None."""
    exclude = exclude_titles or set()
    pool = [
        c for c in candidates
        if _is_chatworthy(c) and str(c.get("title") or "").strip() not in exclude
    ]
    if not pool:
        return None
    random.shuffle(pool)  # 4-1 要"随机": 打乱顺序, 模型不会总挑排第一的
    pool = pool[:_MAX_CANDIDATES]
    formatted = _format_candidates(pool)

    hobbies = await _load_user_hobbies(user_id, workspace_id)
    if hobbies and random.random() < INTEREST_FIRST_PROBABILITY:
        raw = await _run_pick("proactive.trending_pick_interest", {
            "user_hobbies": "\n".join(f"- {h}" for h in hobbies),
            "candidates": formatted,
        })
        pick = _parse_pick(raw, pool, "interest")
        if pick:
            return pick
        # 4-2 兜底: 无匹配热点, 自动切换随机热点

    raw = await _run_pick("proactive.trending_pick_random", {"candidates": formatted})
    return _parse_pick(raw, pool, "random")
