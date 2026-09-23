"""思绪碎片分档预生成（PM #5/#6/#7）。

为每个拍摄物品 × 3 档（rare/epic/legendary）预生成一段「AI 自己的回忆」，素材取自
AI 经历库（memories_ai）。命中时由 recognition 取用并口语化交付。后台执行、幂等。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from app.config import settings
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text

try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo(getattr(settings, "schedule_timezone", "Asia/Shanghai"))
except Exception:  # pragma: no cover - tz db 缺失兜底
    _TZ = None

logger = logging.getLogger(__name__)

# 档位语气（对齐 PM 普通/重要/秘密三档）。
TIER_GUIDANCE: dict[str, str] = {
    "rare": "这是一段普通的、日常的过去回忆，轻、短、自然。",
    "epic": "这是一段过去的、很重要的回忆，情感更重、更私密、稍长。",
    "legendary": "这是一段关于自己的秘密，从来没有告诉过别人，最深、最稀有，带秘密感。",
}


def _time_period_now() -> str:
    now = datetime.now(_TZ) if _TZ else datetime.now()
    h = now.hour
    if h < 6:
        return "凌晨"
    if h < 9:
        return "清晨"
    if h < 12:
        return "上午"
    if h < 14:
        return "中午"
    if h < 18:
        return "下午"
    if h < 20:
        return "傍晚"
    return "夜晚"


async def pregenerate_for_activity(
    activity: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, int]:
    """为传入的每个物品预生成 3 档回忆并入池。幂等：已有池则跳过。"""
    recommendation_id = activity["id"]
    if not items:
        return {"generated": 0, "missing": 0}
    existing = await repo.list_prewritten_condition_tiers(recommendation_id)
    ai_experience = await repo.ai_memory_brief(
        activity["user_id"], activity.get("workspace_id")
    )
    theme = activity.get("title") or activity.get("category") or "线下漫步"
    time_period = _time_period_now()
    total = 0
    for item in items:
        tier_texts: dict[str, str] = {}
        for tier, guidance in TIER_GUIDANCE.items():
            if (str(item["id"]), tier) in existing:
                continue
            text = await _prewrite(item, theme, time_period, ai_experience, guidance)
            if text:
                tier_texts[tier] = text
        if tier_texts:
            await repo.create_prewritten_fragments(
                recommendation_id, item["id"], tier_texts
            )
            total += len(tier_texts)
    logger.info(
        "[offline-pregen] activity=%s 预生成 %d 段分档回忆", recommendation_id, total
    )
    final_pairs = await repo.list_prewritten_condition_tiers(recommendation_id)
    expected = {
        (str(item["id"]), tier)
        for item in items
        for tier in TIER_GUIDANCE
    }
    return {
        "generated": total,
        "missing": len(expected - final_pairs),
    }


async def _prewrite(
    item: dict[str, Any],
    theme: str,
    time_period: str,
    ai_experience: str,
    tier_guidance: str,
) -> str:
    try:
        prompt = (await get_prompt_text("offline.fragment_prewrite")).format(
            item=item.get("short_name") or item.get("category") or "",
            theme=theme,
            time_period=time_period,
            ai_experience=ai_experience or "（暂无）",
            tier_guidance=tier_guidance,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_prewritten(raw)
    except Exception as exc:
        logger.warning("[offline-pregen] 预生成失败 err=%s", exc)
        return ""


def _parse_prewritten(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("prewritten_text"):
            return str(data["prewritten_text"]).strip()
    except Exception:
        match = re.search(r'"prewritten_text"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1).strip()
    return ""
