"""到达后生成 3-5 条拍摄条件集合（spec §3.4/§7）。后台执行，永不下发前端明文。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

# 生成失败时的兜底条件池（保证识图链路有可匹配集合）。
_FALLBACK_CONDITIONS: list[dict[str, str]] = [
    {"short_name": "植物", "criteria": "画面里有明显的植物、绿叶或花"},
    {"short_name": "天空", "criteria": "拍到天空、云或天光"},
    {"short_name": "光影", "criteria": "有明显的光影、逆光或暖光"},
    {"short_name": "细节", "criteria": "某个有意思的小细节、纹理或物件特写"},
]


async def generate_conditions_for_activity(activity: dict[str, Any]) -> None:
    """幂等：已有条件则跳过；否则生成并落库 + 标记 ready。"""
    recommendation_id = activity["id"]
    user_id = activity["user_id"]
    if await repo.count_shooting_conditions(recommendation_id) > 0:
        return
    conditions = await _generate(activity) or _FALLBACK_CONDITIONS
    await repo.create_shooting_conditions(recommendation_id, conditions[:5])
    await repo.mark_conditions_ready(recommendation_id, user_id)
    logger.info(
        "[offline-conditions] activity=%s 生成 %d 条拍摄条件",
        recommendation_id, len(conditions[:5]),
    )


async def _generate(activity: dict[str, Any]) -> list[dict[str, str]]:
    try:
        scene = activity.get("summary") or activity.get("description") or ""
        prompt = (await get_prompt_text("offline.shooting_conditions")).format(
            name=activity.get("title") or activity.get("location_name") or "",
            address=activity.get("address") or activity.get("location_name") or "",
            category=activity.get("category") or "",
            scene=scene,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_conditions(raw)
    except Exception as exc:
        logger.warning(
            "[offline-conditions] 生成失败 activity=%s err=%s",
            activity.get("id"), exc,
        )
        return []


def _parse_conditions(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    # 剥 markdown code fence
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        short_name = str(item.get("short_name") or "").strip()
        criteria = str(item.get("criteria") or "").strip()
        if short_name and criteria:
            out.append({"short_name": short_name, "criteria": criteria})
    return out[:5]
