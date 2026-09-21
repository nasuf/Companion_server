"""到达后生成 3-5 个拍摄「物品」大类，并触发分档回忆预生成（PM #3/#4 → #5-7）。

不披露给用户。物品用于后续识图主体匹配；生成完成后后台为每个物品预生成 3 档 AI 回忆。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import fragment_pregen
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

# 生成失败时的兜底物品池（保证识图链路有可匹配集合）。
_FALLBACK_ITEMS: list[dict[str, str]] = [
    {"category": "植物", "short_name": "花"},
    {"category": "天空", "short_name": "天空"},
    {"category": "光影", "short_name": "光影"},
    {"category": "建筑", "short_name": "建筑"},
]


async def generate_items_for_activity(activity: dict[str, Any]) -> None:
    """幂等：已有物品则跳过；否则生成、落库、标 ready，并后台触发分档回忆预生成。"""
    recommendation_id = activity["id"]
    user_id = activity["user_id"]
    if await repo.count_shooting_conditions(recommendation_id) > 0:
        return
    items = _ensure_min_items(await _generate_items(activity))
    created = await repo.create_shooting_items(recommendation_id, items[:5])
    await repo.mark_conditions_ready(recommendation_id, user_id)
    logger.info(
        "[offline-items] activity=%s 生成 %d 个拍摄物品", recommendation_id, len(created)
    )
    # 后台为每个物品预生成 3 档 AI 回忆（不阻塞到达响应）。
    fire_background(fragment_pregen.pregenerate_for_activity(activity, created))


def _ensure_min_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """spec §3.4/§4-6：拍摄物品必须 3–5 个。LLM 少于 3 个（或空）时用兜底池补齐，
    去重 short_name，保证识图链路始终有 ≥3 个可匹配物品。"""
    out = list(items or [])
    seen = {str(i.get("short_name") or "").strip() for i in out}
    for fb in _FALLBACK_ITEMS:
        if len(out) >= 3:
            break
        if fb["short_name"] not in seen:
            out.append(dict(fb))
            seen.add(fb["short_name"])
    return out


async def _generate_items(activity: dict[str, Any]) -> list[dict[str, str]]:
    try:
        scene = activity.get("summary") or activity.get("description") or ""
        prompt = (await get_prompt_text("offline.shooting_items")).format(
            name=activity.get("title") or activity.get("location_name") or "",
            category=activity.get("category") or "",
            scene=scene,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_items(raw)
    except Exception as exc:
        logger.warning(
            "[offline-items] 生成失败 activity=%s err=%s", activity.get("id"), exc
        )
        return []


def _parse_items(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        short_name = str(item.get("short_name") or "").strip()
        category = str(item.get("category") or "").strip()
        if short_name:
            out.append({"short_name": short_name, "category": category})
    return out[:5]
