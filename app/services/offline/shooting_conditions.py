"""到达后生成 3-5 个拍摄「物品」大类，并触发分档回忆预生成（PM #3/#4 → #5-7）。

不披露给用户。物品用于后续识图主体匹配；生成完成后后台为每个物品预生成 3 档 AI 回忆。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import settings
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import fragment_pregen
from app.services.offline import repository as repo
from app.services.offline.guidance import normalize_guidance_profile
from app.services.prompting.store import get_prompt_text
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

# 生成失败时的兜底物品池（保证识图链路有可匹配集合）。
_FALLBACK_ITEMS: list[dict[str, str]] = [
    {"category": "植物", "short_name": "花", "criteria": "画面主体清楚呈现花朵"},
    {"category": "天空", "short_name": "天空", "criteria": "画面有明显开阔天空"},
    {"category": "光影", "short_name": "光影", "criteria": "画面主体是明显的明暗或倒影"},
    {"category": "建筑", "short_name": "建筑", "criteria": "画面主体是建筑或结构细节"},
]


async def generate_items_for_activity(activity: dict[str, Any]) -> None:
    """Idempotently create a complete target set and initialize companionship."""
    recommendation_id = activity["id"]
    if activity.get("conditions_ready_at"):
        existing = await repo.list_untriggered_conditions(recommendation_id)
        if existing and not activity.get("focus_condition_id"):
            await _initialize_companion(activity, existing)
        return
    items = _normalize_items(_ensure_min_items(await _generate_items(activity)))
    selected = items[:5]
    focus_index = random.randrange(len(selected)) if selected else 0
    created, created_new = await repo.replace_shooting_items_and_initialize(
        recommendation_id=recommendation_id,
        user_id=activity["user_id"],
        items=selected,
        focus_index=focus_index,
        next_companion_at=_initial_companion_due(),
    )
    logger.info(
        "[offline-items] activity=%s targets=%d created=%s",
        recommendation_id,
        len(created),
        created_new,
    )
    # The pre-generator is tier-idempotent and repairs partial prior runs.
    fire_background(fragment_pregen.pregenerate_for_activity(activity, created))


async def _initialize_companion(
    activity: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    if not items:
        return
    focus = random.choice(items)
    await repo.mark_conditions_ready(
        activity["id"],
        activity["user_id"],
        focus_condition_id=str(focus["id"]),
        next_companion_at=_initial_companion_due(),
    )


def _initial_companion_due() -> datetime:
    low = max(1, int(settings.offline_activity_companion_min_interval_minutes))
    high = max(low, int(settings.offline_activity_companion_max_interval_minutes))
    return datetime.now(UTC) + timedelta(minutes=random.randint(low, high))


async def recover_unready_reached_activities() -> dict[str, int]:
    activities = await repo.list_unready_reached_activities()
    semaphore = asyncio.Semaphore(3)

    async def _recover(activity: dict[str, Any]) -> bool:
        async with semaphore:
            try:
                await generate_items_for_activity(activity)
                return True
            except Exception as exc:  # noqa: BLE001 - isolate repair failures
                logger.warning(
                    "[offline-items] recovery failed activity=%s err=%s",
                    activity.get("id"),
                    exc,
                )
                return False

    results = await asyncio.gather(*(_recover(activity) for activity in activities))
    recovered = sum(1 for result in results if result)
    failed = len(results) - recovered
    return {"scanned": len(activities), "recovered": recovered, "failed": failed}


async def recover_missing_prewritten_fragments() -> dict[str, int]:
    activities = await repo.list_ready_activities_with_missing_prewritten()
    semaphore = asyncio.Semaphore(3)

    async def _repair(activity: dict[str, Any]) -> bool:
        async with semaphore:
            try:
                items = await repo.list_all_conditions(activity["id"])
                result = await fragment_pregen.pregenerate_for_activity(
                    activity,
                    items,
                )
                return result["missing"] == 0
            except Exception as exc:  # noqa: BLE001 - isolate repair failures
                logger.warning(
                    "[offline-pregen] recovery failed activity=%s err=%s",
                    activity.get("id"),
                    exc,
                )
                return False

    results = await asyncio.gather(*(_repair(activity) for activity in activities))
    repaired = sum(1 for result in results if result)
    failed = len(results) - repaired
    return {"scanned": len(activities), "repaired": repaired, "failed": failed}


def _ensure_min_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        short_name = str(item.get("short_name") or "").strip()
        category = str(item.get("category") or "").strip()
        if not short_name:
            continue
        criteria = str(item.get("criteria") or "").strip()
        normalized.append(
            {
                "short_name": short_name,
                "category": category,
                "criteria": criteria,
                "guidance_profile": normalize_guidance_profile(
                    short_name=short_name,
                    category=category,
                    criteria=criteria,
                    raw_profile=item.get("guidance_profile"),
                ),
            }
        )
    return normalized


async def _generate_items(activity: dict[str, Any]) -> list[dict[str, Any]]:
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


def _parse_items(raw: str) -> list[dict[str, Any]]:
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
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        short_name = str(item.get("short_name") or "").strip()
        category = str(item.get("category") or "").strip()
        if short_name:
            guidance = item.get("guidance")
            aliases = item.get("aliases")
            out.append(
                {
                    "short_name": short_name,
                    "category": category,
                    "criteria": str(item.get("criteria") or "").strip(),
                    "guidance_profile": {
                        "aliases": aliases if isinstance(aliases, list) else [],
                        "guidance": guidance if isinstance(guidance, dict) else {},
                    },
                }
            )
    return out[:5]
