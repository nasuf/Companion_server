"""记忆手札「旅途小记」生成（spec §5.4-8/§7，可选 LLM，失败有兜底）。"""

from __future__ import annotations

import logging
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_FALLBACK_NOTE = "这一趟走得很轻，把当下的光和风都收进了心里。回头再看，会记得来过。"


async def generate_travel_note(
    activity: dict[str, Any], fragments: list[dict[str, Any]]
) -> str:
    try:
        frag_summary = "\n".join(
            f"- {f.get('text', '')}" for f in fragments[:5]
        ) or "（这趟没有收集到思绪）"
        time_text = (
            activity.get("arrival_confirmed_at")
            or activity.get("created_at")
            or ""
        )
        prompt = (await get_prompt_text("offline.memory_note")).format(
            place=activity.get("title") or activity.get("location_name") or "",
            time=time_text,
            fragments=frag_summary,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return (raw or "").strip()
    except Exception as exc:
        logger.warning(
            "[offline-memory-note] 生成失败 activity=%s err=%s",
            activity.get("id"), exc,
        )
        return ""


def fallback_note() -> str:
    return _FALLBACK_NOTE
