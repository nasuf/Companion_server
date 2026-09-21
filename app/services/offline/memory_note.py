"""记忆手札 / 活动总结（PM #14）。输出 {title, body, mood_tags, fragment_tags}；失败兜底。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_FALLBACK_BODY = "这一趟走得很轻，把当下的光和风都收进了心里。回头再看，会记得来过。"


async def generate_note(
    *,
    activity_info: str,
    dialogue: str,
    voice_transcripts: str,
    photo_keywords: str,
    fragments: str,
) -> dict[str, Any]:
    try:
        prompt = (await get_prompt_text("offline.memory_note")).format(
            activity_info=activity_info,
            dialogue=dialogue or "（无）",
            voice_transcripts=voice_transcripts or "（无）",
            photo_keywords=photo_keywords or "（无）",
            fragments=fragments or "（无）",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_note(raw)
    except Exception as exc:
        logger.warning("[offline-memory-note] 生成失败 err=%s", exc)
        return {}


def _parse_note(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except Exception:
            return {}
    if not isinstance(data, dict):
        return {}
    return {
        "title": str(data.get("title") or "").strip(),
        "body": str(data.get("body") or "").strip(),
        "mood_tags": [str(t).strip() for t in (data.get("mood_tags") or []) if str(t).strip()],
        "fragment_tags": [
            str(t).strip() for t in (data.get("fragment_tags") or []) if str(t).strip()
        ],
    }


def fallback_body() -> str:
    return _FALLBACK_BODY
