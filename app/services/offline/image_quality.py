"""Visual quality filtering is separate from source/branch identity evidence."""

from __future__ import annotations

import asyncio
import base64
import json

from app.config import settings
from app.redis_client import get_redis
from app.services.chat_media.vision import _call_doubao_vision
from app.services.prompting.store import get_prompt_text


async def scene_photo(blob: bytes, digest: str) -> bool:
    key = "offline:scene-photo:v1:" + digest
    try:
        redis = await get_redis()
        cached = await redis.get(key)
        if cached in {"photo", "other"}:
            return cached == "photo"
        if not settings.ark_api_key:
            return True  # Native binding and download checks still apply.
        async with asyncio.timeout(7):
            prompt = (await get_prompt_text("offline.image_quality")).format()
            text = await _call_doubao_vision(
                data_url="data:image/jpeg;base64," + base64.b64encode(blob).decode(),
                user_text=prompt,
            )
        start, end = text.find("{"), text.rfind("}")
        result = json.loads(text[start : end + 1]).get("kind")
        if result in {"photo", "other"}:
            await redis.set(key, result, ex=7 * 86400)
            return result == "photo"
    except Exception:
        pass
    # Model outage must not erase source-bound galleries. This never relaxes
    # wrong-venue/branch evidence or SSRF/MIME/image limits.
    return True
