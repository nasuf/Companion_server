"""CleverSee transport. Native POIs are facts; QA prose never supplies coordinates.

Cache only public search responses. Keys contain a digest rather than GPS/user
text, credentials never enter logs, and every request has a bounded deadline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from email.utils import formatdate
from uuid import uuid4

import httpx

from app.config import settings
from app.redis_client import get_redis
from app.services.offline.providers.search import SearchResult

logger = logging.getLogger(__name__)
_slots = asyncio.Semaphore(2)


async def _rate_slot() -> None:
    # Shared across HTTP workers. The purchased default service allows 5 QPS.
    redis = await get_redis()
    for _ in range(8):
        key = "offline:cleversee:rate:" + str(int(time.time()))
        count = await redis.eval(
            "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],2) end; return n",
            1,
            key,
        )
        if count <= 5:
            return
        await asyncio.sleep(0.25)
    raise TimeoutError("CleverSee rate budget")


async def _cached(
    kind: str, body: dict, call, *, ttl: int = 900, fresh: bool = False
) -> dict:
    key = (
        "offline:cleversee:v1:"
        + kind
        + ":"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )
    try:
        cached = await (await get_redis()).get(key)
        if cached and not fresh:
            return json.loads(cached)
    except Exception:
        pass
    try:
        async with asyncio.timeout(18), _slots:
            await _rate_slot()
            data = await call()
    except Exception as exc:
        # Provider exception messages can include request authentication details.
        logger.warning("[cleversee] %s failed (%s)", kind, type(exc).__name__)
        return {}
    if isinstance(data, dict) and data:
        try:
            await (await get_redis()).set(
                key, json.dumps(data, ensure_ascii=False), ex=ttl
            )
        except Exception:
            pass
        return data
    return {}


async def _sdk_query(body: dict) -> dict:
    from alibabacloud_iqs20240712.client import Client
    from alibabacloud_iqs20240712.models import (
        CommonAgentQuery,
        CommonQueryBySceneRequest,
    )
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    client = Client(
        Config(
            access_key_id=settings.ali_cloud_access_key_id,
            access_key_secret=settings.ali_cloud_access_key_secret,
            endpoint="iqs.cn-zhangjiakou.aliyuncs.com",
        )
    )
    result = await client.common_query_by_scene_with_options_async(
        CommonQueryBySceneRequest(body=CommonAgentQuery().from_map(body)),
        {},
        RuntimeOptions(read_timeout=14000, connect_timeout=4000, autoretry=False),
    )
    return result.body.to_map()


async def _http(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=14, trust_env=False) as client:
        response = await client.post(
            "https://cloud-iqs.aliyuncs.com" + path,
            json=body,
            headers={
                "Authorization": "Bearer " + settings.ali_cleversee_api_key,
                "Date": formatdate(usegmt=True),
            },
        )
        response.raise_for_status()
        return response.json()


async def places(query: str, *, limit: int = 12) -> list[dict]:
    if not settings.ali_cloud_access_key_id or not settings.ali_cloud_access_key_secret:
        return []
    # No forced scene: bookshops/libraries otherwise become entertainment POIs.
    body = dict(query=query[:500], limit=max(1, min(limit, 25)), searchModel="normal")
    data = await _cached("poi", body, lambda: _sdk_query(body))
    return [p for p in data.get("data", []) if isinstance(p, dict)]


async def qa_places(
    query: str, *, lat: float | None = None, lng: float | None = None
) -> list[dict]:
    if not settings.ali_cleversee_api_key:
        return []
    # Stable cache body excludes random session identifiers. Never send user memory.
    body = dict(
        engineType="LiteAgentAdvanced",
        stream=False,
        locationInfo={"location": query[:160]},
        message={"parts": [{"type": "text", "text": query[:500]}]},
    )
    if lat is not None and lng is not None:
        body["locationInfo"].update(latitude=lat, longitude=lng)

    async def call():
        return await _http(
            "/qa/chat", {**body, "userId": "offline-discovery", "deviceId": uuid4().hex}
        )

    data = await _cached("qa-poi", body, call)
    output = []
    # Only native places in cards, never coordinates extracted from answer text.
    for card in data.get("cards", []):
        if isinstance(card, dict):
            payload = card.get("cardData") or {}
            if isinstance(payload, dict):
                output.extend(
                    p for p in payload.get("places", []) if isinstance(p, dict)
                )
    return output


async def web_search(query: str, *, max_results: int = 8) -> list[SearchResult]:
    if not settings.ali_cleversee_api_key:
        return []
    body = dict(
        query=query[:500],
        engineType="CNAuto",
        contents={"mainText": True, "rerankScore": True},
        advancedParams={"numResults": min(max_results, 10)},
    )
    data = await _cached("web", body, lambda: _http("/search/unified", body), ttl=180)
    return [
        SearchResult(
            title=str(p.get("title") or ""),
            url=str(p["link"]),
            content=str(p.get("snippet") or "")[:2000],
            raw_content=str(p.get("mainText") or "")[:16000],
        )
        for p in data.get("pageItems", [])
        if isinstance(p, dict) and p.get("link")
    ]


async def read_page(url: str, *, fresh: bool = False) -> str:
    from app.services.offline.activity_images import _public_url

    if not settings.ali_cleversee_api_key or not await _public_url(url):
        return ""
    body = dict(url=url, maxAge=0)
    data = await _cached(
        "read", body, lambda: _http("/readpage/basic", body), ttl=180, fresh=fresh
    )
    page = data.get("data") or {}
    if not isinstance(page, dict) or page.get("statusCode") != 200:
        return ""
    return str(page.get("text") or page.get("markdown") or "")[:24000]


async def image_search(query: str) -> list[SearchResult]:
    if not settings.ali_cleversee_api_key:
        return []
    body = dict(
        query=query[:500],
        engineType="MultimodalSpeed",
        advancedParams={"numResults": "20"},
    )
    data = await _cached("images", body, lambda: _http("/search/multimodal", body))
    images = [
        dict(
            url=p["imageUrl"],
            source_title=str(p.get("title") or ""),
            source_url=str(p.get("hostPageUrl") or ""),
            description=str(p.get("title") or ""),
        )
        for p in data.get("imageItems", [])
        if isinstance(p, dict) and p.get("imageUrl")
    ]
    return (
        [SearchResult(title=query, url="", content="", query_images=images)]
        if images
        else []
    )
