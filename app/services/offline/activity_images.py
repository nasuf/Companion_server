"""Source-bound place galleries with bounded refill and shared public caching."""
from __future__ import annotations

import asyncio
import hashlib
import io
import ipaddress
import logging
import socket
import re
from itertools import zip_longest
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

from app.services.offline import activity_media_storage as storage, place_catalog
from app.services.offline.content import canonical_url, place_source_matches
from app.services.offline.image_evidence import page_image_evidence, indexed_image_evidence
from app.services.offline.providers.search import SearchResult, tavily_place_images

logger = logging.getLogger(__name__)
_MAX_BYTES = 8 * 1024 * 1024


def _image_queries(card: dict, city: str) -> list[str]:
    name = str(card.get('location_name') or '').strip()
    address = str(card.get('address') or '').strip()
    return [f'{city} {name} {address} 实景照片', f'{city} {name} 官方 图片'] if name else []


def _is_bad_image_url(url: str) -> bool:
    p = urlsplit(url)
    return not canonical_url(url) or any(word in p.path.lower() for word in
        ('staticmap', '/logo', '/icon', '/sprite', 'placeholder', 'avatar'))


def _source_images(card: dict, sources: list[SearchResult]) -> list[dict]:
    candidates = []
    for source in sources:
        if not place_source_matches(card, source.title, source.content):
            continue
        images = source.images or ([{'url': source.image_url}] if source.image_url else [])
        for image in images:
            url = image.get('url')
            if isinstance(url, str) and not _is_bad_image_url(url):
                candidates.append({**image, 'url': url, 'source_url': source.url,
                                   'source_title': source.title, 'evidence': 'place_page'})
    return candidates


async def persist_activity_images(*, user_id: str, card: dict, city: str,
                                  search_results: list[SearchResult], limit: int = 3) -> list[str]:
    # Personal media and LLM-proposed URLs never enter this public cache.
    card = card if card.get('city') else {**card, 'city': city}
    cached = await place_catalog.load_place(card)
    accepted = []
    for item in (cached or {}).get('images', []):
        key = str(item.get('storage_key') or '')
        if key.startswith('place_') and storage.storage_path(key).is_file():
            accepted.append(item)
    accepted = accepted[:limit]
    if len(accepted) >= limit:
        card['image_provenance'] = accepted
        return [item['local_url'] for item in accepted]
    seen = {item['url'] for item in accepted}
    hashes = {item.get('sha256') for item in accepted}
    perceptual = [int(item['dhash']) for item in accepted if item.get('dhash') is not None]
    try:
        async with asyncio.timeout(35):
            async with httpx.AsyncClient(timeout=8, trust_env=False, follow_redirects=False) as client:
                page_cache: dict[str, dict[str, str]] = {}
                page_slots = asyncio.Semaphore(3)

                async def candidates(source: SearchResult) -> list[dict]:
                    if not place_source_matches(card, source.title, source.content + source.raw_content):
                        return []
                    if re.search(r'效果图|设计方案|规划图|拟建', source.title):
                        return []
                    evidence = indexed_image_evidence(source.raw_content, card['location_name'], source.images, source.url)
                    if not evidence:
                        async with page_slots:
                            if source.url not in page_cache:
                                try:
                                    async with asyncio.timeout(4):
                                        page_cache[source.url] = await page_image_evidence(client, source.url, card['location_name'], _public_url)
                                except TimeoutError:
                                    page_cache[source.url] = {}
                        evidence = page_cache[source.url]
                    return [dict(url=url, source_url=source.url, source_title=source.title, evidence=kind)
                            for url, kind in evidence.items() if not _is_bad_image_url(url)]

                async def collect(sources: list[SearchResult]) -> None:
                    # Apply the window after binding the chosen destination.
                    # A diverse discovery pool can put this place after page 8.
                    bound = [source for source in sources if place_source_matches(
                        card, source.title, source.content + source.raw_content)]
                    groups = await asyncio.gather(*(candidates(source) for source in bound[:8]))
                    attempts = 0
                    # Alternate sources so a blocked image CDN cannot monopolize the budget.
                    for candidate in (item for row in zip_longest(*groups) for item in row if item):
                        if len(accepted) >= limit:
                            break
                        if candidate['url'] in seen:
                            continue
                        seen.add(candidate['url'])
                        if attempts >= 12:
                            break
                        attempts += 1
                        image = await _download_image(client, candidate['url'])
                        if image is None:
                            continue
                        blob, digest, dhash = image
                        if digest in hashes or any((dhash ^ old).bit_count() <= 4 for old in perceptual):
                            continue
                        key = f'place_{digest}.jpg'
                        storage._MEDIA_DIR.mkdir(parents=True, exist_ok=True)
                        # Same content => same bytes across workers; atomic replace prevents partial reads.
                        import os, tempfile
                        with tempfile.NamedTemporaryFile(dir=storage._MEDIA_DIR, delete=False) as tmp:
                            tmp.write(blob)
                            tmp_name = tmp.name
                        os.replace(tmp_name, storage.storage_path(key))
                        accepted.append({**candidate, 'storage_key': key, 'sha256': digest,
                                         'dhash': str(dhash), 'local_url': storage.media_url(key)})
                        hashes.add(digest)
                        perceptual.append(dhash)
                await collect(search_results)
                for query in _image_queries(card, city):
                    if len(accepted) >= limit:
                        break
                    await collect(await tavily_place_images(query))
    except TimeoutError:
        logger.info("[offline-images] gallery budget reached; preserving verified images")
    if accepted:
        await place_catalog.save_place(card, accepted)
    card['image_provenance'] = accepted
    return [item['local_url'] for item in accepted]


async def _public_url(url: str) -> bool:
    try:
        p = urlsplit(url)
        if p.scheme not in {'https', 'http'} or not p.hostname or p.username or p.port not in {None, 80, 443}:
            return False
        infos = await asyncio.get_running_loop().getaddrinfo(p.hostname, p.port or 443, type=socket.SOCK_STREAM)
        return bool(infos) and all(ipaddress.ip_address(info[4][0]).is_global for info in infos)
    except (ValueError, OSError):
        return False


async def _download_image(client: httpx.AsyncClient, url: str) -> tuple[bytes, str, int] | None:
    try:
        for _ in range(4):
            if not await _public_url(url):
                return None
            async with client.stream('GET', url, headers={'accept': 'image/*'}) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers.get('location', '')))
                    continue
                response.raise_for_status()
                if not response.headers.get('content-type', '').startswith('image/'):
                    return None
                blob = bytearray()
                async for chunk in response.aiter_bytes():
                    blob.extend(chunk)
                    if len(blob) > _MAX_BYTES:
                        return None
            with Image.open(io.BytesIO(blob)) as original:
                if min(original.size) < 240 or original.width * original.height > 20_000_000:
                    return None
                picture = ImageOps.exif_transpose(original).convert('RGB')
                small = picture.resize((9, 8)).convert('L')
                thumb = [small.getpixel((x,y)) for y in range(8) for x in range(9)]
                dhash = sum((thumb[y*9+x] > thumb[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                picture.thumbnail((1600, 1600))
                target = io.BytesIO()
                picture.save(target, format='JPEG', quality=85)
                output = target.getvalue()
                return output, hashlib.sha256(output).hexdigest(), dhash
    except Exception:
        logger.debug('[offline-images] rejected or unavailable candidate')
    return None
