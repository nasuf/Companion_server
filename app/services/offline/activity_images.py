"""Source-bound place galleries with bounded refill and shared public caching."""
from __future__ import annotations

import asyncio
import hashlib
import io
import ipaddress
import logging
import socket
import re
import os
import tempfile
from collections import Counter
from itertools import zip_longest
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

from app.config import settings
from app.services.offline import activity_media_storage as storage, place_catalog
from app.services.offline.content import canonical_url, photo_source_matches, normalized, place_name_in_text
from app.services.offline.image_evidence import page_image_evidence, indexed_image_evidence, query_image_evidence, native_image_evidence
from app.services.offline.providers.search import SearchResult, tavily_place_images

logger = logging.getLogger(__name__)
_MAX_BYTES = 8 * 1024 * 1024
_GALLERY_BUDGET_S = 35.0


def _image_queries(card: dict, city: str) -> list[str]:
    name = str(card.get('location_name') or '').strip()
    address = str(card.get('address') or '').strip()
    return [f'{city} {name} 实景照片', f'{city} {name} {address} 图片'] if name else []


def _is_bad_image_url(url: str) -> bool:
    p = urlsplit(url)
    return not canonical_url(url) or any(word in p.path.lower() for word in
        ('staticmap', '/logo', '/icon', '/sprite', 'placeholder', 'avatar'))


def _source_images(card: dict, sources: list[SearchResult]) -> list[dict]:
    candidates = []
    for source in sources:
        if not photo_source_matches(card, source.title, source.content, source.url):
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
        if card.get('native_poi_id') and item.get('poi_id') != card['native_poi_id']:
            continue  # Old inferred albums need native identity revalidation.
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
    stats = Counter()
    try:
        async with asyncio.timeout(_GALLERY_BUDGET_S):
            async with httpx.AsyncClient(timeout=8, trust_env=False, follow_redirects=False) as client:
                page_cache: dict[str, dict[str, str]] = {}
                page_slots = asyncio.Semaphore(3)

                async def candidates(source: SearchResult) -> list[dict]:
                    if any(native_image_evidence(card, item) for item in source.images):
                        return []
                    if not photo_source_matches(card, source.title, source.content + source.raw_content, source.url):
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
                    bound = [source for source in sources if photo_source_matches(
                        card, source.title, source.content + source.raw_content, source.url)]
                    groups = await asyncio.gather(*(candidates(source) for source in bound[:8]))
                    native = [{**item, 'source_url': source.url, 'source_title': source.title}
                              for source in sources for item in source.images
                              if native_image_evidence(card, item) and not _is_bad_image_url(str(item.get('url') or ''))]
                    # Don't spend page-read time validating a native POI album.
                    groups.insert(0, native)
                    query_candidates = {}
                    counted_query_urls = set()
                    async def query_bound(item: dict) -> bool:
                        if query_image_evidence(card, item):
                            return True
                        # Short image titles commonly omit the city. Verify the
                        # actual host page and selected branch/address instead.
                        url = canonical_url(item.get('source_url'))
                        if not card.get('native_poi_id') or not url or not place_name_in_text(card['location_name'], item.get('source_title')):
                            return False
                        from app.services.offline.providers.cleversee import read_page
                        text = await read_page(url)
                        address = normalized(card.get('address'))
                        return bool(len(address) >= 4 and address in normalized(text)
                                    and photo_source_matches(card, item.get('source_title', ''), text, url))
                    for source in sources:
                        for item in source.query_images[:8]:
                            if item['url'] in query_candidates:
                                continue
                            if item['url'] not in counted_query_urls:
                                stats['query_candidates'] += 1
                                counted_query_urls.add(item['url'])
                            if not _is_bad_image_url(item['url']) and await query_bound(item):
                                # A title identifies the image document; it does
                                # not give us permission to invent its page URL.
                                query_candidates[item['url']] = {**item, 'evidence': 'image_search_document',
                                    **({'poi_id': card['native_poi_id']} if card.get('native_poi_id') else {})}
                    groups.append(list(query_candidates.values()))
                    stats['bound_pages'] += len(bound)
                    stats['eligible_candidates'] += sum(len(group) for group in groups)
                    attempts = 0
                    # Alternate sources so a blocked image CDN cannot monopolize the budget.
                    ordered = (item for row in zip_longest(*groups) for item in row if item)
                    while len(accepted) < limit and attempts < 18:
                        batch = []
                        capacity = min(3, 18 - attempts)
                        for candidate in ordered:
                            if candidate['url'] in seen:
                                continue
                            seen.add(candidate['url'])
                            batch.append(candidate)
                            attempts += 1
                            if len(batch) >= capacity:
                                break
                        if not batch:
                            break
                        # Bounded parallel downloads let slow CDNs share one
                        # timeout window without monopolizing gallery refill.
                        async def download(candidate):
                            return candidate, await _download_image(client, candidate['url'])
                        tasks = [asyncio.create_task(download(c)) for c in batch]
                        try:
                            # Persist each completed photo immediately: a deadline
                            # must not discard fast downloads waiting on a peer.
                            for completed in asyncio.as_completed(tasks):
                                candidate, image = await completed
                                stats['downloads'] += 1
                                if image is None:
                                    stats['unavailable'] += 1
                                    continue
                                blob, digest, dhash = image
                                if card.get('native_poi_id'):
                                    from app.services.offline.image_quality import scene_photo
                                    if not await scene_photo(blob, digest):
                                        stats['non_photo'] += 1
                                        continue
                                if digest in hashes or any((dhash ^ old).bit_count() <= 4 for old in perceptual):
                                    stats['duplicates'] += 1
                                    continue
                                if len(accepted) >= limit:
                                    break
                                key = f'place_{digest}.jpg'
                                storage._MEDIA_DIR.mkdir(parents=True, exist_ok=True)
                                with tempfile.NamedTemporaryFile(dir=storage._MEDIA_DIR, delete=False) as tmp:
                                    tmp.write(blob)
                                    tmp_name = tmp.name
                                try:
                                    os.replace(tmp_name, storage.storage_path(key))
                                finally:
                                    if os.path.exists(tmp_name):
                                        os.unlink(tmp_name)
                                accepted.append({**candidate, 'storage_key': key, 'sha256': digest,
                                                 'dhash': str(dhash), 'local_url': storage.media_url(key)})
                                hashes.add(digest)
                                perceptual.append(dhash)
                                if len(accepted) >= limit:
                                    break
                        finally:
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                await collect(search_results)
                for query in _image_queries(card, city):
                    if len(accepted) >= limit:
                        break
                    if card.get('native_poi_id'):
                        from app.services.offline.providers.cleversee import image_search
                        found = await image_search(query)
                        if not found and settings.offline_tavily_fallback:
                            found = await tavily_place_images(query)
                        await collect(found)
                    else:
                        await collect(await tavily_place_images(query))
    except TimeoutError:
        logger.info("[offline-images] gallery budget reached; preserving verified images")
    if accepted:
        await place_catalog.save_place(card, accepted)
    card['image_provenance'] = accepted
    logger.info('[offline-images] gallery accepted=%s stages=%s', len(accepted), dict(stats))
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
