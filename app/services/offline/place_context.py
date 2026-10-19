"""Bounded, place-bound text enrichment; photo matching stays stricter."""
from __future__ import annotations

import asyncio
import re

from app.services.offline.content import canonical_url, is_collection_title, normalized, place_source_matches
from app.services.offline.activity_discovery import source_text
from app.services.offline.providers import cleversee
from app.services.offline.providers.search import SearchResult


def context_matches(card: dict, title: str, body: str) -> bool:
    if is_collection_title(title) or re.search(r"搜索结果|商户列表|店铺列表", title):
        return False
    name = str(card.get("location_name") or "")
    parts = re.fullmatch(r"(.+?)[（(]([^()（）]+)[)）]", name)
    branches = re.findall(r"[（(]([^()（）]{1,20}店)[)）]", title)
    if parts and any(normalized(b).removesuffix("店") != normalized(parts[2]).removesuffix("店") for b in branches):
        return False
    if place_source_matches(card, title, body):
        return True
    if not parts:
        return False
    core, branch = normalized(parts[1]), normalized(parts[2]).removesuffix("店")
    heading, excerpt = normalized(title), normalized(body)
    combined = heading + excerpt[:2200]
    city = normalized(card.get("city")).removesuffix("市")
    # Brand, city and the selected branch/address must all agree. A title for
    # another branch cannot be rescued by a footer listing every branch.
    address = normalized(card.get("address"))
    return bool(len(core) >= 2 and (core in heading or 0 <= excerpt.find(core) < 1200) and city and city in combined
                and ((len(branch) >= 2 and branch in combined) or (len(address) >= 6 and address in combined)))


async def load_context(card: dict, city: str, usable) -> list[SearchResult]:
    metadata = card.setdefault("discovery_metadata", {})
    if metadata.get("kind") != "place":
        return []
    name, address = str(card.get("location_name") or ""), str(card.get("address") or "")
    core = re.split(r"[（(]", name, maxsplit=1)[0]
    queries = [f"{city} {name} 介绍 体验", f"{city} {core} {address} 介绍"]
    kept: dict[str, SearchResult] = {}
    searches = reads = 0
    states = []
    status = "empty"
    try:
        async with asyncio.timeout(14):
            for query in dict.fromkeys(queries):
                searches += 1
                pages = await cleversee.web_search(query, max_results=4)
                states.append(getattr(pages, "source_status", "available" if pages else "empty"))
                for page in pages[:4]:
                    url = canonical_url(page.url)
                    if (not url or url in kept or normalized(core) not in normalized(page.title + source_text(page)[:1200])):
                        continue
                    body = source_text(page)
                    if reads < 2 and len(page.raw_content) < 120:
                        reads += 1
                        full = await cleversee.read_page(url)
                        if full:
                            page = SearchResult(title=page.title, url=url, content=page.content, raw_content=full[:12000])
                            body = source_text(page)
                    if usable(page, city) and context_matches(card, page.title, body):
                        kept[url] = page
                    if len(kept) == 2:
                        break
                if kept and any(len(source_text(p)) >= 100 for p in kept.values()):
                    break
            status = ("available" if any(len(source_text(p)) >= 100 for p in kept.values()) else "sparse") if kept else (
                "unavailable" if "failed" in states else "not_configured" if "not_configured" in states
                else "no_matching_sources" if "available" in states else "empty")
    except TimeoutError:
        status = "partial_timeout" if kept else "timeout"
    except Exception:
        status = "partial_failure" if kept else "unavailable"
    metadata.update(context_source_status=status, context_source_count=len(kept),
                    context_search_count=searches, context_read_count=reads)
    return list(kept.values())[:2]
