from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.services.chat_links.extraction import extract_link_metadata
from app.services.chat_links.recommendation import (
    _search_endpoint_urls,
    configured_search_provider,
    search_provider_configured,
)


async def _run(query: str, *, require_live: bool) -> int:
    provider = configured_search_provider()
    configured, reason = search_provider_configured()
    if not configured:
        payload = {
            "status": "skipped",
            "provider": provider,
            "reason": reason,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2 if require_live else 0

    urls = await _search_endpoint_urls(query=query)
    if not urls:
        payload = {
            "status": "failed" if require_live else "empty",
            "provider": provider,
            "query": query,
            "results": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if require_live else 0

    metadata = await extract_link_metadata(url=urls[0], shared_text=query)
    payload = {
        "status": "ok",
        "provider": provider,
        "query": query,
        "results": urls,
        "first_card": {
            "platform": metadata.platform,
            "title": metadata.title,
            "status": metadata.status,
            "final_url": metadata.final_url,
            "has_content": bool(metadata.content_text or metadata.summary),
            "has_image": bool(metadata.image_url),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test configured proactive chat link search provider.",
    )
    parser.add_argument(
        "--query",
        default="周末咖啡 分享",
        help="Search query to send to the configured provider.",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Exit non-zero when no live provider is configured or no result is found.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.query, require_live=args.require_live))


if __name__ == "__main__":
    sys.exit(main())
