"""Credential/response-shape probe; never stores activities or sends chat messages.

Uses at most one external request per invocation, with a 30-second timeout.
Image search and Ark chat tools are distinct products; a model Key is not
silently retried as a search Key in the production recommendation path.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.db import db
from app.services.runtime_config import load_caches
from app.services.llm.models import _chat_model_name


async def probe(args) -> dict:
    key = (settings.ark_api_key if args.key_source == 'ark'
           else os.environ.get('DOUBAO_SEARCH_API_KEY', '')).strip()
    if not key:
        return {'route': args.route, 'status': 'missing_credential'}
    if args.route == 'doubao-image':
        endpoint = 'https://open.feedcoopapi.com/search_api/web_search'
        payload = dict(Query=args.query[:100], SearchType='image', Count=5)
    else:
        endpoint = settings.ark_base_url.rstrip('/') + '/responses'
        payload = dict(model=_chat_model_name(), input=[dict(role='user', content=args.query)],
                       tools=[dict(type='web_search', **({'sources': ['doubao']} if args.route == 'ark-doubao' else {}))],
                       tool_choice='required', max_tool_calls=1, max_output_tokens=600)
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(endpoint, json=payload, headers={'Authorization': 'Bearer ' + key})
        data = response.json()
        if not isinstance(data, dict):
            return dict(route=args.route, http=response.status_code, status='invalid_response_shape')
        error = data.get('error') or (data.get('ResponseMetadata') or {}).get('Error') or {}
        output = data.get('output') or []
        # Only report schema/status, never model prose, signed URLs, keys or
        # request headers. HTTP 200 can still contain a provider auth error.
        return dict(route=args.route, http=response.status_code,
                    error_code=error.get('code') or error.get('Code'),
                    result_keys=list((data.get('Result') or {}).keys()),
                    output_types=[item.get('type') for item in output if isinstance(item, dict)])
    except (httpx.HTTPError, ValueError) as exc:
        return dict(route=args.route, status='request_failed', error_type=type(exc).__name__)


async def main(args):
    await db.connect()
    try:
        await load_caches()
        print(json.dumps(await probe(args), ensure_ascii=False))
    finally:
        await db.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=['doubao-image', 'ark-doubao', 'ark-plugin'], required=True)
    parser.add_argument('--key-source', choices=['ark', 'search'], default='ark')
    parser.add_argument('--query', default='搜索镇江博物馆的实景照片，返回图片原图链接及来源网页，不要生成图片。')
    asyncio.run(main(parser.parse_args()))
