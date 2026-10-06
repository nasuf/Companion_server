"""Repair legacy public galleries; default dry run, --apply requires a backup path.

Never reads or modifies user-uploaded media. Unverifiable images are retired;
bounded source-bound discovery may supply replacements. Activity and historical
chat-card cover updates are atomic, guarded against concurrent gallery changes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import db
from app.services.offline.activity_images import persist_activity_images
from app.services.offline.repository import activity_from_row


async def run(args):
    await db.connect()
    try:
        rows = await db.query_raw("""SELECT * FROM offline_activity_recommendations
            WHERE status NOT IN ('cancelled','expired') AND ($1::boolean OR image_urls <> '[]'::jsonb)
            AND ($2::boolean = FALSE OR image_urls = '[]'::jsonb)
            AND NOT search_sources @> '[{"kind":"image"}]'::jsonb ORDER BY created_at""", getattr(args, "include_empty", False) or getattr(args, "fill_only", False), getattr(args, "fill_only", False))
        print(json.dumps({'legacy_galleries': len(rows), 'apply': args.apply}), flush=True)
        if not args.apply:
            return
        if not args.backup:
            raise ValueError('--backup is required')
        # Exclusive create prevents overwriting the rollback snapshot on reruns.
        with os.fdopen(os.open(args.backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as backup:
            json.dump(rows, backup, ensure_ascii=False, default=str)
        semaphore = asyncio.Semaphore(3)

        async def repair(row):
            async with semaphore:
                card = activity_from_row(row)
                try:
                    urls = await asyncio.wait_for(persist_activity_images(
                        user_id=card['user_id'], card=card, city=card.get('city') or '',
                        search_results=[]), timeout=45)
                except Exception:
                    urls = []  # Never retain an unverified legacy cover after a failed refill.
                if getattr(args, 'fill_only', False) and not urls:
                    print(json.dumps({'activity_id':card['id'],'images':0,'updated':False}), flush=True)
                    return
                sources = card['search_sources'] + [dict(item, kind='image') for item in card.get('image_provenance', [])]
                async with db.tx() as tx:
                    updated = await tx.query_raw('''UPDATE offline_activity_recommendations
                        SET image_urls=$1::jsonb, search_sources=$2::jsonb, updated_at=CURRENT_TIMESTAMP
                        WHERE id=$3 AND image_urls=$4::jsonb RETURNING id''',
                        json.dumps(urls), json.dumps(sources), card['id'], json.dumps(card['image_urls']))
                    if updated:
                        await tx.execute_raw('''UPDATE messages SET metadata=jsonb_set(metadata,
                            '{component_card,payload,image_url}', $1::jsonb)
                            WHERE metadata->'component_card'->>'type'='offline_activity'
                            AND metadata->'component_card'->'payload'->>'activity_id'=$2''',
                            json.dumps(urls[0] if urls else None), card['id'])
                print(json.dumps({'activity_id':card['id'],'images':len(urls),'updated':bool(updated)}), flush=True)
        await asyncio.gather(*(repair(row) for row in rows))
    finally:
        await db.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup')
    parser.add_argument('--include-empty', action='store_true', help='Retry previously unverified empty galleries')
    parser.add_argument('--fill-only', action='store_true', help='Fill empty galleries only; preserve all existing photos and skip failed discovery')
    asyncio.run(run(parser.parse_args()))
