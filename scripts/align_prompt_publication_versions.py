"""Publish unchanged, unverified current text via the guarded Web save path.

Default preflight only. --manifest contains reviewed keys and SHA-256 hashes;
--apply creates actual manual_save publications, never fabricates old history.
Reruns skip content which already has a proven Web publication.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from string import Formatter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import db
from app.redis_client import get_redis
from app.services.prompting import store


def validate_current(entry, row):
    if hashlib.sha256(row.content.encode()).hexdigest() != entry['expected_sha256']:
        raise ValueError(f'{row.key}: current text differs from reviewed content')
    fields = {f for _, f, _, _ in Formatter().parse(row.content) if f is not None}
    if fields != set(entry['allowed_fields']):
        raise ValueError(f'{row.key}: unexpected placeholders')
    row.content.format(**{f: '验证素材' for f in fields})


async def align(entries, *, apply=False):
    if not entries or len({e['key'] for e in entries}) != len(entries):
        raise ValueError('Manifest must contain distinct, reviewed keys')
    redis = await get_redis()
    snapshots = {}
    # Complete history and consistent content identity before any write.
    async with db.tx() as tx:
        await tx.execute_raw('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        for entry in entries:
            key = entry['key']
            if key not in store.PROMPT_DEFINITION_MAP:
                raise ValueError(f'{key}: unknown registry key')
            row = await tx.prompttemplate.find_unique(where={'key': key})
            if row is None:
                raise ValueError(f'{key}: missing template')
            validate_current(entry, row)
            history, numbers = await store._publication_data(tx, key)
            meta = store._publication_metadata(row, history, numbers)
            if meta['content_version_type'] == 'default':
                raise ValueError(f'{key}: default template does not need alignment')
            snapshots[key] = row, history, meta
    receipts = []
    for entry in entries:
        key = entry['key']
        row, history, meta = snapshots[key]
        if not apply or meta['content_version_type'] == 'web':
            receipts.append(dict(key=key, action='already_published' if meta['content_version_type']=='web' else 'preflight', **meta))
            continue
        result = await store.update_prompt_text(key, row.content, publish_version=True,
            expected_updated_at=row.updatedAt.isoformat(), expected_revision=row.revision)
        after = await db.prompttemplate.find_unique(where={'key': key})
        if not result['cache_synced'] or await redis.get(store._redis_key(key)) != row.content:
            raise RuntimeError(f'{key}: committed publication requires cache recovery')
        assert after.content == row.content and after.isEnabled == row.isEnabled
        assert after.defaultContent == row.defaultContent
        versions = await db.prompttemplateversion.find_many(where={'promptKey': key})
        by_id = {v.id: v for v in versions}
        # Background evaluation may fill only the newly created record.
        assert all(by_id[v.id].model_dump() == v.model_dump() for v in history)
        added = [v for v in versions if v.id not in {old.id for old in history}]
        assert len(added)==1 and added[0].changeType=='manual_save' and added[0].content==row.content
        assert result['content_version_type']=='web' and result['content_version_id']==added[0].id
        if row.isEnabled:
            assert row.content in str(await store.get_prompt_text(key))
        receipts.append(dict(key=key,action='published',web_version=result['web_version'],content_version_id=added[0].id))
    return receipts


async def main(manifest, apply):
    entries = json.loads(manifest.read_text())['prompts']
    await db.connect()
    try:
        for receipt in await align(entries, apply=apply):
            print(json.dumps(receipt, ensure_ascii=False))
    finally:
        await db.disconnect()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    asyncio.run(main(args.manifest,args.apply))
