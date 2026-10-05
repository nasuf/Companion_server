"""Publish the reviewed release through the same versioned service as the Web UI.

Run in the deployed server after code rollout. Default is read-only preflight;
--apply performs manual_save with optimistic locking. Never changes defaults or
existing history. Save stdout as the release receipt. Reruns are idempotent.
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
from app.services.prompting.store import get_prompt_text, update_prompt_text

MANIFEST = Path(__file__).with_name('prompt_releases') / '20261005_offline_quality.json'


def validate_entry(entry: dict) -> None:
    content = entry['content']
    allowed = set(entry['allowed_fields'])
    fields = {field for _, field, _, _ in Formatter().parse(content) if field is not None}
    if not fields <= allowed:
        raise ValueError('Unsupported placeholders: ' + str(fields - allowed))
    content.format(**{field: '验证素材' for field in allowed})


async def run(apply: bool) -> None:
    entries = json.loads(MANIFEST.read_text())['prompts']
    for entry in entries:
        # Web saves trim surrounding whitespace before persisting.
        entry['content'] = entry['content'].strip()
        validate_entry(entry)
    await db.connect()
    try:
        redis = await get_redis()
        snapshots = {}
        # Read all current text and FULL history before any mutation.
        for entry in entries:
            key = entry['key']
            row = await db.prompttemplate.find_unique(where={'key': key})
            versions = await db.prompttemplateversion.find_many(where={'promptKey': key}, order={'createdAt':'asc'})
            if row is None:
                raise RuntimeError(f'{key}: deploy/bootstrap registry before publication')
            current_hash = hashlib.sha256(row.content.encode()).hexdigest()
            if row.content != entry['content'] and current_hash != entry['expected_sha256']:
                raise RuntimeError(f'{key}: current text differs from reviewed snapshot; re-audit required')
            cached = await redis.get(f'prompt_template:{key}')
            if cached is not None and cached != row.content:
                raise RuntimeError(f'{key}: DB/Redis mismatch before publication')
            snapshots[key] = row, versions
            print(json.dumps({'phase':'preflight','key':key,'current_sha256':current_hash,
                              'enabled':row.isEnabled,'history_count':len(versions),
                              'history_actions':[v.changeType for v in versions]},ensure_ascii=False),flush=True)
        if not apply:
            return
        for entry in entries:
            key = entry['key']
            before, history = snapshots[key]
            await update_prompt_text(key,entry['content'],expected_updated_at=before.updatedAt.isoformat())
            after = await db.prompttemplate.find_unique(where={'key':key})
            versions = await db.prompttemplateversion.find_many(where={'promptKey':key},order={'createdAt':'asc'})
            assert after.content == entry['content'] == await redis.get(f'prompt_template:{key}')
            assert after.isEnabled == before.isEnabled and after.defaultContent == before.defaultContent
            by_id = {v.id: v.model_dump(mode='json') for v in versions}
            assert all(by_id[v.id] == v.model_dump(mode='json') for v in history)
            added = [v for v in versions if v.id not in {old.id for old in history}]
            assert len(added) == (0 if before.content == entry['content'] else 1)
            assert all(v.changeType == 'manual_save' and v.content == entry['content'] for v in added)
            if after.isEnabled:
                assert entry['content'] in str(await get_prompt_text(key))
            print(json.dumps({'phase':'published','key':key,'version_ids':[v.id for v in added],
                              'change_type':'manual_save' if added else 'already_published',
                              'enabled':after.isEnabled},ensure_ascii=False),flush=True)
        common = await db.prompttemplate.find_unique(where={'key':'proactive.common_rules'})
        if common.isEnabled:
            for key in ('proactive.silence_plain', 'offline.activity_invite_message', 'offline.arrival_guide',
                        'offline.activity_companion_observe', 'offline.activity_companion_feeling', 'offline.activity_companion_casual'):
                row = await db.prompttemplate.find_unique(where={'key':key})
                if row and row.isEnabled:
                    assert str(await get_prompt_text(key)).startswith(common.content+'\n\n')
        print(json.dumps({'phase':'verified','db_redis_history_hot_path':True}))
    finally:
        await db.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    asyncio.run(run(parser.parse_args().apply))
