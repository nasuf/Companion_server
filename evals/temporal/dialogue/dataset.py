"""Frozen synthetic multi-turn corpus, split before model outputs are inspected."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_cases(root: Path = ROOT) -> tuple[list[dict], dict]:
    raw = (root / 'cases.jsonl').read_bytes()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('version') != 1 or hashlib.sha256(raw).hexdigest() != manifest.get('sha256'):
        raise ValueError('Corpus checksum or schema mismatch')
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(cases) != manifest['cases'] or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Case count or unique ID mismatch')
    if len({c['scenario_family'] for c in cases}) != len(cases):
        raise ValueError('Scenario families must not cross partitions or duplicate cases')
    content_keys = set()
    counts = Counter()
    for case in cases:
        if case['split'] not in {'development', 'holdout'} or case['category'] not in manifest['categories']:
            raise ValueError('Unknown split or category')
        if type(case['critical']) is not bool or not case['criterion'].strip() or not case['message'].strip():
            raise ValueError('Missing criterion or critical classification')
        now = datetime.fromisoformat(case['now'])
        if now.tzinfo is None or not 2 <= len(case['history']) <= 120:
            raise ValueError('Invalid clock or history bound')
        previous = None
        for message in case['history']:
            stamp = datetime.fromisoformat(message['createdAt'])
            if stamp.tzinfo is None or stamp >= now or (previous and stamp <= previous):
                raise ValueError('History must precede the endpoint in order with explicit zones')
            if message['role'] not in {'user', 'assistant'} or not message['content'].strip():
                raise ValueError('Invalid history message')
            previous = stamp
        key = ''.join(m['content'] for m in case['history']) + case['message']
        key = ''.join(key.split())
        if key in content_keys:
            raise ValueError('Duplicate dialogue, possibly split leakage')
        content_keys.add(key)
        counts[case['split'], case['category']] += 1
    if any(counts[split, category] != manifest['per_split_category']
           for split in manifest['split_counts'] for category in manifest['categories']):
        raise ValueError('Unbalanced category or partition')
    return cases, manifest
