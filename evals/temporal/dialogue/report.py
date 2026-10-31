"""Fail-closed completeness and quality report; repetitions share a scenario cluster."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import hashlib
import math
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evals.temporal.dialogue.dataset import load_cases


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)] if values else None


def summarize(cases: list[dict], rows: list[dict], samples: int) -> dict:
    index = {c['id']: c for c in cases}
    seen = set()
    grouped = defaultdict(list)
    failures = []
    critical = unknown = 0
    latency, tokens = [], 0
    for row in rows:
        key = row['case'], row['sample']
        if row['case'] not in index or type(row['sample']) is not int or not 0 <= row['sample'] < samples or key in seen:
            raise ValueError('Unknown, duplicate, or out-of-range sample')
        seen.add(key)
        verdict = row['judge']['pass']
        if verdict is not None and type(verdict) is not bool:
            raise ValueError('Invalid verdict')
        case = index[row['case']]
        # A judge verdict without an actual answer is never sufficient.
        passed = verdict is True and bool(row.get('reply'))
        grouped[row['case']].append(int(passed))
        if not passed:
            failures.append(dict(case=row['case'], sample=row['sample'], critical=case['critical'],
                                 reason=row['judge'].get('reason')))
            critical += case['critical']
        unknown += verdict is None or not row.get('reply')
        calls = row.get('calls', [])
        durations = [c['duration_seconds'] for c in calls if 'duration_seconds' in c]
        if durations:
            latency.append(sum(durations))
        tokens += sum((c.get('usage') or {}).get('total_tokens', 0) for c in calls)
    expected = len(cases)*samples
    missing = expected-len(seen)
    per_category = {}
    for category in sorted({c['category'] for c in cases}):
        subset = [c for c in cases if c['category']==category]
        passed = sum(sum(grouped[c['id']]) for c in subset)
        denominator = len(subset)*samples
        per_category[category] = dict(passed=passed, expected=denominator, rate=passed/denominator)
    rates = [sum(grouped[c['id']])/samples for c in cases]
    passed = sum(sum(v) for v in grouped.values())
    # Cluster bootstrap: resample scenarios, keep their repeated answers together.
    rng = random.Random(20261020)
    boot = [sum(rng.choices(rates, k=len(rates)))/len(rates) for _ in range(2000)]
    complete = missing == 0 and unknown == 0
    return dict(cases=len(cases), expected=expected, recorded=len(rows), missing=missing,
        unknown=unknown, passed=passed, rate=passed/expected,
        scenario_cluster_ci95=[percentile(boot,.025), percentile(boot,.975)],
        critical_nonpass=critical, per_category=per_category,
        complete=complete, quality_pass=complete and passed/expected >= .95 and critical == 0
            and all(v['rate'] >= .90 for v in per_category.values()),
        provider_reply_and_guard_tokens=tokens, reply_and_guard_p95_seconds=percentile(latency,.95),
        latency_scope='Provider reply and guard calls only, excludes judge and production network/delivery',
        failures=failures)


def load_runs(paths: list[Path]):
    identity = None
    rows = []
    shards = set()
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        current = records[0]['identity']
        seal_path = path.with_suffix('.complete.json')
        if not seal_path.exists():
            raise ValueError('Run lacks a verified source completion seal')
        seal = json.loads(seal_path.read_text())
        if (seal.get('identity') != current or seal.get('records') != len(records)-1
                or seal.get('journal_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Run completion seal mismatch')
        shard = current['shard']
        comparable = {k:v for k,v in current.items() if k!='shard'}
        if shard in shards or not 0 <= shard < current['shards']:
            raise ValueError('Duplicate or invalid shard')
        if identity is not None and identity != comparable:
            raise ValueError('Cannot mix runs with different source, prompts, inputs, or settings')
        shards.add(shard)
        identity = comparable
        rows.extend(records[1:])
    if identity is None:
        raise ValueError('No run files')
    return identity, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journals', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases, manifest = load_cases()
    identity, rows = load_runs(args.journals)
    if manifest['sha256'] != identity['corpus_sha256']:
        raise ValueError('Corpus mismatch')
    result = summarize([c for c in cases if c['split']==identity['split']], rows, identity['samples'])
    result['identity'] = identity
    result['qualification_scope'] = manifest['scope']
    result['full_sample_schedule'] = identity['split']=='holdout' and identity['samples']==5
    result['qualified'] = result['quality_pass'] and result['full_sample_schedule']
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('expected','recorded','passed','missing','unknown','quality_pass','qualified')}))
    raise SystemExit(0 if result['qualified'] else 1)
