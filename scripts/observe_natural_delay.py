"""Accumulate bounded, private per-message proofs across qualified deployments.

Usage: python scripts/observe_natural_delay.py --state PRIVATE.json --output REPORT.json
       --commit SHA [--commit SHA ...] --identity ~/.ssh/companion_tencent
Commits must be present locally. No production mutation, chat content, or app imports.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

FLOOR = 'db8f60f963e960d688d277915e139139a95d2a25'
LIMIT = 500


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)


def manifest(repo, commit):
    commit = git(repo, 'rev-parse', '--verify', commit + '^{commit}').decode().strip()
    subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', FLOOR, commit], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = git(repo, 'archive', commit, 'app', 'jobs', 'prisma', 'scripts', 'pyproject.toml', 'runtime-constraints.txt')
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        hashes = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                  for m in archive.getmembers() if m.isfile()}
    return commit, hashes


def empty_state():
    return dict(version=1, windows={}, messages={}, events={}, conversations={}, problems=[], sampling_failures=0)


def qualifying_commit(manifests, probe):
    matches = [c for c, hashes in manifests.items() if set(probe['source_files']) == set(hashes)
               and all(probe['source_hashes'].get(path) == value for path, value in hashes.items())]
    if len(matches) != 1:
        raise ValueError('running image source is not uniquely qualified by approved commit manifests')
    return matches[0]


def merge(state, probe, commit):
    """Fail closed and retain previous proofs; polling one log event is not a new start."""
    if probe['truncated']:
        state['problems'].append({'kind': 'evidence_truncated', 'at': probe['sampled_at']})
        return state
    cid = probe['container']
    prior = state['windows'].get(cid)
    if prior and prior['last_anchor'] and prior['last_anchor'] not in probe['log_anchors']:
        state['problems'].append({'kind': 'retained_log_gap', 'at': probe['sampled_at'], 'container': cid})
    if len(state['windows']) >= 64 and not prior:
        state['problems'].append({'kind': 'window_bound_exceeded', 'at': probe['sampled_at']})
        return state
    window = state['windows'].setdefault(cid, dict(image=probe['image'], commit=commit,
         started_at=probe['started_at'], first_observed_at=probe['sampled_at'],
         boundary='Container transition gaps are unobserved, never declared error-free.'))
    if window['image'] != probe['image'] or window['commit'] != commit:
        raise ValueError('immutable container identity changed')
    window.update(last_observed_at=probe['sampled_at'], last_anchor=probe['log_anchors'][-1] if probe['log_anchors'] else None)
    events = dict(state['events'])
    messages = dict(state['messages'])
    for event in probe['samples']:
        events[event['event']] = dict(event, image=probe['image'], commit=commit, container=cid)
        conv = event['conversation_id']
        state['conversations'][conv] = min(state['conversations'].get(conv, probe['started_at']), probe['started_at'])
        messages.setdefault(event['message_id'], dict(conversation_id=conv, status='source_missing'))
    if len(events) > LIMIT or len(messages) > LIMIT or len(state['conversations']) > LIMIT:
        state['problems'].append({'kind': 'accumulation_bound_exceeded', 'at': probe['sampled_at']})
        return state
    queried = {r['id']: r for r in probe['rows']}
    for mid in messages.keys() | queried.keys():
        row = queried.get(mid)
        previous = messages.get(mid, {})
        status = ('source_missing' if not row else 'reply_source_mismatch'
                  if previous.get('conversation_id', row['conversation_id']) != row['conversation_id']
                  else 'replied' if row['reply_id'] else 'pending' if row['age_seconds'] < 300 else 'reply_not_proven')
        messages[mid] = {**previous, **(row or {}), 'status': status, 'last_checked_at': probe['sampled_at']}
        if status not in ('replied', 'pending'):
            problem = dict(kind=status, message_id=mid)
            if problem not in state['problems']:
                state['problems'].append(problem)
    if len(messages) > LIMIT:
        state['problems'].append({'kind': 'accumulation_bound_exceeded', 'at': probe['sampled_at']})
        return state
    state.update(messages=messages, events=events, sampled_at=probe['sampled_at'],
                 queue_conversations=probe['queue_conversations'], sampling_failures=0)
    return state


def distribution(values):
    values = list(values)
    missing = sum(v is None for v in values)
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(unit='seconds', count=0, missing=missing, min=None, p50=None, p90=None, max=None)
    return dict(unit='seconds', count=len(values), missing=missing, min=values[0], p50=values[(len(values)-1)//2],
                p90=values[min(len(values)-1, (9*len(values)+9)//10-1)], max=values[-1])


def report(state):
    def hashed(x):
        return hashlib.sha256(x.encode()).hexdigest()[:16]
    starts = {}
    reasons = {}
    for event in state['events'].values():
        starts[event['message_id']] = starts.get(event['message_id'], 0) + 1
        reasons[event['reason']] = reasons.get(event['reason'], 0) + 1
    items = [dict(message_hash=hashed(mid), conversation_hash=hashed(m['conversation_id']),
                  status=m['status'], generation_starts=starts.get(mid, 0),
                  generation_images=sorted({e['image'] for e in state['events'].values() if e['message_id']==mid}),
                  received_to_reply_seconds=m.get('reply_wait_seconds')) for mid, m in state['messages'].items()]
    duplicate = any(n > 1 for n in starts.values())
    convs = len(state['conversations'])
    status = ('evidence_check_needed' if state['problems'] else 'sampling_failed' if state['sampling_failures']
              else 'duplicate_start_check_needed' if duplicate else 'passed'
              if convs >= 3 and items and all(m['status'] == 'replied' for m in items)
              else 'awaiting_natural_traffic')
    return dict(status=status, sampled_at=state.get('sampled_at'), natural_conversations=convs,
                minimum_conversations=3, qualified_minimum_commit=FLOOR,
                saved_messages=len(items), exact_replies=sum(m['status']=='replied' for m in items),
                messages=items, windows=list(state['windows'].values()),
                due_to_generator_start=distribution(e['due_to_start_seconds'] for e in state['events'].values()),
                received_to_persisted_reply=distribution(m.get('reply_wait_seconds') for m in state['messages'].values()),
                reasons=dict(denominator=len(state['events']), counts=reasons),
                problems=[dict(p, message_id=hashed(p['message_id'])) if 'message_id' in p else p for p in state['problems']],
                consecutive_sampling_failures=state['sampling_failures'],
                queue_conversations=state.get('queue_conversations'),
                limits='Qualified retained log starts only; each selected conversation is checked from its earliest selected container start through the query time, including prior messages after redeploy. Transition gaps are not error-free proof; unsampled conversations are excluded. No whole-site invariant or first-token latency claim.')


def private_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', action='append', required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--host', default='ubuntu@106.52.115.80')
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with open(args.state.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text()) if args.state.exists() else empty_state()
        try:
            manifests = dict(manifest(args.repo, c) for c in args.commit)
            request = dict(source_files=sorted(set().union(*(set(h) for h in manifests.values()))),
                           conversations=state['conversations'], message_ids=list(state['messages']))
            source = Path(__file__).with_name('natural_delay_probe.py').read_text()
            source += '\nprint(json.dumps(collect(' + repr(request) + ')))\n'
            process = subprocess.run(['ssh', '-i', str(args.identity), '-o', 'BatchMode=yes',
                        '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', args.host, 'python3 -'],
                        input=source, text=True, capture_output=True, timeout=150)
            if process.returncode:
                raise RuntimeError('native SSH collection failed (no production mutation)')
            probe = json.loads(process.stdout)
            # Preserve diagnostics even for an incompatible image, but never merge
            # unqualified samples. Raw metadata is private and contains no log text.
            private_json(args.output.with_suffix('.private.json'), probe)
            commit = qualifying_commit(manifests, probe)
            private_json(args.output.with_suffix('.private.json'), dict(probe, qualified_commit=commit))
            merge(state, probe, commit)
        except Exception as exc:
            state['sampling_failures'] += 1
            state['last_sampling_error'] = type(exc).__name__ + ': ' + str(exc)
            state['sampled_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        private_json(args.state, state)
        result = report(state)
        if state.get('last_sampling_error') and state['sampling_failures']:
            result['error'] = state['last_sampling_error']
        private_json(args.output, result)
        print(json.dumps({k:result[k] for k in ('status','natural_conversations','saved_messages','exact_replies')}))
        return 1 if result['status'] in ('sampling_failed', 'evidence_check_needed', 'duplicate_start_check_needed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
