"""Version gate only: does not replace image, DB, receipt or queue qualification."""
import argparse
import json
from pathlib import Path
import subprocess

NORMAL_FLOOR = '410f2ba12cc64639b4a5998e93488e841ae75956'
DELAY_FLOOR = 'db8f60f963e960d688d277915e139139a95d2a25'


def ancestor(repo, base, target):
    result = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', base, target],
                            capture_output=True, timeout=10)
    if result.returncode not in (0, 1):
        raise ValueError('revision unavailable; fetch and qualify the exact image source first')
    return result.returncode == 0


def qualify(repo, target, *, emergency=False, delay_enabled=None, queued=None, inflight=None):
    if ancestor(repo, NORMAL_FLOOR, target):
        return {'version_gate': 'passed', 'mode': 'normal', 'minimum_commit': NORMAL_FLOOR}
    if not emergency:
        raise ValueError('normal rollback requires 410f2ba1 or descendant')
    # Even db8..410 has lease/cancellation defects. Use the stricter emergency rule
    # for all pre-410 images; before db8 the delay path is explicitly unsafe.
    if delay_enabled is not False or queued != 0 or inflight != 0:
        raise ValueError('old-image emergency requires admin delay OFF and verified empty queue/inflight')
    return {'version_gate': 'passed', 'mode': 'emergency_delay_disabled',
            'older_than_delay_fix': not ancestor(repo, DELAY_FLOOR, target),
            'warning': 'Keep delay disabled. Must separately prove DB/receipts/image compatibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--target-commit', required=True)
    parser.add_argument('--emergency', action='store_true')
    parser.add_argument('--delay-enabled', choices=('true', 'false'))
    parser.add_argument('--queued', type=int)
    parser.add_argument('--inflight', type=int)
    args = parser.parse_args()
    try:
        result = qualify(args.repo, args.target_commit, emergency=args.emergency,
                         delay_enabled=None if args.delay_enabled is None else args.delay_enabled == 'true',
                         queued=args.queued, inflight=args.inflight)
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
