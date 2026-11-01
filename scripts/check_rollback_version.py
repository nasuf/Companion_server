"""Version gate only: does not replace image, DB, receipt or queue qualification."""
import argparse
import json
from pathlib import Path
import subprocess

# Routine floor: emoji-only filler replies stay paused (probability 0 from the DB).
NORMAL_FLOOR = 'c302aa613b4092011aba95179968e3ef562b37b1'
# Lease retries and post-dequeue-only cancellation notifications.
DELAY_SAFE_FLOOR = '410f2ba12cc64639b4a5998e93488e841ae75956'
DELAY_FLOOR = 'db8f60f963e960d688d277915e139139a95d2a25'


def ancestor(repo, base, target):
    result = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', base, target],
                            capture_output=True, timeout=10)
    if result.returncode not in (0, 1):
        raise ValueError('revision unavailable; fetch and qualify the exact image source first')
    return result.returncode == 0


def supports_topic_policy(repo, target):
    """Check source capability, not image qualification or state truth."""
    for path, marker in (
        ('app/services/time_domain/topic_policy.py', 'topic_policy_v1'),
        ('app/services/chat/graph_phases.py', 'record_turn_preferences'),
        ('app/services/proactive/emit.py', 'resolve_policy'),
    ):
        result = subprocess.run(['git', '-C', str(repo), 'show', f'{target}:{path}'],
                                capture_output=True, timeout=10, text=True)
        if result.returncode != 0 or marker not in result.stdout:
            return False
    return True


def supports_fact_changes(repo, target):
    """TIME05: retracted sources stay fenced and confirmed changes stay current."""
    for path, marker in (
        ('app/services/memory/evidence.py', 'memory_source_retractions'),
        ('app/services/time_domain/validity_context.py', 'memory_source_retractions'),
        ('app/services/memory/interaction/contradiction_write.py', 'persist_change'),
    ):
        result = subprocess.run(['git', '-C', str(repo), 'show', f'{target}:{path}'],
                                capture_output=True, timeout=10, text=True)
        if result.returncode != 0 or marker not in result.stdout:
            return False
    return True


def qualify(repo, target, *, emergency=False, accept_filler_shortcut=False,
            delay_enabled=None, queued=None, inflight=None, topic_policy_rows=None,
            fact_change_rows=None):
    if not supports_topic_policy(repo, target) and topic_policy_rows != 0:
        raise ValueError('topic-unaware target requires verified topic-policy row count 0; '
                         'existing or unknown preferences forbid this rollback, including emergency')
    if not supports_fact_changes(repo, target) and fact_change_rows != 0:
        raise ValueError('fact-change-unaware target requires verified fact-change row count 0; '
                         'existing or unknown corrections forbid this rollback, including emergency')
    if ancestor(repo, NORMAL_FLOOR, target):
        return {'version_gate': 'passed', 'mode': 'normal', 'minimum_commit': NORMAL_FLOOR}
    # Older images ignore the stored probability and hardcode a 30% emoji-only
    # shortcut, which the product owner paused. Schema compatibility does not
    # preserve that decision, so the operator must accept the regression.
    if not accept_filler_shortcut:
        raise ValueError('targets before c302aa61 restore the 30% emoji-only filler shortcut; '
                         'record that regression and pass --accept-filler-shortcut')
    if ancestor(repo, DELAY_SAFE_FLOOR, target):
        return {'version_gate': 'passed', 'mode': 'filler_shortcut_restored',
                'minimum_commit': DELAY_SAFE_FLOOR,
                'warning': 'Emoji-only filler replies resume at 30% until rolled forward.'}
    if not emergency:
        raise ValueError('rollback before 410f2ba1 requires --emergency')
    # Even db8..410 has lease/cancellation defects. Use the stricter emergency rule
    # for all pre-410 images; before db8 the delay path is explicitly unsafe.
    if delay_enabled is not False or queued != 0 or inflight != 0:
        raise ValueError('old-image emergency requires admin delay OFF and verified empty queue/inflight')
    return {'version_gate': 'passed', 'mode': 'emergency_delay_disabled',
            'older_than_delay_fix': not ancestor(repo, DELAY_FLOOR, target),
            'filler_shortcut_restored': True,
            'warning': 'Keep delay disabled. Must separately prove DB/receipts/image compatibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--target-commit', required=True)
    parser.add_argument('--emergency', action='store_true')
    parser.add_argument('--accept-filler-shortcut', action='store_true',
                        help='Acknowledge that a pre-c302aa61 image restores 30%% emoji-only replies')
    parser.add_argument('--delay-enabled', choices=('true', 'false'))
    parser.add_argument('--queued', type=int)
    parser.add_argument('--inflight', type=int)
    parser.add_argument('--topic-policy-rows', type=int,
                        help='Fresh native read-only count of conversation_topics, including reopened rows')
    parser.add_argument('--fact-change-rows', type=int,
                        help='Fresh native read-only count of memory_fact_changes plus '
                             'memory_source_retractions')
    args = parser.parse_args()
    try:
        result = qualify(args.repo, args.target_commit, emergency=args.emergency,
                         accept_filler_shortcut=args.accept_filler_shortcut,
                         delay_enabled=None if args.delay_enabled is None else args.delay_enabled == 'true',
                         queued=args.queued, inflight=args.inflight, topic_policy_rows=args.topic_policy_rows,
                         fact_change_rows=args.fact_change_rows)
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
