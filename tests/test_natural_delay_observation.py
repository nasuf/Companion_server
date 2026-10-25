"""Cross-deployment evidence must not turn missing or repeated starts into success."""
import copy
import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load('observe_natural_delay')
probe_module = load('natural_delay_probe')


def uid(n):
    return f'00000000-0000-0000-0000-{n:012d}'


def snapshot(n, *, replied=True, container=None):
    container = container or str(n)
    return dict(sampled_at='2026-10-25T07:00:00+00:00', container=container,
                image='image-' + container, started_at='2026-10-25T06:00:00+00:00',
                truncated=False, log_anchors=['anchor'], queue_conversations=0,
                samples=[dict(event=container+':event', conversation_id=uid(n), message_id=uid(n+100),
                              due_to_start_seconds=0.2, reason='idle')],
                rows=[dict(id=uid(n+100), conversation_id=uid(n), age_seconds=30,
                           reply_id=uid(n+200) if replied else None, reply_wait_seconds=5 if replied else None)])


def accumulate_three():
    state = observer.empty_state()
    rows = []
    for n in range(1, 4):
        snap = snapshot(n)
        rows += snap['rows']
        snap['rows'] = copy.deepcopy(rows)
        observer.merge(state, snap, 'sha' + str(n))
    return state


def test_three_conversations_accumulate_across_distinct_images():
    state = accumulate_three()
    result = observer.report(state)
    assert result['status'] == 'passed'
    assert result['natural_conversations'] == result['exact_replies'] == 3
    assert len(result['windows']) == 3
    assert result['due_to_generator_start']['count'] == 3
    assert result['reasons'] == {'denominator': 3, 'counts': {'idle': 3}}


def test_polling_same_start_is_not_duplicate_generation():
    state = observer.empty_state()
    snap = snapshot(1)
    observer.merge(state, snap, 'sha1')
    observer.merge(state, snap, 'sha1')
    assert len(state['events']) == 1
    assert observer.report(state)['messages'][0]['generation_starts'] == 1


def test_same_message_started_again_on_new_image_is_failure():
    state = accumulate_three()
    snap = snapshot(1, container='four')
    snap['rows'] = [dict(m, id=mid) for mid,m in state['messages'].items()]
    observer.merge(state, snap, 'sha4')
    assert observer.report(state)['status'] == 'duplicate_start_check_needed'


def test_same_conversation_across_deploys_counts_once():
    state = observer.empty_state()
    for n in range(3):
        snap = snapshot(1, container=str(n))
        observer.merge(state, snap, 'sha'+str(n))
    assert observer.report(state)['natural_conversations'] == 1


def test_unreplied_old_message_is_still_queried_and_blocks_pass():
    state = accumulate_three()
    snap = snapshot(4)
    snap['rows'] += [dict(m, id=mid) for mid,m in state['messages'].items()]
    snap['rows'][1].update(reply_id=None, age_seconds=301, reply_wait_seconds=None)
    observer.merge(state, snap, 'sha4')
    assert observer.report(state)['status'] == 'evidence_check_needed'
    assert any(p['kind']=='reply_not_proven' for p in state['problems'])


def test_transient_pending_cannot_pass_but_can_later_be_proven():
    state = accumulate_three()
    snap = snapshot(4, replied=False)
    snap['rows'] += [dict(m, id=mid) for mid,m in state['messages'].items()]
    observer.merge(state, snap, 'sha4')
    assert observer.report(state)['status'] == 'awaiting_natural_traffic'
    snap['rows'][0].update(reply_id=uid(999), reply_wait_seconds=35)
    observer.merge(state, snap, 'sha4')
    assert observer.report(state)['status'] == 'passed'


@pytest.mark.parametrize('failure', ['truncation', 'missing_source', 'wrong_conversation', 'log_gap'])
def test_failures_cannot_be_cleared_by_next_deployment(failure):
    state = accumulate_three()
    snap = snapshot(3)
    snap['rows'] = [dict(m, id=mid) for mid,m in state['messages'].items()]
    if failure == 'truncation':
        snap['truncated'] = True
    elif failure == 'missing_source':
        snap['rows'].pop(0)
    elif failure == 'wrong_conversation':
        snap['rows'][0]['conversation_id'] = uid(99)
    else:
        snap['log_anchors'] = ['rotated']
    observer.merge(state, snap, 'sha3')
    assert observer.report(state)['status'] == 'evidence_check_needed'
    next_snap = snapshot(4)
    next_snap['rows'] += [dict(m, id=mid) for mid,m in state['messages'].items()]
    observer.merge(state, next_snap, 'sha4')
    assert observer.report(state)['status'] == 'evidence_check_needed'


def test_empty_and_two_conversations_never_pass():
    state = observer.empty_state()
    assert observer.report(state)['status'] == 'awaiting_natural_traffic'
    for n in (1,2):
        snap = snapshot(n)
        snap['rows'] += [dict(m,id=mid) for mid,m in state['messages'].items()]
        observer.merge(state,snap,'sha'+str(n))
    assert observer.report(state)['status'] == 'awaiting_natural_traffic'


def test_accumulation_bound_is_failure_not_silent_eviction():
    state = accumulate_three()
    with patch.object(observer, 'LIMIT', 2):
        observer.merge(state, snapshot(4), 'sha4')
    assert observer.report(state)['status'] == 'evidence_check_needed'
    assert len(state['messages']) == 3


def test_private_state_atomic_and_report_contains_no_raw_message_ids(tmp_path):
    state = accumulate_three()
    path = tmp_path / 'private.json'
    observer.private_json(path,state)
    assert path.stat().st_mode & 0o777 == 0o600
    result = str(observer.report(state))
    assert uid(101) not in result


def test_sql_rechecks_all_saved_messages_from_earliest_window_and_exact_receipts():
    sql = probe_module.reply_query({uid(1):'2026-10-25T04:12:16Z'}, [uid(101)])
    assert 'BEGIN READ ONLY' in sql
    assert 't.user_message_ids ? m.id' in sql
    assert "a.role='assistant' AND a.conversation_id=m.conversation_id" in sql
    assert 'LIMIT 501' in sql
    assert 'm.created_at>=to_timestamp' in sql


@pytest.mark.parametrize('conversations,ids', [({'bad': '2026-10-25'}, []), ({},["';DELETE"]),
                                            ({uid(1):'not a date'}, []), ({},[uid(1)]*501)])
def test_sql_rejects_unbounded_or_invalid_input(conversations, ids):
    with pytest.raises(ValueError):
        probe_module.reply_query(conversations, ids)


def test_qualification_requires_git_ancestry():
    with patch.object(observer,'git',return_value=b'oldsha\n'), patch.object(observer.subprocess,'run',
            side_effect=subprocess.CalledProcessError(1,['git'])):
        with pytest.raises(subprocess.CalledProcessError):
            observer.manifest(ROOT,'oldsha')


def test_sampling_failure_cannot_leave_previous_pass_visible():
    state = accumulate_three()
    state['sampling_failures'] = 1
    assert observer.report(state)['status'] == 'sampling_failed'


@pytest.mark.parametrize('hashes,files', [({'app/a.py':'wrong'}, ['app/a.py']),
                                       ({'app/a.py':'ok'}, ['app/a.py','app/new.py']),
                                       ({}, [])])
def test_changed_missing_or_extra_image_source_is_not_qualified(hashes, files):
    with pytest.raises(ValueError):
        observer.qualifying_commit({'sha':{'app/a.py':'ok'}}, {'source_files':files,'source_hashes':hashes})


def test_exact_source_qualification_and_ambiguous_versions():
    probe = {'source_files':['app/a.py'],'source_hashes':{'app/a.py':'ok'}}
    assert observer.qualifying_commit({'sha':probe['source_hashes']},probe) == 'sha'
    with pytest.raises(ValueError):
        observer.qualifying_commit({'sha':probe['source_hashes'],'other':probe['source_hashes']},probe)


def test_missing_latency_is_reported_not_zero():
    result = observer.distribution([None,2])
    assert result['count']==1 and result['missing']==1 and result['min']==2
