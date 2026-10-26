from copy import deepcopy
import pytest
from evals.context_embedding.calibrate import calibrate, build_gate_policy


def pair():
    a = {k: 'same' for k in ('source_files_sha256', 'cases_sha256', 'prompt_snapshot_sha256', 'model', 'model_options', 'judge_protocol', 'cache_protocol', 'prefix_protocol', 'repeats')}
    a.update(started_at='a', runs=[dict(case='case', sample=0, turn=0, replay_id='a', reply='回答', critical=True,
                                      calls=[dict(route='chat.completions', reply_path='main_llm')],
                                      judge={'pass': True}, judge_repeats=[{'pass': True}, {'pass': True}])])
    b = deepcopy(a); b['started_at'] = 'b'; b['runs'][0]['replay_id'] = 'b'
    return a, b


def test_distinguishes_generation_and_same_output_judge_noise():
    a, b = pair()
    b['runs'][0].update(reply='另一个回答', judge={'pass': False}, judge_repeats=[{'pass': False}, {'pass': True}])
    r = calibrate(a, b)
    assert r['status'] == 'measured'
    g = r['routes'][0]
    assert g['generation_verdict_flip_rate'] == 1
    assert g['same_output_judge_flip_rate'] == .5
    assert g['critical_pass_to_fail'] == [('case', 0, 0)]
    assert 'passed' not in r  # Calibration never authorizes a product release.


@pytest.mark.parametrize('field', ['source_files_sha256', 'cases_sha256', 'prompt_snapshot_sha256', 'judge_protocol', 'prefix_protocol'])
def test_mismatched_protocol_rejected(field):
    a, b = pair(); b[field] = 'different'
    with pytest.raises(ValueError, match='identical baseline'):
        calibrate(a, b)


def test_rewarming_and_single_judgment_are_not_repeatability_proof():
    a, b = pair(); b['runs'][0]['replay_id'] = 'a'
    with pytest.raises(ValueError, match='warming'):
        calibrate(a, b)
    b['runs'][0]['replay_id'] = 'b'; b['runs'][0]['judge_repeats'].pop()
    with pytest.raises(ValueError, match='two judgments'):
        calibrate(a, b)


def test_policy_uses_generation_noise_without_adding_judge_noise():
    a,b=pair()
    b['runs'][0]['judge_repeats'][1]['pass']=False
    measured=calibrate(a,b)
    policy=build_gate_policy(measured)
    assert policy['noise_allowances'][0]['max_other_category_drop']==0
    changed=deepcopy(measured); changed['routes'][0]['generation_verdict_flip_rate']=.25
    assert build_gate_policy(changed)['aa_sha256'] != policy['aa_sha256']


@pytest.mark.parametrize('rate', [None,True,-.1,1.1,float('nan'),float('inf')])
def test_policy_rejects_unknown_or_invalid_noise(rate):
    a,b=pair(); measured=calibrate(a,b)
    measured['routes'][0]['generation_verdict_flip_rate']=rate
    with pytest.raises(ValueError,match='flip evidence'):
        build_gate_policy(measured)
