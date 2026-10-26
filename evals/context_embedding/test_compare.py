from copy import deepcopy
import pytest
from evals.context_embedding.compare import paired_quality, compare, paired_repetition, select_variant
from evals.context_embedding.calibrate import build_gate_policy


def reports():
    a = {"runs": []}
    b = {"runs": []}
    for n in range(8):
        for sample in range(3):
            row = dict(
                case=str(n),
                sample=sample,
                turn=0,
                category="target",
                critical=False,
                reply='新回答', previous_assistant_reply='旧话题', replay_id=f'replay-{n}-{sample}',
                calls=[{'route': 'chat.completions', 'reply_path': 'tier'}],
                judge={"pass": False},
            )
            a["runs"].append(row)
            row = deepcopy(row)
            row["judge"]["pass"] = True
            b["runs"].append(row)
    return a, b


def test_bootstrap_uses_conversations_and_no_quality_gain_fails():
    a, b = reports()
    q = paired_quality(a, b, ["target"], 100)
    assert q["passed"] and q["categories"][0]["scenario_count"] == 8
    assert q["categories"][0]["paired_cluster_95_ci"] == [1, 1]
    assert q["categories"][0]["turn_pairs"] == 24
    assert q["categories"][0]["baseline_passed"] == 0
    assert q["categories"][0]["candidate_passed"] == 24
    assert not paired_quality(b, b, ["target"], 100)["passed"]


@pytest.mark.parametrize('failures,expected', [(1, False), (2, True), (3, True)])
def test_critical_scene_uses_three_repeat_majority(failures, expected):
    a, b = reports()
    for rows in (a["runs"], b["runs"]):
        for row in rows[:3]:
            row['critical'] = True
            row['judge']['pass'] = True
    for row in b['runs'][:failures]:
        row['judge']['pass'] = False
    assert bool(paired_quality(a, b, ['target'], 100)['critical_regressions']) is expected


def test_critical_baseline_failure_is_not_new_regression_and_incomplete_repeats_rejected():
    a,b=reports()
    for row in a['runs'][:3]+b['runs'][:3]:
        row['critical']=True
    for row in b['runs'][:3]:
        row['judge']['pass']=False
    assert not paired_quality(a,b,['target'],100)['critical_regressions']
    a['runs'].pop(0); b['runs'].pop(0)
    with pytest.raises(ValueError,match='exactly three'):
        paired_quality(a,b,['target'],100)


def register(a,b,rate=0):
    paths={(c['route'],c['reply_path']) for r in a['runs'] for c in r['calls']}
    calibration={'status':'measured', 'routes': [dict(provider_route=route, reply_path=path,
        generation_pairs=96,generation_verdict_flip_rate=rate) for route,path in sorted(paths)],
        'protocol':{k:a[k] for k in ('cases_sha256','prompt_snapshot_sha256','model','judge_protocol','cache_protocol')}}
    policy=build_gate_policy(calibration)
    a['gate_policy']=deepcopy(policy); b['gate_policy']=deepcopy(policy)
    return calibration


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_incomplete_judgements_never_pass(mutation):
    a, b = reports()
    if mutation == "missing":
        b["runs"].pop()
    elif mutation == "duplicate":
        b["runs"].append(b["runs"][0])
    else:
        b["runs"][0]["judge"]["pass"] = None
    with pytest.raises(ValueError):
        paired_quality(a, b, ["target"], 100)


def test_extra_generation_cost_cannot_hide_in_per_call_average():
    from evals.context_embedding.measurement import ARK_MODEL, priced_call

    a, b = reports()
    m = dict(
        status="measured",
        input_tokens=1000,
        output_tokens=10,
        cached_input_tokens=500,
        web_search_calls=0,
    )
    call = dict(
        model=ARK_MODEL,
        route="chat.completions",
        reply_path="main_llm",
        measurement=m,
        price=priced_call(ARK_MODEL, "chat.completions", m),
        ttft_seconds=1, replay_id='synthetic', turn_index=1, cache_eligible=True,
    )
    for report in (a, b):
        report.update(
            cases_sha256="same", prompt_snapshot_sha256='same', cache_protocol='same', prefix_protocol='same',
            model=ARK_MODEL,
            repeats=3,
            model_options={"temperature": 0.5},
            judge_protocol={"model": "fixture", "instruction_sha256": "fixed"},
        )
        for row in report["runs"]:
            row["calls"] = [deepcopy(call)]
    calibration = register(a,b)
    assert compare(a, b, ["target"], calibration)["passed"]
    for row in b["runs"]:
        row["calls"].append(deepcopy(call))
    result = compare(a, b, ["target"], calibration)
    assert not result["passed"]
    assert (
        "normalized_cost_cny_over_10_percent" in result["resource_gates"][0]["reasons"]
    )


def test_changed_judge_cannot_be_compared_as_product_quality_gain():
    a, b = reports()
    for report in (a, b):
        report.update(cases_sha256="same", model="same", repeats=3, model_options={})
    a["judge_protocol"] = {"model": "old"}
    b["judge_protocol"] = {"model": "new"}
    with pytest.raises(ValueError, match="judge_protocol"):
        compare(a, b, ["target"])
    del b["judge_protocol"]
    with pytest.raises(ValueError, match="Missing evaluation protocol"):
        compare(a, b, ["target"])


def test_main_path_only_requires_non_regression():
    a,b=reports()
    for r in a['runs']+b['runs']:
        r['calls'][0]['reply_path']='main_llm'
        r['judge']['pass']=True
    q=paired_quality(a,b,['target'],100)
    assert q['passed'] and q['categories'][0]['criterion']=='non_regression'
    b['runs'][0]['judge']['pass']=False
    assert not paired_quality(a,b,['target'],100)['passed']


def test_actual_route_change_cannot_hide_regression():
    a,b=reports()
    b['runs'][0]['calls'][0]['reply_path']='main_llm'
    with pytest.raises(ValueError,match='Actual route changed'):
        paired_quality(a,b,['target'],100)


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('latency,expected', [
    (1.1, None), (1.1001, 'over_10_percent'),
    (None, 'missing'), (-1, 'missing'), (float('nan'), 'missing'),
    (float('inf'), 'missing'), (True, 'missing'),
])
def test_approved_search_completion_and_streaming_ttft_gate(search, latency, expected):
    from evals.context_embedding.measurement import ARK_MODEL, priced_call
    a,b=reports()
    route='responses.web_search' if search else 'chat.completions'
    metric='time_to_complete_seconds' if search else 'ttft_seconds'
    usage=dict(status='measured',input_tokens=1000,output_tokens=10,cached_input_tokens=500,web_search_calls=int(search))
    for report,value in [(a,1),(b,latency)]:
        report.update(cases_sha256='same',prompt_snapshot_sha256='same',cache_protocol='same',prefix_protocol='same',
                      model=ARK_MODEL,repeats=3,model_options={},judge_protocol={'model':'fixture'})
        for row in report['runs']:
            row['calls']=[dict(model=ARK_MODEL,route=route,reply_path='main_llm',measurement=usage,
                price=priced_call(ARK_MODEL,route,usage),replay_id='synthetic',turn_index=1,
                cache_eligible=True,**{metric:value})]
    calibration=register(a,b)
    result=compare(a,b,['target'],calibration)
    assert result['passed'] is (expected is None)
    gate=result['resource_gates'][0]
    assert gate['latency_metric']==('p90_nonstream_complete_seconds' if search else 'p90_ttft_seconds')
    if expected:
        assert any(expected in reason for reason in gate['reasons'])


def test_noise_allowance_is_exact_path_fixed_and_not_applied_to_main():
    a,b=reports()
    for row in a['runs'][:3]+b['runs'][:3]:
        row['category']='other'; row['judge']['pass']=True
    b['runs'][0]['judge']['pass']=False
    policy={'noise_allowances':[dict(provider_route='chat.completions',reply_path='tier',max_other_category_drop=1/3)]}
    assert paired_quality(a,b,['target'],100,policy=policy)['passed']
    policy['noise_allowances'][0]['max_other_category_drop']=1/3-.001
    assert not paired_quality(a,b,['target'],100,policy=policy)['passed']
    policy['noise_allowances'][0]['provider_route']='responses.web_search'
    with pytest.raises(ValueError,match='actual-path'):
        paired_quality(a,b,['target'],100,policy=policy)


def test_repetition_regression_fails_even_when_quality_improves():
    a,b=reports()
    assert paired_repetition(a,b)['passed']
    b['runs'][0]['reply']='旧话题'
    assert not paired_repetition(a,b)['passed']
    del b['runs'][0]['previous_assistant_reply']
    with pytest.raises(ValueError,match='Missing previous'):
        paired_repetition(a,b)


def state_reports(count):
    from evals.context_embedding.measurement import ARK_MODEL, priced_call
    a,b=reports()
    for report in (a,b):
        report.update(cases_sha256='same',prompt_snapshot_sha256='same',cache_protocol='same',prefix_protocol='same',
                      source_files_sha256={'file':'hash'},model=ARK_MODEL,repeats=3,model_options={},judge_protocol={'model':'fixture'})
        report['runs']=[deepcopy(report['runs'][i % 24]) for i in range(count)]
        usage=dict(status='measured',input_tokens=1000,output_tokens=10,cached_input_tokens=0,web_search_calls=0)
        for i,row in enumerate(report['runs']):
            row.update(case=str(i),sample=0,replay_id=f'{id(report)}-{i}')
            row['calls']=[dict(model=ARK_MODEL,route='chat.completions',reply_path='current_state_shortcircuit',
                measurement=usage,price=priced_call(ARK_MODEL,'chat.completions',usage),ttft_seconds=1)]
    return a,b


@pytest.mark.parametrize('count,passed', [(35,False),(36,True)])
def test_state_latency_needs_36_calls_in_each_arm(count,passed):
    a,b=state_reports(count)
    calibration=register(a,b)
    result=compare(a,b,['target'],calibration)
    assert result['passed'] is passed
    assert result['resource_gates'][0]['latency_population']['candidate_samples']==count


def test_aa_latency_is_deduplicated_and_cannot_pad_candidate_or_fake_distribution():
    from evals.context_embedding.calibrate import latency_observations
    a,b=state_reports(35)
    calibration=register(a,b)
    calibration['latency_observations']=latency_observations(a)*2
    calibration['latency_protocol']={k:a[k] for k in ('source_files_sha256','prompt_snapshot_sha256','model','model_options','cache_protocol','prefix_protocol')}
    a['gate_policy']=b['gate_policy']=build_gate_policy(calibration)
    result=compare(a,b,['target'],calibration)
    population=result['resource_gates'][0]['latency_population']
    assert population['baseline_samples']==population['candidate_samples']==35
    assert not result['passed']
    extra=deepcopy(calibration['latency_observations'][0]); extra['identity'][0]='extra'; extra['ttft_seconds']=20
    calibration['latency_observations']=[dict(extra, identity=[f'extra-{n}',*extra['identity'][1:]]) for n in range(10)]
    a['gate_policy']=b['gate_policy']=build_gate_policy(calibration)
    result=compare(a,b,['target'],calibration)
    population=result['resource_gates'][0]['latency_population']
    assert population['baseline_samples']==45 and population['baseline_p90_seconds']==20
    assert not result['passed']  # candidate still has only 35 samples


def test_policy_must_be_registered_before_reports_and_remain_unchanged():
    a,b=state_reports(36); calibration=register(a,b)
    b['gate_policy']['state_latency_minimum_per_arm']=12
    with pytest.raises(ValueError,match='preregistered'):
        compare(a,b,['target'],calibration)


def test_unmatched_aa_is_excluded_but_36_fresh_calls_can_qualify():
    from evals.context_embedding.calibrate import latency_observations
    a,b=state_reports(36); calibration=register(a,b)
    calibration['latency_observations']=latency_observations(a)
    calibration['latency_protocol']={'source_files_sha256':{'file':'old'}}
    a['gate_policy']=b['gate_policy']=build_gate_policy(calibration)
    result=compare(a,b,['target'],calibration)
    assert result['passed']
    population=result['resource_gates'][0]['latency_population']
    assert population['aa_supplement_status']=='excluded_protocol_mismatch'
    assert population['baseline_samples']==36


def test_missing_actual_route_aa_cannot_pass_even_if_target_improves():
    a,b=state_reports(36); calibration=register(a,b)
    calibration['routes'][0]['reply_path']='tier'
    a['gate_policy']=b['gate_policy']=build_gate_policy(calibration)
    with pytest.raises(ValueError,match='actual-path'):
        compare(a,b,['target'],calibration)


def test_variant_selection_defaults_a_and_requires_b_complete_gate_and_both_metrics():
    before,a=state_reports(36)
    for row in before['runs']:
        row['reply']='旧话题'
    for row in a['runs'][:2]:
        row['reply']='旧话题'
    calibration=register(before,a)
    b=deepcopy(a)
    assert select_variant(before,a,b,['target'],calibration)['selected']=='A'
    b['runs'][0]['reply']='新回答'
    assert select_variant(before,a,b,['target'],calibration)['selected']=='B'
    b['runs'][0]['calls'][0]['ttft_seconds']=float('nan')
    result=select_variant(before,a,b,['target'],calibration)
    assert result['selected']=='A' and result['selected_passed']
