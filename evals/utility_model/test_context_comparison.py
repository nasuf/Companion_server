from pathlib import Path
import json

from evals.utility_model.context_comparison import (
    cases, history, old_context, prepare, render_context, score, summarize, synthetic_cases,
)


def test_corpus_declares_actual_multiturn_and_budget():
    corpus = cases()
    synthetic = synthetic_cases()
    assert len(synthetic) == 24
    assert len(corpus) == 111
    assert all(sum(row['role'] == 'user' for row in case.history) == 5 for case in synthetic)
    assert len({case.id for case in corpus}) == len(corpus)
    assert sum(9 if c.group == 'synthetic_multi_turn' else 3 for c in corpus) == 477


def test_old3_drops_old_anchor_but_tight5_preserves_it():
    case = next(case for case in synthetic_cases() if case.id == 'referent_film')
    assert '星际穿越' not in old_context(case.history)
    assert '星际穿越' in render_context(case.history, 'tight')
    assert '[10-26 09:00]' in render_context(case.history, 'tight')


def test_caps_and_tail_preserve_latest_assistant_question():
    rows = history(*[('长用户消息' * 200, '长AI消息' * 200 + '你困了吗？') for _ in range(5)])
    for variant, limit in [('old',900), ('wide',3000), ('tight',1500)]:
        text = render_context(rows, variant)
        assert len(text) <= limit
        assert text.endswith('你困了吗？')


def test_query_grounding_is_separate_from_level():
    result = score('memory.relevance', '{"level":"中","enhanced_query":"那个乐队"}', '中', ['痛仰'])
    assert result['correct'] is True
    assert result['query_grounded'] is False


def test_prepare_is_frozen_and_independent_of_api(tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps([{'key':'intent.unified','content':'{context}\n{user_message}'},
                                    {'key':'memory.relevance','content':'{context}\n{message}'}]))
    manifest = tmp_path / 'manifest.json'
    prepare(snapshot, manifest)
    value = json.loads(manifest.read_text())
    assert value['calls_max'] == 477
    assert all(job['prompt_sha256'] for job in value['jobs'])
    import pytest
    with pytest.raises(SystemExit, match='Refusing'):
        prepare(snapshot, manifest)


def test_summary_keeps_missing_cache_unknown_and_errors_in_denominator(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'jobs':[{'job_id':'one'},{'job_id':'two'}], 'selection_policy':{}}))
    source = tmp_path / 'raw.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in [
        {'job_id':'one','case_id':'x','task':'memory.relevance','group':'synthetic_multi_turn',
         'variant':'tight','correct':True,'query_grounded':True,'input_tokens':1000,
         'output_tokens':10,'cached_tokens':None,'latency_seconds':.3},
        {'job_id':'two','case_id':'x','task':'memory.relevance','group':'synthetic_multi_turn',
         'variant':'tight','error_type':'RateLimitError','latency_seconds':.3},
    ]))
    target = tmp_path / 'summary.json'
    summarize(manifest,source,target)
    result = json.loads(target.read_text())
    row = result['by_task_group_variant'][0]
    assert result['status'] == 'incomplete_or_failed'
    assert row['accuracy_over_all_calls'] == .5
    assert row['cache_missing_calls'] == 1
    assert row['estimated_cny']['lower'] < row['estimated_cny']['upper']


def test_refinement_preserves_all_cases_and_old_prompt(tmp_path):
    original = [{'key':'intent.unified','content':'INTENT {context}\n{user_message}'},
                {'key':'memory.relevance','content':'ORIGINAL {context}\n{message}'}]
    revised = [original[0], {'key':'memory.relevance','content':'CLARIFIED {context}\n{message}'}]
    before, after, manifest = (tmp_path / name for name in ('before.json','after.json','manifest.json'))
    before.write_text(json.dumps(original))
    after.write_text(json.dumps(revised))
    prepare(before, manifest, candidate_snapshot=after, all_repeats=3, variants=('old','tight'))
    value = json.loads(manifest.read_text())
    assert value['calls_max'] == 666
    assert len({job['case']['id'] for job in value['jobs']}) == 111
    for job in value['jobs']:
        expected_prefix = ('INTENT' if job['case']['task'] == 'intent.unified' else
                           'ORIGINAL' if job['variant'] == 'old' else 'CLARIFIED')
        assert job['prompt'].startswith(expected_prefix)
    revised[0] = {'key':'intent.unified','content':'unauthorized intent revision'}
    after.write_text(json.dumps(revised))
    import pytest
    with pytest.raises(SystemExit, match='outside'):
        prepare(before, tmp_path / 'denied.json', candidate_snapshot=after, all_repeats=3, variants=('old','tight'))


def test_authorized_intent_experiment_adds_guards_and_keeps_old_prompt(tmp_path):
    original = [{'key':'intent.unified','content':'INTENT {context}\n{user_message}'},
                {'key':'memory.relevance','content':'ORIGINAL {context}\n{message}'}]
    revised = [{'key':'intent.unified','content':'CLARIFIED INTENT {context}\n{user_message}'}, original[1]]
    before, after, manifest = (tmp_path / name for name in ('before.json','after.json','manifest.json'))
    before.write_text(json.dumps(original))
    after.write_text(json.dumps(revised))
    prepare(before, manifest, candidate_snapshot=after, all_repeats=3, variants=('old','tight'),
            allow_intent_change=True)
    value = json.loads(manifest.read_text())
    ids = {job['case']['id'] for job in value['jobs']}
    assert value['intent_prompt_changed'] is True and len(ids) == 113
    assert {'ack_unrelated_question', 'own_sleep_plan'} <= ids
    for job in value['jobs']:
        if job['case']['task'] == 'intent.unified':
            assert job['prompt'].startswith('INTENT' if job['variant'] == 'old' else 'CLARIFIED INTENT')
