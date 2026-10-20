"""Evaluation integrity: frozen inputs, exact denominators, incomplete != passed."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'evals/temporal_dialogue'


def load(name):
    spec = importlib.util.spec_from_file_location('temporal_dialogue_'+name, ROOT / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    # Evaluation scripts are also runnable directly; isolate their sibling imports.
    sys.path.insert(0,str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT))
    return module


def test_frozen_split_is_balanced_and_distinct():
    cases, manifest = load('dataset').load_cases()
    assert len(cases)==240 and manifest['split_counts']=={'development':120,'holdout':120}
    assert len({c['scenario_family'] for c in cases})==240


def test_dataset_rejects_tampered_inputs(tmp_path):
    for name in ('manifest.json','cases.jsonl'):
        (tmp_path/name).write_bytes((ROOT/name).read_bytes())
    with (tmp_path/'cases.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        load('dataset').load_cases(tmp_path)


def test_missing_unknown_and_critical_failures_cannot_pass():
    report = load('report')
    cases=[dict(id='one',category='identity',critical=True),dict(id='two',category='gap',critical=False)]
    good=dict(case='one',sample=0,reply='answer',judge={'pass':True})
    result=report.summarize(cases,[good],2)
    assert result['expected']==4 and result['missing']==3 and not result['quality_pass']
    bad=dict(case='one',sample=1,reply='answer',judge={'pass':None})
    result=report.summarize(cases,[good,bad],2)
    assert result['unknown']==1 and result['critical_nonpass']==1
    with pytest.raises(ValueError, match='Duplicate|duplicate'):
        report.summarize(cases,[good,good],2)
    good['judge']['pass']=1
    with pytest.raises(ValueError, match='verdict'):
        report.summarize(cases,[good],2)


def test_resume_preserves_failed_samples_and_rejects_mismatched_source(tmp_path):
    run=load('run')
    path=tmp_path/'run.jsonl'
    identity={'source_sha256':'old'}
    record=dict(case='a',sample=0,judge={'pass':False})
    path.write_text(json.dumps({'identity':identity})+'\n'+json.dumps(record)+'\n')
    assert run.read_journal(path,identity)[('a',0)]['judge']['pass'] is False
    with pytest.raises(ValueError,match='identity'):
        run.read_journal(path,{'source_sha256':'new'})
    with path.open('a') as f:f.write('{')
    with pytest.raises(json.JSONDecodeError):run.read_journal(path,identity)


def test_aggregator_rejects_mixed_prompt_versions(tmp_path):
    files=[]
    for i in range(2):
        path=tmp_path/f'{i}.jsonl'
        path.write_text(json.dumps({'identity':{'shard':i,'shards':2,'prompt_sha256':str(i)}})+'\n')
        identity={'shard':i,'shards':2,'prompt_sha256':str(i)}
        path.with_suffix('.complete.json').write_text(json.dumps({'identity':identity,'records':0,
            'journal_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
        files.append(path)
    with pytest.raises(ValueError,match='mix'):
        load('report').load_runs(files)


def test_report_rejects_unsealed_run(tmp_path):
    path=tmp_path/'unsealed.jsonl'
    path.write_text(json.dumps({'identity':{'shard':0,'shards':1}})+'\n')
    with pytest.raises(ValueError, match='completion seal'):
        load('report').load_runs([path])
