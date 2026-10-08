"""Readiness must distinguish unfair socket acceptance from a dead worker."""
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

# CLI imports sibling verifier; load it without changing application sys.path.
root = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('verify_server_workers', root / 'verify_server_workers.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
prior = sys.modules.get('verify_server_workers')
sys.modules['verify_server_workers'] = verifier
try:
    spec = importlib.util.spec_from_file_location('worker_probe_harness', root / 'test_server_workers_e2e.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
finally:
    if prior is None:
        del sys.modules['verify_server_workers']
    else:
        sys.modules['verify_server_workers'] = prior


def test_two_responsive_workers_can_serve_separate_bursts(monkeypatch):
    responses = iter([{15}, {409}])
    monkeypatch.setattr(harness, 'serving_workers', lambda _: next(responses))
    monkeypatch.setattr(harness, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None))
    assert harness.wait_expected_pair('owned-test-container', {15, 409}) == {15, 409}


@pytest.mark.parametrize('samples,reason', [([{15}, {700}], 'identity changed'), ([{15}, set()], 'HTTP probe failed')])
def test_replacement_or_unreachable_probe_fails_immediately(monkeypatch, samples, reason):
    responses = iter(samples)
    monkeypatch.setattr(harness, 'serving_workers', lambda _: next(responses))
    monkeypatch.setattr(harness, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None))
    with pytest.raises(AssertionError, match=reason):
        harness.wait_expected_pair('owned-test-container', {15, 409})


def test_alive_but_nonresponding_second_worker_fails_at_deadline(monkeypatch):
    times = iter([0, 0, 2])
    monkeypatch.setattr(harness, 'time', SimpleNamespace(monotonic=lambda: next(times), sleep=lambda _: None))
    monkeypatch.setattr(harness, 'serving_workers', lambda _: {15})
    with pytest.raises(AssertionError, match='did not both respond'):
        harness.wait_expected_pair('owned-test-container', {15, 409}, timeout=1)
