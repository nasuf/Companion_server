"""The live qualification harness must reject a silently degraded fixture."""
import logging

import pytest

from evals.temporal_awareness.run_incident_eval import _LocationFailures, validate_location_evidence
from tests.g02_harness_support import configure_pair


def test_qualification_cli_can_run_by_file_path_outside_checkout(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "evals/temporal_awareness/run_incident_eval.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--prompt-snapshot" in result.stdout
    assert "--cases-json" in result.stdout


def test_time_context_uses_production_renderer_with_frozen_clock(monkeypatch):
    from datetime import datetime, timezone
    from app.services.chat import orchestrator as chat
    from evals.temporal_awareness.run_incident_eval import configure_time_context
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 10, 23, 9, 32, tzinfo=timezone.utc)
    configure_time_context(monkeypatch, chat, io)
    assert chat.build_time_context() == "当前时间：2026年10月23日 17时 星期五"


@pytest.mark.asyncio
async def test_fixture_reads_real_location_loader_with_correct_clock_and_ownership(monkeypatch):
    from app.services.chat import orchestrator as chat
    io = configure_pair(monkeypatch)
    io.agent.city = '浙江省 杭州市 西湖区'
    context = await chat.load_location_context(io.agent, 'u-1', 'w-1')
    assert io.location_queries.await_count == 2
    assert context.ai_city == '杭州市' and context.user_city == '南京市'
    assert context.age_days == 0 and not context.diagnostics()['user_stale']
    assert context.relation == 'different'
    assert context.web_search_location()['city'] == '南京市'
    validate_location_evidence('normal', {'location_context': context.diagnostics()}, 2, [])


@pytest.mark.asyncio
async def test_fixture_db_failure_is_not_qualified_as_success(monkeypatch):
    from app.services.chat import orchestrator as chat
    from app.services.chat import location_context
    io = configure_pair(monkeypatch)
    io.location_queries.side_effect = ConnectionError('synthetic unavailable')
    logger = logging.getLogger(location_context.__name__)
    failures = _LocationFailures()
    logger.addHandler(failures)
    try:
        result = await chat.load_location_context(io.agent, 'u-1', 'w-1')
        assert result.user_source == 'unknown'
        with pytest.raises(RuntimeError, match='failed during qualification'):
            validate_location_evidence('normal', {}, io.location_queries.await_count, failures.failures)
    finally:
        logger.removeHandler(failures)


def test_missing_reads_rejected_but_current_state_short_circuit_is_explicit():
    with pytest.raises(RuntimeError, match='not exercised'):
        validate_location_evidence('normal', {}, 0, [])
    validate_location_evidence('current_state', {}, 0, [])
    with pytest.raises(RuntimeError, match='not exercised'):
        validate_location_evidence('current_state', {'location_context': {'user_source': 'device'}}, 0, [])
